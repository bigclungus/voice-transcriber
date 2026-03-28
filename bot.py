#!/usr/bin/env python3
"""Discord voice transcription bot.

Joins a voice channel, captures per-user audio, runs Whisper on buffered
chunks, and posts speaker-labelled transcriptions to a text channel.

Uses py-cord (discord.py fork) for voice receive support.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import discord
from discord.ext import commands
from dotenv import load_dotenv

import transcriber

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-18s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice-bot")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ["BOT_TOKEN"]
TRANSCRIPT_CHANNEL_ID = int(os.environ["TRANSCRIPT_CHANNEL_ID"])
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
CHUNK_DURATION = int(os.environ.get("CHUNK_DURATION", "8"))

transcriber.set_model(WHISPER_MODEL)

DB_PATH = Path(__file__).parent / "transcripts.db"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            speaker TEXT NOT NULL,
            speaker_id TEXT NOT NULL,
            text TEXT NOT NULL,
            timestamp TEXT DEFAULT (datetime('now')),
            audio_duration_ms INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            guild_id TEXT NOT NULL,
            voice_channel TEXT NOT NULL,
            text_channel_id TEXT NOT NULL,
            started_at TEXT DEFAULT (datetime('now')),
            ended_at TEXT
        )
        """
    )
    conn.commit()
    return conn


db_lock = Lock()
db = init_db()

# ---------------------------------------------------------------------------
# Audio buffer — collects per-user PCM and flushes when chunk is ready
# ---------------------------------------------------------------------------

# Discord sends 48kHz 16-bit stereo PCM.
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # bytes per sample per channel
BYTES_PER_SEC = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH  # 192000
CHUNK_BYTES = CHUNK_DURATION * BYTES_PER_SEC


class UserAudioBuffer:
    """Thread-safe per-user PCM accumulator."""

    def __init__(self, user: discord.User | discord.Member):
        self.user = user
        self.lock = Lock()
        self._buf = bytearray()

    def write(self, data: bytes) -> bytes | None:
        """Append PCM data. Returns a full chunk (bytes) when threshold is reached, else None."""
        with self.lock:
            self._buf.extend(data)
            if len(self._buf) >= CHUNK_BYTES:
                chunk = bytes(self._buf[:CHUNK_BYTES])
                self._buf = self._buf[CHUNK_BYTES:]
                return chunk
        return None

    def flush(self) -> bytes | None:
        """Return whatever PCM remains (for session end)."""
        with self.lock:
            if len(self._buf) > BYTES_PER_SEC:  # only if >1s of audio
                data = bytes(self._buf)
                self._buf.clear()
                return data
            self._buf.clear()
            return None


# ---------------------------------------------------------------------------
# Custom AudioSink for py-cord voice receive
# ---------------------------------------------------------------------------


class TranscriptionSink(discord.sinks.Sink):
    """Py-cord audio sink that buffers per-user PCM and dispatches to Whisper."""

    def __init__(self, bot_instance: "TranscriberBot", session_id: str):
        super().__init__()
        self.bot_instance = bot_instance
        self.session_id = session_id
        self.buffers: dict[int, UserAudioBuffer] = {}
        self._user_map: dict[int, discord.User | discord.Member] = {}

    @discord.sinks.Filters.container
    def write(self, data: bytes, user: int):
        """Called by py-cord with PCM data for a specific user ID."""
        if user not in self.buffers:
            member = self._user_map.get(user)
            if member is None:
                # Try to resolve from guild later; use placeholder
                member = None
            self.buffers[user] = UserAudioBuffer(member)

        chunk = self.buffers[user].write(data)
        if chunk is not None:
            member = self._user_map.get(user)
            display = member.display_name if member else f"User#{user}"
            uid = str(user)
            asyncio.run_coroutine_threadsafe(
                self.bot_instance.process_chunk(chunk, display, uid, self.session_id),
                self.bot_instance.bot.loop,
            )

    def cleanup(self):
        """Flush remaining buffers on disconnect."""
        for uid, buf in self.buffers.items():
            remaining = buf.flush()
            if remaining:
                member = self._user_map.get(uid)
                display = member.display_name if member else f"User#{uid}"
                asyncio.run_coroutine_threadsafe(
                    self.bot_instance.process_chunk(remaining, display, str(uid), self.session_id),
                    self.bot_instance.bot.loop,
                )

    def register_user(self, user_id: int, member: discord.User | discord.Member):
        self._user_map[user_id] = member
        if user_id in self.buffers:
            self.buffers[user_id].user = member


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


class TranscriberBot:
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.members = True

        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.session_id: str | None = None
        self.sink: TranscriptionSink | None = None
        self.transcript_channel: discord.TextChannel | None = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="whisper")

        self._register_commands()
        self._register_events()

    def _register_events(self):
        @self.bot.event
        async def on_ready():
            log.info("Logged in as %s (id=%s)", self.bot.user, self.bot.user.id)
            channel = self.bot.get_channel(TRANSCRIPT_CHANNEL_ID)
            if channel:
                self.transcript_channel = channel
                log.info("Transcript channel: #%s", channel.name)
            else:
                log.warning("Could not find transcript channel %s", TRANSCRIPT_CHANNEL_ID)

        @self.bot.event
        async def on_voice_state_update(member: discord.Member, before, after):
            # Register user mapping when someone joins the voice channel
            if self.sink and after.channel:
                self.sink.register_user(member.id, member)

    def _register_commands(self):
        @self.bot.command(name="join")
        async def join(ctx: commands.Context):
            """Join the voice channel the user is in and start transcribing."""
            if ctx.author.voice is None or ctx.author.voice.channel is None:
                await ctx.send("You need to be in a voice channel first.")
                return

            vc_channel = ctx.author.voice.channel

            if ctx.voice_client and ctx.voice_client.is_connected():
                await ctx.send("Already connected to a voice channel. Use `!leave` first.")
                return

            # Connect
            try:
                vc = await vc_channel.connect()
            except Exception as e:
                await ctx.send(f"Failed to join voice channel: {e}")
                log.error("Failed to join voice: %s", e, exc_info=True)
                return

            # Set up session
            self.session_id = f"session-{uuid.uuid4().hex[:12]}"
            self.sink = TranscriptionSink(self, self.session_id)

            # Register all current members
            for member in vc_channel.members:
                if not member.bot:
                    self.sink.register_user(member.id, member)

            # Record session in DB
            with db_lock:
                db.execute(
                    "INSERT INTO sessions (session_id, guild_id, voice_channel, text_channel_id) VALUES (?, ?, ?, ?)",
                    (self.session_id, str(ctx.guild.id), vc_channel.name, str(ctx.channel.id)),
                )
                db.commit()

            # Resolve transcript channel (prefer config, fall back to command channel)
            self.transcript_channel = self.bot.get_channel(TRANSCRIPT_CHANNEL_ID) or ctx.channel

            # Start recording
            vc.start_recording(self.sink, self._recording_finished, ctx.channel)

            await ctx.send(
                f"Joined **{vc_channel.name}** and started transcribing.\n"
                f"Session: `{self.session_id}`\n"
                f"Transcriptions will appear in <#{self.transcript_channel.id}>.\n"
                f"Use `!leave` to stop."
            )
            log.info("Started recording in %s (session %s)", vc_channel.name, self.session_id)

        @self.bot.command(name="leave")
        async def leave(ctx: commands.Context):
            """Stop transcribing and leave the voice channel."""
            if not ctx.voice_client or not ctx.voice_client.is_connected():
                await ctx.send("Not connected to a voice channel.")
                return

            session_id = self.session_id

            # Stop recording — this triggers cleanup and _recording_finished
            ctx.voice_client.stop_recording()
            await ctx.voice_client.disconnect()

            # Mark session ended
            if session_id:
                with db_lock:
                    db.execute(
                        "UPDATE sessions SET ended_at = datetime('now') WHERE session_id = ?",
                        (session_id,),
                    )
                    db.commit()

            # Count transcript lines
            count = 0
            if session_id:
                with db_lock:
                    row = db.execute(
                        "SELECT COUNT(*) FROM transcripts WHERE session_id = ?", (session_id,)
                    ).fetchone()
                    count = row[0] if row else 0

            await ctx.send(
                f"Stopped transcribing and disconnected.\n"
                f"Session `{session_id}` saved with **{count}** transcript entries."
            )
            self.session_id = None
            self.sink = None
            log.info("Stopped recording (session %s, %d entries)", session_id, count)

        @self.bot.command(name="status")
        async def status(ctx: commands.Context):
            """Show current transcription status."""
            if not ctx.voice_client or not ctx.voice_client.is_connected():
                await ctx.send("Not currently transcribing.")
                return

            vc = ctx.voice_client
            channel_name = vc.channel.name if vc.channel else "unknown"
            user_count = len(self.sink.buffers) if self.sink else 0

            count = 0
            if self.session_id:
                with db_lock:
                    row = db.execute(
                        "SELECT COUNT(*) FROM transcripts WHERE session_id = ?",
                        (self.session_id,),
                    ).fetchone()
                    count = row[0] if row else 0

            await ctx.send(
                f"**Transcription active**\n"
                f"Channel: {channel_name}\n"
                f"Session: `{self.session_id}`\n"
                f"Users detected: {user_count}\n"
                f"Transcripts so far: {count}\n"
                f"Chunk duration: {CHUNK_DURATION}s | Model: {WHISPER_MODEL}"
            )

    async def _recording_finished(self, sink: discord.sinks.Sink, channel: discord.TextChannel):
        """Callback when recording stops (via !leave or disconnect)."""
        log.info("Recording finished callback fired.")

    async def process_chunk(self, pcm: bytes, speaker: str, speaker_id: str, session_id: str):
        """Run Whisper on a PCM chunk in the thread pool, then post the result."""
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                transcriber.transcribe_pcm,
                pcm,
            )
        except Exception:
            log.error("Whisper transcription failed for %s", speaker, exc_info=True)
            return

        text = result["text"]
        duration_ms = result["duration_ms"]

        if not text:
            return

        # Store in DB
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with db_lock:
            db.execute(
                "INSERT INTO transcripts (session_id, speaker, speaker_id, text, timestamp, audio_duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, speaker, speaker_id, text, now, duration_ms),
            )
            db.commit()

        # Post to text channel
        ts = datetime.datetime.utcnow().strftime("%H:%M:%S")
        msg = f"`[{ts}]` **{speaker}**: {text}"
        if self.transcript_channel:
            try:
                await self.transcript_channel.send(msg)
            except Exception:
                log.error("Failed to post transcript to channel", exc_info=True)

        log.info("[%s] %s: %s", ts, speaker, text[:80])

    def run(self):
        self.bot.run(BOT_TOKEN)


if __name__ == "__main__":
    bot = TranscriberBot()
    bot.run()
