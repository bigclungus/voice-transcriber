# voice-transcriber

Discord voice transcription bot. Joins a voice channel, captures per-user audio streams, runs Whisper on buffered chunks, and posts speaker-labelled transcriptions to a text channel. Full transcripts are stored in SQLite.

## Requirements

- Python 3.12+
- FFmpeg (for audio processing)
- A Discord bot token with voice permissions

## Setup

```bash
cd /mnt/data/voice-transcriber
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your bot token and channel IDs
```

The Whisper small model (~500MB) downloads automatically on first use.

## Configuration (.env)

| Variable | Required | Description |
|---|---|---|
| `BOT_TOKEN` | Yes | Discord bot token |
| `TRANSCRIPT_CHANNEL_ID` | Yes | Text channel ID for posting transcriptions |
| `WHISPER_MODEL` | No | Whisper model size: tiny/base/small/medium/large (default: small) |
| `CHUNK_DURATION` | No | Seconds of audio per chunk, 5-15 recommended (default: 8) |

## Bot permissions

The bot needs these Discord permissions:
- Connect (voice)
- Speak (voice, needed for receive)
- Send Messages (text)
- Read Message History (text)

Plus the **Message Content** privileged intent enabled in the bot settings.

## Commands

| Command | Description |
|---|---|
| `!join` | Join your current voice channel and start transcribing |
| `!leave` | Stop transcribing, save session, and disconnect |
| `!status` | Show current transcription status |

## Running

```bash
python3 bot.py
```

Or via systemd:

```bash
cp voice-transcriber.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start voice-transcriber
```

## Architecture

1. Bot connects to voice using py-cord's voice receive
2. `TranscriptionSink` captures per-user PCM audio (48kHz stereo s16le)
3. Audio is buffered per-user in `CHUNK_DURATION`-second chunks
4. Full chunks are sent to a thread pool running Whisper
5. Whisper output is posted to the transcript text channel with speaker labels
6. All transcriptions are stored in `transcripts.db` (SQLite)

## Database

SQLite database at `transcripts.db` with two tables:

- `sessions` — one row per recording session (start/end times, channels)
- `transcripts` — one row per transcription chunk (speaker, text, timestamp, duration)

Query example:
```sql
SELECT timestamp, speaker, text
FROM transcripts
WHERE session_id = 'session-abc123'
ORDER BY timestamp;
```
