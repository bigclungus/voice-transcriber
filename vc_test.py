import discord
import asyncio
import os
from dotenv import load_dotenv

# Load token
load_dotenv('/home/clungus/.claude/channels/discord/.env')
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

intents = discord.Intents.default()
intents.voice_states = True
bot = discord.Client(intents=intents)

@bot.event
async def on_ready():
    print(f"Connected as {bot.user}")
    channel = bot.get_channel(1325567700029931560)
    if not channel:
        print("Channel not found!")
        await bot.close()
        return
    vc = await channel.connect()
    print(f"Joined {channel.name}")
    await asyncio.sleep(30)
    await vc.disconnect()
    print("Disconnected")
    await bot.close()

bot.run(TOKEN)
