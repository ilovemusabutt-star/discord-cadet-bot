# Discord Cadet Bot

A Discord bot that extracts clone trooper usernames from Roblox Star Wars RP screenshots using OCR (Optical Character Recognition). It can filter results to show only Cadets.

## Features

- `!cadets` - Upload a screenshot to extract only **Cadet** usernames
- `!scan` - Upload a screenshot to see **all** clone usernames with their ranks
- `!ocrraw` - Show raw OCR output from a screenshot (for debugging)

## Setup

### 1. Create a Discord Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Go to the "Bot" tab and click "Add Bot"
4. Under "Privileged Gateway Intents", enable **Message Content Intent**
5. Copy the bot token

### 2. Invite the Bot to Your Server

1. In the Developer Portal, go to "OAuth2" > "URL Generator"
2. Select scopes: `bot`
3. Select permissions: `Send Messages`, `Read Message History`, `Attach Files`
4. Copy the generated URL and open it in your browser to invite the bot

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Bot

Create a `.env` file in the project root:

```
DISCORD_BOT_TOKEN=your_bot_token_here
```

Or set the environment variable directly:

```bash
export DISCORD_BOT_TOKEN=your_bot_token_here
```

### 5. Run the Bot

```bash
python bot.py
```

> **Note:** Tesseract OCR must be installed on your system. On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`

## Usage

1. In any Discord channel where the bot is present, type `!cadets` and attach a screenshot
2. The bot will process the image and return a list of Cadets found
3. Use `!scan` to see all clones regardless of rank
4. Use `!ocrraw` to see the raw OCR output for debugging

## Requirements

- Python 3.10+
- discord.py
- pytesseract (+ Tesseract OCR system package)
- numpy
