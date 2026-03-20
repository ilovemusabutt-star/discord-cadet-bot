import os
import re
import io
import logging
import discord
from discord.ext import commands
import pytesseract
import numpy as np
from PIL import Image, ImageFilter, ImageOps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cadet-bot")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


def _load_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load image bytes and return (gray, r, g, b) numpy arrays."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image)
    r_ch, g_ch, b_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = (0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch).astype(np.uint8)
    return gray, r_ch, g_ch, b_ch


def preprocess_bright_text(image_bytes: bytes) -> Image.Image:
    """Pass 1: Isolate bright white text (clone IDs and nicknames).

    Targets the top line of name tags which has white text on
    blue translucent banners.
    """
    gray, _, _, _ = _load_image(image_bytes)
    binary = (gray > 155).astype(np.uint8) * 255
    bin_img = Image.fromarray(binary)
    w, h = bin_img.size
    bin_img = bin_img.resize((w * 5, h * 5), Image.LANCZOS)
    bin_img = bin_img.filter(ImageFilter.MedianFilter(3))
    bin_img = ImageOps.invert(bin_img)
    return bin_img


def preprocess_banner_text(image_bytes: bytes) -> Image.Image:
    """Pass 2: Isolate dimmer text on blue banners (usernames and ranks).

    Targets the bottom line of name tags which has smaller, dimmer text
    in the format 'Username | Role | Rank'.
    Uses a strict blue filter (blue >> green) to avoid picking up
    blue-ish sky or background pixels.
    """
    gray, r_ch, g_ch, b_ch = _load_image(image_bytes)

    # Strict blue banner detection: blue must be significantly higher
    # than green to isolate actual UI banners from sky/background
    blue_banner = (
        (b_ch.astype(np.int16) > g_ch.astype(np.int16) + 60)
        & (b_ch > 60)
        & (gray >= 40)
        & (gray <= 160)
    )

    # Within blue banner regions, isolate text pixels
    text_mask = blue_banner & (gray > 40)
    text_arr = np.where(text_mask, 255, 0).astype(np.uint8)

    pil_img = Image.fromarray(text_arr)
    w, h = pil_img.size
    pil_img = pil_img.resize((w * 8, h * 8), Image.LANCZOS)

    # Dilate to thicken thin text strokes
    pil_img = pil_img.filter(ImageFilter.MaxFilter(3))
    pil_img = pil_img.filter(ImageFilter.MedianFilter(3))
    pil_img = ImageOps.invert(pil_img)
    return pil_img


def run_ocr_lines(image_bytes: bytes) -> list[str]:
    """Run OCR on bright text and return cleaned text lines."""
    processed = preprocess_bright_text(image_bytes)
    result = pytesseract.image_to_string(
        processed, config="--psm 11 --oem 3",
    ).strip()
    return [line.strip() for line in result.split("\n") if len(line.strip()) > 2]


def run_ocr_banner_lines(image_bytes: bytes) -> list[str]:
    """Run OCR on blue-banner text (usernames / ranks) and return lines."""
    processed = preprocess_banner_text(image_bytes)
    result = pytesseract.image_to_string(
        processed, config="--psm 11 --oem 3",
    ).strip()
    return [line.strip() for line in result.split("\n") if len(line.strip()) > 2]


def _looks_like_cadet(text: str) -> bool:
    """Fuzzy check whether *text* contains the word 'Cadet'.

    OCR may misread it as Cadel, Cader, Cades, Cacket, Cachet,
    Codet, Cackt, etc.
    """
    lowered = text.lower()
    # Match common OCR misreadings of 'Cadet'
    return bool(
        re.search(r"\bcade[tlrs]?\b", lowered)
        or re.search(r"\bcac[hk]e?t\b", lowered)
        or re.search(r"\bco?det\b", lowered)
    )


def _parse_banner_line(line: str) -> dict[str, str] | None:
    """Try to parse 'Username | Role | Rank' from a banner OCR line."""
    if "|" not in line:
        return None

    parts = [p.strip() for p in line.split("|")]
    # We need at least a username and one other part
    parts = [p for p in parts if len(p) > 1]
    if not parts:
        return None

    username = parts[0]
    # Clean common OCR artefacts from the username
    username = re.sub(r"[^A-Za-z0-9_]", "", username)
    if len(username) < 2:
        return None

    rank = parts[-1] if len(parts) > 1 else ""
    role = parts[1] if len(parts) > 2 else ""

    return {"username": username, "role": role, "rank": rank}


def parse_clone_from_line(line: str) -> dict[str, str] | None:
    """Try to parse a clone ID and nickname from a bright-text OCR line."""
    clone_pattern = re.compile(r"C[-\s]?\d{3,4}")
    nickname_pattern = re.compile(
        r'["\u201c\u201d\'](.*?)["\u201c\u201d\']'
    )

    clone_match = clone_pattern.search(line)
    if not clone_match:
        return None

    clone_id = clone_match.group(0).replace(" ", "-")

    nickname = ""
    nick_match = nickname_pattern.search(line)
    if nick_match:
        nickname = nick_match.group(1)
    else:
        remaining = line[clone_match.end():].strip()
        if remaining:
            nickname = remaining.strip("\"' ")

    return {"clone_id": clone_id, "nickname": nickname, "raw_line": line}


def extract_cadets_from_image(image_bytes: bytes) -> list[dict[str, str]]:
    """Extract Roblox usernames of clones with 'Cadet' rank.

    Uses two OCR passes:
    - Pass 1 (bright text): detects clone IDs and nicknames.
    - Pass 2 (banner text): detects 'Username | Role | Rank' lines.

    Returns usernames where the rank contains 'Cadet'.
    """
    # --- Pass 2: extract usernames and ranks from banner text ---
    banner_lines = run_ocr_banner_lines(image_bytes)
    cadets: list[dict[str, str]] = []
    seen_usernames: set[str] = set()

    for line in banner_lines:
        parsed = _parse_banner_line(line)
        if not parsed:
            continue

        username = parsed["username"]
        if username.lower() in seen_usernames:
            continue

        if _looks_like_cadet(parsed["rank"]) or _looks_like_cadet(line):
            seen_usernames.add(username.lower())
            cadets.append({
                "username": username,
                "role": parsed["role"],
                "rank": parsed["rank"],
                "raw_text": line,
            })

    # --- Fallback: if pass 2 found nothing, check pass 1 lines ---
    if not cadets:
        bright_lines = run_ocr_lines(image_bytes)
        for i, line in enumerate(bright_lines):
            if _looks_like_cadet(line):
                clone = parse_clone_from_line(line)
                if clone and clone["clone_id"] not in seen_usernames:
                    seen_usernames.add(clone["clone_id"])
                    cadets.append({
                        "username": clone["clone_id"],
                        "role": "",
                        "rank": "Cadet",
                        "raw_text": line,
                    })

            # Also check neighbours
            for offset in [-2, -1, 1, 2]:
                ni = i + offset
                if 0 <= ni < len(bright_lines) and _looks_like_cadet(bright_lines[ni]):
                    clone = parse_clone_from_line(line)
                    if clone and clone["clone_id"] not in seen_usernames:
                        seen_usernames.add(clone["clone_id"])
                        cadets.append({
                            "username": clone["clone_id"],
                            "role": "",
                            "rank": "Cadet",
                            "raw_text": line,
                        })

    return cadets


def extract_all_clones(image_bytes: bytes) -> list[dict[str, str]]:
    """Extract all clone IDs, nicknames, and banner info from an image."""
    bright_lines = run_ocr_lines(image_bytes)
    banner_lines = run_ocr_banner_lines(image_bytes)

    clones: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    clone_pattern = re.compile(r"C[-\s]?\d{3,4}")

    for line in bright_lines:
        clone = parse_clone_from_line(line)
        if not clone or clone["clone_id"] in seen_ids:
            continue
        seen_ids.add(clone["clone_id"])
        clones.append({
            "clone_id": clone["clone_id"],
            "nickname": clone["nickname"],
        })

    # Attach banner info (usernames / ranks) parsed from pass 2
    parsed_banners = []
    for bline in banner_lines:
        parsed = _parse_banner_line(bline)
        if parsed:
            parsed_banners.append(parsed)

    # Add banner summary to output
    for clone in clones:
        clone["username"] = ""
        clone["rank"] = "Unknown"

    # If we got any parsed banner lines, attach them in order
    for idx, pb in enumerate(parsed_banners):
        if idx < len(clones):
            clones[idx]["username"] = pb["username"]
            clones[idx]["rank"] = pb["rank"]

    return clones


@bot.event
async def on_ready():
    logger.info(f"Bot is ready! Logged in as {bot.user}")


@bot.command(name="cadets", help="Upload a screenshot to extract Cadet usernames.")
async def cadets(ctx: commands.Context):
    """Extract Roblox usernames of Cadets from an attached screenshot."""
    if not ctx.message.attachments:
        await ctx.send(
            "Please attach a screenshot to your message.\n"
            "Usage: `!cadets` (with an image attached)"
        )
        return

    attachment = ctx.message.attachments[0]
    if not attachment.content_type or not attachment.content_type.startswith("image/"):
        await ctx.send("The attachment must be an image (PNG, JPG, etc.).")
        return

    await ctx.send("Processing screenshot... This may take a moment.")

    try:
        image_bytes = await attachment.read()
        cadets_found = extract_cadets_from_image(image_bytes)

        if not cadets_found:
            await ctx.send(
                "No cadets found in the screenshot.\n"
                "*Tip: closer screenshots with larger name tags "
                "give better results.*"
            )
            return

        lines = ["**Cadet usernames found:**\n"]
        for i, cadet in enumerate(cadets_found, 1):
            extra = f" ({cadet['role']})" if cadet.get("role") else ""
            lines.append(f"{i}. **{cadet['username']}**{extra}")

        lines.append(f"\n**Total Cadets: {len(cadets_found)}**")
        response = "\n".join(lines)

        if len(response) > 2000:
            chunks = [response[i:i + 1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await ctx.send(chunk)
        else:
            await ctx.send(response)

    except Exception as e:
        logger.error(f"Error processing screenshot: {e}", exc_info=True)
        await ctx.send(f"Error processing the screenshot: {e}")


@bot.command(name="scan", help="Upload a screenshot to see ALL clones detected.")
async def scan(ctx: commands.Context):
    """Extract all clone info from an attached screenshot."""
    if not ctx.message.attachments:
        await ctx.send(
            "Please attach a screenshot to your message.\n"
            "Usage: `!scan` (with an image attached)"
        )
        return

    attachment = ctx.message.attachments[0]
    if not attachment.content_type or not attachment.content_type.startswith("image/"):
        await ctx.send("The attachment must be an image (PNG, JPG, etc.).")
        return

    await ctx.send("Scanning screenshot... This may take a moment.")

    try:
        image_bytes = await attachment.read()
        clones = extract_all_clones(image_bytes)

        if not clones:
            await ctx.send("No clone usernames found in the screenshot.")
            return

        lines = ["**All clones found in screenshot:**\n"]
        for i, entry in enumerate(clones, 1):
            nickname = entry["nickname"] or "?"
            username = entry.get("username") or "?"
            rank = entry.get("rank", "Unknown")
            lines.append(
                f"{i}. **{entry['clone_id']}** \"{nickname}\" "
                f"| user: {username} | rank: {rank}"
            )

        lines.append(f"\n**Total: {len(clones)}**")
        response = "\n".join(lines)

        if len(response) > 2000:
            chunks = [response[i:i + 1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await ctx.send(chunk)
        else:
            await ctx.send(response)

    except Exception as e:
        logger.error(f"Error scanning screenshot: {e}", exc_info=True)
        await ctx.send(f"Error scanning the screenshot: {e}")


@bot.command(name="ocrraw", help="Show raw OCR output from a screenshot (debug).")
async def ocrraw(ctx: commands.Context):
    """Show raw OCR text detected in a screenshot for debugging."""
    if not ctx.message.attachments:
        await ctx.send("Please attach a screenshot. Usage: `!ocrraw` (with image)")
        return

    attachment = ctx.message.attachments[0]
    if not attachment.content_type or not attachment.content_type.startswith("image/"):
        await ctx.send("The attachment must be an image.")
        return

    await ctx.send("Running OCR... This may take a moment.")

    try:
        image_bytes = await attachment.read()
        bright_lines = run_ocr_lines(image_bytes)
        banner_lines = run_ocr_banner_lines(image_bytes)

        lines = ["**Pass 1 — Bright text (clone IDs):**\n```"]
        lines.extend(bright_lines)
        lines.append("```")
        lines.append("\n**Pass 2 — Banner text (usernames/ranks):**\n```")
        lines.extend(banner_lines)
        lines.append("```")

        response = "\n".join(lines)

        if len(response) > 2000:
            chunks = [response[i:i + 1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await ctx.send(chunk)
        else:
            await ctx.send(response)

    except Exception as e:
        logger.error(f"Error in OCR: {e}", exc_info=True)
        await ctx.send(f"Error: {e}")


def main():
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
            token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("Error: DISCORD_BOT_TOKEN not set.")
        print("Set it as an environment variable or add it to a .env file.")
        raise SystemExit(1)
    bot.run(token)


if __name__ == "__main__":
    main()
