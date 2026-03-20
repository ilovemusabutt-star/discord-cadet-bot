"""Quick test script to verify OCR works on the sample screenshot."""

import sys
from bot import (
    extract_cadets_from_image,
    extract_all_clones,
    run_ocr_lines,
    run_ocr_banner_lines,
)

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/ubuntu/attachments/23042b6e-17d4-4bf6-aa06-73805830b4b1/"
    "Screenshot+2026-03-14+120256.png"
)


def main():
    print(f"Reading image: {IMAGE_PATH}")
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    # Pass 1: bright text (clone IDs + nicknames)
    print("\n=== Pass 1: Bright Text (Clone IDs) ===")
    bright_lines = run_ocr_lines(image_bytes)
    for i, line in enumerate(bright_lines):
        print(f"  {i}: {line}")

    # Pass 2: banner text (usernames + ranks)
    print("\n=== Pass 2: Banner Text (Usernames/Ranks) ===")
    banner_lines = run_ocr_banner_lines(image_bytes)
    for i, line in enumerate(banner_lines):
        print(f"  {i}: {line}")

    # Show all clones
    print("\n=== All Clones ===")
    clones = extract_all_clones(image_bytes)
    if not clones:
        print("  No clones found.")
    else:
        for i, clone in enumerate(clones, 1):
            username = clone.get("username") or "?"
            rank = clone.get("rank", "Unknown")
            print(
                f"  {i}. {clone['clone_id']} "
                f"\"{clone['nickname']}\" | user: {username} | rank: {rank}"
            )

    print(f"\nTotal clones: {len(clones)}")

    # Show extracted cadets (Roblox usernames)
    print("\n=== Extracted Cadet Usernames ===")
    cadets = extract_cadets_from_image(image_bytes)
    if not cadets:
        print("  No cadets found.")
    else:
        for i, cadet in enumerate(cadets, 1):
            print(f"  {i}. {cadet['username']} (rank: {cadet['rank']})")

    print(f"\nTotal cadets: {len(cadets)}")


if __name__ == "__main__":
    main()
