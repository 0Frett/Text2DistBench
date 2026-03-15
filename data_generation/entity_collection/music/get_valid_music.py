import os
import json
import argparse
from pathlib import Path
import re
from tqdm import tqdm
from googleapiclient.errors import HttpError
from youtube_client import YouTubeClient


def main(
    music_pool_file: str, 
    output_file: str, 
    min_comments: int, 
    max_snippets: int
):

    with open(music_pool_file, "r", encoding="utf-8") as f:
        music_pool = json.load(f)

    yt = YouTubeClient()
    all_videos = []

    for snippet in tqdm(music_pool, desc="Fetching comments"):
        vid = snippet.get("vid")

        try:
            video_data = yt.fetch_snippet_with_comments(
                vid, 
                max_page=50, 
                max_comments=min_comments,
            )
            comment_count = len(video_data.get("comments", []))

            # Skip videos with too few comments
            if comment_count < min_comments:
                print(f"[Skip] {vid} only has {comment_count} comments.")
                continue

            video_data['title'] = re.sub(r"[^\w\-_. ]", "_", video_data['title'])
            all_videos.append(video_data)

            if len(all_videos) >= max_snippets:
                print(f"[Early Stop] Already get {max_snippets} snippets.")
                break

        except HttpError as e:
            print(f"[Error] Failed to fetch {vid}: {e}")
            continue

    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_videos, f, indent=2, ensure_ascii=False)

    print(f"[End] Saved {len(all_videos)} videos to {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Music Video comments")
    parser.add_argument(
        "--entity_pool_file",
        type=str,
        default="data/music/music_pool/2025-07-01_2025-09-30.json",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/music/opinion_entity/2025-07-01_2025-09-30.json",
    )
    parser.add_argument(
        "--min_comments",
        type=int,
        default=1000,
        help="Minimum number of comments required to keep the video."
    )
    parser.add_argument(
        "--max_entity",
        type=int,
        default=20,
    )


    args = parser.parse_args()
    main(
        music_pool_file=args.entity_pool_file, 
        output_file=args.output_file, 
        min_comments=args.min_comments, 
        max_snippets=args.max_entity,
    )
