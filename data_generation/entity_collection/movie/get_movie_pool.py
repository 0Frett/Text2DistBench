import json
import os
import re
import argparse
from tqdm import tqdm
from typing import Dict, Any
from movie_client import MovieClient


def main(
    start_date: str, 
    end_date: str, 
    output_dir: str,
    max_movies: int
) -> Dict[str, Any]:
    
    print(f"Initializing MovieClient...")
    client = MovieClient()

    print(f"\nFetching movies with theatrical (wide) release in: US")
    print(f"Time range: {start_date} to {end_date}\n")

    movies = client.get_movies(
        time_range=(start_date, end_date),
        ISOs=["US"],
        max_movies=max_movies,
    )
    print(movies)
    print(f"Found {len(movies)} valid movies.\n")

    movie_info = []
    cnt = 0

    for movie_title in tqdm(movies, desc="Downloading movie details"):
        try:
            info = client.get_movie_info(movie_title)
            # print(info)
            # Required fields check
            # if not info.get("title"):
            #     print(f"Skipping '{movie_title}': missing Title")
            #     continue
            # if not info.get("summary"):
            #     print(f"Skipping '{movie_title}': missing Summary")
            #     continue
            if not info.get("top5cast"):
                print(f"Skipping '{movie_title}': missing Casts")
                continue
            if not info.get("release_dates"):
                print(f"Skipping '{movie_title}': missing Release_Dates")
                continue

            info["release_dates"] = info.get("release_dates").get("US")
            info["aka"] = info.get("aka").get("US")
            info['title'] = re.sub(r"[^\w\-_. ]", "_", movie_title)

            movie_info.append(info)
            cnt += 1

            if cnt >= max_movies:
                break
            
        except Exception as e:
            print(f"Error processing '{movie_title}': {e}")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{start_date}_{end_date}.json")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(movie_info, f, indent=2, ensure_ascii=False)

    print(f"\nMovie Pool saved to: {save_path}")
    return movie_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_str",
        type=str,
        default="2025-04-01",
        help="Start date for movie collection in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end_str",
        type=str,
        default="2025-06-30",
        help="End date for movie collection in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/movie/movie_pool",
        help="Directory to save the collected movie data."
    )
    parser.add_argument(
        "--max_entity",
        type=int,
        default=50,
        help="Maximum number of movies."
    )
    args = parser.parse_args()

    main(
        start_date=args.start_str, 
        end_date=args.end_str,
        output_dir=args.output_dir,
        max_movies=args.max_entity
    )
