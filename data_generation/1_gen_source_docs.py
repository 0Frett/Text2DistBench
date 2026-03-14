import json
import os
import argparse
from prompts import music_datagen_prompts, movie_datagen_prompts


def get_meta_doc(unit, domain, templates):
    # --- Domain unit unpacking ---
    if domain == "music":
        doc = templates.DOC_TEMPLATE.format(
            title=unit['title'],
            date=unit["published_time"][:10],
            description=unit['description']
        )
    elif domain == "movie":
        doc = templates.DOC_TEMPLATE.format(
            title=unit['title'],
            date=unit['release_dates'],
            casts=", ".join(unit.get("top5cast", [])),
            summary=" ".join(unit.get("summary", [])),
            synopsis=" ".join(unit.get("synopsis", [])),
        )
    else:
        raise ValueError(f"Unsupported domain: {domain}")
    return doc



def main(opinion_entity_dir, output_dir, source_lang, domain):
    if domain == "music":
        templates = music_datagen_prompts
    elif domain == "movie":
        templates = movie_datagen_prompts
    else:
        raise ValueError("domain must be 'music' or 'movie' ")

    time_stamp = os.path.basename(opinion_entity_dir)

    with open(os.path.join(opinion_entity_dir, f"{source_lang}.json"), 'r', encoding='utf-8') as f:
        units = json.load(f)
    
    save_dir = os.path.join(output_dir, time_stamp, source_lang)
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, unit in enumerate(units):
        title = unit.get('title')
        save_path = os.path.join(save_dir, f"{title}.json")

        if os.path.exists(save_path):
            print(f"[{idx}/{len(units)}] Skipping... Train docs already saved to {save_path}")
            continue
        else:
            print(f"[{idx+1}/{len(units)}] Processing {domain}: {title}")
            doc = get_meta_doc(unit, domain, templates)

        unit_train_doc = {
            "meta_data": doc,
            "comments": [c["text"] for c in unit.get('comments')]
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(unit_train_doc, f, indent=2, ensure_ascii=False)
        print(f"Finished: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", type=str,
        default="movie", choices=["movie", "music"]
    )
    parser.add_argument(
        "--opinion_entity_dir", type=str, 
        default="data/movie/opinion_entity/2025-07-01_2025-09-30"
    )
    parser.add_argument(
        "--output_dir", type=str, 
        default="data/movie/training_docs"
    )
    parser.add_argument(
        "--source_lang", type=str, default="en"
    )
    args = parser.parse_args()

    main(
        args.opinion_entity_dir, 
        args.output_dir,
        args.source_lang,
        args.domain,
    )
