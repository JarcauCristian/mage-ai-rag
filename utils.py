from pathlib import Path
from rapidfuzz import process


def clean_files(directory: str) -> None:
    for path in Path(directory).rglob("*.py"):
        print(f"Cleaning {path.name}")
        with open(path, 'r') as f:
            res = ""
            lines = f.readlines()
            for line in lines:
                if "{%" in line or '{{' in line:
                    continue

                res += line

            with open(path, 'w') as w:
                w.write(res)


def get_similar_names(target_name, name_list, score_cutoff=80):
    similar_names = process.extract(target_name, name_list, limit=5, score_cutoff=score_cutoff)
    return [name[0] for name in similar_names]


def get_similar_block_types(target_category, categories, score_cutoff=80):
    target_names = categories.get(target_category, [])
    similar_block_types = []
    for name in target_names:
        similar_block_types.extend(get_similar_names(name, target_names, score_cutoff))
    return list(set(similar_block_types))


def return_filter():
    target_categories = ["loaders", "transformers", "exporters", "sensors"]
    categories = {
        "loaders": ["importers", "fetchers", "readers", "ingesters", "gatherers", "retrievers", "loaders", "acquirers"],
        "transformers": ["converters", "modifiers", "adjusters", "processors", "alterers", "changers", "reformers",
                         "shapers"],
        "exporters": ["writers", "savers", "dispatchers", "transmitters", "distributors", "forwarders", "deliverers",
                      "emitters"],
        "sensors": ["detectors", "monitors", "gauges", "probes", "trackers", "readers", "analyzers", "scanners"]
    }

    similar_block_types = []
    for target_category in target_categories:
        similar_block_types.extend(get_similar_block_types(target_category, categories))

    similar_block_types = list(set(similar_block_types))

    return similar_block_types
