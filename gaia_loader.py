import json
import os
import datasets
from datasets import load_dataset

# ============ Loader Definition =============

_NAMES = [
    "2023_all",
    "2023_level1",
    "2023_level2",
    "2023_level3",
]

VERSION = datasets.Version("0.0.1")
YEAR_TO_LEVELS = {"2023": [1, 2, 3]}
separator = "_"

class GAIA_dataset(datasets.GeneratorBasedBuilder):
    VERSION = VERSION

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=name, version=VERSION, description=name)
        for name in _NAMES
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="GAIA Dataset Loader (Local)",
            features=datasets.Features({
                "task_id": datasets.Value("string"),
                "Question": datasets.Value("string"),
                "Level": datasets.Value("string"),
                "Final answer": datasets.Value("string"),
                "file_name": datasets.Value("string"),
                "file_path": datasets.Value("string"),
                "Annotator Metadata": {
                    k: datasets.Value("string") for k in [
                        "Steps", "Number of steps", "How long did this take?", "Tools", "Number of tools"
                    ]
                }
            }),
        )

    def _split_generators(self, dl_manager):
        year, level_name = self.config.name.split(separator)
        levels = YEAR_TO_LEVELS[year] if level_name == "all" else [int(level_name.replace("level", ""))]

        output = []
        for split in ["validation", "test"]:
            root_file = dl_manager.download(os.path.join("GAIA_dataset", year, split, "metadata.jsonl"))
            test_attached_files = {}
            with open(root_file, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    fname = record.get("file_name", "")
                    if fname:
                        test_attached_files[fname] = dl_manager.download(os.path.join("GAIA_dataset", year, split, fname))

            output.append(
                datasets.SplitGenerator(
                    name=getattr(datasets.Split, split.upper()),
                    gen_kwargs={"root_file": root_file, "attached_files": test_attached_files, "levels": levels}
                )
            )
        return output

    def _generate_examples(self, root_file, attached_files, levels):
        with open(root_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                level_value = int(ex["Level"]) if isinstance(ex["Level"], str) else ex["Level"]
                if level_value in levels:
                    ex["file_path"] = attached_files.get(ex["file_name"], "")
                    yield i, ex

# ============ Perform loading and presentation =============

if __name__ == "__main__":
    print("📂 load Level all's GAIA validation/test dataset...")

    val_ds = load_dataset(
        path=os.path.abspath(__file__),  # current script path
        name="2023_all",
        split="validation",
        trust_remote_code=True,
    )
    test_ds = load_dataset(
        path=os.path.abspath(__file__),
        name="2023_all",
        split="test",
        trust_remote_code=True,
    )

    print(f"✅ Sample size of validation set: {len(val_ds)}")
    print(f"✅ Sample size of test set: {len(test_ds)}")
