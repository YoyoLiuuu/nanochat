"""
MetaMathQA: Bootstrapped mathematical questions for improved math reasoning.
https://huggingface.co/datasets/meta-math/MetaMathQA

Reference: Yu et al., "MetaMath: Bootstrap Your Own Mathematical Questions
for Large Language Models", ICLR 2024. arXiv:2309.12284

Contains ~395K math problems created via answer augmentation, question
rephrasing, and backward reasoning from GSM8K and MATH training sets.
"""

from datasets import load_dataset
from tasks.common import Task


class MetaMathQA(Task):
    """MetaMathQA dataset. 395K rows of augmented math Q&A pairs."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "MetaMathQA only has a train split"
        self.ds = load_dataset("meta-math/MetaMathQA", split=split).shuffle(seed=42)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        conversation = {
            "messages": [
                {"role": "user", "content": row["query"]},
                {"role": "assistant", "content": row["response"]},
            ],
        }
        return conversation
