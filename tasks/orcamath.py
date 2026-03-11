"""
Orca-Math: 200K grade school math word problems for small language models.
https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k

Reference: Mitra et al., "Orca-Math: Unlocking the potential of SLMs
in Grade School Math", 2024. arXiv:2402.14830

Generated via AgentInstruct multi-agent framework with GPT-4 Turbo.
Designed specifically for small language models (SLMs).
"""

from datasets import load_dataset
from tasks.common import Task


class OrcaMath(Task):
    """Orca-Math dataset. 200K rows of math word problems with solutions."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "Orca-Math only has a train split"
        self.ds = load_dataset("microsoft/orca-math-word-problems-200k", split=split).shuffle(seed=42)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        conversation = {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
        }
        return conversation
