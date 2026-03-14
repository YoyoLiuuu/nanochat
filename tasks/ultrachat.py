"""
UltraChat-200k: Filtered multi-turn conversations for SFT.
https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

Reference: Ding et al., "Enhancing Chat Language Models by Scaling
High-quality Instructional Conversations", 2023. arXiv:2305.14233

Heavily filtered subset of UltraChat (1.4M dialogues). Used to train
Zephyr-7B-beta. Quality improvements include truecasing and removal of
robotic responses.
"""

from datasets import load_dataset
from tasks.common import Task


class UltraChat(Task):
    """UltraChat-200k SFT split. ~208K multi-turn conversations."""

    def __init__(self, split="train_sft", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train_sft", "test_sft"], "UltraChat split must be train_sft|test_sft"
        self.ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split).shuffle(seed=42)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        # Validate structure (same pattern as SmolTalk)
        assert len(messages) >= 2, "UltraChat messages must have at least 2 messages"
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"
        conversation = {
            "messages": messages,
        }
        return conversation
