"""
Reward systems for GSM8K RL experiments.

Part 4 uses the original exact-match reward as baseline and adds
additional reward shaping systems that can be toggled from chat_rl.py.
"""

from __future__ import annotations

import math
import re
from typing import Callable

from tasks.gsm8k import extract_answer


def reward_baseline(task, conversation, assistant_response: str):
    """Original nanochat reward: exact final-answer match (0/1)."""
    exact = float(task.evaluate(conversation, assistant_response))
    return exact, {
        "exact_match": exact,
    }


def reward_numeric_distance(task, conversation, assistant_response: str):
    """
    Baseline + bounded numeric-distance shaping to the gold final answer.

        - exact match: 1.0
        - otherwise:
                reward = 0.10 * parseable_hash_answer
                             + 0.35 * exp(-|pred-ref| / (|ref| + 1))
            where parseable_hash_answer is 1 if output has parseable #### number.
        - non-parseable/missing numeric answer: only the parseable_hash term can apply.

    This is harder to reward-hack than formatting rewards because the signal is
    anchored directly to numeric closeness to the gold answer.
    """
    exact = float(task.evaluate(conversation, assistant_response))

    assistant_message = conversation["messages"][-1]
    assert (
        assistant_message["role"] == "assistant"
    ), "Last message must be from assistant"
    ref_text = assistant_message["content"][-1]["text"]
    ref_num = extract_answer(ref_text)

    pred_num = extract_answer(assistant_response)
    parseable = float(pred_num is not None)

    if exact > 0.0:
        reward = 1.0
        closeness = 1.0
    else:
        closeness = 0.0
        try:
            if pred_num is None or ref_num is None:
                raise ValueError("non-numeric")
            pred_val = float(pred_num)
            ref_val = float(ref_num)
            distance = abs(pred_val - ref_val)
            scale = abs(ref_val) + 1.0
            closeness = math.exp(-distance / scale)
        except Exception:
            closeness = 0.0
        reward = 0.10 * parseable + 0.35 * closeness

    return reward, {
        "exact_match": exact,
        "parseable_answer": parseable,
        "numeric_closeness": closeness,
    }


def reward_completion_brevity(task, conversation, assistant_response: str):
    """
    Baseline + shaping for answer completion and concise outputs.

    reward = 1.0 (if exact)
           else 0.20 * parseable_answer
              + 0.10 * has_step_words
              + 0.15 * brevity_score

    brevity_score = 1.0 if output length <= 220
                    linearly decays to 0.0 by length 520
                    0.0 above 520

    This targets the observed failure mode of long rambling outputs that never
    finish with a usable answer marker.
    """
    exact = float(task.evaluate(conversation, assistant_response))
    pred_num = extract_answer(assistant_response)
    parseable = float(pred_num is not None)
    length = len(assistant_response)
    low, high = 220, 520
    if length <= low:
        brevity_score = 1.0
    elif length >= high:
        brevity_score = 0.0
    else:
        brevity_score = 1.0 - ((length - low) / (high - low))

    has_step_words = float(
        bool(
            re.search(
                r"\b(first|then|therefore|so|total|finally)\b",
                assistant_response.lower(),
            )
        )
    )

    if exact > 0.0:
        reward = 1.0
    else:
        reward = 0.20 * parseable + 0.10 * has_step_words + 0.15 * brevity_score

    return reward, {
        "exact_match": exact,
        "parseable_answer": parseable,
        "brevity_score": brevity_score,
        "has_step_words": has_step_words,
    }


def reward_numeric_brevity(task, conversation, assistant_response: str):
    """
    Combined numeric-distance + brevity shaping.

    reward = 1.0 (if exact)
           else 0.10 * parseable_answer
              + 0.25 * numeric_closeness
              + 0.08 * has_step_words
              + 0.12 * brevity_score

    numeric_closeness = exp(-|pred-ref| / (|ref| + 1))
    brevity_score = 1.0 if length <= 220, decays to 0.0 by length 520
    """
    exact = float(task.evaluate(conversation, assistant_response))

    assistant_message = conversation["messages"][-1]
    assert assistant_message["role"] == "assistant", "Last message must be from assistant"
    ref_text = assistant_message["content"][-1]["text"]
    ref_num = extract_answer(ref_text)

    pred_num = extract_answer(assistant_response)
    parseable = float(pred_num is not None)

    if exact > 0.0:
        return 1.0, {
            "exact_match": 1.0,
            "parseable_answer": parseable,
            "numeric_closeness": 1.0,
            "has_step_words": 0.0,
            "brevity_score": 0.0,
        }

    closeness = 0.0
    try:
        if pred_num is not None and ref_num is not None:
            pred_val = float(pred_num)
            ref_val = float(ref_num)
            closeness = math.exp(-abs(pred_val - ref_val) / (abs(ref_val) + 1.0))
    except Exception:
        closeness = 0.0

    length = len(assistant_response)
    low, high = 220, 520
    if length <= low:
        brevity_score = 1.0
    elif length >= high:
        brevity_score = 0.0
    else:
        brevity_score = 1.0 - ((length - low) / (high - low))

    has_step_words = float(
        bool(
            re.search(
                r"\b(first|then|therefore|so|total|finally)\b",
                assistant_response.lower(),
            )
        )
    )

    reward = 0.10 * parseable + 0.25 * closeness + 0.08 * has_step_words + 0.12 * brevity_score

    return reward, {
        "exact_match": exact,
        "parseable_answer": parseable,
        "numeric_closeness": closeness,
        "has_step_words": has_step_words,
        "brevity_score": brevity_score,
    }


REWARD_SYSTEMS: dict[str, Callable] = {
    "baseline": reward_baseline,
    "numeric_distance": reward_numeric_distance,
    "completion_brevity": reward_completion_brevity,
    "numeric_brevity": reward_numeric_brevity,
}


def get_reward_fn(name: str) -> Callable:
    if name not in REWARD_SYSTEMS:
        valid = ", ".join(sorted(REWARD_SYSTEMS.keys()))
        raise ValueError(f"Unknown reward system '{name}'. Valid options: {valid}")
    return REWARD_SYSTEMS[name]
