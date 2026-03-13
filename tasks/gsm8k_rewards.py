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


CALC_RE = re.compile(r"<<([^>]+)>>")
ALLOWED_CALC_CHARS = set("0123456789*+-/.() ")


def _safe_eval_arithmetic(expr: str):
    expr = expr.replace(",", "")
    if not expr:
        return None
    if "**" in expr:
        return None
    if not set(expr).issubset(ALLOWED_CALC_CHARS):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None


def _calc_consistency_score(text: str) -> float:
    """
    Score in [0, 1] for arithmetic consistency of <<expr=result>> snippets.
    If there are no calculator snippets, returns 0.
    """
    snippets = CALC_RE.findall(text)
    if not snippets:
        return 0.0

    num_valid = 0
    for snippet in snippets:
        if "=" not in snippet:
            continue
        expr, claimed = snippet.rsplit("=", 1)
        actual = _safe_eval_arithmetic(expr.strip())
        if actual is None:
            continue
        try:
            claimed_float = float(claimed.strip().replace(",", ""))
            actual_float = float(actual)
        except Exception:
            continue
        if abs(actual_float - claimed_float) <= 1e-6:
            num_valid += 1
    return num_valid / max(len(snippets), 1)


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
    - otherwise (if both numeric):
        reward = 0.4 * exp(-|pred-ref| / (|ref| + 1))
      which is in (0, 0.4]
    - non-parseable/missing numeric answer: 0.0

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
    else:
        try:
            if pred_num is None or ref_num is None:
                raise ValueError("non-numeric")
            pred_val = float(pred_num)
            ref_val = float(ref_num)
            distance = abs(pred_val - ref_val)
            scale = abs(ref_val) + 1.0
            reward = 0.4 * math.exp(-distance / scale)
        except Exception:
            reward = 0.0

    return reward, {
        "exact_match": exact,
        "parseable_answer": parseable,
    }


def reward_calc_consistency(task, conversation, assistant_response: str):
    """
    Baseline + shaping from answer formatting and calculator consistency.

    reward = 1.0 (if exact)
           else 0.15 * parseable_answer + 0.35 * calc_consistency

    This keeps exact-match as the dominant signal while giving denser
    intermediate feedback on structured reasoning behavior.
    """
    exact = float(task.evaluate(conversation, assistant_response))
    pred_num = extract_answer(assistant_response)
    parseable = float(pred_num is not None)
    calc_consistency = _calc_consistency_score(assistant_response)

    if exact > 0.0:
        reward = 1.0
    else:
        reward = 0.15 * parseable + 0.35 * calc_consistency

    return reward, {
        "exact_match": exact,
        "parseable_answer": parseable,
        "calc_consistency": calc_consistency,
    }


REWARD_SYSTEMS: dict[str, Callable] = {
    "baseline": reward_baseline,
    "numeric_distance": reward_numeric_distance,
    "calc_consistency": reward_calc_consistency,
}


def get_reward_fn(name: str) -> Callable:
    if name not in REWARD_SYSTEMS:
        valid = ", ".join(sorted(REWARD_SYSTEMS.keys()))
        raise ValueError(f"Unknown reward system '{name}'. Valid options: {valid}")
    return REWARD_SYSTEMS[name]
