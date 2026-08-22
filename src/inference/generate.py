"""Single-prompt generation used by the serving API."""

from __future__ import annotations

from typing import Any

INFERENCE_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def generate_response(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    prompt = INFERENCE_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=max(temperature, 1e-5),
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:\n")[-1].strip()