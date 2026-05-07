#!/usr/bin/env python3
"""Generate responses from an OpenAI-compatible API for IFBench evaluation."""

import json
import argparse
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from tqdm import tqdm

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(text: str) -> tuple[str, str]:
    """Return (response, reasoning_content).

    Mirrors mind-eval's separate_thinking_from_response:
      1. <think>...</think> present — reasoning is the inner text, response
         is what follows </think>.
      2. Only </think> present (no opening tag) — reasoning is everything
         before </think> (with any stray <think> stripped), response is
         after.
      3. Neither tag present — warn (IFBench is run against thinking models,
         so a missing close tag usually means a malformed/truncated
         generation), and return the full text as response.
    """
    match = _THINK_RE.search(text)
    if match is not None:
        return text[match.end() :].strip(), match.group(1).strip()

    parts = text.split("</think>", 1)
    if len(parts) == 2:
        return parts[1].strip(), parts[0].replace("<think>", "").strip()

    tqdm.write(
        f"WARNING: thinking split failed — no </think> in response "
        f"(len={len(text)}, head={text[:80]!r})",
        file=sys.stderr,
    )
    return text.strip(), ""


def load_prompts(input_file: str) -> list[dict]:
    """Load prompts from IFBench test file."""
    prompts = []
    with open(input_file, "r") as f:
        for line in f:
            example = json.loads(line)
            prompts.append({"key": example["key"], "prompt": example["prompt"]})
    return prompts


def generate_response(
    client: httpx.Client,
    api_base: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
    seed: int | None,
) -> str:
    """Generate a response from the API."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if seed is not None:
        payload["seed"] = seed

    response = client.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses from an OpenAI-compatible API for IFBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-base",
        required=True,
        help="Base URL for the OpenAI-compatible API",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use",
    )
    parser.add_argument(
        "--input-file",
        default="data/IFBench_test.jsonl",
        help="Path to IFBench test file",
    )
    parser.add_argument(
        "--output-file",
        help="Output file for responses (defaults to data/{model}-responses.jsonl)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (omitted from request if unset — server picks default)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (omit for random)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (if required)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )

    args = parser.parse_args()

    prompts = load_prompts(args.input_file)
    print(f"Loaded {len(prompts)} prompts from {args.input_file}")

    if not args.output_file:
        safe_model_name = args.model.replace("/", "-")
        args.output_file = f"data/{safe_model_name}-responses.jsonl"

    print(f"Model: {args.model}")
    print(f"API: {args.api_base}")

    # Load existing responses if resuming
    existing_prompts = set()
    existing_responses = []
    if args.resume and Path(args.output_file).exists():
        with open(args.output_file, "r") as f:
            for line in f:
                resp = json.loads(line)
                existing_prompts.add(resp["prompt"])
                existing_responses.append(resp)
        print(f"Resuming: {len(existing_prompts)} prompts already completed")

    # Filter out completed prompts
    remaining = [p for p in prompts if p["prompt"] not in existing_prompts]
    print(f"Generating responses for {len(remaining)} prompts...")

    # Generate responses
    results = list(existing_responses)
    errors = []

    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_prompt = {
                executor.submit(
                    generate_response,
                    client,
                    args.api_base,
                    args.model,
                    p["prompt"],
                    args.temperature,
                    args.max_tokens,
                    args.api_key,
                    args.seed,
                ): p
                for p in remaining
            }

            with tqdm(total=len(remaining), desc="Generating") as pbar:
                for future in as_completed(future_to_prompt):
                    prompt_data = future_to_prompt[future]
                    try:
                        response = future.result()
                        response_text, reasoning_content = split_thinking(response)
                        results.append(
                            {
                                "prompt": prompt_data["prompt"],
                                "response": response_text,
                                "reasoning_content": reasoning_content,
                            }
                        )
                    except Exception as e:
                        errors.append(
                            {
                                "key": prompt_data["key"],
                                "error": str(e),
                            }
                        )
                        # Add empty response so eval can still run
                        results.append(
                            {
                                "prompt": prompt_data["prompt"],
                                "response": "",
                                "reasoning_content": "",
                            }
                        )
                    pbar.update(1)

                    # Save incrementally
                    if len(results) % 10 == 0:
                        with open(args.output_file, "w") as f:
                            for r in results:
                                f.write(json.dumps(r) + "\n")

    # Final save
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(results)} responses to {args.output_file}")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  - Key {e['key']}: {e['error']}")

    print(f"\nRun evaluation with:")
    print(
        f"  uv run python3 -m run_eval --input_data={args.input_file} --input_response_data={args.output_file} --output_dir=eval"
    )


if __name__ == "__main__":
    main()
