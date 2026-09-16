# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
import json
from typing import Dict, Optional, Union

from ifbench import instructions_registry


@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: list[bool]


def empty_output(inp):
  """Builds a failed output for prompts with no generated response."""
  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response="",
      follow_all_instructions=False,
      follow_instruction_list=[False] * len(inp.instruction_id_list),
  )


def read_prompt_list(input_jsonl_filename):
  """Read inputs from jsonl."""
  inputs = []
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      inputs.append(
          InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"]))
  return inputs


def write_outputs(output_jsonl_filename, outputs):
  """Writes outputs to jsonl."""
  assert outputs
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      f.write(
          json.dumps(
              {
                  attr_name: o.__getattribute__(attr_name)
                  for attr_name in [
                      name for name in dir(o) if not name.startswith("_")
                  ]
              }
          )
      )
      f.write("\n")


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
  """Tests response to see if instrutions are followed."""
  response = get_response_for_prompt(inp, prompt_to_response)
  if response is None:
    return empty_output(inp)

  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)
    inp.kwargs[index] = {key: value for key, value in inp.kwargs[index].items() if value is not None}
    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    if response and response.strip() and instruction.check_following(response):
      is_following_list.append(True)
    else:
      is_following_list.append(False)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
  """Tests response for an upper bound for following instructions."""
  response = get_response_for_prompt(inp, prompt_to_response)
  if response is None:
    return empty_output(inp)

  r = response.split("\n")
  response_remove_first = "\n".join(r[1:]).strip()
  response_remove_last = "\n".join(r[:-1]).strip()
  response_remove_both = "\n".join(r[1:-1]).strip()
  revised_response = response.replace("*", "")
  revised_response_remove_first = response_remove_first.replace("*", "")
  revised_response_remove_last = response_remove_last.replace("*", "")
  revised_response_remove_both = response_remove_both.replace("*", "")
  all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
  ]
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def read_prompt_to_response_dict(input_jsonl_filename):
  """Creates dictionary matching prompt and response."""
  return_dict = {}
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      return_dict[example["prompt"]] = example["response"]
      stripped_prompt = example["prompt"].strip()
      if stripped_prompt != example["prompt"]:
        return_dict[stripped_prompt] = example["response"]
  return return_dict


def get_response_for_prompt(inp, prompt_to_response):
  """Returns a prompt response, tolerating harmless leading/trailing spaces."""
  if inp.prompt in prompt_to_response:
    return prompt_to_response[inp.prompt]
  return prompt_to_response.get(inp.prompt.strip())


def compute_report(outputs):
  """Return the accuracy report as a structured dict (same numbers print_report shows)."""

  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)

  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example.follow_instruction_list
    instruction_id_list = example.instruction_id_list

    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      tier0_id = instruction_id.split(":")[0]
      tier0_total[tier0_id] += 1
      if followed_or_not:
        tier0_correct[tier0_id] += 1

      tier1_total[instruction_id] += 1
      if followed_or_not:
        tier1_correct[instruction_id] += 1

  return {
      "prompt_level_accuracy": prompt_correct / prompt_total,
      "instruction_level_accuracy": instruction_correct / instruction_total,
      "prompt_total": prompt_total,
      "prompt_correct": prompt_correct,
      "instruction_total": instruction_total,
      "instruction_correct": instruction_correct,
      "tier0": {
          k: tier0_correct[k] / tier0_total[k] for k in sorted(tier0_total)
      },
      "tier1": {
          k: tier1_correct[k] / tier1_total[k] for k in sorted(tier1_total)
      },
  }


def print_report(outputs):
  """Prints a report on accuracy scores."""

  report = compute_report(outputs)
  print(f"prompt-level: {report['prompt_level_accuracy']}")
  print(f"instruction-level: {report['instruction_level_accuracy']}")
  print()
  for instruction_id, accuracy in report["tier0"].items():
    print(f"{instruction_id} {accuracy}")
  print()
  for instruction_id, accuracy in report["tier1"].items():
    print(f"{instruction_id} {accuracy}")
