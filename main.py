import os
import fnmatch
import json
import argparse
import pathlib
from warnings import warn

import torch
import openai
import datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from eval.args import RunnerArguments, HFArguments, OAIArguments, GenerationArguments
from eval.evaluator import HFEvaluator, OAIEvaluator
from eval.tasks import ALL_TASKS

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()


def main():
    # args = HfArgumentParser(
    #     [RunnerArguments, HFArguments, OAIArguments, GenerationArguments]
    # ).parse_args()

    # args.output_dir = pathlib.Path(__file__).parent / args.output_dir
    # args.save_generations_raw_path = args.output_dir / args.save_generations_raw_path
    # args.save_generations_prc_path = args.output_dir / args.save_generations_prc_path
    # args.save_references_path = args.output_dir / args.save_references_path
    # args.save_results_path = args.output_dir / args.save_results_path
    # args.save_generations_raw_path.parent.mkdir(parents=True, exist_ok=True)
    # args.save_generations_prc_path.parent.mkdir(parents=True, exist_ok=True)
    # args.save_references_path.parent.mkdir(parents=True, exist_ok=True)
    # args.save_results_path.parent.mkdir(parents=True, exist_ok=True)
    args.tasks = ...
    args.generations_path = ...
    args.model = ...
    parser = argparse.ArgumentParser('LLM Reasoner')
    args = parser.parse_args()

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = set()
        for pattern in args.tasks.split(","):
            for matching in fnmatch.filter(ALL_TASKS, pattern):
                task_names.add(matching)
        task_names = list(task_names)

    results = {}

    evaluator = None
    env_key = args.openai_api_env_keys[0]  # use any key to get list of models
    openai.api_key = os.environ[env_key]
    comp_models = {
        "code-davinci-002",
        "text-davinci-003",
        "text-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
    }
    chat_models = {
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613"
    }
    if any(model == args.model for model in comp_models):
        print(f"Using OpenAI Completion API for model {args.model}")
        evaluator = OAIEvaluator(args)
    elif any(model == args.model for model in chat_models):
        print(f"Using OpenAI Chat API for model {args.model}")
        evaluator = OAIEvaluator(args, chat=True)
    else:
        print(
            f"Model {args.model} not found in OpenAI API. Assuming HuggingFace locally."
        )

    for task in task_names:
        if args.generation_only:
            print("Generation mode only")
            generations_prc, generations_raw, references = evaluator.generate_text(task)
            if args.save_generations_raw:
                with open(args.save_generations_raw_path, "w") as fp:
                    json.dump(generations_raw, fp)
                    print("raw generations were saved")
            if args.save_generations_prc:
                with open(args.save_generations_prc_path, "w") as fp:
                    json.dump(generations_prc, fp)
                    print("processed generations were saved")
            if args.save_references:
                with open(args.save_references_path, "w") as fp:
                    json.dump(references, fp)
                    print("references were saved")
        else:
            results[task] = evaluator.evaluate(task)

    results["config"] = {"model": args.model}
    if not args.generation_only:
        dumped = json.dumps(results, indent=2, sort_keys=True)
        print(dumped)

        if args.save_results:
            with open(args.save_results_path, "w") as f:
                f.write(dumped)


if __name__ == "__main__":
    main()
