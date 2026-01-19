import argparse
import logging
from pathlib import Path

from utils import read_json, save_dict_to_json

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_inference_function(model_checkpoint: str):
    """
    Maps model checkpoints to the correct inference backend.
    Allows clean extension when new models are added.
    """
    if model_checkpoint in [
        'Qwen/Qwen2.5-Coder-32B-Instruct',
        'Qwen/Qwen2.5-Coder-14B-Instruct',
        'Qwen/Qwen2.5-Coder-7B-Instruct',
        'Qwen/Qwen2.5-Coder-3B-Instruct',
        'Qwen/Qwen3-Coder-480B-A35B-Instruct',
        'Qwen/Qwen3-Coder-30B-A3B-Instruct',
    ]:
        from model_utils.qwen_coder_utils import qwen_coder_inference
        return qwen_coder_inference

    elif model_checkpoint in [
        "deepseek-ai/deepseek-coder-33b-instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "deepseek-ai/deepseek-coder-1.3b-instruct",
    ]:
        from model_utils.deepseek_coder_utils import deepseek_coder_inference
        return deepseek_coder_inference

    raise ValueError(f"Unknown model checkpoint: {model_checkpoint}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default='Qwen/Qwen2.5-Coder-32B-Instruct')
    parser.add_argument("--chart_type", type=str, default='h_bar')
    parser.add_argument("--dataset", type=str, default='PlotQA')
    parser.add_argument("--partition", type=str, default='test')
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # Few-shot configuration
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--zero_shot", type=str2bool, default=False)
    parser.add_argument("--version", type=str, default="complete")

    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--prompt_dir", type=str, default="prompts")
    parser.add_argument("--response_dir", type=str, default="responses")

    return parser.parse_args()

def log_args(args):
    logging.info(f"Attacker model: {args.model_checkpoint}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Partition: {args.partition}")
    logging.info(f"Chart type: {args.chart_type}")
    logging.info(f"Max new tokens: {args.max_new_tokens}")
    logging.info(f"Zero-shot: {args.zero_shot}")
    logging.info(f"Few-shot examples: {args.shots}")
    logging.info(f"Version: {args.version}")

def main():
    args = set_arguments()
    log_args(args)

    inference_fn = get_inference_function(args.model_checkpoint)

    project_root = Path(__file__).resolve().parent

    # Build portable paths
    data_dir = project_root / args.data_dir
    prompt_dir = project_root / args.prompt_dir
    response_dir = project_root / args.response_dir

    # Load dataset questions
    question_path = (
        data_dir / args.dataset / args.partition / args.chart_type /
        f"questions_{args.chart_type}_few_shot_{args.shots}_{args.version}.json"
    )

    questions = read_json(question_path)
    logging.info(f"{len(questions)} questions loaded.")

    # Load prompt template
    prompt_path = (
        prompt_dir / f"few_shot_prompt_v2_complete_{args.chart_type}.txt"
    )

    # Run inference
    responses = inference_fn(args, questions, prompt_path)

    # Determine output file
    model_name = args.model_checkpoint.split('/')[-1]

    if args.zero_shot:
        filename = (
            f"zero_shot_{args.shots}_attacker_{model_name}_"
            f"{args.chart_type}_{args.version}.json"
        )
    else:
        filename = (
            f"few_shot_{args.shots}_attacker_{model_name}_"
            f"{args.chart_type}_{args.version}.json"
        )

    output_path = response_dir / args.dataset / args.partition / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict_to_json(responses, output_path)
    logging.info(f"Responses saved: {output_path}")
    logging.info("Execution finished.")


if __name__ == "__main__":
    main()
