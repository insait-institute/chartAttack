import os
import logging
import argparse
import math
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from utils import read_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")       # for saving fine-tuned models
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")  # for intermediate checkpoints

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

def load_training_dataset_path(args):
    version = args.dataset_version
    dataset_folder = os.path.join(args.base_path, args.dataset, args.partition, "tuple_dataset")
    
    if version == "ef-anchor_postive_median_balanced":
        return os.path.join(dataset_folder, f"tuple_{args.chart_type}_essential_flatten_full_balanced.csv")
    elif version == "ef-anchor_positive_balanced":
        return os.path.join(dataset_folder, f"tuple_{args.chart_type}_essential_flatten_remove_high_freq.csv")
    elif version == "ef-anchor_balanced":
        return os.path.join(dataset_folder, f"tuple_{args.chart_type}_essential_flatten_balanced.csv")
    else:
        raise ValueError("Wrong dataset version")

def logging_args(args):
    logging.info(f"Model checkpoint: {args.model_checkpoint}")
    logging.info(f"Chart type: {args.chart_type}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Training dataset version: {args.dataset_version}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Gradient accumulation steps: {args.gradient_acc_steps}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Patience: {args.patience}")
    logging.info(f"Learning rate: {args.learning_rate}")

def set_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=DATA_DIR)
    parser.add_argument("--model_checkpoint", type=str, default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument("--finetuned_model", type=str, default=MODELS_DIR)
    parser.add_argument("--chart_type", type=str, default='h_bar')
    parser.add_argument("--dataset", type=str, default='PlotQA')
    parser.add_argument("--dataset_version", type=str, default='v1')
    parser.add_argument("--partition", type=str, default='train')
    parser.add_argument("--val_partition", type=str, default='val')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_acc_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--n_evals_per_epoch", type=int, default=10)
    parser.add_argument("--metric", type=str, default="eval_cosine_accuracy@5")
    
    args = parser.parse_args()
    return args

def main():
    args = set_arguments()
    model_name = args.model_checkpoint.split('/')[-1]

    train_dataset_path = load_training_dataset_path(args)
    train_dataset = load_dataset("csv", data_files=train_dataset_path)["train"]

    model = SentenceTransformer(args.model_checkpoint)

    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    evaluation_steps = max(1, steps_per_epoch // args.n_evals_per_epoch)

    loss = MultipleNegativesRankingLoss(model)

    output_dir = os.path.join(CHECKPOINTS_DIR, args.chart_type,
                              f"{model_name}_{args.chart_type}_MNR_{args.batch_size}_{args.learning_rate}_"
                              f"{args.gradient_acc_steps}_{args.dataset_version}_IRE")
    os.makedirs(output_dir, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        gradient_accumulation_steps=args.gradient_acc_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=evaluation_steps,
        save_strategy="steps",
        save_steps=evaluation_steps,
        metric_for_best_model=args.metric,
        load_best_model_at_end=True,
        weight_decay=args.weight_decay,
        fp16=True,
        disable_tqdm=True
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=args.patience,
        early_stopping_threshold=0.0
    )

    ir_dir = os.path.join(args.base_path, args.dataset, args.val_partition, "informationRetrieval_dataset")
    queries_path = os.path.join(ir_dir, f"queries_{args.chart_type}_essential_flatten.json")
    corpus_path = os.path.join(ir_dir, f"corpus_{args.chart_type}_essential_flatten.json")
    relevant_docs_path = os.path.join(ir_dir, f"relevant_docs_{args.chart_type}_essential_flatten.json")

    queries = read_json(queries_path)
    corpus = read_json(corpus_path)
    relevant_docs = read_json(relevant_docs_path)

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="eval"
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=[early_stopping]
    )

    logging_args(args)

    trainer.train()
    logging.info(f"Val evaluation:\n{evaluator(model)}")

    save_model_dir = os.path.join(args.finetuned_model, args.chart_type,
                                  f"{model_name}_{args.chart_type}_MNR_{args.batch_size}_{args.learning_rate}_"
                                  f"{args.gradient_acc_steps}_{args.dataset_version}_IRE")
    
    os.makedirs(save_model_dir, exist_ok=True)
    trainer.save_model(save_model_dir)

    logging.info('Model training completed.')

if __name__ == '__main__':
    main()

