import argparse
import os
import random
from typing import Optional

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from torch import nn
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    BertModel,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_DISABLED"] = "true"


def generate_examples(filepath):
    print(f"Generating example from {filepath}")
    guid, tokens, ner_tags = 0, [], []
    with open(filepath, encoding="utf-8") as rf:
        for line in rf:
            if line == "" or line =="\n":
                if tokens:
                    yield {
                        "id": str(guid),
                        "tokens": tokens,
                        "ner_tags": ner_tags
                    }
                    guid += 1
                    tokens, ner_tags = [], []
            else:
                splits = line.rstrip().split()
                tokens.append(splits[0])
                ner_tags.append(splits[-1])
        if tokens:
            yield {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags
            }


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_col_name],
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_length,
        is_split_into_words=True,
    )
    labels, words = [], []
    for i, label in enumerate(examples[label_col_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        tokens = tokenized_inputs.tokens(batch_index=i)
        previous_word_idx = None
        label_ids, new_word_ids = [], []
        for word_idx, t in zip(word_ids, tokens):
            if word_idx is None:
                label_ids.append(-100)
                word_idx = -100
            elif t in ['[CLS]', '[SEP]']:
                label_ids.append(-100)
                word_idx = -100
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
            new_word_ids.append(word_idx)
        labels.append(label_ids)
        words.append(new_word_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_dataset_length(dataset):
    dataset_len = len(dataset["tokens"])
    for i in range(dataset_len-1, -1, -1):
        if len(dataset["tokens"][i]) != 0:
            dataset_len = i+1
            break
    return dataset_len


class EarlyStoppingCallback_minEpoch(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0,
                 min_epochs: int = 0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience_counter = -min_epochs

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if self.early_stopping_patience_counter > 0 and (
                state.best_metric is None or (
                operator(metric_value, state.best_metric)
                and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        )):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1


class SCELoss(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, ignore_index=-100):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        mask = (target != self.ignore_index)

        filtered_pred = pred[mask]
        filtered_target = target[mask]

        if filtered_target.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        ce_loss = nn.CrossEntropyLoss()(filtered_pred, filtered_target)

        pred_softmax = torch.softmax(filtered_pred, dim=1)
        target_one_hot = torch.zeros_like(filtered_pred).scatter(1, filtered_target.unsqueeze(1), 1)
        rce_loss = -torch.mean(torch.sum(pred_softmax * torch.log(target_one_hot + 1e-12), dim=1))

        loss = self.alpha * ce_loss + self.beta * rce_loss
        return loss


class NER(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        hidden_size = config.hidden_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        if config.loss_func == "sce":
            self.loss_fct = SCELoss(alpha=config.sce_alpha)
        else:
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # print(input_ids)
        # print(labels)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        #print("labels max:", labels.max().item(), "num_labels:", self.num_labels)
        #print("logits max:", logits.max().item(), "min:", logits.min().item())

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        results = torch.argmax(logits, dim=-1)
        return ((loss,) + (results,)) if loss is not None else results


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--predict_file", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--label_list", default="Disease")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--loss_func", default="ce")
    parser.add_argument("--sce_alpha", default=0.2, type=float)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--min_epochs", default=25, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--early_stopping_patience", default=5, type=int)
    parser.add_argument("--early_stopping_threshold", default=0.0001, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite_cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    set_seed(args.seed)

    text_col_name, label_col_name = "tokens", "ner_tags"
    data_files = {}

    if args.train_file is not None:
        examples = list(generate_examples(args.train_file))
        data_files["train"] = Dataset.from_pandas(pd.DataFrame(examples))
    if args.predict_file is not None:
        examples = list(generate_examples(args.predict_file))
        data_files["predict"] = Dataset.from_pandas(pd.DataFrame(examples))
    raw_data = DatasetDict(data_files)

    if "train" in raw_data:
        column_names = raw_data["train"].column_names
        features = raw_data["train"].features

    label_list = sorted(args.label_list.split(","))
    #print(label_list)
    label_to_id = {l: i for i, l in enumerate(label_list)}
    #print(label_to_id)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        finetuning_task="ner"
    )
    config.label_list = label_list
    config.loss_func = args.loss_func
    config.sce_alpha = args.sce_alpha
    config.label2id = {l: i for i, l in enumerate(label_list)}
    config.id2label = dict(enumerate(label_list))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        add_prefix_space=True,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    mapping_func = tokenize_and_align_labels
    if args.do_train:
        if "train" not in raw_data:
            raise ValueError("--do_train requires a train dataset")
        train_data = raw_data["train"]
        data_len = get_dataset_length(train_data)
        train_data = train_data.map(
            mapping_func,
            batched=True,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        ).shuffle(seed=args.seed)

    if args.do_predict:
        if "predict" not in raw_data:
            raise ValueError("--do_predict requires a predict dataset")
        predict_data = raw_data["predict"]
        data_len = get_dataset_length(predict_data)
        predict_data = predict_data.map(
            mapping_func,
            batched=True,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on prediction dataset"
        )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    metric = evaluate.load("seqeval")

    ert_cb = EarlyStoppingCallback_minEpoch(early_stopping_patience=args.early_stopping_patience,
                                            early_stopping_threshold=args.early_stopping_threshold,
                                            min_epochs=args.min_epochs)

    def remove_ignored_index(predictions, labels):
        true_predictions = []
        true_labels = []

        for pred, label in zip(predictions, labels):
            pred_labels = []
            true_labels_seq = []
            for p_i, l_i in zip(pred, label):
                if l_i != -100:  # ignore padding
                    pred_labels.append(label_list[p_i])
                    true_labels_seq.append(label_list[l_i])
            true_predictions.append(pred_labels)
            true_labels.append(true_labels_seq)

        return true_predictions, true_labels

    def compute_metrics(results):
        predict, answer = results
        predict, answer = remove_ignored_index(predict, answer)
        results = metric.compute(predictions=predict, references=answer)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        # return metrics

    def model_init(trial):
        model = NER.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        model.classifier.reset_parameters()
        return model

    if args.do_train:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=5,
            num_train_epochs=args.epochs,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            adam_epsilon=1e-6,
            weight_decay=0.01,
            seed=args.seed,
        )
    else:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
        )
    trainer = Trainer(
        model=model_init(None),
        args=training_args,
        train_dataset=train_data if args.do_train else None,
        eval_dataset=train_data if args.do_train else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[ert_cb],
    )

    if args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        metrics["train_samples"] = len(train_data)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        tokenizer.save_pretrained(trainer.args.output_dir)

    if args.do_predict:
        predictions, labels, metrics = trainer.predict(predict_data, metric_key_prefix="predict")
        true_predictions, true_labels = remove_ignored_index(predictions, labels)
        results = metric.compute(predictions=true_predictions, references=true_labels)
        metrics = {
            "predict_precision": results["overall_precision"],
            "predict_recall": results["overall_recall"],
            "predict_f1": results["overall_f1"],
        }
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        with open(os.path.join(args.output_dir, "predictions.txt"), "w") as writer:
            tokens = predict_data["tokens"]
            i = 0
            for token, prediction in zip(tokens, true_predictions):
                sentence_idx = 0
                for t, p in zip(token, prediction):
                    writer.write(t + " " + p + "\n")
                writer.write("\n")
                i += 1

