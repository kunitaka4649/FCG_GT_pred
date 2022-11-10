# -*- coding: utf-8 -*-
"""
"""

import argparse
import os
import pickle as origin_pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

import utils


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--train_set", default="data/train_dev/TRAIN.prep_feedback_comment.public.tsv"
    )
    parser.add_argument(
        "--dev_set", default="data/train_dev/DEV.prep_feedback_comment.public.tsv"
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--model",
        default="roberta-large",
        choices=["bert-base-uncased", "roberta-base", "roberta-large"],
    )
    parser.add_argument("--grammar_term_set", default="data/grammar_terms/grammar_terms.small.set")
    parser.add_argument(
        "--given_offset_phrase",
        action="store_true",
        help="Flag to give offset phrase to source.",
    )
    parser.add_argument("--top_n", type=int, default=0)
    parser.add_argument("--exclude_grammar_terms_by_count", type=int, default=0)
    parser.add_argument("--exclude_preposition_tag", action="store_true")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("-lr", "--learning_rate", type=float, default=4e-5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--class_weight_pkl", default="data/grammar_terms_weight.pkl")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--dont_save_model", action="store_true")
    parser.add_argument("--thresh", type=float, default=0)
    parser.add_argument("--without_idf", action="store_true")
    args = parser.parse_args()
    return args


def dataset_to_df(
    dataset_path,
    gt2id,
    columns,
    exclude_grammar_terms_by_count,
    offset_phrase=False,
):
    data_list = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            source, offset, target = line.strip().split("\t")
            target = utils.clean_up_data(target)
            grammar_terms = utils.extract_grammar_terms(target)
            grammar_terms = utils.exclude_grammar_terms_by_count(
                grammar_terms, gt2id, 0
            )
            grammar_term_ids = []
            for term in grammar_terms:
                grammar_term_ids.append(gt2id[term])
            source = utils.add_info_to_source(
                source,
                offset=offset,
                offset_phrase=offset_phrase,
                add_prefix=False,
            )
            off_s, off_e = map(int, offset.split(":"))
            new_off_s = source[:off_s].count(" ")
            new_off_e = source[:off_e].count(" ") + 1
            new_data = [(new_off_s, new_off_e), source.lower()] + [grammar_term_ids]
            data_list.append(new_data)
    df = pd.DataFrame(data_list, columns=columns)
    return df


def one_hot_encoder(df, n_labels):
    one_hot_encoding = []
    for i in tqdm(range(len(df))):
        temp = [0] * n_labels
        label_indices = df.iloc[i]["labels"]
        for index in label_indices:
            temp[index] = 1
        one_hot_encoding.append(temp)
    return pd.DataFrame(one_hot_encoding)


class Dataset:
    def __init__(self, texts, labels, offsets, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.offsets = offsets

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        offset = self.offsets[index]

        inputs = self.tokenizer.__call__(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        noff_s = -1
        noff_e = -1
        for idx, id in enumerate(inputs.word_ids()):
            if noff_s == -1 and id == offset[0]:
                noff_s = idx
            if noff_e == -1 and id == offset[1]:
                noff_e = idx
        noffset = [noff_s, noff_e]
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
            "offsets": torch.tensor(noffset, dtype=torch.long),
        }


class Classifier(nn.Module):
    def __init__(self, n_train_steps, n_classes, do_prob, bert_model):
        super(Classifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(do_prob)
        self.out = nn.Linear(1024, n_classes)
        self.n_train_steps = n_train_steps
        self.step_scheduler_after = "batch"

    def forward(self, ids, mask, offsets):
        hiddens = self.bert(ids, attention_mask=mask)["last_hidden_state"]
        output_1 = []
        for hidden, offset in zip(hiddens, offsets):
            output_1.append(torch.mean(hidden[offset[0] : offset[1]], 0))
        output_1 = torch.stack(output_1)
        output_2 = self.dropout(output_1)
        output = self.out(output_2)
        return output


def build_dataloader(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_data_loader, valid_data_loader


def ret_optimizer(model, learning_rate):
    opt = AdamW(model.parameters(), lr=learning_rate)
    return opt

def ret_scheduler(optimizer, num_train_steps):
    sch = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_steps * 0.1,
        num_training_steps=num_train_steps,
    )
    return sch

def loss_fn(outputs, labels):
    if labels is None:
        return None
    return nn.BCEWithLogitsLoss()(outputs, labels.float())

def log_metrics(preds, labels):
    preds = torch.stack(preds)
    preds = preds.cpu().detach().numpy()
    labels = torch.stack(labels)
    labels = labels.cpu().detach().numpy()
    fpr_micro, tpr_micro, _ = metrics.roc_curve(labels.ravel(), preds.ravel())

    auc_micro = metrics.auc(fpr_micro, tpr_micro)
    return {"auc_micro": auc_micro}


def train_fn(data_loader, model, optimizer, device, scheduler, weight=None):
    train_loss = 0.0
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["labels"]
        offsets = d["offsets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, offsets=offsets)
        outputs = outputs * torch.tensor(weight).to(device)
        loss = loss_fn(outputs, targets)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()
    return train_loss


def eval_fn(data_loader, model, device, weight=None):
    eval_loss = 0.0
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["labels"]
            offsets = d["offsets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask, offsets=offsets)
            outputs = outputs * torch.tensor(weight).to(device)
            loss = loss_fn(outputs, targets)
            eval_loss += loss.item()
            fin_targets.extend(targets)
            fin_outputs.extend(torch.sigmoid(outputs))
    return eval_loss, fin_outputs, fin_targets


def load_weight():
    return weight

def main(config):
    args = init_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gts = utils.read_label(
        args.grammar_term_set
    )
    if args.top_n != 0: # if set top_n
        gts = gts[:args.top_n]
    gt2id = {gt: i for i, gt in enumerate(gts)}

    columns = ["offset", "text", "labels"]
    train = dataset_to_df(
        args.train_set,
        gt2id,
        columns,
        args.exclude_grammar_terms_by_count,
        args.given_offset_phrase,
    )
    valid = dataset_to_df(
        args.dev_set,
        gt2id,
        columns,
        args.exclude_grammar_terms_by_count,
        args.given_offset_phrase,
    )
    test = valid.copy()

    train.head()
    mapping = {value: key for key, value in gt2id.items()}

    n_labels = len(mapping)

    train_ohe_labels = one_hot_encoder(train, n_labels)
    valid_ohe_labels = one_hot_encoder(valid, n_labels)
    test_ohe_labels = one_hot_encoder(test, n_labels)

    train = pd.concat([train, train_ohe_labels], axis=1)
    valid = pd.concat([valid, valid_ohe_labels], axis=1)
    test = pd.concat([test, test_ohe_labels], axis=1)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, do_lower_case=True
    )

    bert_model = transformers.AutoModel.from_pretrained(args.model)

    # prepare datasets
    train_dataset = Dataset(
        train.text.tolist(),
        train[range(n_labels)].values.tolist(),
        train.offset.tolist(),
        tokenizer,
        config["tokenizer_max_len"],
    )
    valid_dataset = Dataset(
        valid.text.tolist(),
        valid[range(n_labels)].values.tolist(),
        valid.offset.tolist(),
        tokenizer,
        config["tokenizer_max_len"],
    )
    train_data_loader, valid_data_loader = build_dataloader(
        train_dataset, valid_dataset, config["batch_size"]
    )
    print("Length of Train Dataloader: ", len(train_data_loader))
    print("Length of Valid Dataloader: ", len(valid_data_loader))

    # calculate weight loss of each class
    with open(args.class_weight_pkl, "rb") as f:
        weight = origin_pickle.load(f)
    if args.without_idf:
        weight = [1] * len(weight)

    # use gpu if it can be used.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_train_steps = int(len(train_dataset) / config["batch_size"] * config["epochs"])

    model = Classifier(n_train_steps, n_labels, config["dropout"], bert_model=bert_model)
    optimizer = ret_optimizer(model, config["learning_rate"])
    scheduler = ret_scheduler(optimizer, n_train_steps)
    model.to(device)
    model = nn.DataParallel(model)

    n_epochs = config["epochs"]

    best_val_loss = 100
    for epoch in tqdm(range(n_epochs)):

        train_loss = train_fn(
            train_data_loader, model, optimizer, device, scheduler, weight=weight
        )
        eval_loss, preds, labels = eval_fn(
            valid_data_loader, model, device, weight=weight
        )

        auc_score = log_metrics(preds, labels)["auc_micro"]
        print("AUC score: ", auc_score)
        avg_train_loss, avg_val_loss = (
            train_loss / len(train_data_loader),
            eval_loss / len(valid_data_loader),
        )
        print("Average Train loss: ", avg_train_loss)
        print("Average Valid loss: ", avg_val_loss)

        out_path = os.path.join(args.out_dir, f"lr_{config['learning_rate']}.bin")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), out_path)
            print("Model saved as current val_loss is: ", best_val_loss)


if __name__ == "__main__":
    seed_everything(1234)

    config = {
        "method": "grid",
        "metric": {"name": "auc_score", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"values": [1e-5, 3e-5, 1e-4]},
            "batch_size": {"values": [8]},
            "epochs": {"value": 10},
            "dropout": {"values": [0.1]},
            "tokenizer_max_len": {"value": 256},
        },
    }

    def gridsearch(parameters):
        for lr in parameters['learning_rate']['values']:
            for bs in parameters['batch_size']['values']:
                for d in parameters['dropout']['values']:
                    param = {
                        "learning_rate": lr,
                        "batch_size": bs,
                        "epochs": parameters['epochs']['value'],
                        "dropout": d,
                        "tokenizer_max_len": parameters['tokenizer_max_len']['value']
                    }
                    yield param

    for param in gridsearch(config["parameters"]):
        main(config=param)