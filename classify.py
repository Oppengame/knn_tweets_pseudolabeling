import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments

from knn_pseudolabel import train_val_split, run_net_knn


def simple_accuracy(preds, labels):
    return (preds == labels).mean().item()


def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=-1)
    acc = simple_accuracy(preds, labels)
    f1 = sklearn.metrics.f1_score(y_true=labels, y_pred=preds)
    return {
        "accuracy": acc,
        "f1": f1,
    }


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_dataset(df, tokenizer, text_field='tweet_cleaned', label_field='label_knn'):
    texts = df[text_field].to_list()
    labels = df[label_field].to_list()
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = TweetDataset(encodings, labels)
    return dataset


@torch.no_grad()
def get_logits(texts, tokenizer, model, device='cpu', batch_size=128):
    logits = []
    for i in trange(0, len(texts), batch_size):
        encoded_input = tokenizer(texts[i:i + batch_size], return_tensors='pt',
                                  padding=True, truncation=True).to(device)
        output = model(**encoded_input)
        logits.append(output.logits.cpu())
    return torch.cat(logits, dim=0)


def run_net_class(knn_filename='knn_labeled.csv.gzip', train_share=0.8, use_val=True, dropout=0.1,
                  model_name="cardiffnlp/twitter-xlm-roberta-base", train_batch_size=64,
                  result_filename='class_labeled.csv.gzip', final_sample_count=5000,
                  final_filename='filtered_twitter_data.csv'):
    df = pd.read_parquet(knn_filename, engine='pyarrow')
    df_labeled = df[df['label_knn'] != -1]
    df_train, df_val = train_val_split(df_labeled, train_share=train_share, use_val=use_val)

    num_labels = df_labeled['label_knn'].max() + 1

    kwargs = {'hidden_dropout_prob': dropout, 'attention_probs_dropout_prob': dropout, 'classifier_dropout': dropout, }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, **kwargs)

    train_dataset = get_dataset(df_train, tokenizer)
    val_dataset = get_dataset(df_val, tokenizer)

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        output_dir='./results',  # output directory
        max_steps=150,
        per_device_train_batch_size=train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=train_batch_size * 8,  # batch size for evaluation
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        evaluation_strategy='steps',
        save_total_limit=5,  # Only last 5 models are saved.
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_texts = df['tweet_cleaned'].to_list()
    # test_labels = np.zeros(len(test_texts), dtype=np.int64)

    model = torch.nn.DataParallel(trainer.model)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logits = get_logits(test_texts, tokenizer, model, device=device, batch_size=train_batch_size * 8)
    probs = torch.softmax(logits, dim=-1)
    df['prob'] = probs[:, -1].cpu().numpy()

    df.to_csv(result_filename, compression='gzip')
    t = df.prob.quantile(1 - final_sample_count / len(df))
    df[(df['prob'] > t)].to_csv(final_filename)


def full_process(data_folder='assignment', train_tweet_column='clean_tweet', train_share=0.8, use_val=True,
                 knn_model="cardiffnlp/twitter-roberta-base-dec2021", inference_batch_size=2048, emb_type='max',
                 dist_type='cosine', knn_k=5, num_knn_pseudolabels=10000, like_thres=10,
                 knn_result_filename='knn_labeled.parquet.gzip', dropout=0.1,
                 class_model_name="cardiffnlp/twitter-xlm-roberta-base", train_batch_size=64,
                 class_results_filename='class_labeled.csv.gzip', final_sample_count=5000,
                 final_filename='filtered_twitter_data.csv'):
    run_net_knn(data_folder=data_folder, orig_tweet_col=train_tweet_column, train_share=train_share, use_val=use_val,
                model_name=knn_model, batch_size=inference_batch_size, emb_type=emb_type, dist_type=dist_type, k=knn_k,
                num_pos=num_knn_pseudolabels, like_thres=like_thres, result_filename=knn_result_filename)
    run_net_class(knn_filename=knn_result_filename, train_share=train_share, use_val=use_val, dropout=dropout,
                  model_name=class_model_name, train_batch_size=train_batch_size,
                  result_filename=class_results_filename, final_sample_count=final_sample_count,
                  final_filename=final_filename)
    
    full_process()
