import json
import os
import re
import time
import urllib.request

import pandas as pd
import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel


def replace_names(tweet, cashtag, name):
    cashtag = '\$' + cashtag
    tweet = re.sub(cashtag, 'COMPANY_CASHTAG', tweet, flags=re.I)
    if name != '':
        tweet = re.sub(re.escape(name), 'COMPANY_NAME', tweet, flags=re.I)
    return tweet


def get_yahoo_shortname(symbol):
    url = f'https://query2.finance.yahoo.com/v1/finance/search?q={symbol}'
    response = urllib.request.urlopen(url)
    content = response.read()
    data = json.loads(content.decode('utf8'))['quotes'][0]['shortname']
    return data


def get_clean_shortname(symbol):
    try:
        sn = get_yahoo_shortname(symbol)
    except:
        return ''
    drop = [', Inc.', "Inc.", ', Incorporated', ', Ltd.']
    for d in drop:
        sn = sn.replace(d, '')
    time.sleep(2)  # hack
    return sn.strip()


def read_and_fix(path):
    df = pd.read_parquet(path, engine='pyarrow')
    # list saved as string
    df['cashtags'] = df['cashtags'].apply(lambda x: eval(eval(x[0])[0]))
    df['urls'] = df['urls'].apply(lambda x: eval(eval(x[0])[0]))
    return df


def process_df(df, orig_tweet_col, shortname_dict):
    tweet_col = 'tweet_cleaned'
    df = df[df[orig_tweet_col].apply(len) > 4]
    df = df.drop_duplicates(subset=[orig_tweet_col])
    df['company_name'] = df['cashtags'].apply(lambda x: shortname_dict[x[0]])
    df['tweet_cleaned'] = df.apply(
        lambda x: replace_names(x[orig_tweet_col], x['cashtags'][0], x['company_name']), axis=1)

    df = df.drop_duplicates(subset=['tweet_cleaned'])
    return df, tweet_col


def get_preprocessed_data(data_folder='assignment', orig_tweet_col='clean_tweet'):
    unlabeled = os.path.join(data_folder, 'tweets.parquet')
    labeled = os.path.join(data_folder, 'filtered.parquet')

    df_labeled = read_and_fix(labeled)
    df_unlabeled = read_and_fix(unlabeled)

    all_cashtags = list(df_labeled['cashtags'].apply(lambda x: x[0]).unique()) + \
                   list(df_unlabeled['cashtags'].apply(lambda x: x[0]).unique())

    shortname_dict = {x: get_clean_shortname(x) for x in tqdm(all_cashtags)}

    df_labeled, tweet_col = process_df(df_labeled, orig_tweet_col, shortname_dict)
    df_unlabeled, _ = process_df(df_unlabeled, orig_tweet_col, shortname_dict)
    return df_labeled, df_unlabeled, tweet_col


def train_val_split(df_labeled, train_share=0.8, use_val=True):
    last_train = df_labeled.reset_index()['date'].quantile(train_share)
    if use_val:
        df_labeled_train = df_labeled[df_labeled.index <= last_train]
    else:
        df_labeled_train = df_labeled
    df_labeled_val = df_labeled[df_labeled.index > last_train]
    return df_labeled_train, df_labeled_val


@torch.no_grad()
def get_embeddings(texts, tokenizer, model, device='cpu', batch_size=128):
    mean_embedding, max_embedding = [], []
    for i in trange(0, len(texts), batch_size):
        encoded_input = tokenizer(texts[i:i + batch_size], return_tensors='pt',
                                  padding=True, truncation=True).to(device)
        output = model(**encoded_input)
        mean_embedding.append(output.last_hidden_state.mean(-2).cpu())
        max_embedding.append(output.last_hidden_state.max(-2)[0].cpu())
    return torch.cat(mean_embedding, dim=0), torch.cat(max_embedding, dim=0)


def cosine_dist(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return -sim_mt


def get_recall(df_unlabeled, df_val, num_pos):
    thres = df_unlabeled['kth_dist'].quantile(num_pos / len(df_unlabeled))
    chosen = len(df_val[df_val['kth_dist'] <= thres])
    tot = len(df_val)
    return chosen / tot


def run_net_knn(data_folder='assignment', orig_tweet_col='clean_tweet', train_share=0.8, use_val=True,
            model_name="cardiffnlp/twitter-roberta-base-dec2021", batch_size=2048, emb_type='max',
            dist_type='cosine', k=5, num_pos=10000, like_thres=10, result_filename='knn_labeled.parquet.gzip'):
    df_labeled, df_unlabeled, tweet_col = get_preprocessed_data(data_folder=data_folder, orig_tweet_col=orig_tweet_col)
    df_labeled_train, df_labeled_val = train_val_split(df_labeled, train_share=train_share, use_val=use_val)

    unlabeled_text = df_unlabeled[tweet_col].to_list()
    labeled_train_text = df_labeled_train[tweet_col].to_list()
    labeled_val_text = df_labeled_val[tweet_col].to_list()

    val_idx = df_unlabeled[tweet_col].isin(set(labeled_val_text))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model = torch.nn.DataParallel(model)

    labeled_train_mean_emb, labeled_train_max_emb = get_embeddings(labeled_train_text, tokenizer, model,
                                                                   device, batch_size=batch_size)
    unlabeled_mean_emb, unlabeled_max_emb = get_embeddings(unlabeled_text, tokenizer, model,
                                                           device, batch_size=batch_size)

    if emb_type == 'max':
        labeled_train_emb, unlabeled_emb = labeled_train_max_emb, unlabeled_max_emb
    elif emb_type == 'mean':
        labeled_train_emb, unlabeled_emb = labeled_train_mean_emb, unlabeled_mean_emb
    else:
        raise ValueError("Wrong embedding type")

    if dist_type == 'cosine':
        distances = cosine_dist(labeled_train_emb, unlabeled_emb)
    elif dist_type == 'l2':
        distances = torch.cdist(labeled_train_emb.unsqueeze(0), unlabeled_emb.unsqueeze(0)).squeeze()
    else:
        raise ValueError("Wrong distance type")

    closest, indices = torch.kthvalue(distances, k, dim=0)
    df_unlabeled['kth_dist'] = closest.cpu().numpy()

    df_val = df_unlabeled[val_idx]
    df_unlabeled['original_val'] = 0
    df_unlabeled.loc[val_idx, 'original_val'] = 1

    recall = get_recall(df_unlabeled, df_val, num_pos)
    print(f'k-nn recall is {recall}')

    cond = df_unlabeled['nlikes'] > like_thres
    df_liked = df_unlabeled[cond]
    thres_pos = df_liked['kth_dist'].quantile(10000 / len(df_liked))
    thres_neg = df_liked['kth_dist'].quantile(1 - 10000 / len(df_liked))
    df_unlabeled['label_knn'] = -1
    df_unlabeled.loc[cond & (df_unlabeled['kth_dist'] < thres_pos) & ~val_idx, 'label_knn'] = 1
    df_unlabeled.loc[cond & (df_unlabeled['kth_dist'] > thres_neg) & ~val_idx, 'label_knn'] = 0

    df_unlabeled.to_parquet(result_filename, compression='gzip')
