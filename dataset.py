import numpy as np
import pandas as pd
import tldextract

table = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '.': 10,
    'a': 11,
    'b': 12,
    'c': 13,
    'd': 14,
    'e': 15,
    'f': 16,
    'g': 17,
    'h': 18,
    'i': 19,
    'j': 20,
    'k': 21,
    'l': 22,
    'm': 23,
    'n': 24,
    'o': 25,
    'p': 26,
    'q': 27,
    'r': 28,
    's': 29,
    't': 30,
    'u': 31,
    'v': 32,
    'w': 33,
    'x': 34,
    'y': 35,
    'z': 36,
    '-': 37,
    '_': 38
}

suffixes = ['.com', '.net', '.biz', '.ru', '.org', '.co.uk', '.info', '.cc', '.ws', '.cn']

pad_value = 40

def text2seq(text):
    return [table[c] for c in text]

def pad_seq(s, max_len):
    if len(s) > max_len:
        return s[:max_len]
    else:
        return s + ([pad_value]*(max_len-len(s)))

def remove_tld(domain):
    return tldextract.extract(domain).domain

def load_data(val_number, max_len, tld=True):
    df_pos = pd.read_csv('data/all_dga.txt', names=['domain', 'label'])
    df_neg = pd.read_csv('data/all_legit.txt', names=['domain', 'label'])
    df_pos['target'] = 1
    df_neg['target'] = 0
    if tld:
        df_pos['domain'] = df_pos.domain.apply(remove_tld)
        df_neg['domain'] = df_neg.domain.apply(remove_tld)

    all_data = pd.concat([df_pos, df_neg])

    all_data['feature'] = all_data.domain.apply(lambda x: pad_seq(text2seq(x), max_len))

    idx = list(range(len(all_data)))
    np.random.shuffle(idx)

    train = all_data.iloc[idx[:val_number]]
    test = all_data.iloc[idx[val_number:]]

    return (np.array(list(train.feature.values)), np.array(list(train.target.values))),(np.array(list(test.feature.values)), np.array(list(test.target.values)))

if __name__ == '__main__':
    print("dataset main")
