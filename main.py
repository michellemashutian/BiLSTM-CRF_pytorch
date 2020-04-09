#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 下午1:39
# @Author  : shutian
# @File    : train.py


import codecs
import torch
import os
import itertools
import pickle
import numpy as np
from config import Config
from data_util import load_sentences, char_mapping, tag_mapping, augment_with_pretrained, prepare_dataset
from utils import BatchManager
from conlleval import return_report
from model import BiLSTM_CRF
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def get_predictions(model, data_set, id_to_tag):
    model.eval()
    with torch.no_grad():
        results = []
        for data in data_set:
            strings, inputs, segs, tags = data
            score, preds = model(inputs)
            result = []
            for i in range(len(strings)):
                string = strings[i]
                gold = id_to_tag[int(tags[i])]
                pred = id_to_tag[int(preds[i])]
                result.append(" ".join([string, gold, pred]))
            results.append(result)
        return results


def evaluate_ner(results, conf):
    with open(conf.result_file, "w", encoding='utf-8') as f:
        to_write = []
        for block in results:
            print(block)
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")
        f.writelines(to_write)
    eval_lines = return_report(conf.result_file)
    for line in eval_lines:
        print(line)
    f1 = float(eval_lines[1].strip().split()[-1])
    return f1


def train(conf):
    train_sentences = load_sentences(conf.train_file, conf.zeros)
    dev_sentences = load_sentences(conf.dev_file, conf.zeros)
    test_sentences = load_sentences(conf.test_file, conf.zeros)

    dico_chars_train = char_mapping(train_sentences, conf.lower)[0]
    dico_chars, char_to_id, id_to_char = augment_with_pretrained(
        dico_chars_train.copy(),
        conf.emb_file,
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in test_sentences])
        )
    )
    _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, conf.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, conf.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, conf.lower
    )

    #loading word embeddings
    all_word_embeds = {}
    for i, line in enumerate(codecs.open(conf.emb_file, 'r', 'utf-8')):
        s = line.strip().split()
        if len(s) == conf.embedding_dim + 1:
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])
    word_embeds_dict = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(char_to_id), conf.embedding_dim))
    for w in char_to_id:
        if w in all_word_embeds:
            word_embeds_dict[char_to_id[w]] = all_word_embeds[w]
        elif w.lower() in all_word_embeds:
            word_embeds_dict[char_to_id[w]] = all_word_embeds[w.lower()]
    print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

    train_manager = BatchManager(train_data, conf.batch_size)

    model = BiLSTM_CRF(conf, tag_to_id, char_to_id, word_embeds_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, weight_decay=1e-4)
    epoch = conf.epochs
    dev_f1_ = 0
    for epoch in range(1, epoch+1):
        print(f'train on epoch {epoch}')
        j = 1
        for batch in train_manager.iter_batch(shuffle=True):
            batch_loss = 0.0
            sentences = batch[1]
            tags = batch[-1]
            for i, index in enumerate(np.random.permutation(len(sentences))):
                model.zero_grad()
                sentence_in = sentences[index]
                tags_in = tags[index]
                loss = model.neg_log_likelihood(sentence_in, tags_in)
                loss.backward()
                optimizer.step()
                batch_loss += loss.data
            print(
                f'[batch {j},batch size:{conf.batch_size}] On this batch loss: {batch_loss}')
            j = j+1
        print(f'Begin validing result on [epoch {epoch}] valid dataset ...')
        dev_results = get_predictions(model, dev_data, id_to_tag)
        dev_f1 = evaluate_ner(dev_results, conf)
        if dev_f1 > dev_f1_:
            torch.save(model, conf.model_file)
            print('save model success.')
        test_results = get_predictions(model, test_data, id_to_tag)
        test_f1 = evaluate_ner(test_results, conf)
        print(f'[epoch {epoch}] On test dataset] f1: {test_f1:3f}')


def predict_data(conf):
    test_sentences = load_sentences(conf.test_file, conf.zeros)
    with open(conf.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, conf.lower
    )
    model = torch.load(conf.model_file)
    results = get_predictions(model, test_data, id_to_tag)
    print(results)


if __name__ == "__main__":
    con_path = r'/data/mashutian/workspace/NER/data/con.json'
    config = Config(config_file=con_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.visible_device_list)
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    train(config)
    predict_data(config)
