#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:44
# @Author  : Tianyu Liu

import sys
import os
import jittor as jt
import argparse
import time
from SeqUnit import *
from DataLoader import DataLoader
import numpy as np
from PythonROUGE import PythonROUGE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from preprocess import *
from util import *

parser = argparse.ArgumentParser()

parser.add_argument("--hidden_size", type=int, default=500, help="Size of each layer.")
parser.add_argument("--emb_size", type=int, default=400, help="Size of embedding.")
parser.add_argument("--field_size", type=int, default=50, help="Size of embedding.")
parser.add_argument("--pos_size", type=int, default=5, help="Size of embedding.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size of train set.")
parser.add_argument("--epoch", type=int, default=50, help="Number of training epoch.")
parser.add_argument("--source_vocab", type=int, default=20003, help="vocabulary size.")
parser.add_argument("--field_vocab", type=int, default=1480, help="vocabulary size.")
parser.add_argument("--position_vocab", type=int, default=31, help="vocabulary size.")
parser.add_argument("--target_vocab", type=int, default=20003, help="vocabulary size.")
parser.add_argument("--report", type=int, default=5000, help="report valid results after some steps.")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="learning rate.")

parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--load', type=str, default='0', help='load directory')
parser.add_argument('--dir', type=str, default='processed_data', help='data set directory')
parser.add_argument('--limits', type=int, default=0, help='max data set size')

parser.add_argument("--dual_attention", type=bool, default=False, help='dual attention layer or normal attention')
parser.add_argument("--fgate_encoder", type=bool, default=True, help='add field gate in encoder lstm')

parser.add_argument("--field", type=bool, default=False, help='concat field information to word embedding')
parser.add_argument("--position", type=bool, default=False, help='concat position information to word embedding')
parser.add_argument("--encoder_pos", type=bool, default=True, help='position information in field-gated encoder')
parser.add_argument("--decoder_pos", type=bool, default=True, help='position information in dual attention decoder')

args = parser.parse_args()

# FLAGS = tf.app.flags.FLAGS
last_best = 0.0

gold_path_test = 'processed_data/test/test_split_for_rouge/gold_summary_'
gold_path_valid = 'processed_data/valid/valid_split_for_rouge/gold_summary_'

# test phase
if args.load != "0":
    save_dir = 'results/res/' + args.load + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = 'results/evaluation/' + args.load + '/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'
# train phase
else:
    prefix = str(int(time.time() * 1000))
    save_dir = 'results/res/' + prefix + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = 'results/evaluation/' + prefix + '/'
    os.mkdir(save_dir)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'

log_file = save_dir + 'log.txt'


def train(dataloader, model):
    model.train()
    write_log("#######################################################")
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    write_log("#######################################################")
    trainset = dataloader.train_set
    k = 0
    loss, start_time = 0.0, time.time()
    for _ in range(args.epoch):
        for x in dataloader.batch_iter(trainset, args.batch_size, True):
            loss += model(x)
            k += 1
            progress_bar(k%args.report, args.report)
            if (k % args.report == 0):
                cost_time = time.time() - start_time
                write_log("%d : loss = %.3f, time = %.3f " % (k // args.report, loss, cost_time))
                loss, start_time = 0.0, time.time()
                if k // args.report >= 1:
                    ksave_dir = save_model(model, save_dir, k // args.report)
                    write_log(evaluate(dataloader, model, ksave_dir, 'valid'))
                    


def test(dataloader, model):
    evaluate(dataloader, model, save_dir, 'test')

def save_model(model, save_dir, cnt):
    new_dir = save_dir + 'loads' + '/' 
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    nnew_dir = new_dir + str(cnt) + '/'
    if not os.path.exists(nnew_dir):
        os.mkdir(nnew_dir)
    model.save(nnew_dir)
    return nnew_dir

def evaluate(dataloader, model, ksave_dir, mode='valid'):
    model.eval()
    if mode == 'valid':
        # texts_path = "original_data/valid.summary"
        texts_path = "processed_data/valid/valid.box.val"
        gold_path = gold_path_valid
        evalset = dataloader.dev_set
    else:
        # texts_path = "original_data/test.summary"
        texts_path = "processed_data/test/test.box.val"
        gold_path = gold_path_test
        evalset = dataloader.test_set

    # for copy words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]
    v = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []

    k = 0
    for x in dataloader.batch_iter(evalset, args.batch_size, False):
        predictions, atts = model(x)
        atts = np.squeeze(atts)
        idx = 0
        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum, unk_sum, mask_sum = [], [], []
                for tk, tid in enumerate(summary):
                    if tid == 3:
                        sub = texts[k][np.argmax(atts[tk, : len(texts[k]), idx])]
                        real_sum.append(sub)
                        mask_sum.append("**" + str(sub) + "**")
                    else:
                        real_sum.append(v.id2word(tid))
                        mask_sum.append(v.id2word(tid))
                    unk_sum.append(v.id2word(tid))
                sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                pred_list.append([str(x) for x in real_sum])
                pred_unk.append([str(x) for x in unk_sum])
                pred_mask.append([str(x) for x in mask_sum])
                k += 1
                idx += 1
    write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
    write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")

    for tk in range(k):
        with open(gold_path + str(tk), 'r') as g:
            gold_list.append([g.read().strip().split()])

    gold_set = [[gold_path + str(i)] for i in range(k)]
    pred_set = [pred_path + str(i) for i in range(k)]

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_list)
    copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
                  (str(F_measure), str(recall), str(precision), str(bleu))
    # print copy_result

    for tk in range(k):
        with open(pred_path + str(tk), 'w') as sw:
            sw.write(" ".join(pred_unk[tk]) + '\n')

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_unk)
    nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
                    (str(F_measure), str(recall), str(precision), str(bleu))
    # print nocopy_result
    result = copy_result + nocopy_result
    # print result
    if mode == 'valid':
        print(result)

    return result


def write_log(s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')


def main():

    copy_file(save_file_dir)
    dataloader = DataLoader(args.dir, args.limits)
    model = SeqUnit(hidden_size=args.hidden_size, emb_size=args.emb_size,
                    field_size=args.field_size, pos_size=args.pos_size, field_vocab=args.field_vocab,
                    source_vocab=args.source_vocab, position_vocab=args.position_vocab,
                    target_vocab=args.target_vocab, name="seq2seq",
                    field_concat=args.field, position_concat=args.position,
                    fgate_enc=args.fgate_encoder, dual_att=args.dual_attention, decoder_add_pos=args.decoder_pos,
                    encoder_add_pos=args.encoder_pos, lr=args.learning_rate)
    # copy_file(save_file_dir)
    if args.load != '0':
        model.load(save_dir)
    if args.mode == 'train':
        train(dataloader, model)
    else:
        test(dataloader, model)


if __name__=='__main__':
    # with tf.device('/gpu:' + FLAGS.gpu):
    print(np.version.version)
    main()
