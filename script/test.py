import tensorflow as tf
from data_load import data
import os
from model import Model
from nltk.translate.bleu_score import corpus_bleu
from hyperparameters import Hyperparameters
hp = Hyperparameters()

os.environ['CUDA_VISIBLE_DEVICES']=str(0)
data = data()
model = Model(mode="test")
session_config = tf.ConfigProto(
            log_device_placement=False,
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0,
            allow_soft_placement=True)
session_config.gpu_options.allow_growth = True
session_config.gpu_options.allocator_type = 'BFC'
with tf.Session(config=session_config) as sess:
    saver = tf.train.Saver()
    saver.restore(sess,"../model/result.ckpt")
    predicts = model.infer(sess,data.source_test_datas,data.target_test_datas,hp.batch_size)
    fout = open("./result.txt", "w", encoding="utf-8")
    list_of_refs = []
    hypotheses = []
    source_id2word = dict(zip(data.source_word2id.values(),data.source_word2id.keys()))
    target_id2word = dict(zip(data.target_word2id.values(),data.target_word2id.keys()))
    for source, target, pred in zip(data.source_test_datas, data.target_test_datas, predicts):
        source = list(source)
        source.reverse()
        source = " ".join(source_id2word[idx] for idx in source).split("<end>")[0].strip()
        target = " ".join(target_id2word[idx] for idx in target).split("<end>")[0].strip()
        pred = " ".join(target_id2word[idx] for idx in pred).split("<end>")[0].strip()
        fout.write("- source: " + source + "\n")
        fout.write("- expected: " + target + "\n")
        fout.write("- got: " + pred + "\n\n")
        # bleu score
        ref = target.split()
        hypothesis = pred.split()
        if len(ref) > 3 and len(hypothesis) > 3:
            list_of_refs.append([ref])
            hypotheses.append(hypothesis)
        ## Calculate bleu score
    score = corpus_bleu(list_of_refs, hypotheses)
    fout.write("Bleu Score = " + str(100 * score))



