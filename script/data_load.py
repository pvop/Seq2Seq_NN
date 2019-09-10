import nltk
import re
from collections import Counter
import pickle
import os
from hyperparameters import Hyperparameters
hp = Hyperparameters()

import numpy as np
class data():
    def __init__(self):
        if not os.path.exists("../data/iwslt14/source_train_datas"):
            source_train_datas, target_train_datas = self.get_train_data()
            source_dev_datas, target_dev_datas, source_test_datas, target_test_datas = self.get_dev_test_data()
            source_word2id = self.get_vocab(source_train_datas,hp.source_vocab_size)
            target_word2id = self.get_vocab(target_train_datas,hp.target_vocab_size)
            self.source_word2id = source_word2id
            self.target_word2id = target_word2id
            self.source_train_datas,_ = self.convert_word_to_id(source_train_datas,source_word2id,is_source=True)
            print ("source_train_datas:",self.source_train_datas.shape)
            self.target_train_datas,self.target_train_length = self.convert_word_to_id(target_train_datas,target_word2id,is_source=False)
            print("target_train_datas:", self.target_train_datas.shape)
            self.source_dev_datas,_ = self.convert_word_to_id(source_dev_datas,source_word2id,is_source=True)
            print("source_dev_datas:", self.source_dev_datas.shape)
            self.target_dev_datas,_ = self.convert_word_to_id(target_dev_datas,target_word2id,is_source=False)
            print("target_dev_datas:", self.target_dev_datas.shape)
            self.source_test_datas,_ = self.convert_word_to_id(source_test_datas,source_word2id,is_source=True)
            print("source_test_datas:", self.source_test_datas.shape)
            self.target_test_datas,_ = self.convert_word_to_id(target_test_datas,target_word2id,is_source=False)
            print("target_test_datas:", self.target_test_datas.shape)
            pickle.dump(self.source_train_datas,open("../data/iwslt14/source_train_datas","wb"))
            pickle.dump(self.target_train_datas,open("../data/iwslt14/target_train_datas","wb"))
            pickle.dump(self.target_train_length, open("../data/iwslt14/target_train_length", "wb"))
            pickle.dump(self.source_dev_datas,open("../data/iwslt14/source_dev_datas","wb"))
            pickle.dump(self.target_dev_datas,open("../data/iwslt14/target_dev_datas","wb"))
            pickle.dump(self.source_test_datas,open("../data/iwslt14/source_test_datas","wb"))
            pickle.dump(self.target_test_datas,open("../data/iwslt14/target_test_datas","wb"))
            pickle.dump(self.source_word2id,open("../data/iwslt14/source_word2id","wb"))
            pickle.dump(self.target_word2id,open("../data/iwslt14/target_word2id","wb"))

        else:
            self.source_train_datas = pickle.load(open("../data/iwslt14/source_train_datas","rb"))
            self.target_train_datas = pickle.load(open("../data/iwslt14/target_train_datas","rb"))
            self.target_train_length = pickle.load(open("../data/iwslt14/target_train_length","rb"))
            self.source_dev_datas = pickle.load(open("../data/iwslt14/source_dev_datas","rb"))
            self.target_dev_datas = pickle.load(open("../data/iwslt14/target_dev_datas","rb"))
            self.source_test_datas = pickle.load(open("../data/iwslt14/source_test_datas","rb"))
            self.target_test_datas = pickle.load(open("../data/iwslt14/target_test_datas","rb"))
            self.source_word2id = pickle.load(open("../data/iwslt14/source_word2id","rb"))
            self.target_word2id = pickle.load(open("../data/iwslt14/target_word2id","rb"))

            print("source_train_datas:", self.source_train_datas.shape)
            print("target_train_datas:", self.target_train_datas.shape)
            print("source_dev_datas:", self.source_dev_datas.shape)
            print("target_dev_datas:", self.target_dev_datas.shape)
            print("source_test_datas:", self.source_test_datas.shape)
            print("target_test_datas:", self.target_test_datas.shape)
    def read_train_files(self,path):
        datas = open(path,"r").read().split("\n")[0:-1]
        datas = [data for data in datas if not (data.endswith("</talkid>") or  data.endswith("</title>") or data.endswith("</description>") or data.endswith("</url>"))]
        for i,data in enumerate(datas):
            data = data.lower()
            if i%10000==0:
                print (i,len(datas))
            datas[i] = nltk.word_tokenize(data)
        return datas
    def read_test_files(self,path):
        text = open(path,"r").read()
        datas = re.findall("<seg id=\".*?\"> (.*?) </seg>",text)
        for i,data in enumerate(datas):
            data = data.lower()
            if i%10000==0:
                print (i,len(datas))
            datas[i] = nltk.word_tokenize(data)
        return datas
    def get_train_data(self):
        source_train_datas = self.read_train_files("../data/iwslt14/train.tags.de-en.en")
        target_train_datas = self.read_train_files("../data/iwslt14/train.tags.de-en.de")
        new_source_train_datas = []
        new_target_train_datas = []
        for i in range(len(source_train_datas)):
            if len(source_train_datas[i])<hp.max_source_length and len(target_train_datas[i])<hp.max_target_length:
                new_source_train_datas.append(source_train_datas[i])
                new_target_train_datas.append(target_train_datas[i])
        return new_source_train_datas,new_target_train_datas
    def get_dev_test_data(self):
        source_dev_datas = self.read_test_files("../data/iwslt14/IWSLT14.TEDX.dev2012.de-en.en.xml")
        target_dev_datas = self.read_test_files("../data/iwslt14/IWSLT14.TEDX.dev2012.de-en.de.xml")
        source_test_datas = self.read_test_files("../data/iwslt14/IWSLT14.TED.tst2012.de-en.en.xml")
        target_test_datas = self.read_test_files("../data/iwslt14/IWSLT14.TED.tst2012.de-en.de.xml")
        return source_dev_datas,target_dev_datas,source_test_datas,target_test_datas
    def get_vocab(self,datas,vocab_size):
        words = []
        for data in datas:
            words.extend(data)
        word2id = {"<pad>":0,"<unk>":1,"<start>":2,"<end>":3}
        words = dict(Counter(words).most_common(vocab_size-len(word2id)))
        for word in words:
            word2id[word] = len(word2id)
        return word2id
    def convert_word_to_id(self,datas,word2id,is_source):
        datas_length = [len(data) for data in datas]
        max_length = max(datas_length)
        for i,data in enumerate(datas):
            if i%10000==0:
                print (i)
            for j,word in enumerate(data):
                datas[i][j] = word2id.get(word,1)
            datas[i] = datas[i]+[3] +[0]*(max_length-len(datas[i]))
            if is_source:
                datas[i].reverse()
        datas = np.array(datas)
        datas_length = np.array(datas_length)
        datas_length = datas_length+1
        return datas,datas_length



if __name__=="__main__":
    data = data()
