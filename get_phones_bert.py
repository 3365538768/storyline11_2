# -*- coding: utf-8 -*-
import os
import sys, numpy as np, traceback, pdb
import os.path
from glob import glob
from tqdm import tqdm
from text.cleaner import clean_text
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
import shutil
import torch
bert_pretrained_dir = "models/chinese-roberta-wwm-ext-large"
i_part = 0
all_parts = 1
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
version = os.environ.get('version', None)
opt_dir = "resources/phones_bert"
txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
bert_dir = "%s/3-bert" % (opt_dir)
os.makedirs(opt_dir, exist_ok=True)  # 递归创建目录，当 exist_ok=True 时，如果目标目录已经存在，os.makedirs 不会抛出 FileExistsError 异常
os.makedirs(bert_dir, exist_ok=True)
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
if os.path.exists(bert_pretrained_dir):
    ...  # 不存在则报错
else:
    raise FileNotFoundError(bert_pretrained_dir)
tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)  # 加载预训练模型的分词器——把句子拆分成可识别的单元，再把单元转化为数字
bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)  # 加载预训练模型
if is_half == True:  # bert半精度设置
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)
language_v1_to_language_v2 = {
    "ZH": "zh",
    "zh": "zh",
    "JP": "ja",
    "jp": "ja",
    "JA": "ja",
    "ja": "ja",
    "EN": "en",
    "en": "en",
    "En": "en",
    "KO": "ko",
    "Ko": "ko",
    "ko": "ko",
    "yue": "yue",
    "YUE": "yue",
    "Yue": "yue",
}



def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


def get_phones_and_bert(text, language, version):
    phones, word2ph, norm_text = clean_text_inf(text, language, version)
    bert = get_bert_feature(norm_text, word2ph).to(device)
    return phones, bert, norm_text


def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(
        path)  # os.path.dirname() 返回指定路径中的目录部分。例如，如果 path 是 '/home/user/file.pth'，那么 dir 将是 '/home/user'。
    name = os.path.basename(
        path)  # os.path.basename(path):os.path.basename() 返回指定路径中的文件名部分。例如，如果 path 是 '/home/user/file.pth'，那么 name 将是 'file.pth'。
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path = "%s%s.pth" % (ttime(), i_part)
    torch.save(fea, tmp_path)  # torch.save(fea, tmp_path) 表示将 fea（可能是一个张量或模型的状态字典）保存到 tmp_path 路径指向的文件中。
    shutil.move(tmp_path,
                "%s/%s" % (dir, name))  # shutil.move(tmp_path, "%s/%s" % (dir, name)) 将 tmp_path 指向的文件移动到新的位置。

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs,
                         output_hidden_states=True)  # **inputs 是一种解包操作，表示将 inputs 字典中的键值对作为关键字参数传递给 bert_model。output_hidden_states 是 BERT 模型的一个参数。当设置为 True 时，模型将返回所有隐藏层的状态（hidden states），而不仅仅是最后一层的输出。这些隐藏层状态包括输入经过每一层 Transformer 的输出。
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[
              1:-1]  # [-3:-2] 代表获取倒数第三层到倒数第二层之间的隐藏状态。不过，由于这里的范围仅包括 -3，实际提取的只是倒数第三层的隐藏状态，返回的是一个长度为 1 的列表。
    # 在-1最后一个维度进行拼接（但不是只有一层-3:-2）,[0]表示取出第一个元素BERT 的输出张量形状为 (batch_size, sequence_length, hidden_size），所以输出(sequence_length, hidden_size)
    # 这是另一个切片操作，用于去除序列的第一个和最后一个标记对应的隐藏状态。在BERT的输出中，通常第一个标记是[CLS]标记，最后一个标记是[SEP]标记。[1: -1] 只保留中间的实际文本部分，去除了[CLS]和[SEP]的部分。
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i],
                                       1)  # res[i] 表示提取 res 张量中第 i 个元素（通常是第 i 个单词的特征向量）。如果 res 的形状是 (sequence_length, feature_dim)，那么 res[i] 的形状是 (feature_dim,)。
        # repeat(n, m) 是 PyTorch 张量的一个方法，用于将张量在指定维度上重复 n 次。res[i].repeat(word2ph[i], 1) 将 res[i] 张量在第一个维度上重复 word2ph[i] 次，在第二个维度上保持不变。
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T


def process(data, res):
    for name, text, lan in data:
        try:
            name = os.path.basename(name)
            print(name)
            phones, word2ph, norm_text = clean_text(  # 文本变成音素、每个字占的音素（中文特有）、规范化文本
                text.replace("%", "-").replace("￥", ","), lan, version
            )
            path_bert = "%s/%s.pt" % (bert_dir, name)
            if os.path.exists(path_bert) == False and lan == "zh":
                bert_feature = get_bert_feature(norm_text, word2ph)
                assert bert_feature.shape[-1] == len(
                    phones)  # shape[-1] 表示获取张量的最后一个维度的大小。通常，对于 BERT 特征矩阵，这个最后的维度表示序列的长度或特征维度。#assert 是一个内置的断言语句，用于调试时检查程序状态。它后面跟随一个表达式（或条件）。如果表达式为 True，程序继续正常运行。如果为 False，程序会引发 AssertionError。
                # torch.save(bert_feature, path_bert)
                my_save(bert_feature, path_bert)
            phones = " ".join(phones)
            # res.append([name,phones])
            res.append([name, phones, word2ph, norm_text])
        except:
            print(name, text, traceback.format_exc())



def get_all_phones_bert(train_file_name):
    inp_text = "resources/asr/" + train_file_name + "/" + train_file_name + ".list"
    inp_wav_dir = "resources/slice/" + train_file_name
    exp_name = "exp_" + train_file_name
    todo = []
    res = []
    with open(inp_text, "r", encoding="utf8") as f:  # inp_text就是asr出来的列表
        lines = f.read().strip("\n").split("\n")
    for line in lines[int(i_part):: int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")  # 根据|分割内容
            # todo.append([name,text,"zh"])
            if language in language_v1_to_language_v2.keys():
                todo.append(  # todo就是吧list里的东西拆开来做了个列表
                    [wav_name, text, language_v1_to_language_v2.get(language, language)]
                )
            else:
                print(f"\033[33m[Waring] The {language = } of {wav_name} is not supported for training.\033[0m")
        except:
            print(line, traceback.format_exc())

    process(todo, res)
    opt = []
    for name, phones, word2ph, norm_text in res:
        opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")

if __name__ == "__main__":
    get_all_phones_bert("shoulinrui.m4a")
    phones, bert, norm_text = get_phones_and_bert("你好", "zh", "v1")
    print(phones,bert,norm_text)