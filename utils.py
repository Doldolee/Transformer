from typing import Sequence
from unittest import result
import pandas as pd
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from konlpy.tag import Okt

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCE = 25


def load_data(path):
  data_df = pd.read_csv(path)
  question, answer = list(data_df["Q"]), list(data_df["A"])

  return question, answer

# 특수 문자를 없애고 모든 단어를 포함하는 단어 리스트를 만듬.
def data_tokenizer(data):
  words=[]
  for sentence in data:
    sentence = re.sub(CHANGE_FILTER, "", sentence)
    for word in sentence.split():
      words.append(word)
  
  return list(set(words))

#형태소를 기준으로 텍스트 데이터를 토크나이징함.
def prepro_like_morphlized(data):
  morph_analyzer=Okt()
  result_data = []
  for seq in tqdm(data):
    morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(" ","")))
    result_data.append(morphlized_seq)
  
  return result_data

def load_vocabulary(path, vocab_path="./chatdic.csv", tokenize_as_morph=False):
  vocabulary_list = []

  if not os.path.exists(vocab_path):
    if (os.path.exists(path)):
      question, answer = load_data(path)

      if tokenize_as_morph:
        question = prepro_like_morphlized(question)
        answer = prepro_like_morphlized(answer)

      words_list = []
      words_list.extend(question)
      words_list.extend(answer)

      words_list = data_tokenizer(words_list)
      words_list[:0] = MARKER
      # print(words_list[:3])
    
    #단어 사전 저장
    with open(vocab_path, "w", encoding="utf-8")as vocabulary_file:
      for word in words_list:
        vocabulary_file.write(word + "\n")
  
  with open(vocab_path, "r", encoding="utf-8") as vocabulary_file:
    for line in vocabulary_file:
      vocabulary_list.append(line.strip())

  word2idx = { word: idx for idx, word in enumerate(vocabulary_list)}
  idx2word = { idx: word for idx, word in enumerate(vocabulary_list)}

  return word2idx, idx2word, len(word2idx)

def enc_processing(value, dictionary, tokenize_as_morph=False):

  sequences_input_index = []
  sequences_length = []
  if tokenize_as_morph:
    value = prepro_like_morphlized(value)

  for sequence in value:
    sequence_index = []
    sequence = re.sub(CHANGE_FILTER, "", sequence)
    for word in sequence.split():
      if dictionary.get(word) is not None:
        sequence_index.extend([dictionary[word]])
      else:
        sequence_index.extend([dictionary[UNK]])
    
    if len(sequence_index) > MAX_SEQUENCE:
      sequence_index = sequence_index[:MAX_SEQUENCE]
    
    sequences_length.append(len(sequence_index))

    sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]] #리스트에 음수를 곱하면 그냥 0이고 양수를 곱하면 그 수 만큼 갯수가 늘어남.
    
    sequences_input_index.append(sequence_index)

  return np.asarray(sequences_input_index), sequences_length

def dec_input_processing(value, dictionary, tokenize_as_morph=False):
  sequences_input_index = []
  sequences_length = []
  if tokenize_as_morph:
    value = prepro_like_morphlized(value)

  for sequence in value:
    sequence_index = []
    sequence = re.sub(CHANGE_FILTER, "", sequence)
    sequence_index.extend([dictionary[STD]])
    for word in sequence.split():
      if dictionary.get(word) is not None:
        sequence_index.extend([dictionary[word]])
      else:
        sequence_index.extend([dictionary[UNK]])
    
    if len(sequence_index) > MAX_SEQUENCE:
      sequence_index = sequence_index[:MAX_SEQUENCE]

    sequences_length.append(len(sequence_index))

    sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]] #리스트에 음수를 곱하면 그냥 0이고 양수를 곱하면 그 수 만큼 갯수가 늘어남.
    
    sequences_input_index.append(sequence_index)

  return np.asarray(sequences_input_index), sequences_length

def dec_target_processing(value, dictionary, tokenize_as_morph=False):
  sequences_target_index = []
  if tokenize_as_morph:
    value = prepro_like_morphlized(value)

  for sequence in value:
    sequence_index = []
    sequence = re.sub(CHANGE_FILTER, "", sequence)
    for word in sequence.split():
      if dictionary.get(word) is not None:
        sequence_index.extend([dictionary[word]])
      else:
        sequence_index.extend([dictionary[UNK]])
    sequence_index.extend([dictionary[END]])

    sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]] #리스트에 음수를 곱하면 그냥 0이고 양수를 곱하면 그 수 만큼 갯수가 늘어남.
    sequences_target_index.append(sequence_index)

  return np.asarray(sequences_target_index)

# 이것만 호출해가면 데이터 전처리 완료
def prepro_dataset():
  inputs, outputs = load_data("./ChatbotData.csv")
  char2idx, idx2char, vocab_size = load_vocabulary("./ChatbotData.csv", "./chatdic.csv", tokenize_as_morph=False)
  index_inputs, input_seq_len = enc_processing(inputs, char2idx, tokenize_as_morph=False)
  index_outputs, output_seq_len = dec_input_processing(outputs, char2idx, tokenize_as_morph=False)
  index_targets = dec_target_processing(outputs, char2idx, tokenize_as_morph=False)

  data_configs = {}
  data_configs['char2idx'] = char2idx
  data_configs['idx2char'] = idx2char
  data_configs['vocab_size'] = vocab_size
  data_configs['pad_symbol'] = PAD
  data_configs['std_symbol'] = STD
  data_configs['end_symbol'] = END
  data_configs['unk_symbol'] = UNK


  return index_inputs, index_outputs, index_targets, data_configs
  

    
# if __name__ == "__main__":
#   # word2idx, idx2word, length1 = load_vocabulary("./ChatbotData.csv")
#   # input, output  = load_data("./ChatbotData.csv")
#   # dec_input_processing(output, word2idx)
#   prepro_dataset()
