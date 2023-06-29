from gensim.models import KeyedVectors
from torch.utils.data import Dataset
import torch

#该文件用于编写一些功能函数


def Word_to_Vec():
    model = KeyedVectors.load_word2vec_format('/home/w/Program/emotion/Dataset/wiki_word2vec_50.bin', binary=True)
    word2vec = {}
    for word in model.index_to_key:
        word2vec[word] = model[word]
    return word2vec #返回词向量字典


class TextDataset(Dataset):
    def __init__(self, file_path, word2vec):
        self.data = []
        self.word2vec = word2vec
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                text_vectors = []
                for word in text.split()[:64]:  # 限定text_vector的长度为64，即每个句子只取64个词
                    try:
                        text_vectors.append(torch.tensor(self.word2vec[word]))
                    except KeyError:
                        continue
                if len(text_vectors) < 64:  # 使用torch.cat补齐缺失的位置，RNN需要在句首填充0
                    padding_vectors = torch.zeros(64 - len(text_vectors), 50)
                    text_vector = torch.cat((padding_vectors, torch.stack(text_vectors)), dim=0) if len(text_vectors) > 0 else torch.zeros(64, 50)
                else:
                    text_vector = torch.stack(text_vectors)
                self.data.append((int(label), text_vector))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, text_vector = self.data[index]
        return torch.tensor(label), text_vector

