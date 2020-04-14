
## 读入文本
import collections
import re

def read_time_machine():
    with open('/home/kesci/input/timemachine7163/timemachine.txt', 'r') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines

    lines = read_time_machine()
    print("# sentences %d"%len(lines))

    ## 分词
    def tokenize(sentences,token = 'word'):
        if token == 'word':
            return [sentence.split(' ') for sentence in sentences]
        elif token == 'char':
            return list[sentence for sentence in sentences]
        else:
            print("ERROR:unkown token type"+token)

tokens = tokensize(lines)
tokens[0:2]

## 建立字典
class Vocab():
    def __init__(self,tokens,min_freq =0 ,use_special_tokens = False):
        counter = count_corpus(tokens)
        self.token_freqs = list(counter.item())
        self.idx_to_token = []
        if use_special_tokens:
            # padding,begin of sentence,end of sentence,unknown
            self.pad,self.bos,self.eos.self.unk = (0,1,2,3)
            self.idx_to_token += ['','','']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token  += [token for token,freq in self.token_freqs if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx,token in enumerate(self.idx_to_token):
            self.token_to_idx[token]  = idx
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self,tokens):
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)
        retirn [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self,indices):
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):
    tokens = [tk for st in senetences for tk in st]
    return collections.Counter(tokens)

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:10])

## 将词转为索引
for i in range(8,10):
    print("words:",tokens[i])
    print("indices:",vocab[tokens[i]])

## example 
text = "Mr. Chen doesn't agree with my suggestion."
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
print([token.text for token in doc])

##nltk
from nltk.tokenize import word_tokensize
from nltk import data
data.path.append('/home/kesci/input/nltk_data3784/nltk_data')
print(word_tokenize(text))




