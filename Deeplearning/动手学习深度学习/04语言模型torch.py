import torch
import random
## 读取数据集
with open('/home/kesci/input/jaychou_lyrics4703/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()
print(len(corpus_chars))
print(corpus_chars[: 40])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[: 10000]

## 建立字符索引
idx_to_char = list(set(corpus_chars))
char_to_idx = {char: i for i,char in enumerate(idx_to_char)}
vocab_size = len(char_to_idx)
print(vocab_size)

corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:',''.join([idx_to_char[idx] for idx in sample]))
print('indices',sample)

def load_data_jay_lyrics():
    with open('/home/kesci/input/jaychou_lyrics4703/jaychou_lyrics.txt') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

## 随机采样
def data_iter_random(corpus_indices,batch_size,num_steps,device = None):
    num_examples = (len(corpus_indices) - 1) //  num_steps
    example_indices = [i * num_steps for i in range(num_examples)] # 每个样本的第一个字符在corpus_indices中的下标
    random.shuffle(example_indices)

    def _data(i):
        return corpus_indices[i:i+num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(0,num_examples,batch_size):
        batch_indices  = example_indices[i:i+ batch_size]
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)
my_seq =  list(range(30))
for X,Y in data_iter_random(my_seq,batch_size = 2,num_steps = 6):
    print('X:',X,'\nY:',Y,'\n')

## 相邻采样

def data_iter_consecutive(corpus_indices,batch_size,num_steps,device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_avaiable() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size *  batch_size
    corpus_indices = corpus_indices[:corpus_len]
    indices = torch.tensor(corpus_indices,device = device)
    indices  = indices.view(batch_size,-1)
    batch_num = (indices.shape[1] -1) // num_steps
    for i in range(batch_num):
        i = i *  num_steps
        X  = indices[:,i:i+num_steps]
        Y =  indices[:,i+1:i+num_steps + 1]
        yield X,Y

for X,Y in data_iter_consecutive(my_seq,batch_size =2,num_steps = 6):
    print('X: ', X, '\nY:', Y, '\n')

