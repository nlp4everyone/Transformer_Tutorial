import torch.cuda

list_sentences = ["Today is not a good day","It's terrible day","Weather is excellent","How it going with this terrible today "]
# Max length = 10
max_length = 6
dictionary_vocab = 100

# TF vocab
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

# Tensorflow tokenizer
tf_tokenizer = Tokenizer(oov_token="UNK")
tf_tokenizer.fit_on_texts(list_sentences)
embedding = tf_tokenizer.texts_to_sequences(list_sentences)
embedding = pad_sequences(embedding,maxlen=max_length,padding="post",truncating="post")

# Torchtext tokenizer
from torchtext.vocab import build_vocab_from_iterator
# from torchtext.data.utils import get_tokenizer
# torch_tokenizer = get_tokenizer("basic_english")
# vocab = [torch_tokenizer(sentence) for sentence in list_sentences]
# print(vocab)
# torch_vocabulary = build_vocab_from_iterator(list_sentences)
# print(torch_vocabulary)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
from torch.backends.cudnn import is_available,version
print(version())