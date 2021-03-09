import os

import numpy as np
from ds_ctcdecoder import Alphabet, ctc_beam_search_decoder, Scorer

classes = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
text_file = open("chars_small.txt", "w", encoding='utf-8')
text_file.write('\n'.join([x if x != '#' else '\\#' for x in list(classes)]))
text_file.close()


def softmax(matrix):
    time_steps, _ = matrix.shape
    result = np.zeros(matrix.shape)
    for t in range(time_steps):
        e = np.exp(matrix[t, :])
        result[t, :] = e / np.sum(e)
    return result


def load_rnn_output(fn):
    return np.genfromtxt(fn, delimiter=';')[:, : -1]


alphabet = Alphabet(os.path.abspath("chars_small.txt"))
crnn_output = softmax(load_rnn_output('./rnn_output.csv'))
res = ctc_beam_search_decoder(probs_seq=crnn_output, alphabet=alphabet, beam_size=25, scorer=Scorer(alphabet=alphabet, scorer_path='iam.scorer', alpha=0.75, beta=1.85))
# predicted: the fake friend of the family has to
# actual: the fake friend of the family, like the
print(res[0][1])
