"""Informativeness model.

Loads a bi-directional language model and computes various
entropy-based informativeness measures.
"""

import numpy
import spacy
import copy
import torch
import torch.nn.functional as F

import bdlm.utils as bdlm_utils

from bdlm.language_model import BDLM  # TODO change to language ultimately
import bdlm.bilstm2
from bdlm.models.corpus import Dictionary

__all__ = ('Informativeness')


def entropy(x):
    """Compute the Shannon entropy of a tensor x along its lines.

    Args:
        x (torch.autograd.Variable): A pyTorch Variable: wrapper around a
                                     torch.Tensor of size n x m.

    Returns:
        a torch.Tensor of size n x 1

    """
    plogp = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    shannon_entropy = -1.0 * plogp.sum(dim=1)
    return shannon_entropy


class Informativeness():
    """Informativeness class relying on a bi-directional language model."""

    def __init__(self, torch_model_path, vocab_path, cuda=False):
        """Initialize the Informativeness instance.

        Args:
            torch_model_path (str): The absolute path to the pickled
                                    bdlm torch model.
            vocab_path (str): The absolute path to the .txt file storing the
                              bdlm model vocabulary.

        Returns:
            model: a bdlm.BDLM instance.
            vocab: a bdlm.Dictionary instance
            nlp: a spacy.nlp loaded instance (for English)
            cuda: set to True to use GPUs with pyTorch
        """
        self.model = BDLM.load(torch_model_path)
        self.vocab = Dictionary.load(vocab_path)
        #self.nlp = spacy.load('en_core_web_sm')
        self.cuda = cuda

    def _tokenize(self, sentence):
        return self.nlp(sentence).to_array()

    @classmethod
    def _add_eos(cls, tokens):
        preppend = numpy.append(['<eos>'], tokens)
        append = numpy.append(preppend, ['<eos>'])
        return append

    def _map_to_idx(self, tokens_with_eos):
        ids = torch.LongTensor(len(tokens_with_eos))
        idx = 0
        for token in tokens_with_eos:
            if token not in self.vocab.word2idx:
                ids[idx] = self.vocab.word2idx['<unk>']
            else:
                ids[idx] = self.vocab.word2idx[token]
            idx += 1
        return ids

    def _preprocess(self, sentence):
        """Preprocess input sentence.

        Tokenize and add <eos> symbols to the beginning and the end of the
        string. Map to ids with vocab and batchify with bdlm utils.

        Args:
            sentence (str): The input sentence string.

        Returns:
            tokens_batch: a batchified BDLM sentence as a torch.Tensor.
            tokens: an iterable spacy.Doc wrapping the tokenized sentence
        """
        tokens = self._tokenize(sentence)
        tokens_with_eos = Informativeness._add_eos(tokens)
        tokens_as_ints = self._map_to_idx(tokens_with_eos)
        tokens_batch = bdlm_utils.batchify(tokens_as_ints, bsz=1,
                                           cuda=self.cuda)
        return tokens_batch, tokens

    def _update_word_index(self, tokens, sentence, word_index):
        # Don't forget to add 1 for start <eos>
        return

    def sentence2sentence(self):
        """Get sentence-to-sentence informativeness."""
        pass

    def sentence2word(self, tokens, word_index, seq_len=0):
        """Get sentence-to-word informativeness."""
        if not isinstance(word_index, int) or word_index < 0 \
         or word_index >= len(tokens):
            raise Exception('Invalid input word_index = {}. Should be a '
                            'positive integer within input tokens length '
                            '= {}'.format(word_index, len(tokens)))
        _tokens = copy.deepcopy(tokens)
        _tokens.insert(0, '<eos>')
        _tokens.append('<eos>')
        word_index += 1
        tokens_as_ints = self._map_to_idx(_tokens)
        assert len(_tokens) == len(tokens) + 2
        assert _tokens[word_index] == tokens[word_index - 1]
        batches = bdlm_utils.batchify(tokens_as_ints, bsz=1, cuda=self.cuda)
        assert batches.size(0) == len(_tokens)
        assert batches[0][0] == batches[0][-1]  # start/end with <eos>
        # TODO: maybe change this to a max seq_length ultimately?
        #seq_len = len(batches) - 2
        #assert seq_len <= len(batches) - 2
        hidden = self.model.init_hidden(bsz=1)
        for i in range(0, batches.size(0), seq_len):
            hidden = bdlm_utils.update_hidden(self.model, mode='bidir',
                                              hidden=hidden,
                                              batch_size=1)
            if batches.size(0) - i < seq_len + 2:
                seq_len = batches.size(0) - i - 2
            if seq_len < 3:
                continue  # TODO: raise exception
            if word_index > i and word_index < i + seq_len + 1:
                data, targets = bdlm_utils.get_batch(batches, i, seq_len,
                                                     mode='bidir',
                                                     evaluation=True)
                print(data)
                assert data[0].size(0) == seq_len
                assert data[1].size(0) == seq_len
                assert targets.size(0) == seq_len
                predictions, hidden = self.model(data, hidden)
                assert 0 <= word_index-i-1 <= seq_len - 1
                pred_tens_variable = predictions[word_index-i-1]
                # Ultimately 1 should be replaced by the number of batches for w2wi
                shannon_entropy = float(entropy(pred_tens_variable))
                # as-is the code supposes that entropy returns a float and not a tensor
                return 1 - (shannon_entropy / numpy.log(len(self.vocab)))
            else:
                continue
        return  # TODO: raise Exception

    def word2word(self):
        """Get word-to-word informativeness."""
        # Use torch.where with condition
        # Better: use masked_select with ge(s2w) condition
        pass
