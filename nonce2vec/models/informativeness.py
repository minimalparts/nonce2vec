"""Informativeness model.

Loads a bi-directional language model and computes various
entropy-based informativeness measures.
"""

import copy

import numpy
import scipy

from gensim.models import Word2Vec

# import bdlm.utils as bdlm_utils
#
# from bdlm.language_model import BDLM  # TODO change to language ultimately
# import bdlm.bilstm2
# from bdlm.models.corpus import Dictionary

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

    def __init__(self, mode, w2v_model_path=None,
                 torch_model_path=None, vocab_path=None,
                 cuda=False, entropy=None):
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
        assert mode == 'cbow' or mode == 'bidir'
        assert entropy == 'shannon' or 'weighted'
        self._mode = mode
        if self._mode == 'cbow':
            if not w2v_model_path:
                raise Exception('Unspecified w2v model path for informativeness '
                                'in CBOW mode')
            self._model = Word2Vec.load(w2v_model_path)
        self._entropy = entropy
        # self._model = BDLM.load(torch_model_path)
        # self._vocab = Dictionary.load(vocab_path)
        # #self.nlp = spacy.load('en_core_web_sm')
        # self._cuda = cuda

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
            if token not in self._vocab.word2idx:
                ids[idx] = self._vocab.word2idx['<unk>']
            else:
                ids[idx] = self._vocab.word2idx[token]
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
                                           cuda=self._cuda)
        return tokens_batch, tokens

    def _update_word_index(self, tokens, sentence, word_index):
        # Don't forget to add 1 for start <eos>
        return

    def sentence2sentence(self):
        """Get sentence-to-sentence informativeness."""
        pass

    def _get_bidir_sentence2word(self, tokens, word_index, seq_len=0):
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
        batches = bdlm_utils.batchify(tokens_as_ints, bsz=1, cuda=self._cuda)
        print(batches)
        assert batches.size(0) == len(_tokens)
        assert batches[0][0] == batches[0][-1]  # start/end with <eos>
        if seq_len == 0 or seq_len > 18:
            seq_len = 18  # with Kristina's pretrained model its the max we can do
            # seq_len = len(bathes) - 2  # currently not possible as the limit is 18
        hidden = self._model.init_hidden(bsz=1)
        for i in range(0, batches.size(0), seq_len):
            hidden = bdlm_utils.update_hidden(self._model, mode='bidir',
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
                print(targets)
                assert data[0].size(0) == seq_len
                assert data[1].size(0) == seq_len
                assert targets.size(0) == seq_len
                predictions, hidden = self._model(data, hidden)
                assert 0 <= word_index-i-1 <= seq_len - 1
                pred_tens_variable = predictions[word_index-i-1]
                # Ultimately 1 should be replaced by the number of batches for w2wi
                shannon_entropy = float(entropy(pred_tens_variable))
                # as-is the code supposes that entropy returns a float and not a tensor
                return 1 - (shannon_entropy / numpy.log(len(self._vocab)))
            else:
                continue
        return  # TODO: raise Exception

    def _get_cbow_s2w_with_weighted_entropy(self, tokens, word_index):
        _tokens = copy.deepcopy(tokens)
        del _tokens[word_index]
        words_and_probs = self._model.predict_output_word(
            _tokens, topn=len(self._model.wv.vocab))
        # shannon_entropy = scipy.stats.entropy(probs)
        shannon_entropy = 0
        total_count = sum(self._model.wv.vocab[w].count for w in self._model.wv.vocab)
        _alpha = 0
        for w in self._model.wv.vocab:
            pw = self._model.wv.vocab[w].count / total_count
            _alpha += (1 / (len(self._model.wv.vocab) * pw)) * numpy.log(1 / len(self._model.wv.vocab))
        alpha = 1 / _alpha
        for (w, prob) in words_and_probs:
            abs_prob = self._model.wv.vocab[w].count / total_count
            #shannon_entropy -= (prob / abs_prob) * numpy.log((prob / abs_prob))
            shannon_entropy += (prob / abs_prob) * numpy.log(prob)
        #s2w = 1 - (shannon_entropy / numpy.log(len(self._model.wv.vocab)))
        s2w = 1 - (shannon_entropy * alpha)
        return s2w

    def _get_cbow_s2w_with_shannon_entropy(self, tokens, word_index):
        _tokens = copy.deepcopy(tokens)
        del _tokens[word_index]
        words_and_probs = self._model.predict_output_word(
            _tokens, topn=len(self._model.wv.vocab))
        probs = [item[1] for item in words_and_probs]
        shannon_entropy = scipy.stats.entropy(probs)
        s2w = 1 - (shannon_entropy / numpy.log(len(self._model.wv.vocab)))
        return s2w

    def _get_cbow_sentence2word(self, tokens, word_index):
        if self._entropy == 'shannon':
            return self._get_cbow_s2w_with_shannon_entropy(tokens, word_index)
        if self._entropy == 'weighted':
            return self._get_cbow_s2w_with_weighted_entropy(tokens, word_index)


    def sentence2word(self, tokens, word_index, seq_len=0):
        """Get sentence-to-word informativeness."""
        if self._mode == 'bidir':
            return self._get_bidir_sentence2word(tokens, word_index, seq_len)
        if self._mode == 'cbow':
            return self._get_cbow_sentence2word(tokens, word_index)


    def _get_bidir_word2word(self, tokens, source_word_index,
                             target_word_index, seq_len=0):
        # Use torch.where with condition
        # Better: use masked_select with ge(s2w) condition
        _tokens = copy.deepcopy(tokens)
        _tokens.insert(0, '<eos>')
        _tokens.append('<eos>')
        source_word_index += 1
        target_word_index += 1
        tokens_as_ints = self._map_to_idx(_tokens)
        source_word_id = tokens_as_ints[source_word_index]
        target_word_id = tokens_as_ints[target_word_index]
        source_tensor = bdlm_utils.batchify(tokens_as_ints, bsz=1, cuda=self._cuda)
        batches = torch.LongTensor(source_tensor.size(0), len(self._vocab) - 1).zero_()
        index_tensor = torch.LongTensor([[i for i in range(len(self._vocab) - 1)]])
        for i in range(source_tensor.size(0)):
            batches.index_fill_(0, torch.LongTensor([i]), source_tensor[i][0])
        batches[source_word_index].put_(
            torch.LongTensor(
                [j for j in range(len(self._vocab) - 1)]),
            torch.LongTensor(
                [k for k in self._vocab.word2idx.values() if k != source_word_id]))
        print(batches)

        for j in range(batches.size(1)):
            if seq_len == 0 or seq_len > 18:
                seq_len = 18  # with Kristina's pretrained model its the max we can do
                # seq_len = len(bathes) - 2  # currently not possible as the limit is 18
            batch = batches[:, j:j+1].contiguous()
            hidden = self._model.init_hidden(bsz=1)
            for i in range(0, batch.size(0), seq_len):
                hidden = bdlm_utils.update_hidden(self._model, mode='bidir',
                                                  hidden=hidden,
                                                  batch_size=1)
                if batch.size(0) - i < seq_len + 2:
                    seq_len = batch.size(0) - i - 2
                if seq_len < 3:
                    continue  # TODO: raise exception
                if target_word_index > i and target_word_index < i + seq_len + 1:
                    data, targets = bdlm_utils.get_batch(batch, i,
                                                         bptt=seq_len,
                                                         mode='bidir',
                                                         evaluation=True)
                    predictions, hidden = self._model(data, hidden)

    def _get_cbow_word2word(self, tokens, source_word_index, target_word_index):
        swi_with_source = self._get_cbow_sentence2word(tokens, target_word_index)
        _tokens = copy.deepcopy(tokens)
        del _tokens[source_word_index]
        if target_word_index > source_word_index:
            target_word_index -= 1
        swi_without_source = self._get_cbow_sentence2word(_tokens, target_word_index)
        #return numpy.abs(swi_without_source - swi_with_source) / swi_with_source
        #return (swi_without_source - swi_with_source) / swi_with_source
        return swi_with_source - swi_without_source

    def word2word(self, tokens, source_word_index, target_word_index,
                  seq_len=0):
        """Get word-to-word informativeness."""
        if self._mode == 'bidir':
            return self._get_bidir_word2word(tokens, source_word_index,
                                             target_word_index, seq_len)
        if self._mode == 'cbow':
            return self._get_cbow_word2word(tokens, source_word_index,
                                            target_word_index)
