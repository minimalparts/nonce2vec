"""Files utils."""

import os
import smart_open
import natsort

__all__ = ('Samples', 'get_zipped_sentences', 'get_sentences',
           'get_model_path', 'get_input_filepaths', 'get_output_filepath',
           'get_tmp_filepaths')


def _get_tmp_dirpath(output_txt_filepath):
    """Return absolute path to output_txt_dirpath/tmp/."""
    return os.path.join(os.path.dirname(output_txt_filepath), 'tmp')


def get_tmp_filepaths(output_txt_filepath):
    """Return all .txt files under the output_txt_dirpath/tmp/ dir."""
    tmp_dirpath = _get_tmp_dirpath(output_txt_filepath)
    return natsort.natsorted([os.path.join(tmp_dirpath, filename) for filename
                              in os.listdir(tmp_dirpath)],
                             alg=natsort.ns.IGNORECASE)


def get_output_filepath(input_xml_filepath, output_txt_filepath):
    """Return filepath to output_txt_dirpath/tmp/input_xml_filename.

    Create tmp dir if not exists.
    """
    tmp_dirpath = _get_tmp_dirpath(output_txt_filepath)
    os.makedirs(tmp_dirpath, exist_ok=True)
    output_filename = os.path.basename(input_xml_filepath)
    output_txt_filepath = os.path.join(tmp_dirpath, '{}.txt'.format(output_filename))
    return output_txt_filepath


def get_input_filepaths(dirpath):
    """Return a list of absolute filepaths from a given dirpath.

    List all the files under a specific directory.
    """
    return [os.path.join(dirpath, filename) for filename in
            os.listdir(dirpath) if '.xml' in filename]


def get_model_path(datadir, outputdir, train_mode, alpha, neg, window_size,
                   sample, epochs, min_count, size):
    """Return absolute path to w2v model file.

    Model absolute path is computed from the outputdir and the
    datadir name.
    """
    os.makedirs(outputdir, exist_ok=True)
    return os.path.join(
        outputdir,
        '{}.{}.alpha{}.neg{}.win{}.sample{}.epochs{}.mincount{}.size{}.model'
        .format(os.path.basename(datadir), train_mode, alpha, neg, window_size,
                sample, epochs, min_count, size))


def get_zipped_sentences(datazip):
    """Return generator over sentence split of wikidump for gensim.word2vec.

    datazip should be the absolute path to the wikidump.gzip file.
    """
    for filename in os.listdir(smart_open.smart_open(datazip)):
        with open(filename, 'r') as input_stream:
            for line in input_stream:
                yield line.strip().split()


def get_sentences(data):
    """Return generator over sentence split of wikidata for gensim.word2vec."""
    for filename in os.listdir(data):
        if filename.startswith('.'):
            continue
        with open(os.path.join(data, filename), 'r') as input_stream:
            for line in input_stream:
                yield line.strip().split()


class Samples(object):
    """An iterable class (with generators) for gensim and n2v."""

    def __init__(self, input_data, source):
        if source != 'wiki' and source != 'nonces' and source != 'chimeras':
            raise Exception('Invalid source parameter \'{}\''.format(source))
        self._source = source
        self._datafile = input_data

    def _iterate_over_wiki(self):
        with open(self._datafile, 'rt') as input_stream:
            for line in input_stream:
                yield line.strip().split()

    def _iterate_over_nonces(self):
        with open(self._datafile, 'rt') as input_stream:
            for line in input_stream:
                fields = line.rstrip('\n').split('\t')
                nonce = fields[0]
                sentence = fields[1].replace('___', nonce).split()
                probe = '{}_true'.format(nonce)
                yield [sentence], nonce, probe

    def _iterate_over_chimeras(self):
        with open(self._datafile, 'rt') as input_stream:
            for line in input_stream:
                fields = line.rstrip('\n').split('\t')
                sentences = []
                for sent in fields[1].split('@@'):
                    tokens = sent.strip().split(' ')
                    if '___' in tokens:
                        sentences.append(tokens)
                probes = fields[2].split(',')
                responses = fields[3].split(',')
                yield sentences, probes, responses

    def __iter__(self):
        if self._source == 'wiki':
            return self._iterate_over_wiki()
        if self._source == 'nonces':
            return self._iterate_over_nonces()
        if self._source == 'chimeras':
            return self._iterate_over_chimeras()
        raise Exception('Invalid source parameter \'{}\''.format(self._source))
