"""Utilities for Wikipedia XML dump content extractionself.

Relies heavily on a custom version of wikiextractor.
See: https://github.com/akb89/wikiextractor
"""

import logging

import spacy
import wikiextractor

import nonce2vec.utils.files as futils

logger = logging.getLogger(__name__)

__all__ = ('extract')


def extract(output_txt_filepath, input_xml_filepath):
    """Extract content of wikipedia XML file.

    Extract content of json.text as given by wikiextractor and tokenize
    content with spacy. Output one-sentence-per-line, lowercase, tokenized
    text.
    """
    logger.info('Extracting content of wikipedia file {}'
                .format(input_xml_filepath))
    output_filepath = futils.get_output_filepath(input_xml_filepath,
                                                 output_txt_filepath)
    spacy_nlp = spacy.load('en_core_web_sm')
    spacy_nlp.max_length = 10000000  # avoid bug with very long input
    with open(output_filepath, 'w', encoding='utf-8') as output_stream:
        logger.info('Writing output to file {}'.format(output_filepath))
        for json_object in wikiextractor.extract(input_xml_filepath):
            doc = spacy_nlp(json_object['text'])
            for sent in doc.sents:
                #output_sent = ' '.join([token.text.lower() for token in sent])
                output_sent = ' '.join([token.text.lower().strip() for token
                                        in sent])
                print(output_sent, file=output_stream)
                logger.debug('Printed to {}'.format(output_filepath))
    return input_xml_filepath
