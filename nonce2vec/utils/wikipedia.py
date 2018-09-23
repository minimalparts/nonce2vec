"""Utilities for Wikipedia XML dump content extractionself.

Relies heavily on a custom version of wikiextractor.
See: https://github.com/akb89/wikiextractor
"""

import wikiextractor
import spacy

__all__ = ('extract')

def extract(input_xml_filepath, output_txt_filepath):
    """Extract content of wikipedia XML file.

    Extract content of json.text as given by wikiextractor and tokenize
    content with spacy. Output one-sentence-per-line, lowercase, tokenized
    text.
    """
    with open(output_txt_filepath, 'w', encoding='utf-8') as output_stream:
        for json_object in wikiextractor.extract(input_xml_filepath):
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(json_object['text'])
            for sent in doc.sents:
                output_sent = ' '.join([token.text.lower() for token in sent])
                print(output_sent.strip(), file=output_stream)
