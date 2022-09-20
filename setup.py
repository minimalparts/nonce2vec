#!/usr/bin/env python3
"""nonce2vec setup.py.

This file details modalities for packaging the nonce2vec application.
"""

from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='nonce2vec',
    description='A python module to generate word embeddings from tiny data',
    author=' Alexandre Kabbach and Aurélie Herbelot',
    author_email='akb@3azouz.net',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='2.0.2',
    url='https://github.com/minimalparts/nonce2vec',
    download_url='https://github.com/minimalparts/nonce2vec/#files',
    license='MIT',
    keywords=['word2vec', 'word-embeddings', 'incremental-learning'],
    platforms=['any'],
    packages=['nonce2vec', 'nonce2vec.utils', 'nonce2vec.models',
              'nonce2vec.exceptions', 'nonce2vec.logging',
              'nonce2vec.resources'],
    package_data={'nonce2vec': ['logging/*.yml', 'resources/*']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'n2v = nonce2vec.main:main'
        ],
    },
    install_requires=['pyyaml~=6.0', 'gensim~=4.2.0', 'numpy>=1.21',
                      'scipy>=1.7.0'],
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Environment :: Web Environment',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Text Processing :: Linguistic'],
    zip_safe=False,
)
