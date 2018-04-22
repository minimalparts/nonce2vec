#!/usr/bin/env python3
"""nonce2vec setup.py.

This file details modalities for packaging the nonce2vec application.
"""

from setuptools import setup

setup(
    name='nonce2vec',
    description='A python module to generate word embeddings from tiny data',
    author='Alexandre Kabbach',
    author_email='akb@3azouz.net',
    version='2.0.0',
    url='https://github.com/akb89/nonce2vec',
    download_url='https://github.com/akb89/nonce2vec/archive/2.0.0.tar.gz',
    license='MIT',
    keywords=['word2vec', 'embeddings', 'nonce', 'once-shot'],
    platforms=['any'],
    packages=['nonce2vec', 'nonce2vec.utils', 'nonce2vec.models',
              'nonce2vec.exceptions'],
    entry_points={
        'console_scripts': [
            'n2v = nonce2vec.main:main'
        ],
    },
    install_requires=['PyYAML==3.12'],
    setup_requires=['pytest-runner==4.0', 'pytest-pylint==0.8.0'],
    tests_require=['pytest==3.4.1', 'pylint==1.8.2', 'pytest-cov==2.5.1'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Environment :: Web Environment',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development :: Libraries :: Python Modules',
                 'Topic :: Text Processing :: Linguistic'],
    zip_safe=True,
)
