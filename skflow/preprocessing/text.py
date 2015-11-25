"""Implements a number of text preprocessing utilities."""

#  Copyright 2015 Google Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import re
import collections
import numpy as np

TOKENIZER_RE = re.compile("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+")

def tokenizer(iterator):
    """Tokenizer generator.

    Args:
        iterator: Input iterator with strings.

    Yields:
        array of tokens per each value in the input.
    """
    for value in iterator:
        yield TOKENIZER_RE.findall(value)


class WordVocabulary(object):
    """Word vocabulary class.

    Accumulates and provides mapping from words to indexes.
    """

    def __init__(self, unknown_token='<UNK>'):
        self._mapping = {unknown_token: 0}
        self._freq = collections.defaultdict(int)
        self._freeze = False

    def freeze(self, freeze=True):
        self._freeze = freeze

    def get(self, word):
        if word not in self._mapping:
            if self._freeze:
                return 0
            self._mapping[word] = len(self._mapping)
        return self._mapping[word]

    def add(self, word, count=1):
        word_id = self.get(word)
        if word_id <= 0: return
        self._freq[word] += count

    def trim(self, min_frequency, max_frequency=-1):
        """Trims vocabulary for minimum frequency.

        Args:
            min_frequency: minimum frequency to keep.
            max_frequency: optional, maximum frequency to keep.
                Useful to remove stop words.
        """
        for word, count in self._freq.iteritems():
            if count <= min_frequency and (max_frequency < 0 or 
               count >= max_frequency):
               self._mapping.pop(word)

class VocabularyProcessor(object):
    """Maps documents to sequences of word ids.

    Parameters:
        max_document_length: Maximum length of documents.
            if documents are longer, they will be trimmed, if shorter - padded.
        min_frequency: Minimum frequency of words in the vocabulary.
        vocabulary: WordVocabulary object.

    Attributes:
        vocabulary_: WordVocabulary object.
    """
    
    def __init__(self, max_document_length,
                 min_frequency=0, vocabulary=None,
                 tokenizer_fn=None):
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        if vocabulary:
            self.vocabulary_ = vocabulary
        else:
            self.vocabulary_ = WordVocabulary()
        if tokenizer_fn:
            self._tokenizer = tokenizer_fn
        else:
            self._tokenizer = tokenizer

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.
        
        Args:
            raw_documents: iterable
                An iterable which yield either str or unicode.

        Returns:
            self
        """
        for tokens in self._tokenizer(raw_documents):
            for token in tokens:
                self.vocabulary_.add(token)
        if self.min_frequency > 0:
            self.vocabulary_.trim(self.min_frequency)
        self.vocabulary_.freeze()
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return indexies of words.
        
        Args:
            raw_documents: iterable
                An iterable which yield either str or unicode.

        Returns:
            X: iterable, [n_samples, max_document_length]
                Word-id matrix.
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def transform(self, raw_documents):
        """Transform documents to word-id matrix.

        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.

        Args:
            raw_documents: iterable.
                An iterable which yield either str or unicode.

        Returns:
            X: iterable, [n_samples, max_document_length]
                Word-id matrix.
        """
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids

