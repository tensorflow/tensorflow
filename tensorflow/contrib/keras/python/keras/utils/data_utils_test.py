# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for data_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import cycle
import os
import tarfile
import threading
import zipfile

import numpy as np
from six.moves.urllib.parse import urljoin
from six.moves.urllib.request import pathname2url

from tensorflow.contrib.keras.python import keras
from tensorflow.python.platform import test


class TestGetFileAndValidateIt(test.TestCase):

  def test_get_file_and_validate_it(self):
    """Tests get_file from a url, plus extraction and validation.
    """
    dest_dir = self.get_temp_dir()
    orig_dir = self.get_temp_dir()

    text_file_path = os.path.join(orig_dir, 'test.txt')
    zip_file_path = os.path.join(orig_dir, 'test.zip')
    tar_file_path = os.path.join(orig_dir, 'test.tar.gz')

    with open(text_file_path, 'w') as text_file:
      text_file.write('Float like a butterfly, sting like a bee.')

    with tarfile.open(tar_file_path, 'w:gz') as tar_file:
      tar_file.add(text_file_path)

    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
      zip_file.write(text_file_path)

    origin = urljoin('file://', pathname2url(os.path.abspath(tar_file_path)))

    path = keras.utils.data_utils.get_file('test.txt', origin,
                                           untar=True, cache_subdir=dest_dir)
    filepath = path + '.tar.gz'
    hashval_sha256 = keras.utils.data_utils._hash_file(filepath)
    hashval_md5 = keras.utils.data_utils._hash_file(filepath, algorithm='md5')
    path = keras.utils.data_utils.get_file(
        'test.txt', origin, md5_hash=hashval_md5,
        untar=True, cache_subdir=dest_dir)
    path = keras.utils.data_utils.get_file(
        filepath, origin, file_hash=hashval_sha256,
        extract=True, cache_subdir=dest_dir)
    self.assertTrue(os.path.exists(filepath))
    self.assertTrue(keras.utils.data_utils.validate_file(filepath,
                                                         hashval_sha256))
    self.assertTrue(keras.utils.data_utils.validate_file(filepath, hashval_md5))
    os.remove(filepath)

    origin = urljoin('file://', pathname2url(os.path.abspath(zip_file_path)))

    hashval_sha256 = keras.utils.data_utils._hash_file(zip_file_path)
    hashval_md5 = keras.utils.data_utils._hash_file(zip_file_path,
                                                    algorithm='md5')
    path = keras.utils.data_utils.get_file(
        'test', origin, md5_hash=hashval_md5,
        extract=True, cache_subdir=dest_dir)
    path = keras.utils.data_utils.get_file(
        'test', origin, file_hash=hashval_sha256,
        extract=True, cache_subdir=dest_dir)
    self.assertTrue(os.path.exists(path))
    self.assertTrue(keras.utils.data_utils.validate_file(path, hashval_sha256))
    self.assertTrue(keras.utils.data_utils.validate_file(path, hashval_md5))


class ThreadsafeIter(object):

  def __init__(self, it):
    self.it = it
    self.lock = threading.Lock()

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    with self.lock:
      return next(self.it)


def threadsafe_generator(f):

  def g(*a, **kw):
    return ThreadsafeIter(f(*a, **kw))

  return g


class TestSequence(keras.utils.data_utils.Sequence):

  def __init__(self, shape):
    self.shape = shape

  def __getitem__(self, item):
    return np.ones(self.shape, dtype=np.uint8) * item

  def __len__(self):
    return 100


class FaultSequence(keras.utils.data_utils.Sequence):

  def __getitem__(self, item):
    raise IndexError(item, 'item is not present')

  def __len__(self):
    return 100


@threadsafe_generator
def create_generator_from_sequence_threads(ds):
  for i in cycle(range(len(ds))):
    yield ds[i]


def create_generator_from_sequence_pcs(ds):
  for i in cycle(range(len(ds))):
    yield ds[i]


class TestEnqueuers(test.TestCase):

  def test_generator_enqueuer_threads(self):
    enqueuer = keras.utils.data_utils.GeneratorEnqueuer(
        create_generator_from_sequence_threads(TestSequence([3, 200, 200, 3])),
        use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for _ in range(100):
      acc.append(int(next(gen_output)[0, 0, 0, 0]))

    self.assertEqual(len(set(acc) - set(range(100))), 0)
    enqueuer.stop()

  def test_generator_enqueuer_processes(self):
    enqueuer = keras.utils.data_utils.GeneratorEnqueuer(
        create_generator_from_sequence_pcs(TestSequence([3, 200, 200, 3])),
        use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for _ in range(100):
      acc.append(int(next(gen_output)[0, 0, 0, 0]))
    self.assertNotEqual(acc, list(range(100)))
    enqueuer.stop()

  def test_generator_enqueuer_fail_threads(self):
    enqueuer = keras.utils.data_utils.GeneratorEnqueuer(
        create_generator_from_sequence_threads(FaultSequence()),
        use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with self.assertRaises(StopIteration):
      next(gen_output)

  def test_generator_enqueuer_fail_processes(self):
    enqueuer = keras.utils.data_utils.GeneratorEnqueuer(
        create_generator_from_sequence_pcs(FaultSequence()),
        use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with self.assertRaises(StopIteration):
      next(gen_output)

  def test_ordered_enqueuer_threads(self):
    enqueuer = keras.utils.data_utils.OrderedEnqueuer(
        TestSequence([3, 200, 200, 3]), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for _ in range(100):
      acc.append(next(gen_output)[0, 0, 0, 0])
    self.assertEqual(acc, list(range(100)))
    enqueuer.stop()

  def test_ordered_enqueuer_processes(self):
    enqueuer = keras.utils.data_utils.OrderedEnqueuer(
        TestSequence([3, 200, 200, 3]), use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    acc = []
    for _ in range(100):
      acc.append(next(gen_output)[0, 0, 0, 0])
    self.assertEqual(acc, list(range(100)))
    enqueuer.stop()

  def test_ordered_enqueuer_fail_threads(self):
    enqueuer = keras.utils.data_utils.OrderedEnqueuer(
        FaultSequence(), use_multiprocessing=False)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with self.assertRaises(StopIteration):
      next(gen_output)

  def test_ordered_enqueuer_fail_processes(self):
    enqueuer = keras.utils.data_utils.OrderedEnqueuer(
        FaultSequence(), use_multiprocessing=True)
    enqueuer.start(3, 10)
    gen_output = enqueuer.get()
    with self.assertRaises(StopIteration):
      next(gen_output)


if __name__ == '__main__':
  test.main()
