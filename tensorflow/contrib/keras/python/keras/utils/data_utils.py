# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for file download and caching."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import hashlib
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import random
import shutil
import sys
import tarfile
import threading
import time
import zipfile

import numpy as np
import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen

from tensorflow.contrib.keras.python.keras.utils.generic_utils import Progbar

try:
  import queue  # pylint:disable=g-import-not-at-top
except ImportError:
  import Queue as queue  # pylint:disable=g-import-not-at-top


if sys.version_info[0] == 2:

  def urlretrieve(url, filename, reporthook=None, data=None):
    """Replacement for `urlretrive` for Python 2.

    Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
    `urllib` module, known to have issues with proxy management.

    Arguments:
        url: url to retrieve.
        filename: where to store the retrieved data locally.
        reporthook: a hook function that will be called once
            on establishment of the network connection and once
            after each block read thereafter.
            The hook will be passed three arguments;
            a count of blocks transferred so far,
            a block size in bytes, and the total size of the file.
        data: `data` argument passed to `urlopen`.
    """

    def chunk_read(response, chunk_size=8192, reporthook=None):
      content_type = response.info().get('Content-Length')
      total_size = -1
      if content_type is not None:
        total_size = int(content_type.strip())
      count = 0
      while 1:
        chunk = response.read(chunk_size)
        count += 1
        if not chunk:
          reporthook(count, total_size, total_size)
          break
        if reporthook:
          reporthook(count, chunk_size, total_size)
        yield chunk

    response = urlopen(url, data)
    with open(filename, 'wb') as fd:
      for chunk in chunk_read(response, reporthook=reporthook):
        fd.write(chunk)
else:
  from six.moves.urllib.request import urlretrieve  # pylint: disable=g-import-not-at-top


def _extract_archive(file_path, path='.', archive_format='auto'):
  """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

  Arguments:
      file_path: path to the archive file
      path: path to extract the archive file
      archive_format: Archive format to try for extracting the file.
          Options are 'auto', 'tar', 'zip', and None.
          'tar' includes tar, tar.gz, and tar.bz files.
          The default 'auto' is ['tar', 'zip'].
          None or an empty list will return no matches found.

  Returns:
      True if a match was found and an archive extraction was completed,
      False otherwise.
  """
  if archive_format is None:
    return False
  if archive_format is 'auto':
    archive_format = ['tar', 'zip']
  if isinstance(archive_format, six.string_types):
    archive_format = [archive_format]

  for archive_type in archive_format:
    if archive_type is 'tar':
      open_fn = tarfile.open
      is_match_fn = tarfile.is_tarfile
    if archive_type is 'zip':
      open_fn = zipfile.ZipFile
      is_match_fn = zipfile.is_zipfile

    if is_match_fn(file_path):
      with open_fn(file_path) as archive:
        try:
          archive.extractall(path)
        except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
          if os.path.exists(path):
            if os.path.isfile(path):
              os.remove(path)
            else:
              shutil.rmtree(path)
          raise
      return True
  return False


def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
  """Downloads a file from a URL if it not already in the cache.

  By default the file at the url `origin` is downloaded to the
  cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
  and given the filename `fname`. The final location of a file
  `example.txt` would therefore be `~/.keras/datasets/example.txt`.

  Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
  Passing a hash will verify the file after download. The command line
  programs `shasum` and `sha256sum` can compute the hash.

  Arguments:
      fname: Name of the file. If an absolute path `/path/to/file.txt` is
          specified the file will be saved at that location.
      origin: Original URL of the file.
      untar: Deprecated in favor of 'extract'.
          boolean, whether the file should be decompressed
      md5_hash: Deprecated in favor of 'file_hash'.
          md5 hash of the file for verification
      file_hash: The expected hash string of the file after download.
          The sha256 and md5 hash algorithms are both supported.
      cache_subdir: Subdirectory under the Keras cache dir where the file is
          saved. If an absolute path `/path/to/folder` is
          specified the file will be saved at that location.
      hash_algorithm: Select the hash algorithm to verify the file.
          options are 'md5', 'sha256', and 'auto'.
          The default 'auto' detects the hash algorithm in use.
      extract: True tries extracting the file as an Archive, like tar or zip.
      archive_format: Archive format to try for extracting the file.
          Options are 'auto', 'tar', 'zip', and None.
          'tar' includes tar, tar.gz, and tar.bz files.
          The default 'auto' is ['tar', 'zip'].
          None or an empty list will return no matches found.
      cache_dir: Location to store cached files, when None it
          defaults to the [Keras
            Directory](/faq/#where-is-the-keras-configuration-filed-stored).

  Returns:
      Path to the downloaded file
  """
  if cache_dir is None:
    cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
  if md5_hash is not None and file_hash is None:
    file_hash = md5_hash
    hash_algorithm = 'md5'
  datadir_base = os.path.expanduser(cache_dir)
  if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
  datadir = os.path.join(datadir_base, cache_subdir)
  if not os.path.exists(datadir):
    os.makedirs(datadir)

  if untar:
    untar_fpath = os.path.join(datadir, fname)
    fpath = untar_fpath + '.tar.gz'
  else:
    fpath = os.path.join(datadir, fname)

  download = False
  if os.path.exists(fpath):
    # File found; verify integrity if a hash was provided.
    if file_hash is not None:
      if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
        print('A local file was found, but it seems to be '
              'incomplete or outdated because the ' + hash_algorithm +
              ' file hash does not match the original value of ' + file_hash +
              ' so we will re-download the data.')
        download = True
  else:
    download = True

  if download:
    print('Downloading data from', origin)

    class ProgressTracker(object):
      # Maintain progbar for the lifetime of download.
      # This design was chosen for Python 2.7 compatibility.
      progbar = None

    def dl_progress(count, block_size, total_size):
      if ProgressTracker.progbar is None:
        if total_size is -1:
          total_size = None
        ProgressTracker.progbar = Progbar(total_size)
      else:
        ProgressTracker.progbar.update(count * block_size)

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
      try:
        urlretrieve(origin, fpath, dl_progress)
      except URLError as e:
        raise Exception(error_msg.format(origin, e.errno, e.reason))
      except HTTPError as e:
        raise Exception(error_msg.format(origin, e.code, e.msg))
    except (Exception, KeyboardInterrupt) as e:
      if os.path.exists(fpath):
        os.remove(fpath)
      raise
    ProgressTracker.progbar = None

  if untar:
    if not os.path.exists(untar_fpath):
      _extract_archive(fpath, datadir, archive_format='tar')
    return untar_fpath

  if extract:
    _extract_archive(fpath, datadir, archive_format)

  return fpath


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
  """Calculates a file sha256 or md5 hash.

  Example:

  ```python
     >>> from keras.data_utils import _hash_file
     >>> _hash_file('/path/to/file.zip')
     'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
  ```

  Arguments:
      fpath: path to the file being validated
      algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
          The default 'auto' detects the hash algorithm in use.
      chunk_size: Bytes to read at a time, important for large files.

  Returns:
      The file hash
  """
  if (algorithm is 'sha256') or (algorithm is 'auto' and len(hash) is 64):
    hasher = hashlib.sha256()
  else:
    hasher = hashlib.md5()

  with open(fpath, 'rb') as fpath_file:
    for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
      hasher.update(chunk)

  return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
  """Validates a file against a sha256 or md5 hash.

  Arguments:
      fpath: path to the file being validated
      file_hash:  The expected hash string of the file.
          The sha256 and md5 hash algorithms are both supported.
      algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
          The default 'auto' detects the hash algorithm in use.
      chunk_size: Bytes to read at a time, important for large files.

  Returns:
      Whether the file is valid
  """
  if ((algorithm is 'sha256') or
      (algorithm is 'auto' and len(file_hash) is 64)):
    hasher = 'sha256'
  else:
    hasher = 'md5'

  if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
    return True
  else:
    return False


class Sequence(object):
  """Base object for fitting to a sequence of data, such as a dataset.

  Every `Sequence` must implements the `__getitem__` and the `__len__` methods.

  Examples:

  ```python
  from skimage.io import imread
  from skimage.transform import resize
  import numpy as np

  # Here, `x_set` is list of path to the images
  # and `y_set` are the associated classes.

  class CIFAR10Sequence(Sequence):
      def __init__(self, x_set, y_set, batch_size):
          self.X,self.y = x_set,y_set
          self.batch_size = batch_size

      def __len__(self):
          return len(self.X) // self.batch_size

      def __getitem__(self,idx):
          batch_x = self.X[idx*self.batch_size:(idx+1)*self.batch_size]
          batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]

          return np.array([
              resize(imread(file_name), (200,200))
                 for file_name in batch_x]), np.array(batch_y)
  ```
  """

  @abstractmethod
  def __getitem__(self, index):
    """Gets batch at position `index`.

    Arguments:
        index: position of the batch in the Sequence.

    Returns:
        A batch
    """
    raise NotImplementedError

  @abstractmethod
  def __len__(self):
    """Number of batch in the Sequence.

    Returns:
        The number of batches in the Sequence.
    """
    raise NotImplementedError


def get_index(ds, i):
  """Quick fix for Python2, otherwise, it cannot be pickled.

  Arguments:
      ds: a Holder or Sequence object.
      i: index

  Returns:
      The value at index `i`.
  """
  return ds[i]


class SequenceEnqueuer(object):
  """Base class to enqueue inputs.

  The task of an Enqueuer is to use parallelism to speed up preprocessing.
  This is done with processes or threads.

  Examples:

  ```python
  enqueuer = SequenceEnqueuer(...)
  enqueuer.start()
  datas = enqueuer.get()
  for data in datas:
      # Use the inputs; training, evaluating, predicting.
      # ... stop sometime.
  enqueuer.close()
  ```

  The `enqueuer.get()` should be an infinite stream of datas.

  """

  @abstractmethod
  def is_running(self):
    raise NotImplementedError

  @abstractmethod
  def start(self, workers=1, max_queue_size=10):
    """Starts the handler's workers.

    Arguments:
        workers: number of worker threads
        max_queue_size: queue size
            (when full, threads could block on `put()`).
    """
    raise NotImplementedError

  @abstractmethod
  def stop(self, timeout=None):
    """Stop running threads and wait for them to exit, if necessary.

    Should be called by the same thread which called start().

    Arguments:
        timeout: maximum time to wait on thread.join()
    """
    raise NotImplementedError

  @abstractmethod
  def get(self):
    """Creates a generator to extract data from the queue.

    Skip the data if it is `None`.

    Returns:
        Generator yielding tuples `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
    """
    raise NotImplementedError


class OrderedEnqueuer(SequenceEnqueuer):
  """Builds a Enqueuer from a Sequence.

  Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

  Arguments:
      sequence: A `keras.utils.data_utils.Sequence` object.
      use_multiprocessing: use multiprocessing if True, otherwise threading
      scheduling: Sequential querying of datas if 'sequential', random
        otherwise.
  """

  def __init__(self,
               sequence,
               use_multiprocessing=False,
               scheduling='sequential'):
    self.sequence = sequence
    self.use_multiprocessing = use_multiprocessing
    self.scheduling = scheduling
    self.workers = 0
    self.executor = None
    self.queue = None
    self.run_thread = None
    self.stop_signal = None

  def is_running(self):
    return self.stop_signal is not None and not self.stop_signal.is_set()

  def start(self, workers=1, max_queue_size=10):
    """Start the handler's workers.

    Arguments:
        workers: number of worker threads
        max_queue_size: queue size
            (when full, workers could block on `put()`)
    """
    if self.use_multiprocessing:
      self.executor = multiprocessing.Pool(workers)
    else:
      self.executor = ThreadPool(workers)
    self.queue = queue.Queue(max_queue_size)
    self.stop_signal = threading.Event()
    self.run_thread = threading.Thread(target=self._run)
    self.run_thread.daemon = True
    self.run_thread.start()

  def _run(self):
    """Submits requests to the executor and queues the `Future` objects."""
    sequence = list(range(len(self.sequence)))
    while True:
      if self.scheduling is not 'sequential':
        random.shuffle(sequence)
      for i in sequence:
        if self.stop_signal.is_set():
          return
        self.queue.put(
            self.executor.apply_async(get_index, (self.sequence, i)),
            block=True)

  def get(self):
    """Creates a generator to extract data from the queue.

    Skip the data if it is `None`.

    Yields:
        Tuples (inputs, targets)
            or (inputs, targets, sample_weights)
    """
    try:
      while self.is_running():
        inputs = self.queue.get(block=True).get()
        if inputs is not None:
          yield inputs
    except Exception as e:
      self.stop()
      raise StopIteration(e)

  def stop(self, timeout=None):
    """Stops running threads and wait for them to exit, if necessary.

    Should be called by the same thread which called `start()`.

    Arguments:
        timeout: maximum time to wait on `thread.join()`
    """
    self.stop_signal.set()
    with self.queue.mutex:
      self.queue.queue.clear()
      self.queue.unfinished_tasks = 0
      self.queue.not_full.notify()
    self.executor.close()
    self.executor.join()
    self.run_thread.join(timeout)


class GeneratorEnqueuer(SequenceEnqueuer):
  """Builds a queue out of a data generator.

  Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

  Arguments:
      generator: a generator function which endlessly yields data
      use_multiprocessing: use multiprocessing if True, otherwise threading
      wait_time: time to sleep in-between calls to `put()`
      random_seed: Initial seed for workers,
          will be incremented by one for each workers.
  """

  def __init__(self,
               generator,
               use_multiprocessing=False,
               wait_time=0.05,
               random_seed=None):
    self.wait_time = wait_time
    self._generator = generator
    self._use_multiprocessing = use_multiprocessing
    self._threads = []
    self._stop_event = None
    self.queue = None
    self.random_seed = random_seed

  def start(self, workers=1, max_queue_size=10):
    """Kicks off threads which add data from the generator into the queue.

    Arguments:
        workers: number of worker threads
        max_queue_size: queue size
            (when full, threads could block on `put()`)
    """

    def data_generator_task():
      while not self._stop_event.is_set():
        try:
          if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
            generator_output = next(self._generator)
            self.queue.put(generator_output)
          else:
            time.sleep(self.wait_time)
        except Exception:
          self._stop_event.set()
          raise

    try:
      if self._use_multiprocessing:
        self.queue = multiprocessing.Queue(maxsize=max_queue_size)
        self._stop_event = multiprocessing.Event()
      else:
        self.queue = queue.Queue()
        self._stop_event = threading.Event()

      for _ in range(workers):
        if self._use_multiprocessing:
          # Reset random seed else all children processes
          # share the same seed
          np.random.seed(self.random_seed)
          thread = multiprocessing.Process(target=data_generator_task)
          thread.daemon = True
          if self.random_seed is not None:
            self.random_seed += 1
        else:
          thread = threading.Thread(target=data_generator_task)
        self._threads.append(thread)
        thread.start()
    except:
      self.stop()
      raise

  def is_running(self):
    return self._stop_event is not None and not self._stop_event.is_set()

  def stop(self, timeout=None):
    """Stops running threads and wait for them to exit, if necessary.

    Should be called by the same thread which called `start()`.

    Arguments:
        timeout: maximum time to wait on `thread.join()`.
    """
    if self.is_running():
      self._stop_event.set()

    for thread in self._threads:
      if thread.is_alive():
        if self._use_multiprocessing:
          thread.terminate()
        else:
          thread.join(timeout)

    if self._use_multiprocessing:
      if self.queue is not None:
        self.queue.close()

    self._threads = []
    self._stop_event = None
    self.queue = None

  def get(self):
    """Creates a generator to extract data from the queue.

    Skip the data if it is `None`.

    Yields:
        Data arrays.
    """
    while self.is_running():
      if not self.queue.empty():
        inputs = self.queue.get()
        if inputs is not None:
          yield inputs
      else:
        time.sleep(self.wait_time)
