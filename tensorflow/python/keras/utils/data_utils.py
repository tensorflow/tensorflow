# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-import-not-at-top
"""Utilities for file download and caching."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import hashlib
from itertools import cycle
import multiprocessing
from multiprocessing.managers import SyncManager, BaseProxy
import os
import random
import shutil
import sys
import tarfile
import threading
import time
import traceback
from uuid import uuid4
import zipfile

import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen

from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

try:
  import queue
except ImportError:
  import Queue as queue


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
      while True:
        chunk = response.read(chunk_size)
        count += 1
        if reporthook is not None:
          reporthook(count, chunk_size, total_size)
        if chunk:
          yield chunk
        else:
          break

    response = urlopen(url, data)
    with open(filename, 'wb') as fd:
      for chunk in chunk_read(response, reporthook=reporthook):
        fd.write(chunk)
else:
  from six.moves.urllib.request import urlretrieve


def is_generator_or_sequence(x):
  """Check if `x` is a Keras generator type."""
  return tf_inspect.isgenerator(x) or isinstance(x, Sequence)


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
  if archive_format == 'auto':
    archive_format = ['tar', 'zip']
  if isinstance(archive_format, six.string_types):
    archive_format = [archive_format]

  for archive_type in archive_format:
    if archive_type == 'tar':
      open_fn = tarfile.open
      is_match_fn = tarfile.is_tarfile
    if archive_type == 'zip':
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


@tf_export('keras.utils.get_file')
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
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
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
        if total_size == -1:
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
  if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
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
  if (algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64):
    hasher = 'sha256'
  else:
    hasher = 'md5'

  if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
    return True
  else:
    return False


@tf_export('keras.utils.Sequence')
class Sequence(object):
  """Base object for fitting to a sequence of data, such as a dataset.

  Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
  If you want to modify your dataset between epochs you may implement
  `on_epoch_end`.
  The method `__getitem__` should return a complete batch.

  Notes:

  `Sequence` are a safer way to do multiprocessing. This structure guarantees
  that the network will only train once
   on each sample per epoch which is not the case with generators.

  Examples:

  ```python
      from skimage.io import imread
      from skimage.transform import resize
      import numpy as np
      import math

      # Here, `x_set` is list of path to the images
      # and `y_set` are the associated classes.

      class CIFAR10Sequence(Sequence):

          def __init__(self, x_set, y_set, batch_size):
              self.x, self.y = x_set, y_set
              self.batch_size = batch_size

          def __len__(self):
              return math.ceil(len(self.x) / self.batch_size)

          def __getitem__(self, idx):
              batch_x = self.x[idx * self.batch_size:(idx + 1) *
              self.batch_size]
              batch_y = self.y[idx * self.batch_size:(idx + 1) *
              self.batch_size]

              return np.array([
                  resize(imread(file_name), (200, 200))
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

  def on_epoch_end(self):
    """Method called at the end of every epoch.
    """
    pass

  def __iter__(self):
    """Create a generator that iterate over the Sequence."""
    for item in (self[i] for i in range(len(self))):
      yield item


def iter_sequence_infinite(seq):
  """Iterates indefinitely over a Sequence.

  Arguments:
    seq: Sequence instance.

  Yields:
    Batches of data from the Sequence.
  """
  while True:
    for item in seq:
      yield item

@tf_export('keras.utils.SequenceEnqueuer')
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

  def __init__(self):
    self._children = []
    self._queue = None
    self._stop_event = None

  @abstractmethod
  def start(self, workers=1, max_queue_size=10):
    """Starts the workers.

    Arguments:
        workers: number of workers
        max_queue_size: queue size
            (when full, threads could block on `put()`).
    """
    raise NotImplementedError

  def is_running(self):
    """Checks if background workers are still running

    Returns:
        True if workers are still running, False otherwise
    """
    return self._stop_event is not None and not self._stop_event.is_set()

  def get(self):
    """Creates a generator to extract data from the queue.

    Skip the data if it is `None`.

    Returns:
        Generator yielding tuples `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
    """
    while self.is_running():
      if not self._queue.empty():
        success, value = self._queue.get()
        # Rethrow any exceptions found in the queue
        if not success:
          six.reraise(value.__class__, value, value.__traceback__)
        # Yield regular values
        if value is not None:
          yield value
      else:
        all_finished = all([not child.is_alive() for child in self._children])
        if all_finished and self._queue.empty():
          return
        time.sleep(self._wait_time)

    # Make sure to rethrow the first exception in the queue, if any
    while not self._queue.empty():
      success, value = self._queue.get()
      if not success:
        six.reraise(value.__class__, value, value.__traceback__)

  def stop(self, timeout=None):
    """Stops background workers and waits for them to exit, if necessary.

    Should be called by the same thread which called `start()`.

    Arguments:
        timeout: maximum time to wait on `join()`
    """
    if self.is_running():
      # let the children know we're done
      self._stop_event.set()

    for child in self._children:
      while child.is_alive():
        # drain any remaining messages, otherwise join will block
        self._drain_queue()
        time.sleep(self._wait_time)
      child.join(timeout)

    self._children = []
    self._queue = None
    self._stop_event = None

  def _drain_queue(self):
    while not self._queue.empty():
      success, value = self._queue.get()
      if not success:
        six.reraise(value.__class__, value, value.__traceback__)

class DelegateEnqueuer(object):
  """Delegates SequenceEnqeurer operations to an underlying implementation"""

  def __init__(self, instance):
    """Arguments:
        instance: An object that implements SequenceEnqueuer

    See: MultiProcEnqueuer, ThreadedEnqueuer
    """

    self._instance = instance

  def start(self, workers=1, max_queue_size=10):
    self._instance.start(workers=workers, max_queue_size=max_queue_size)

  def is_running(self):
    return self._instance.is_running()

  def get(self):
    return self._instance.get()

  def stop(self):
    self._instance.stop()

@tf_export('keras.utils.GeneratorEnqueuer')
class GeneratorEnqueuer(DelegateEnqueuer):
  """Builds a queue out of a data generator.

  The provided generator can be finite in which case the class will throw
  a `StopIteration` exception.

  Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

  Arguments:
      generator: a generator function which yields data
      use_multiprocessing: use multiprocessing if True, otherwise threading
      wait_time: time to sleep waiting for workers to generate data or exit
  """

  def __init__(self, generator, use_multiprocessing=False, wait_time=0.05):
    if use_multiprocessing:
      instance = MultiProcGeneratorEnqueuer(generator, wait_time=wait_time)
    else:
      instance = ThreadedGeneratorEnqueuer(generator, wait_time=wait_time)

    super(GeneratorEnqueuer, self).__init__(instance)

@tf_export('keras.utils.OrderedEnqueuer')
class OrderedEnqueuer(DelegateEnqueuer):
  """Builds a Enqueuer from a Sequence.

  Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

  Arguments:
      sequence: A `keras.utils.data_utils.Sequence` object.
      use_multiprocessing: use multiprocessing if True, otherwise threading
      wait_time: time to sleep waiting for workers to generate data or exit
      shuffle: whether to shuffle the data at the beginning of each epoch
  """

  def __init__(
      self,
      sequence,
      use_multiprocessing=False,
      wait_time=0.05,
      shuffle=False):

    if use_multiprocessing:
      instance = MultiProcOrderedEnqueuer(
          sequence,
          wait_time=wait_time,
          shuffle=shuffle)
    else:
      instance = ThreadedOrderedEnqueuer(
          sequence,
          wait_time=wait_time,
          shuffle=shuffle)

    super(OrderedEnqueuer, self).__init__(instance)

class MultiProcEnqueuer(SequenceEnqueuer):
  """Enqueuer that implements multiprocessed workers

  Arguments:
      manager: A `multiprocessing.Manager` instance. Used to synchronize
          communication and state between processes
      task: the `target` function processes should run
      task_kwargs: arguments to pass to `task`
      wait_time: time to sleep waiting for workers to generate data or exit
  """
  def __init__(self, manager, task, task_kwargs=None, wait_time=0.05):
    if os.name == 'nt':
      # On Windows, avoid **SYSTEMATIC** error in `multiprocessing`:
      # `TypeError: can't pickle generator objects`
      # => Suggest multithreading instead of multiprocessing on Windows
      raise ValueError('Using a generator with `use_multiprocessing=True`'
                       ' is not supported on Windows (no marshalling of'
                       ' generators across process boundaries). Instead,'
                       ' use single thread/process or multithreading.')

    self._manager = manager
    self._task = task
    self._task_kwargs = task_kwargs if task_kwargs else {}
    self._wait_time = wait_time

    super(MultiProcEnqueuer, self).__init__()

  def start(self, workers=1, max_queue_size=10):
    """Starts the worker processes and initializes multiprocess synchronization
    constructs

    Arguments:
        workers: number of processes
        max_queue_size: queue size
            (when full, processes could block on `put()`).
    """
    try:
      self._stop_event = self._manager.Event()
      self._queue = self._manager.Queue(maxsize=max_queue_size)

      task = self._task
      task_kwargs = self._task_kwargs

      task_kwargs['lock'] = self._manager.Lock()
      task_kwargs['stop_event'] = self._stop_event
      task_kwargs['queue'] = self._queue

      for _ in range(workers):
        child = multiprocessing.Process(target=task, kwargs=task_kwargs)
        child.daemon = True
        self._children.append(child)
        child.start()

    except:
      self.stop()
      raise

  def stop(self, timeout=None):
    """Stops background processes and shuts down the Process Manager

    Arguments:
        timeout: maximum time to wait on `join()`
    """
    super(MultiProcEnqueuer, self).stop(timeout)
    self._manager.shutdown()

class ThreadedEnqueuer(SequenceEnqueuer):
  """Enqueuer that implements multithreaded workers

  Arguments:
      task: the `target` function threads should run
      task_kwargs: arguments to pass to `task`
      wait_time: time to sleep waiting for workers to generate data or exit
  """
  def __init__(self, task, task_kwargs=None, wait_time=0.05):
    self._task = task
    self._task_kwargs = task_kwargs if task_kwargs else {}
    self._wait_time = wait_time

    super(ThreadedEnqueuer, self).__init__()

  def start(self, workers=1, max_queue_size=10):
    """Starts the worker threads and initializes threading synchronization
    constructs

    Arguments:
        workers: number of threads
        max_queue_size: queue size
            (when full, threads could block on `put()`).
    """
    try:
      self._stop_event = threading.Event()
      self._queue = queue.Queue(maxsize=max_queue_size)

      task = self._task
      task_kwargs = self._task_kwargs

      task_kwargs['lock'] = threading.Lock()
      task_kwargs['stop_event'] = self._stop_event
      task_kwargs['queue'] = self._queue

      for _ in range(workers):
        child = threading.Thread(target=task, kwargs=task_kwargs)
        self._children.append(child)
        child.start()

    except:
      self.stop()
      raise

class GeneratorProxy(BaseProxy):
  """`multiprocessing.Manager` Proxy class used to synchronize and dispatch
      calls to the GeneratorEnqueuer data generator object across multiple
      processes"""

  _exposed_ = ('next', '__next__')

  def __iter__(self):
    return self

  def next(self):
    return self._callmethod('next')

  def __next__(self):
    return self._callmethod('__next__')

class GeneratorManager(SyncManager):
  """Manager wrapper used to register and share Proxy objects across processes
  """
  pass

def _data_generator_task(**kwargs):
  """Function used by both threaded and multiprocessed versions of
      GeneratorEnqueuer to get data from a generator and send it to the parent
      process via a synchronized queue
  """
  generator = kwargs['generator']
  lock = kwargs['lock']
  stop_event = kwargs['stop_event']
  taskqueue = kwargs['queue']

  while not stop_event.is_set():
    try:
      with lock:
        generator_output = next(generator)
      taskqueue.put((True, generator_output))
    except StopIteration:
      break
    except Exception as e:  # pylint: disable=broad-except
      # Can't pickle tracebacks.
      # As a compromise, print the traceback and pickle None instead.
      traceback.print_exc()
      setattr(e, '__traceback__', None)
      taskqueue.put((False, e))
      stop_event.set()
      break

class MultiProcGeneratorEnqueuer(MultiProcEnqueuer):
  """Multiprocessed version of GeneratorEnqueuer

  Shares a data generator across N number of worker processes and coordinates
  and synchronizes them to build a stream of datas

  Arguments:
      generator: a generator function which yields data
      wait_time: time to sleep waiting for workers to generate data or exit
  """

  def __init__(self, generator, wait_time=0.05):
    self._uid = str(uuid4())

    # wrap generator with a function we can register with the manager proxy
    # object. it needs a pre-primed generator funcion.
    def g():
      while True:
        try:
          yield next(generator)
        except StopIteration:
          return

    # register the Proxy generator with the multiprocessing Manager to
    # share across processes
    self._generator_typeid = 'generator'+self._uid
    GeneratorManager.register(
        self._generator_typeid, g, proxytype=GeneratorProxy)

    manager = GeneratorManager()

    super(MultiProcGeneratorEnqueuer, self).__init__(
        manager, _data_generator_task, wait_time=wait_time)

  def start(self, workers=1, max_queue_size=10):
    """Starts the Process Manager and intiailizes the Proxy generator"""
    self._manager.start()

    generator = getattr(self._manager, self._generator_typeid)()

    self._task_kwargs = {
        'generator': generator,
    }

    super(MultiProcGeneratorEnqueuer, self).start(
        workers=workers, max_queue_size=max_queue_size)

class ThreadedGeneratorEnqueuer(ThreadedEnqueuer):
  """Multithreaded version of GeneratorEnqueuer

  Shares a data generator across N number of worker threads and coordinates
  and synchronizes them to build a stream of datas

  Arguments:
      generator: a generator function which yields data
      wait_time: time to sleep waiting for workers to generate data or exit
  """
  def __init__(self, generator, wait_time=0.05):

    task_kwargs = {
        'generator': generator,
    }

    super(ThreadedGeneratorEnqueuer, self).__init__(
        _data_generator_task, task_kwargs=task_kwargs, wait_time=wait_time)

class SequenceWrapper(object):
  """Holds the underlying sequence to be shared across OrderedEnqueur workers

  Necessary because Manager Proxy classes cannot wrap instances, only classes
      or functions
  """
  def __init__(self, sequence):
    self._sequence = sequence

  def __getitem__(self, item):
    return self._sequence.__getitem__(item)

  def __len__(self):
    return self._sequence.__len__()

  def on_epoch_end(self):
    return self._sequence.on_epoch_end()

class SequenceProxy(BaseProxy):
  """`multiprocessing.Manager` Proxy class used to synchronize and dispatch
      calls to the OrderedEnqueuer data sequence object across multiple
      processes"""
  _exposed_ = ['__getitem__', '__len__', 'on_epoch_end']

  def __getitem__(self, item):
    return self._callmethod('__getitem__', (item,))

  def __len__(self):
    return self._callmethod('__len__')

  def on_epoch_end(self):
    return self._callmethod('on_epoch_end')

class SequenceManager(SyncManager):
  """Manager wrapper used to register and share Proxy objects across processes
  """
  pass

def seq_next_i(seq_order):
  for i in cycle(seq_order):
    yield i

def _data_sequence_task(**kwargs):
  """Function used by both threaded and multiprocessed versions of
      OrderedEnqueuer to get data from a sequence and send it to the parent
      process via a synchronized queue
  """
  counter = kwargs['counter']
  lock = kwargs['lock']
  next_i_gen = kwargs['next_i_gen']
  sequence = kwargs['sequence']
  stop_event = kwargs['stop_event']
  taskqueue = kwargs['queue']

  while not stop_event.is_set():
    try:
      with lock:
        # put call to queue.put in critical
        # section since we need to maintain order
        i = next(next_i_gen)
        seq_output = sequence[i]
        taskqueue.put((True, seq_output))

        counter.value += 1
        if counter.value % len(sequence) == 0:
          # saw every item in the sequence, end of epoch
          counter.value = 0
          sequence.on_epoch_end()
    except Exception as e:  # pylint: disable=broad-except
      # Can't pickle tracebacks.
      # As a compromise, print the traceback and pickle None instead.
      traceback.print_exc()
      setattr(e, '__traceback__', None)
      taskqueue.put((False, e))
      stop_event.set()
      break

class MultiProcOrderedEnqueuer(MultiProcEnqueuer):
  """Multiprocessed version of OrderedEnqueuer

  Shares a data sequence across N number of worker processes and coordinates
  and synchronizes them to build a stream of datas

  Arguments:
      sequence: A `tf.keras.utils.data_utils.Sequence` object.
      wait_time: time to sleep waiting for workers to generate data or exit
  """

  def __init__(self, sequence, wait_time=0.05, shuffle=False):
    self._sequence = sequence
    self._shuffle = shuffle

    self._seq_order = list(range(len(self._sequence)))
    self._uid = str(uuid4())

    # register the Manager Proxy classes used to share and synchronize access
    # to the underlying sequence
    self._sequence_typeid = 'sequence'+self._uid
    SequenceManager.register(
        self._sequence_typeid, SequenceWrapper, proxytype=SequenceProxy)

    self._seq_next_i_typeid = 'seq_next_i'+self._uid
    SequenceManager.register(
        self._seq_next_i_typeid, seq_next_i, proxytype=GeneratorProxy)

    manager = SequenceManager()

    super(MultiProcOrderedEnqueuer, self).__init__(
        manager, _data_sequence_task, wait_time=wait_time)

  def start(self, workers=1, max_queue_size=10):
    """Starts the Process Manager and intiailizes the Proxy sequence"""
    if self._shuffle:
      random.shuffle(self._seq_order)

    manager = self._manager
    manager.start()

    # equivalent to self._manager.sequence<uuid>(self._sequence)
    # creates and returns instances of the appropriate proxy classes
    sequence = getattr(manager, self._sequence_typeid)(self._sequence)
    next_i_gen = getattr(manager, self._seq_next_i_typeid)(self._seq_order)

    self._task_kwargs = {
        'counter': manager.Value('i', 0),
        'next_i_gen': next_i_gen,
        'sequence': sequence,
    }

    super(MultiProcOrderedEnqueuer, self).start(
        workers=workers, max_queue_size=max_queue_size)

class CounterWrapper(object):
  """Simple wrapper around an integer counter to be used by threads

  This is to provide a consistent counter interface to `_data_sequence_task`
  """
  def __init__(self):
    self.value = 0

class ThreadedOrderedEnqueuer(ThreadedEnqueuer):
  """Multithreaded version of OrderedEnqueuer

  Shares a data sequence across N number of worker threads and coordinates
  and synchronizes them to build a stream of datas

  Arguments:
      sequence: A `tf.keras.utils.data_utils.Sequence` object.
      wait_time: time to sleep waiting for workers to generate data or exit
  """
  def __init__(self, sequence, wait_time=0.05, shuffle=False):
    self._sequence = sequence
    self._shuffle = shuffle

    self._seq_order = list(range(len(self._sequence)))

    super(ThreadedOrderedEnqueuer, self).__init__(
        _data_sequence_task, wait_time=wait_time)

  def start(self, workers=1, max_queue_size=10):
    if self._shuffle:
      random.shuffle(self._seq_order)

    self._task_kwargs = {
        'counter': CounterWrapper(),
        'next_i_gen': seq_next_i(self._seq_order),
        'sequence': self._sequence,
    }

    super(ThreadedOrderedEnqueuer, self).start(
        workers=workers, max_queue_size=max_queue_size)
