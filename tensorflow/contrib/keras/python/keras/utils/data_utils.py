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

import functools
import hashlib
import os
import shutil
import sys
import tarfile
import zipfile

import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen

from tensorflow.contrib.keras.python.keras.utils.generic_utils import Progbar


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
      total_size = response.info().get('Content-Length').strip()
      total_size = int(total_size)
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
    progbar = None

    def dl_progress(count, block_size, total_size, progbar=None):
      if progbar is None:
        progbar = Progbar(total_size)
      else:
        progbar.update(count * block_size)

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
      try:
        urlretrieve(origin, fpath,
                    functools.partial(dl_progress, progbar=progbar))
      except URLError as e:
        raise Exception(error_msg.format(origin, e.errno, e.reason))
      except HTTPError as e:
        raise Exception(error_msg.format(origin, e.code, e.msg))
    except (Exception, KeyboardInterrupt) as e:
      if os.path.exists(fpath):
        os.remove(fpath)
      raise
    progbar = None

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
