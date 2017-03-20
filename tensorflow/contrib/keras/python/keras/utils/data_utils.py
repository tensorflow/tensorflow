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


def get_file(fname, origin, untar=False, md5_hash=None,
             cache_subdir='datasets'):
  """Downloads a file from a URL if it not already in the cache.

  Passing the MD5 hash will verify the file after download
  as well as if it is already present in the cache.

  Arguments:
      fname: name of the file
      origin: original URL of the file
      untar: boolean, whether the file should be decompressed
      md5_hash: MD5 hash of the file for verification
      cache_subdir: directory being used as the cache

  Returns:
      Path to the downloaded file
  """
  datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
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
    if md5_hash is not None:
      if not validate_file(fpath, md5_hash):
        print('A local file was found, but it seems to be '
              'incomplete or outdated.')
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
      print('Untaring file...')
      tfile = tarfile.open(fpath, 'r:gz')
      try:
        tfile.extractall(path=datadir)
      except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(untar_fpath):
          if os.path.isfile(untar_fpath):
            os.remove(untar_fpath)
          else:
            shutil.rmtree(untar_fpath)
        raise
      tfile.close()
    return untar_fpath

  return fpath


def validate_file(fpath, md5_hash):
  """Validates a file against a MD5 hash.

  Arguments:
      fpath: path to the file being validated
      md5_hash: the MD5 hash being validated against

  Returns:
      Whether the file is valid
  """
  hasher = hashlib.md5()
  with open(fpath, 'rb') as f:
    buf = f.read()
    hasher.update(buf)
  if str(hasher.hexdigest()) == str(md5_hash):
    return True
  else:
    return False
