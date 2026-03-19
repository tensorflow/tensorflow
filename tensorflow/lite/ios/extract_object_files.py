# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Module for extracting object files from a compiled archive (.a) file.

This module provides functionality almost identical to the 'ar -x' command,
which extracts out all object files from a given archive file. This module
assumes the archive is in the BSD variant format used in Apple platforms.

See: https://en.wikipedia.org/wiki/Ar_(Unix)#BSD_variant

This extractor has two important differences compared to the 'ar -x' command
shipped with Xcode.

1.  When there are multiple object files with the same name in a given archive,
    each file is renamed so that they are all correctly extracted without
    overwriting each other.

2.  This module takes the destination directory as an additional parameter.

    Example Usage:

    archive_path = ...
    dest_dir = ...
    extract_object_files(archive_path, dest_dir)
"""

import hashlib
import io
import itertools
import os
import struct
from typing import Iterator, Tuple


def extract_object_files(archive_file: io.BufferedIOBase,
                         dest_dir: str) -> None:
  """Extracts object files from the archive path to the destination directory.

  Extracts object files from the given BSD variant archive file. The extracted
  files are written to the destination directory, which will be created if the
  directory does not exist.

  Colliding object file names are automatically renamed upon extraction in order
  to avoid unintended overwriting.

  Args:
    archive_file: The archive file object pointing at its beginning.
    dest_dir: The destination directory path in which the extracted object files
      will be written. The directory will be created if it does not exist.
  """
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

  _check_archive_signature(archive_file)

  # Keep the extracted file names and their content hash values, in order to
  # handle duplicate names correctly.
  extracted_files = dict()

  for name, file_content in _extract_next_file(archive_file):
    digest = hashlib.md5(file_content).digest()

    # Check if the name is already used. If so, come up with a different name by
    # incrementing the number suffix until it finds an unused one.
    # For example, if 'foo.o' is used, try 'foo_1.o', 'foo_2.o', and so on.
    for final_name in _generate_modified_filenames(name):
      if final_name not in extracted_files:
        extracted_files[final_name] = digest

        # Write the file content to the desired final path.
        with open(os.path.join(dest_dir, final_name), 'wb') as object_file:
          object_file.write(file_content)
        break

      # Skip writing this file if the same file was already extracted.
      elif extracted_files[final_name] == digest:
        break


def _generate_modified_filenames(filename: str) -> Iterator[str]:
  """Generates the modified filenames with incremental name suffix added.

  This helper function first yields the given filename itself, and subsequently
  yields modified filenames by incrementing number suffix to the basename.

  Args:
    filename: The original filename to be modified.

  Yields:
    The original filename and then modified filenames with incremental suffix.
  """
  yield filename

  base, ext = os.path.splitext(filename)
  for name_suffix in itertools.count(1, 1):
    yield '{}_{}{}'.format(base, name_suffix, ext)


def _check_archive_signature(archive_file: io.BufferedIOBase) -> None:
  """Checks if the file has the correct archive header signature.

  The cursor is moved to the first available file header section after
  successfully checking the signature.

  Args:
    archive_file: The archive file object pointing at its beginning.

  Raises:
    RuntimeError: The archive signature is invalid.
  """
  signature = archive_file.read(8)
  if signature != b'!<arch>\n':
    raise RuntimeError('Invalid archive file format.')


def _extract_next_file(
    archive_file: io.BufferedIOBase) -> Iterator[Tuple[str, bytes]]:
  """Extracts the next available file from the archive.

  Reads the next available file header section and yields its filename and
  content in bytes as a tuple. Stops when there are no more available files in
  the provided archive_file.

  Args:
    archive_file: The archive file object, of which cursor is pointing to the
      next available file header section.

  Yields:
    The name and content of the next available file in the given archive file.

  Raises:
    RuntimeError: The archive_file is in an unknown format.
  """
  while True:
    header = archive_file.read(60)
    if not header:
      return
    elif len(header) < 60:
      raise RuntimeError('Invalid file header format.')

    # For the details of the file header format, see:
    # https://en.wikipedia.org/wiki/Ar_(Unix)#File_header
    # We only need the file name and the size values.
    name, _, _, _, _, size, end = struct.unpack('=16s12s6s6s8s10s2s', header)
    if end != b'`\n':
      raise RuntimeError('Invalid file header format.')

    # Convert the bytes into more natural types.
    name = name.decode('ascii').strip()
    size = int(size, base=10)
    odd_size = size % 2 == 1

    # Handle the extended filename scheme.
    if name.startswith('#1/'):
      filename_size = int(name[3:])
      name = archive_file.read(filename_size).decode('utf-8').strip(' \x00')
      size -= filename_size

    file_content = archive_file.read(size)
    # The file contents are always 2 byte aligned, and 1 byte is padded at the
    # end in case the size is odd.
    if odd_size:
      archive_file.read(1)

    yield (name, file_content)
