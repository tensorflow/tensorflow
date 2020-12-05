# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Utilities for accessing TFDS GCS buckets."""

import concurrent.futures
import os
import posixpath
from typing import List, Optional

import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core.utils import py_utils
from tensorflow.data.experimental.core.utils import tqdm_utils

GCS_ROOT_DIR = 'gs://tfds-data'

# for dataset_info/
GCS_DATASET_INFO_DIR = 'dataset_info'
GCS_DATASETS_DIR = 'datasets'

_is_gcs_disabled = False


def exists(path: str) -> bool:
  """Checks if path exists. Returns False if issues occur connecting to GCS."""
  try:
    return tf.io.gfile.exists(path)
  # * UnimplementedError: On windows, gs:// isn't supported.
  # * FailedPreconditionError: Raised by TF
  # * PermissionDeniedError: Some environments block GCS access.
  except (
      tf.errors.UnimplementedError,
      tf.errors.FailedPreconditionError,
      tf.errors.PermissionDeniedError,
  ):
    # TODO(tfds): Investigate why windows, gs:// isn't supported.
    # https://github.com/tensorflow/tensorflow/issues/38477
    return False


def gcs_path(suffix: Optional[str] = None) -> str:
  """Returns the GCS URI path.

  Args:
    suffix: Eventual relative path in the bucket. If `None`, returns the root
      GCS bucket uri.

  Returns:
    path: The GCS uri.
  """
  if not suffix:
    path = GCS_ROOT_DIR
  elif suffix.startswith('gs://'):  # Path is already a full path
    path = suffix
  else:
    path = posixpath.join(GCS_ROOT_DIR, suffix)
  return path


@py_utils.memoize()
def gcs_listdir(dir_name: str) -> Optional[List[str]]:
  """List all files in the given GCS dir (`['dataset/1.0.0/file0', ...]`)."""
  root_dir = gcs_path(dir_name)
  if _is_gcs_disabled or not exists(root_dir):
    return None
  return [posixpath.join(dir_name, f) for f in tf.io.gfile.listdir(root_dir)]


def gcs_dataset_info_files(dataset_dir: str) -> Optional[List[str]]:
  """Return paths to GCS files in the given dataset directory."""
  return gcs_listdir(posixpath.join(GCS_DATASET_INFO_DIR, dataset_dir))


def is_dataset_on_gcs(dataset_name: str) -> bool:
  """If the dataset is available on the GCS bucket gs://tfds-data/datasets."""
  dir_name = posixpath.join(GCS_DATASETS_DIR, dataset_name)
  return not _is_gcs_disabled and exists(gcs_path(dir_name))


def download_gcs_dataset(
    dataset_name, local_dataset_dir, max_simultaneous_downloads=25
):
  """Downloads prepared GCS dataset to local dataset directory."""
  if _is_gcs_disabled:
    raise AssertionError('Cannot download from GCS when _is_gcs_disabled')

  prefix = posixpath.join(GCS_DATASETS_DIR, dataset_name)
  gcs_paths_to_dl = gcs_listdir(prefix)

  # Filter out the diffs folder if present
  filter_prefix = posixpath.join(prefix, 'diffs')
  gcs_paths_to_dl = [
      p for p in gcs_paths_to_dl if not p.startswith(filter_prefix)
  ]

  with tqdm_utils.async_tqdm(
      total=len(gcs_paths_to_dl), desc='Dl Completed...', unit=' file') as pbar:

    def _copy_from_gcs(gcs_path_):
      # Copy 'gs://tfds-data/datasets/ds/1.0.0/file' -> `local_dir/file`
      tf.io.gfile.copy(
          gcs_path(gcs_path_),
          os.path.join(local_dataset_dir, posixpath.basename(gcs_path_)),
      )
      pbar.update(1)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_simultaneous_downloads) as executor:
      futures = [
          executor.submit(_copy_from_gcs, path) for path in gcs_paths_to_dl
      ]
      for future in concurrent.futures.as_completed(futures):
        future.result()
