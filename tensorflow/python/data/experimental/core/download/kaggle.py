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

"""Data downloads using the Kaggle CLI."""

import os
import subprocess
import textwrap
from typing import List
import zipfile

from absl import logging
import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.download import extractor
from tensorflow.data.experimental.core.download import resource


def _get_kaggle_type(competition_or_dataset: str) -> str:
  """Returns the kaggle type (competitions/datasets).

  Args:
    competition_or_dataset: Name of the kaggle competition/dataset.

  Returns:
    Kaggle type (competitions/datasets).
  """
  # Dataset are `user/dataset_name`
  return 'datasets' if '/' in competition_or_dataset else 'competitions'


def _kaggle_dir_name(competition_or_dataset: str) -> str:
  """Returns path where the dataset is to be downloaded.

  Args:
    competition_or_dataset: Name of the kaggle competition/dataset.

  Returns:
    Path to the dir where the dataset is to be downloaded.
  """
  return competition_or_dataset.replace('/', '_')


def _run_command(command_args: List[str]) -> str:
  """Run kaggle command with subprocess.

  Args:
    command_args: Arguments to the kaggle api.

  Returns:
    output of the command.

  Raises:
    CalledProcessError: If the command terminates with exit status 1.
  """
  command_str = ' '.join(command_args)
  competition_or_dataset = command_args[-1]
  try:
    return subprocess.check_output(command_args, encoding='UTF-8')
  except (subprocess.CalledProcessError, FileNotFoundError) as err:
    if isinstance(err, subprocess.CalledProcessError) and '404' in err.output:
      raise ValueError(textwrap.dedent("""\
      Error for command: {}

      Competition {} not found. Please ensure you have spelled the name
      correctly.
      """).format(command_str, competition_or_dataset))
    else:
      raise RuntimeError(textwrap.dedent("""\
      Error for command: {}

      To download Kaggle data through TFDS, follow the instructions to install
      the kaggle API and get API credentials:
      https://github.com/Kaggle/kaggle-api#installation

      Additionally, you may have to join the competition through the Kaggle
      website: https://www.kaggle.com/c/{}
      """).format(command_str, competition_or_dataset))


def _download_competition_or_dataset(
    competition_or_dataset: str, output_dir: str
) -> None:
  """Downloads the data and extracts it if it was zipped by the kaggle api.

  Args:
    competition_or_dataset: Name of the kaggle competition/dataset.
    output_dir: Path to the dir where the data is to be downloaded.
  """
  _run_command([
      'kaggle',
      _get_kaggle_type(competition_or_dataset),
      'download',
      '--path',
      output_dir,
      competition_or_dataset,
  ])
  for download in tf.io.gfile.listdir(output_dir):
    fpath = os.path.join(output_dir, download)
    if zipfile.is_zipfile(fpath):
      ext = extractor.get_extractor()
      with ext.tqdm():
        ext.extract(fpath, resource.ExtractMethod.ZIP, output_dir).get()


def download_kaggle_data(
    competition_or_dataset: str,
    download_dir: utils.PathLike,
) -> utils.ReadWritePath:
  """Downloads the kaggle data to the output_dir.

  Args:
    competition_or_dataset: Name of the kaggle competition/dataset.
    download_dir: Path to the TFDS downloads dir.

  Returns:
    Path to the dir where the kaggle data was downloaded.
  """
  kaggle_dir = _kaggle_dir_name(competition_or_dataset)
  download_path = utils.as_path(download_dir) / kaggle_dir
  # If the dataset has already been downloaded, return the path to it.
  if download_path.is_dir():
    logging.info(
        'Dataset %s already downloaded: reusing %s.',
        competition_or_dataset,
        download_path,
    )
    return download_path
  # Otherwise, download the dataset.
  with utils.incomplete_dir(download_path) as tmp_data_dir:
    logging.info(
        'Downloading %s into %s...',
        competition_or_dataset,
        tmp_data_dir,
    )
    _download_competition_or_dataset(competition_or_dataset, tmp_data_dir)
  return download_path
