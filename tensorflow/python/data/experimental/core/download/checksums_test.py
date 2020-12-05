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

"""Tests for tensorflow.data.experimental.core.download.checksums."""

import pathlib
from tensorflow.data.experimental.core.download import checksums


def test_checksums(tmp_path: pathlib.Path):
  path = tmp_path / 'checksums.tsv'
  url_infos = {
      'http://abc.org/data': checksums.UrlInfo(
          checksum='abcd',
          size=1234,
          filename='a.zip',
      ),
      'http://edf.org/data': checksums.UrlInfo(
          checksum='abcd',
          size=1234,
          filename='b.zip',
      ),
  }

  checksums.save_url_infos(path, url_infos)
  loaded_url_infos = checksums.load_url_infos(path)
  assert loaded_url_infos == url_infos
