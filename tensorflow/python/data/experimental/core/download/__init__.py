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

"""`tfds.download.DownloadManager` API."""

from tensorflow.data.experimental.core.download.checksums import add_checksums_dir
from tensorflow.data.experimental.core.download.download_manager import DownloadConfig
from tensorflow.data.experimental.core.download.download_manager import DownloadManager
from tensorflow.data.experimental.core.download.downloader import DownloadError
from tensorflow.data.experimental.core.download.extractor import iter_archive
from tensorflow.data.experimental.core.download.resource import ExtractMethod
from tensorflow.data.experimental.core.download.resource import Resource
from tensorflow.data.experimental.core.download.util import ComputeStatsMode
from tensorflow.data.experimental.core.download.util import GenerateMode

__all__ = [
    "add_checksums_dir",
    "DownloadConfig",
    "DownloadManager",
    "DownloadError",
    "ComputeStatsMode",
    "GenerateMode",
    "Resource",
    "ExtractMethod",
    "iter_archive",
]
