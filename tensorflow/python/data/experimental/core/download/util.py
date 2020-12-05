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

"""Utils functions."""

import enum


class GenerateMode(enum.Enum):
  """`Enum` for how to treat pre-existing downloads and data.

  The default mode is `REUSE_DATASET_IF_EXISTS`, which will reuse both
  raw downloads and the prepared dataset if they exist.

  The generations modes:

  |                                    | Downloads | Dataset |
  | -----------------------------------|-----------|---------|
  | `REUSE_DATASET_IF_EXISTS` (default)| Reuse     | Reuse   |
  | `REUSE_CACHE_IF_EXISTS`            | Reuse     | Fresh   |
  | `FORCE_REDOWNLOAD`                 | Fresh     | Fresh   |
  """

  REUSE_DATASET_IF_EXISTS = 'reuse_dataset_if_exists'
  REUSE_CACHE_IF_EXISTS = 'reuse_cache_if_exists'
  FORCE_REDOWNLOAD = 'force_redownload'


class ComputeStatsMode(enum.Enum):
  """Mode to decide if dynamic dataset info fields should be computed or not.

  Mode can be:

  * AUTO: Compute the DatasetInfo dynamic fields only if they haven't been
    restored from GCS.
  * FORCE: Always recompute DatasetInfo dynamic  fields, even if they are
    already present
  * SKIP: Ignore the dataset dynamic field computation (whether they already
    exist or not)

  """

  AUTO = 'auto'
  FORCE = 'force'
  SKIP = 'skip'
