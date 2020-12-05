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

"""Public API of the proto package."""
# pylint: disable=g-import-not-at-top,g-importing-member, import-outside-toplevel

try:  # pylint: disable=g-statement-before-imports
  from waymo_open_dataset import waymo_dataset_pb2
except ImportError:
  # If original waymo proto is not found, fallback to the pre-generated proto
  from tensorflow_datasets.proto import waymo_dataset_generated_pb2 as waymo_dataset_pb2  # pylint: disable=line-too-long
