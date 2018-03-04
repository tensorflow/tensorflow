#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
GSUTIL_BIN="/var/gcloud/google-cloud-sdk/bin/gsutil"

echo "Got teardown argument $1"

if "${GSUTIL_BIN}" rm "$1"
then
  echo "Cleaned up new tfrecord file in GCS: '$1'"
else
  echo "FAIL: Unable to clean up new tfrecord file in GCS: '$1'"
  exit 1
fi
