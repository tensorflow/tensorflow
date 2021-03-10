#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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


# Compute the MD5 sum.
#
# Parameter(s):
#   ${1} - path to the file
function compute_md5() {
  UNAME_S=`uname -s`
  if [ ${UNAME_S} == Linux ]; then
    tflm_md5sum=md5sum
  elif [ ${UNAME_S} == Darwin ]; then
    tflm_md5sum='md5 -r'
  fi
  ${tflm_md5sum} ${1} | awk '{print $1}'
}

# Check that MD5 sum matches expected value.
#
# Parameter(s):
#   ${1} - path to the file
#   ${2} - expected md5
function check_md5() {
  MD5=`compute_md5 ${1}`

  if [[ ${MD5} != ${2} ]]
  then
    echo "Bad checksum. Expected: ${2}, Got: ${MD5}"
    exit 1
  fi

}

