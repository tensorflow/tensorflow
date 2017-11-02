#!/usr/bin/env bash
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

OUTPUT_FILENAME=$1
if [[ -z "${OUTPUT_FILENAME}"  ]]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

GIT_VERSION=$(git describe --long --tags)
if [[ $? != 0 ]]; then
   GIT_VERSION=unknown;
fi

cat <<EOF > ${OUTPUT_FILENAME}
#include <string>
const char* tf_git_version() {return "${GIT_VERSION}";}
const char* tf_compiler_version() {return __VERSION__;}
const int tf_cxx11_abi_flag() {
#ifdef _GLIBCXX_USE_CXX11_ABI
  return _GLIBCXX_USE_CXX11_ABI;
#else
  return 0;
#endif
}
EOF

