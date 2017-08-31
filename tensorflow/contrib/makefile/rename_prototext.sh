#!/bin/bash -e -x

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

PROTO_TEXT_COMMAND=$1
shift
${PROTO_TEXT_COMMAND} $*

# Assumes a fixed order for the arguments.
PROTO_LAST_FOUR_ARGS=(${@: -4})
PROTO_FILE=${PROTO_LAST_FOUR_ARGS[3]}
CC_FILE=${PROTO_FILE%.proto}.pb_text.cc
H_FILE=${PROTO_FILE%.proto}.pb_text.h
GEN_DIR=${PROTO_LAST_FOUR_ARGS[0]}
GEN_CC=${GEN_DIR}/${CC_FILE#tensorflow/core}
GEN_H=${GEN_DIR}/${H_FILE#tensorflow/core}

sed -i '' 's%protobuf::%protobuf3::%g' ${GEN_CC}
sed -i '' 's%protobuf::%protobuf3::%g' ${GEN_H}
sed -i '' 's%google_2fprotobuf3_2f%google_2fprotobuf_2f%g' ${GEN_CC}
sed -i '' 's%google_2fprotobuf3_2f%google_2fprotobuf_2f%g' ${GEN_H}
