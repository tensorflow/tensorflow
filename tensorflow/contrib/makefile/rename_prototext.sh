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
PROTO_FILE=${@: -1}
CC_FILE=${${PROTO_FILE}/%.proto/.pb_text.h}
H_FILE=${${PROTO_FILE}/%.proto/.pb_text.cc}
GEN_DIR=${@: -4}
GEN_CC=${GEN_DIR}/{`basename $CC_FILE`}
GEN_H=${GEN_DIR}/{`basename $H_FILE`}

sed -i -e 's%protobuf::%protobuf3::%g' ${GEN_CC}
sed -i -e 's%protobuf::%protobuf3::%g' ${GEN_H}
sed -i -e 's%google_2fprotobuf3_2f%google_2fprotobuf_2f%g' ${GEN_CC}
sed -i -e 's%google_2fprotobuf3_2f%google_2fprotobuf_2f%g' ${GEN_H}
