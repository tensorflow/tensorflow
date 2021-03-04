#!/bin/bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

cat <<EOF
TENSORFLOW_COMPATIBLE_OPS = (
EOF

# TODO(b/178456916): Leverage existing op compat definitions/specs in the
# MLIR conversion pipeline in a better way.
# TODO(b/180352158): Validate generated TF op names.
grep 'patterns.insert<Legalize' $1 | awk -F'<Legalize|>' '{printf "    \"%s\",\n", $2}'

cat <<EOF
    # Rules at tensorflow/compiler/mlir/lite/transforms/legalize_tf.cc
    "Assert",
    "ConcatV2",
    "MatMul",
    "MatrixDiagV2",
    "MatrixDiagV3",
    "Pack",
    "Split",
    "SplitV",
    "Unpack",
    "RandomUniform",
)
EOF
