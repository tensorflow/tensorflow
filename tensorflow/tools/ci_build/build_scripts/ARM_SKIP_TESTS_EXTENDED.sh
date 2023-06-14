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
#!/bin/bash
set -x

source tensorflow/tools/ci_build/build_scripts/ARM_SKIP_TESTS.sh

ARM_SKIP_TESTS="${ARM_SKIP_TESTS} \
-//tensorflow/compiler/mlir/tfr/examples/mnist:mnist_ops_test \
-//tensorflow/core/grappler/optimizers:auto_mixed_precision_test_cpu \
-//tensorflow/core/grappler/optimizers:remapper_test_cpu \
"
