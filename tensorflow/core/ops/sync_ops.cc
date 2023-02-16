/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

// SyncDevice is stateful because it has a side effect: it synchronizes the GPU
// steam. If it weren't stateful, optimization passes like dead code elimination
// might incorrectly remove it.
REGISTER_OP("SyncDevice")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow
