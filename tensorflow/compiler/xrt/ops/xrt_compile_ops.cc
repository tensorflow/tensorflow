/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

REGISTER_OP("XRTCompile")
    .Input("computation: string")
    .Output("handle: int64")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(
        R"(
Reads a computation proto, compiles it, and places it in the global compilation
cache.

'computation' is a serialized xrt::XLAComputation proto.
'handle' is an identifier that can be used in other ops to refer to the
computation.
)");

REGISTER_OP("XRTReleaseCompilationHandle")
    .Input("handle: int64")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(
        R"(
Discards a computation from the compilation cache. The handle cannot be
subsequently used.

'handle' is an id returned from a XRTCompile Op.
)");

}  // namespace tensorflow
