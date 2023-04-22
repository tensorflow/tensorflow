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

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

REGISTER_OP("GetCalibrationDataOp")
    .Input("resource_name: string")
    .Output("serialized_resource: string")
    .SetShapeFn(shape_inference::ScalarShape)
    .SetIsStateful()
    .Doc(R"doc(
Returns calibration data for the given resource name
)doc");

}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
