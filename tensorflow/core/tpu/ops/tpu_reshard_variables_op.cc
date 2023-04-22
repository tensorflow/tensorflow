/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("TPUReshardVariables")
    .Attr("N: int >= 0")
    .Input("vars: N * resource")
    .Input("new_format_key: string")
    .Input("format_state_var: resource")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return ::tensorflow::shape_inference::UnknownShape(c);
    });

}  // namespace tensorflow
