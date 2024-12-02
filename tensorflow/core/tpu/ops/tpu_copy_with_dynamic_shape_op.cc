/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// Op that copy dynamic shape tensor to device.

#include "absl/status/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

REGISTER_OP("TPUCopyWithDynamicShape")
    .Input("tensors: T")
    .Input("unpadded_sizes: N * int32")
    .Output("tpu_tensors: T")
    .Attr("N: int >= 0")
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      int n;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
      for (int i = 0; i < c->num_inputs() - n; ++i) {
        c->set_output(i, c->input(i));
      }
      return absl::OkStatus();
    });

REGISTER_OP("TPUAnnotateTensorsWithDynamicShape")
    .Input("tensors: T")
    .Output("tpu_tensors: T")
    .Attr("T: list(type)")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) -> absl::Status {
      for (int i = 0; i < c->num_inputs(); ++i) {
        c->set_output(i, c->input(i));
      }
      return absl::OkStatus();
    });

}  // namespace tensorflow
