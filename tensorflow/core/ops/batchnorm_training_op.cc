/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("BatchNormTraining")
    .Input("inp: T")
    .Input("scale: T")
    .Input("bias: T")
    .Input("running_mean: Ref(T)")
    .Input("running_inv_var: Ref(T)")
    .Output("out: T")
    .Output("save_mean: T")
    .Output("save_inv_var: T")
    .Attr("T: {float}")
    .Attr("epsilon: float")
    .Doc(R"doc(
      TODO
)doc");

REGISTER_OP("BatchNormTrainingGrad")
    .Input("inp: T")
    .Input("output_grad: T")
    .Input("scale: T")
    .Input("saved_mean: T")
    .Input("saved_var: T")
    .Output("input_grad: T")
    .Output("scale_grad: T")
    .Output("bias_grad: T")
    .Attr("T: {float}")
    .Attr("epsilon: float")
    .Doc(R"doc(
      TODO
)doc");
}  // namespace tensorflow
