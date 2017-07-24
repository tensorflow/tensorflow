/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {

REGISTER_OP("CrossReplicaSum")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
An Op to sum inputs across replicated TPU instances. Each
instance supplies its own input, and the output of each is the sum of
all the inputs.

input: The local input to the sum.
output: The sum of all the distributed inputs.
T: The type of elements to be summed.
)doc");

}  // namespace tensorflow
