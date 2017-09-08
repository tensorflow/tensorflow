/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("ObtainNext")
    .Input("list: string")
    .Input("counter: Ref(int64)")
    .Output("out_element: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused_input, input1;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused_input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &input1));
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Takes a list and returns the next based on a counter in a round-robin fashion.

Returns the element in the list at the new position of the counter, so if you
want to circle the list around start by setting the counter value = -1.

list: A list of strings
counter: A reference to an int64 variable
)doc");

}  // namespace tensorflow
