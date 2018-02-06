/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

using shape_inference::InferenceContext;

REGISTER_OP("SymbolicGradient")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type)")
    .Attr("Tout: list(type)")
    .Attr("f: func")
    .SetShapeFn([](InferenceContext* c) {
      if (c->num_inputs() < c->num_outputs()) {
        return errors::InvalidArgument("len(inputs) < len(outputs)");
      }
      // Say, (u, v) = f(x, y, z), _symbolic_gradient(f) is a function of
      // (x, y, z, du, dv) -> (dx, dy, dz). Therefore, shapes of its
      // outputs (dx, dy, dz) are the same as (x, y, z).
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    });

REGISTER_OP("RemoteCall")
    .Input("target: string")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type)")
    .Attr("Tout: list(type)")
    .Attr("f: func")
    .SetShapeFn(shape_inference::UnknownShape);
}  // end namespace tensorflow
