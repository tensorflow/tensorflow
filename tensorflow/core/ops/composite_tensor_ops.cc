/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("CompositeTensorVariantFromComponents")
    .Input("components: Tcomponents")
    .Output("encoded: variant")
    .Attr("metadata: string")
    .Attr("Tcomponents: list(type) >= 0")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("CompositeTensorVariantToComponents")
    .Input("encoded: variant")
    .Output("components: Tcomponents")
    .Attr("metadata: string")
    .Attr("Tcomponents: list(type) >= 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->UnknownShape());
      }
      return absl::OkStatus();
    });

}  // namespace tensorflow
