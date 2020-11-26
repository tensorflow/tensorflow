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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

REGISTER_OP("TPUExecute")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("key: string")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle key;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs() - 1), 1, &key));
      shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(key, 0), 3, &unused));
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->UnknownShape());
      }
      return Status::OK();
    });

REGISTER_OP("TPUExecuteAndUpdateVariables")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("key: string")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("device_var_reads_indices: list(int) >= 0")
    .Attr("device_var_updates_indices: list(int) >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle key;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs() - 1), 1, &key));
      shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(key, 0), 3, &unused));
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->UnknownShape());
      }
      return Status::OK();
    });

}  // namespace tensorflow
