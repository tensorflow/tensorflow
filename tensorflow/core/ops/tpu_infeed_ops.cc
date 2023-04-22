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

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("InfeedDequeue")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("InfeedEnqueue")
    .Input("input: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape = {}")
    .Attr("layout: list(int) = []")
    .Attr("device_ordinal: int = -1")
    .SetShapeFn(shape_inference::NoOutputs)
    .SetIsStateful();

REGISTER_OP("InfeedEnqueueTuple")
    .Input("inputs: dtypes")
    .Attr("dtypes: list(type)")
    .Attr("shapes: list(shape)")
    .Attr("layouts: list(int) = []")
    .Attr("device_ordinal: int = -1")
    .SetShapeFn(shape_inference::NoOutputs)
    .SetIsStateful();

REGISTER_OP("InfeedDequeueTuple")
    .Output("outputs: dtypes")
    .Attr("dtypes: list(type)")
    .Attr("shapes: list(shape)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      std::vector<PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      for (int i = 0; i < shapes.size(); ++i) {
        ShapeHandle out;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shapes[i], &out));
        c->set_output(i, out);
      }
      return Status::OK();
    });

REGISTER_OP("Prelinearize")
    .Input("input: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape = {}")
    .Attr("layout: list(int) = []")
    .Output("output: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("PrelinearizeTuple")
    .Input("inputs: dtypes")
    .Attr("dtypes: list(type)")
    .Attr("shapes: list(shape)")
    .Attr("layouts: list(int) = []")
    .Output("output: variant")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("InfeedEnqueuePrelinearizedBuffer")
    .Input("input: variant")
    .Attr("device_ordinal: int = -1")
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow
