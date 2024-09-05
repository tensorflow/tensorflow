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
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("OutfeedEnqueue")
    .Input("input: dtype")
    .Attr("dtype: type")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("OutfeedEnqueueTuple")
    .Input("inputs: dtypes")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("OutfeedDequeue")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("device_ordinal: int = -1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("OutfeedDequeueTuple")
    .Output("outputs: dtypes")
    .Attr("dtypes: list(type)")
    .Attr("shapes: list(shape)")
    .Attr("device_ordinal: int = -1")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      std::vector<PartialTensorShape> shapes;
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      TF_RETURN_IF_ERROR(c->GetAttr("dtypes", &dtypes));
      if (shapes.size() != dtypes.size()) {
        return errors::InvalidArgument(
            "Incorrect number of output shapes specified");
      }
      for (int i = 0; i < shapes.size(); ++i) {
        ShapeHandle out;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shapes[i], &out));
        c->set_output(i, out);
      }
      return absl::OkStatus();
    });

REGISTER_OP("OutfeedDequeueV2")
    .Input("device_ordinal: int32")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("OutfeedDequeueTupleV2")
    .Input("device_ordinal: int32")
    .Output("outputs: dtypes")
    .Attr("dtypes: list(type)")
    .Attr("shapes: list(shape)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      if (c->Rank(c->input(0)) != 0) {
        return errors::InvalidArgument("device ordinal must be a scalar.");
      }
      std::vector<PartialTensorShape> shapes;
      std::vector<DataType> dtypes;
      TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
      TF_RETURN_IF_ERROR(c->GetAttr("dtypes", &dtypes));
      if (shapes.size() != dtypes.size()) {
        return errors::InvalidArgument(
            "Incorrect number of output shapes specified");
      }
      for (int i = 0; i < shapes.size(); ++i) {
        ShapeHandle out;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shapes[i], &out));
        c->set_output(i, out);
      }
      return absl::OkStatus();
    });

}  // namespace tensorflow
