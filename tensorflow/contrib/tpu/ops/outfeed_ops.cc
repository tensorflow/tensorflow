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

REGISTER_OP("OutfeedEnqueue")
    .Input("input: dtype")
    .Attr("dtype: type")
    .SetIsStateful()
    .Doc(R"doc(
An op which emits a single Tensor value from an XLA computation.

input: A tensor that will be inserted into the outfeed queue.
)doc");

REGISTER_OP("OutfeedEnqueueTuple")
    .Input("inputs: dtypes")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .Doc(R"doc(
An op which emits multiple Tensor values from an XLA computation.

inputs: A list of tensors that will be inserted into the outfeed queue as an
XLA tuple.
)doc");

REGISTER_OP("OutfeedDequeue")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("device_ordinal: int = -1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape)
    .Doc(R"doc(
Retrieves a single tensor from the computation outfeed.  This operation will
block indefinitely until data is available.

output: A tensor that will be read from the device outfeed.
dtype: The type of elements in the tensor.
shape: The shape of the tensor.
device_ordinal: The TPU device to use. This should be -1 when the Op
is running on a TPU device, and >= 0 when the Op is running on the CPU
device.
)doc");

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
      return Status::OK();
    })
    .Doc(R"doc(
Retrieve multiple values that will be emitted by the computation as an XLA
tuple.  This operations will block indefinitely until data is available.
Output `i` corresponds to XLA tuple element `i`.

outputs: A list of tensors that will be read from the outfeed.
dtypes: The element types of each element in `outputs`.
shapes: The shapes of each tensor in `outputs`.
device_ordinal: The TPU device to use. This should be -1 when the Op
is running on a TPU device, and >= 0 when the Op is running on the CPU
device.
)doc");

}  // namespace tensorflow
