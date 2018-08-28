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

namespace tensorflow {

REGISTER_OP("IdentityIndexedDataset")
    .Input("size: uint64")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(
        shape_inference::ScalarShape);  // TODO(saeta): check input shapes.

///////////////////////////////////////////////////////////////////////////////
//     IndexedDataset Internals
///////////////////////////////////////////////////////////////////////////////

// Creates the handle.
REGISTER_OP("MaterializedIndexDatasetHandle")
    .Output("handle: resource")
    .Attr("container: string")
    .Attr("shared_name: string")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

// Actually materialize the materialize handle.
REGISTER_OP("IndexedDatasetMaterialize")
    .Input("dataset: variant")
    .Input("materialized: resource")
    .SetShapeFn(shape_inference::NoOutputs);

namespace {

Status GetShapeFn(shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
  std::vector<PartialTensorShape> output_shapes;
  TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
  if (output_shapes.size() != c->num_outputs()) {
    return errors::InvalidArgument(
        "`output_shapes` must be the same length as `output_types` (",
        output_shapes.size(), " vs. ", c->num_outputs());
  }
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    shape_inference::ShapeHandle output_shape_handle;
    TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
        output_shapes[i], &output_shape_handle));
    c->set_output(static_cast<int>(i), output_shape_handle);
  }
  return Status::OK();
}

}  // namespace

REGISTER_OP("IndexedDatasetGet")
    .Input("materialized: resource")
    .Input("index: uint64")
    .Output("components: output_types")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(GetShapeFn)
    .Doc(R"doc(
Gets the element at `index` from `materialized` IndexedDataset.
)doc");

}  // namespace tensorflow
