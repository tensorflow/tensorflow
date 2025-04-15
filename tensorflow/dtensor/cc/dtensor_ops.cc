/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {
namespace dtensor {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

// Change layout of input to target layout inside the same mesh cluster.
REGISTER_OP("Relayout")
    .Input("input: T")
    .Output("output: T")
    .Attr("layout: string")
    .Attr("T: type")
    .SetShapeFn(UnchangedShape);

// Relayout the input according to the layout of layout_input.
REGISTER_OP("RelayoutLike")
    .Input("input: T")
    .Input("layout_input: U")  // To infer the output mesh.
    .Output("output: T")
    .Attr("T: type")
    .Attr("U: type")
    .SetShapeFn(UnchangedShape);

// FIXME(b/271292250): Add DTensor suffix to signal this is a meta Op
// Op. Or remove this altogether, if there is no use for it.
// Copy `input` to the given mesh and layout.
REGISTER_OP("CopyToMesh")
    .Input("input: T")
    .Output("output: T")
    .Attr("mesh: string")
    .Attr("T: type")
    .SetShapeFn(UnchangedShape);

// FIXME(b/271292250): Remove this Op It is no longer used.
// Gradient of CopyToMesh.
REGISTER_OP("CopyToMeshGrad")
    .Input("input: T")
    .Input("forward_input: T")  // To infer the output mesh.
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(UnchangedShape);

// DTensorRestoreV2 that is pretty much RestoreV2 but with extra global shapes
// and layouts.
REGISTER_OP("DTensorRestoreV2")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Output("tensors: dtypes")
    .Attr("input_shapes: list(shape)")
    .Attr("input_layouts: list(string)")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle shape0, shape1, shape2;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &shape0));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &shape1));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shape2));
      TF_RETURN_IF_ERROR(c->Merge(shape1, shape2, &shape0));

      std::vector<PartialTensorShape> input_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("input_shapes", &input_shapes));
      std::vector<std::string> input_layouts;
      TF_RETURN_IF_ERROR(c->GetAttr("input_layouts", &input_layouts));

      if (input_shapes.size() != input_layouts.size()) {
        return errors::InvalidArgument(
            "Size of input_shapes and input_layouts is expected to match, but "
            "got ",
            input_shapes.size(), " for input_shapes and ", input_layouts.size(),
            " for input_layouts");
      }

      // TODO(hthu): We should be able to infer from layout and global_shape
      // field.
      return UnknownShape(c);
    });

}  // namespace dtensor
}  // namespace tensorflow
