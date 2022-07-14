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

// Copy `input` to the given mesh and layout.
REGISTER_OP("CopyToMesh")
    .Input("input: T")
    .Output("output: T")
    .Attr("layout: string")
    .Attr("source_layout: string = ''")
    .Attr("T: type")
    .SetShapeFn(UnchangedShape);

// Queries the generated sharded prefix that is used to in SaveV2 op in a
// multi-client setup. Should take exact same inputs as the original SaveV2 is
// invoked or the value returned won't match the ones generated.
REGISTER_OP("DTensorShardedPrefix")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Input("mesh: string")
    .Input("layouts: string")
    .Input("tensors: dtypes")
    .Attr("dtypes: list(type)")
    .Output("output: string")
    .SetShapeFn([](InferenceContext* c) {
      // Always output a one d vector of strings.
      // We could calculate the exact numbers of output here as well but that's
      // the whole logic of the op itself.
      c->set_output(0, c->Vector(c->UnknownDim()));
      return OkStatus();
    });

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
