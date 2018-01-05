// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/contrib/tensor_forest/hybrid/core/ops/utils.h"
#include "tensorflow/contrib/tensor_forest/kernels/tree_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("UnpackPath")
    .Input("path: int32")
    .Input("path_values: float")
    .Output("unpacked_path: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input, params;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &params));

      auto num_points = c->Dim(input, 0);

      auto tree_depth = c->Dim(params, 1);
      int64 num_nodes = InferenceContext::kUnknownDim;
      if (c->ValueKnown(tree_depth)) {
        num_nodes = (static_cast<int64>(1) << c->Value(tree_depth)) - 1;
      }

      c->set_output(0, c->Matrix(num_points, num_nodes));
      return Status::OK();
    })
    .Doc(R"doc(
  Takes a batch of paths through a tree and a batch of values along those paths
  and returns a batch_size by num_nodes encoding of the path values.

  path: `path[i][j]` gives the jth node in the path taken by the ith data
   instance.
  path_values: `path_values[i][j]` gives the value associated with node j in the
   path defined by the ith instance

  unpacked_paths: `unpacked_paths[i][path[i][k]]` is path_values[i][k] for k in
   [0, tree_depth).  All other elements of unpacked_paths are zero.
)doc");

class UnpackPath : public OpKernel {
 public:
  explicit UnpackPath(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    VLOG(1) << "unpack start";
    const Tensor& path_tensor = context->input(0);
    const Tensor& path_values_tensor = context->input(1);

    const int32 num_data = static_cast<int32>(path_tensor.shape().dim_size(0));
    const int32 tree_depth = static_cast<int32>(
        path_tensor.shape().dim_size(1));

    const int32 num_nodes = MathUtil::IPow(2, tree_depth) - 1;

    VLOG(1) << "num_data: " << num_data;
    VLOG(1) << "tree_depth: " << tree_depth;
    VLOG(1) << "num_nodes: " << num_nodes;

    Tensor* output = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(num_data);
    output_shape.AddDim(num_nodes);

    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    VLOG(1) << "unpack before init";
    tensorforest::Initialize(*output, 0.0f);
    VLOG(1) << "unpack after init";

    auto out = output->tensor<float, 2>();

    const auto path = path_tensor.tensor<int32, 2>();
    const auto path_values = path_values_tensor.tensor<float, 2>();

    for (int i = 0; i < num_data; i++) {
      for (int j = 0; j < tree_depth; j++) {
        CHECK_LT(path(i, j), num_nodes);
        out(i, path(i, j)) = path_values(i, j);
      }
    }
    VLOG(1) << "unpack end";
  }
};

REGISTER_KERNEL_BUILDER(Name("UnpackPath").Device(DEVICE_CPU),
                        UnpackPath);

}  // namespace tensorflow
