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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/save_restore_util.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {

// An OpKernel that queries prefixes of all generated Save v2 ops. This is
// needed in distributed context to track all save ops inserted by DTensor SPMD,
// and ensures a proper MergeV2 ops afterwards.
class DTensorShardedPrefixOpKernel : public OpKernel {
 public:
  explicit DTensorShardedPrefixOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

 private:
  void Compute(OpKernelContext* ctx) override {
    const Tensor* prefix_tensor;
    const Tensor* tensor_names;
    const Tensor* shape_and_slices;
    const Tensor* mesh_tensor;
    const Tensor* layouts;

    OP_REQUIRES_OK(ctx, ctx->input("prefix", &prefix_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("tensor_names", &tensor_names));
    OP_REQUIRES_OK(ctx, ctx->input("shape_and_slices", &shape_and_slices));
    OP_REQUIRES_OK(ctx, ctx->input("mesh", &mesh_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("layouts", &layouts));

    const std::string& prefix = prefix_tensor->scalar<tstring>()();

    const auto& shape_and_slices_vec = shape_and_slices->flat<tstring>();
    for (int i = 0; i < shape_and_slices->NumElements(); ++i) {
      OP_REQUIRES(
          ctx, shape_and_slices_vec(i).empty(),
          errors::Unimplemented("DTensor save currently does not support "
                                "distributed save with shape_and_slices"));
    }

    const std::string& mesh_str = mesh_tensor->scalar<tstring>()();
    const auto& mesh_or = Mesh::FromString(mesh_str);
    OP_REQUIRES(ctx, mesh_or.ok(),
                errors::InvalidArgument(
                    absl::StrCat("Got invalid mesh string : ", mesh_str)));
    const Mesh& mesh = *mesh_or;

    const auto& layouts_flat = layouts->flat<tensorflow::tstring>();
    OP_REQUIRES(ctx, tensor_names->NumElements() == layouts->NumElements(),
                errors::InvalidArgument(absl::StrCat(
                    "tensor_names must match the size of layouts, "
                    "but got tensor_names size : ",
                    tensor_names->NumElements(),
                    " and layouts size : ", layouts->NumElements())));

    // (prefix, tensor names, shape_and_slices, mesh, layout) are fixed inputs
    // while tensors are variadic inputs.
    const int kFixedInputs = 5;
    const int num_tensors = static_cast<int>(tensor_names->NumElements());

    // Construct a map of the <tensor_idex -> (tensor_shape, Layout)) to build
    // the saving spec.
    std::vector<SavingTensorMetadata> metadata;
    metadata.reserve(num_tensors);

    for (int i = 0; i < num_tensors; ++i) {
      const string& layout_string = layouts_flat(i);
      const Tensor& tensor = ctx->input(i + kFixedInputs);

      // Note that in runtime we always have local shape, so here we recovers to
      // global shape to compute saving specs correctly.
      const TensorShape& shape = tensor.shape();
      std::vector<int64_t> local_shape;
      local_shape.reserve(shape.dims());
      for (int dim = 0; dim < shape.dims(); ++dim) {
        local_shape.push_back(shape.dim_size(dim));
      }

      const auto& layout_or = Layout::FromString(layout_string);
      OP_REQUIRES(ctx, layout_or.ok(),
                  errors::InvalidArgument(absl::StrCat(
                      "Tensor at index : ", i,
                      " has invalid layout string : ", layout_string)));
      std::vector<int64_t> global_shape =
          layout_or->GlobalShapeFromLocalShape(local_shape);
      metadata.push_back(SavingTensorMetadata(i, std::move(global_shape),
                                              std::move(*layout_or)));
    }

    const auto& saving_specs_or = BuildSavingSpec(metadata);
    OP_REQUIRES(ctx, saving_specs_or.ok(),
                errors::Internal(absl::StrCat(
                    "failed to build saving specs for given shapes and "
                    "layouts. This should not happen. Message from stack : ",
                    saving_specs_or.status().error_message())));

    const absl::flat_hash_map<
        int64_t, absl::flat_hash_map<int64_t, std::vector<std::string>>>&
        saving_spec = *saving_specs_or;

    // Construct the mesh and builds per device save ops.
    // We don't need to build the real save ops here. Rather, we just query the
    // shards that would be generated in DTensor SPMD.
    std::vector<std::string> all_shard_prefixes;
    for (int device_id = 0; device_id < mesh.size(); ++device_id) {
      const auto& it = saving_spec.find(device_id);
      if (it == saving_spec.end()) continue;
      SaveOpSpecs specs =
          BuildPerDeviceSave(it->second, device_id, prefix, mesh.size());
      // Add all generated shards into a vector
      for (const std::string& new_prefix : specs.new_prefixes) {
        all_shard_prefixes.push_back(new_prefix);
      }
    }

    Tensor* out;
    auto out_vector_size = all_shard_prefixes.size();
    OP_REQUIRES_OK(
        ctx,
        ctx->allocate_output(
            0, TensorShape({static_cast<int64_t>(out_vector_size)}), &out));
    for (size_t i = 0; i < out_vector_size; ++i) {
      out->flat<tstring>()(i) = all_shard_prefixes[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("DTensorShardedPrefix").Device(DEVICE_CPU),
                        DTensorShardedPrefixOpKernel);

}  // namespace dtensor
}  // namespace tensorflow
