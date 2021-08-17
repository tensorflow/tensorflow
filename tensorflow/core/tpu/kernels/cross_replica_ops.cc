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
#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// Convert 1D group_assignment into 2D replica_groups.
std::vector<xla::ReplicaGroup> Convert(
    const std::vector<int64_t>& group_assignment,
    const TensorShape& group_assignment_shape) {
  VLOG(1) << "group_assignment size: " << group_assignment.size();
  VLOG(1) << "group_assignment_shape: " << group_assignment_shape.DebugString();

  std::vector<xla::ReplicaGroup> replica_groups;
  const int64_t num_groups = group_assignment_shape.dim_size(0);
  const int64_t num_replica_per_group = group_assignment_shape.dim_size(1);

  replica_groups.reserve(num_groups);
  for (int64_t g = 0; g < num_groups; ++g) {
    xla::ReplicaGroup replica_group;

    for (int64_t i = 0; i < num_replica_per_group; ++i) {
      int64_t replica = group_assignment[num_replica_per_group * g + i];
      replica_group.add_replica_ids(replica);
    }
    replica_groups.push_back(replica_group);
  }
  return replica_groups;
}

class CrossReplicaSumOp : public XlaOpKernel {
 public:
  explicit CrossReplicaSumOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<int64_t> flattened_group_assignment;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputReshapedToIntVector(
                            1, &flattened_group_assignment));
    std::vector<xla::ReplicaGroup> replica_groups =
        Convert(flattened_group_assignment, ctx->InputShape(1));
    ctx->SetOutput(0, xla::CrossReplicaSum(ctx->Input(0), replica_groups));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CrossReplicaSumOp);
};

class AllToAllOp : public XlaOpKernel {
 public:
  explicit AllToAllOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("split_dimension", &split_dimension_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("concat_dimension", &concat_dimension_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("split_count", &split_count_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<int64_t> flattened_group_assignment;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputReshapedToIntVector(
                            1, &flattened_group_assignment));

    std::vector<xla::ReplicaGroup> replica_groups =
        Convert(flattened_group_assignment, ctx->InputShape(1));
    ctx->SetOutput(
        0, xla::AllToAll(ctx->Input(0), split_dimension_, concat_dimension_,
                         split_count_, replica_groups));
  }

 private:
  int64_t split_dimension_;
  int64_t concat_dimension_;
  int64_t split_count_;

  TF_DISALLOW_COPY_AND_ASSIGN(AllToAllOp);
};

class CollectivePermuteOp : public XlaOpKernel {
 public:
  explicit CollectivePermuteOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape source_target_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx,
        source_target_shape.dims() == 2 && source_target_shape.dim_size(1) == 2,
        errors::InvalidArgument(
            "CollectivePermuteOp requires source_target_pair's shape to"
            " [num_pairs, 2]. Get ",
            source_target_shape));

    xla::Literal source_target_literal;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsInt64Literal(1, &source_target_literal));
    const int num_pairs = source_target_shape.dim_size(0);
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs(num_pairs);
    for (int i = 0; i < num_pairs; ++i) {
      source_target_pairs[i] = {source_target_literal.Get<int64_t>({i, 0}),
                                source_target_literal.Get<int64_t>({i, 1})};
    }
    ctx->SetOutput(0,
                   xla::CollectivePermute(ctx->Input(0), source_target_pairs));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectivePermuteOp);
};

REGISTER_XLA_OP(Name("AllToAll").CompileTimeConstantInput("group_assignment"),
                AllToAllOp);
REGISTER_XLA_OP(Name("CollectivePermute")
                    .TypeConstraint("T", {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16,
                                          DT_INT32, DT_COMPLEX64})
                    .CompileTimeConstantInput("source_target_pairs"),
                CollectivePermuteOp);
REGISTER_XLA_OP(
    Name("CrossReplicaSum").CompileTimeConstantInput("group_assignment"),
    CrossReplicaSumOp);

}  // anonymous namespace
}  // namespace tensorflow
