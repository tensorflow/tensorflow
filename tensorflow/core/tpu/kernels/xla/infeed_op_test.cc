/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

// `:xla_ops` registers an XLA kernel for `GetItem`, whose op definition is not
// part of open-source TensorFlow. RegisterCompilationKernels() requires an op
// definition for every registered XLA kernel, so register a minimal one here.
REGISTER_OP("GetItem")
    .Input("data: T")
    .Input("index: int32")
    .Output("output: T")
    .Attr("T: type");

absl::Status CreateInfeedDequeueTupleKernel(
    const std::vector<DataType>& dtypes,
    const std::vector<TensorShape>& shapes) {
  XlaOpRegistry::RegisterCompilationKernels();
  NodeDef node_def;
  TF_RETURN_IF_ERROR(NodeDefBuilder("infeed", "InfeedDequeueTuple")
                         .Attr("dtypes", dtypes)
                         .Attr("shapes", shapes)
                         .Finalize(&node_def));
  DeviceBase device(Env::Default());
  absl::Status status;
  std::unique_ptr<OpKernel> kernel =
      CreateOpKernel(DeviceType(DEVICE_CPU_XLA_JIT), &device, cpu_allocator(),
                     node_def, TF_GRAPH_DEF_VERSION, &status);
  return status;
}

TEST(InfeedDequeueTupleOpTest, MatchingShapesAndDtypes) {
  TF_EXPECT_OK(CreateInfeedDequeueTupleKernel(
      {DT_INT32, DT_FLOAT}, {TensorShape({1}), TensorShape({2, 3})}));
}

TEST(InfeedDequeueTupleOpTest, MoreShapesThanDtypes) {
  absl::Status status = CreateInfeedDequeueTupleKernel(
      {DT_INT32}, {TensorShape({1}), TensorShape({1})});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "shapes and dtypes must be the same length. shapes: 2, dtypes: 1"))
      << status;
}

TEST(InfeedDequeueTupleOpTest, FewerShapesThanDtypes) {
  absl::Status status =
      CreateInfeedDequeueTupleKernel({DT_INT32, DT_INT32}, {TensorShape({1})});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "shapes and dtypes must be the same length. shapes: 1, dtypes: 2"))
      << status;
}

}  // namespace
}  // namespace tensorflow
