/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/arg_ret_placement.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/cc/framework/scope.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace tensorflow {

class FullTypeGraphUtilsTest : public ::testing::Test {
 protected:
  FullTypeGraphUtilsTest()
      : graph_(OpRegistry::Global()),
        root_(Scope::NewRootScope().ExitOnError()) {}

  absl::Status MakeArg(Node **arg, DataType dtype) {
    return NodeBuilder("arg", "_Arg", &root_.graph()->flib_def())
        .Attr("T", dtype)
        .Attr("index", 0)
        .Finalize(root_.graph(), arg);
  }

  absl::Status MakeRet(Node *src, Node **ret, DataType dtype) {
    return NodeBuilder("ret", "_Retval", &root_.graph()->flib_def())
        .Input(src, 0)
        .Attr("T", dtype)
        .Attr("index", 0)
        .Finalize(root_.graph(), ret);
  }

 public:
  absl::Status MakeArgRet(Node **arg, Node **ret, DataType dtype) {
    TF_RETURN_IF_ERROR(MakeArg(arg, dtype));
    return MakeRet(*arg, ret, dtype);
  }

  void AddArgFullType(Node *arg, FullTypeId out_id, FullTypeId data_id) {
    FullTypeDef *t = arg->mutable_def()->mutable_experimental_type();
    t->set_type_id(TFT_PRODUCT);
    FullTypeDef out_t;
    out_t.set_type_id(out_id);
    if (data_id != TFT_UNSET) {
      FullTypeDef data_t;
      data_t.set_type_id(data_id);
      (*out_t.add_args()) = data_t;
    }
    (*t->add_args()) = out_t;
  }

 private:
  Graph graph_;
  Scope root_;
};

TEST_F(FullTypeGraphUtilsTest, MemoryTypesArgNoFT) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  MemoryTypeVector memory_types;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT64));
  nodes.push_back(arg);
  dtypes.push_back(DT_INT64);
  TF_ASSERT_OK(
      full_type::WeakSetMemoryTypeForArgs(nodes, dtypes, memory_types));
  ASSERT_EQ(memory_types.size(), 1);
  ASSERT_EQ(memory_types[0], MemoryType::DEVICE_MEMORY);
}

TEST_F(FullTypeGraphUtilsTest, AllocatorAttrsArgNoFT) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT64));
  nodes.push_back(arg);
  dtypes.push_back(DT_INT64);
  TF_ASSERT_OK(full_type::WeakSetAllocAttrsForArgs(nodes, dtypes, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_FALSE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, MemoryTypesArgWithFT) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  MemoryTypeVector memory_types;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  AddArgFullType(arg, TFT_SHAPE_TENSOR, TFT_INT32);

  nodes.push_back(arg);
  dtypes.push_back(DT_INT32);
  TF_ASSERT_OK(full_type::SetMemoryTypeForArgs(nodes, dtypes, memory_types));
  ASSERT_EQ(memory_types.size(), 1);
  ASSERT_EQ(memory_types[0], MemoryType::HOST_MEMORY);
}

TEST_F(FullTypeGraphUtilsTest, AllocatorAttrsArgWithFT) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  AddArgFullType(arg, TFT_SHAPE_TENSOR, TFT_INT32);

  nodes.push_back(arg);
  dtypes.push_back(DT_INT32);
  TF_ASSERT_OK(full_type::SetAllocAttrsForArgs(nodes, dtypes, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_TRUE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, ArgError) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  MemoryTypeVector memory_types;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  AddArgFullType(arg, TFT_TENSOR, TFT_INT32);

  nodes.push_back(arg);
  dtypes.push_back(DT_INT32);
  absl::Status status =
      full_type::SetMemoryTypeForArgs(nodes, dtypes, memory_types);
  EXPECT_FALSE(status.ok());
}

TEST_F(FullTypeGraphUtilsTest, WeakAllocAttrsArgIgnore) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  AddArgFullType(arg, TFT_TENSOR, TFT_INT32);

  nodes.push_back(arg);
  dtypes.push_back(DT_INT32);
  TF_ASSERT_OK(full_type::WeakSetAllocAttrsForArgs(nodes, dtypes, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_TRUE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, RetNoFT) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  MemoryTypeVector memory_types;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT64));
  nodes.push_back(ret);
  dtypes.push_back(DT_INT64);
  TF_ASSERT_OK(
      full_type::WeakSetMemoryTypeForRets(nodes, dtypes, memory_types));
  ASSERT_EQ(memory_types.size(), 1);
  ASSERT_EQ(memory_types[0], MemoryType::DEVICE_MEMORY);
}

TEST_F(FullTypeGraphUtilsTest, MemoryTypeRetWithFT) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  MemoryTypeVector memory_types;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  // `ret` does not have an output, so it has no useful full type information.
  // Add full type information to the input to `ret`, which is `arg`.
  AddArgFullType(arg, TFT_SHAPE_TENSOR, TFT_INT32);
  nodes.push_back(ret);
  dtypes.push_back(DT_INT32);
  TF_ASSERT_OK(full_type::SetMemoryTypeForRets(nodes, dtypes, memory_types));
  ASSERT_EQ(memory_types.size(), 1);
  ASSERT_EQ(memory_types[0], MemoryType::HOST_MEMORY);
}

TEST_F(FullTypeGraphUtilsTest, AllowAttrRetWithFT) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  // `ret` does not have an output, so it has no useful full type information.
  // Add full type information to the input to `ret`, which is `arg`.
  AddArgFullType(arg, TFT_SHAPE_TENSOR, TFT_INT32);
  nodes.push_back(ret);
  dtypes.push_back(DT_INT32);
  TF_ASSERT_OK(full_type::SetAllocAttrsForRets(nodes, dtypes, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_TRUE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, RetError) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  MemoryTypeVector memory_types;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  nodes.push_back(ret);
  dtypes.push_back(DT_INT32);
  absl::Status status =
      full_type::SetMemoryTypeForRets(nodes, dtypes, memory_types);
  EXPECT_FALSE(status.ok());
}

TEST_F(FullTypeGraphUtilsTest, WeakAllocAttrsRetIgnore) {
  absl::InlinedVector<Node *, 4UL> nodes;
  DataTypeVector dtypes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  nodes.push_back(ret);
  dtypes.push_back(DT_INT32);
  TF_ASSERT_OK(full_type::WeakSetAllocAttrsForRets(nodes, dtypes, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_TRUE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, AllocatorAttrsArgWithFTSingleDevice) {
  std::vector<std::pair<Node *, FunctionArgIndex>> arg_nodes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  AddArgFullType(arg, TFT_TENSOR, TFT_INT32);  // numeric INT32

  arg_nodes.push_back(std::make_pair(arg, FunctionArgIndex(0, 0)));
  TF_ASSERT_OK(full_type::SingleDeviceSetAllocAttrsForArgs(
      arg_nodes, /*ints_on_device=*/true, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_FALSE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, AllocatorAttrsArgWithUnsetFTSingleDevice) {
  std::vector<std::pair<Node *, FunctionArgIndex>> arg_nodes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  AddArgFullType(arg, TFT_UNSET, TFT_UNSET);  // numeric INT32

  arg_nodes.push_back(std::make_pair(arg, FunctionArgIndex(0, 0)));
  TF_ASSERT_OK(full_type::SingleDeviceSetAllocAttrsForArgs(
      arg_nodes, /*ints_on_device=*/true, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_FALSE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, WeakAllocatorAttrsArgWithFTSingleDevice) {
  std::vector<std::pair<Node *, FunctionArgIndex>> arg_nodes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  AddArgFullType(arg, TFT_SHAPE_TENSOR, TFT_INT32);

  arg_nodes.push_back(std::make_pair(arg, FunctionArgIndex(0, 0)));
  TF_ASSERT_OK(full_type::WeakSingleDeviceSetAllocAttrsForArgs(
      arg_nodes, /*ints_on_device=*/false, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_TRUE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, SingleDeviceAllocAttrsRetError) {
  std::vector<std::pair<Node *, int>> ret_nodes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  // test TFT_SHAPE_TENSOR and ints_on_device=true mismatch
  AddArgFullType(arg, TFT_SHAPE_TENSOR, TFT_INT32);
  ret_nodes.push_back(std::make_pair(ret, 0));
  absl::Status status = full_type::SingleDeviceSetAllocAttrsForRets(
      ret_nodes, /*ints_on_device=*/true, alloc_attrs);
  EXPECT_FALSE(status.ok());
}

TEST_F(FullTypeGraphUtilsTest, SingleDeviceAllocAttrsNotInt32) {
  std::vector<std::pair<Node *, int>> ret_nodes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_STRING));
  // If dtype is not DT_UINT32, then OK to not have full type information
  ret_nodes.push_back(std::make_pair(ret, 0));
  TF_ASSERT_OK(full_type::SingleDeviceSetAllocAttrsForRets(
      ret_nodes, /*ints_on_device=*/false, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_TRUE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, SingleDeviceWeakAllocAttrsRetIgnore) {
  std::vector<std::pair<Node *, int>> ret_nodes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  ret_nodes.push_back(std::make_pair(ret, 0));
  TF_ASSERT_OK(full_type::WeakSingleDeviceSetAllocAttrsForRets(
      ret_nodes, /*ints_on_device=*/true, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_FALSE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, CheckMemoryTypeOK) {
  Node *node;
  TF_ASSERT_OK(MakeArg(&node, DT_INT32));
  AddArgFullType(node, TFT_SHAPE_TENSOR, TFT_INT32);
  const FullTypeDef &ft = node->def().experimental_type().args()[0];
  TF_ASSERT_OK(full_type::CheckMemoryType(true, ft));
}

TEST_F(FullTypeGraphUtilsTest, CheckMemoryTypeBadFT) {
  Node *node;
  TF_ASSERT_OK(MakeArg(&node, DT_INT32));
  AddArgFullType(node, TFT_SHAPE_TENSOR, TFT_INT32);
  // full type information for the whole node, not for one tensor / one output
  const FullTypeDef &ft = node->def().experimental_type();
  absl::Status status = full_type::CheckMemoryType(true, ft);
  EXPECT_FALSE(status.ok());
}

TEST_F(FullTypeGraphUtilsTest, CheckMemoryTypeWrongFT) {
  Node *node;
  TF_ASSERT_OK(MakeArg(&node, DT_INT32));
  AddArgFullType(node, TFT_SHAPE_TENSOR, TFT_INT32);
  const FullTypeDef &ft = node->def().experimental_type().args()[0];
  // use_host_memory=false does not match TFT_SHAPE_TENSOR
  absl::Status status = full_type::CheckMemoryType(false, ft);
  EXPECT_FALSE(status.ok());
}

TEST_F(FullTypeGraphUtilsTest, LogMemoryTypeMismatchOK) {
  Node *node;
  TF_ASSERT_OK(MakeArg(&node, DT_INT32));
  AddArgFullType(node, TFT_SHAPE_TENSOR, TFT_INT32);
  const FullTypeDef &ft = node->def().experimental_type().args()[0];
  EXPECT_TRUE(full_type::LogMemoryTypeMismatch(true, ft));
}

TEST_F(FullTypeGraphUtilsTest, LogMemoryTypeMismatchBadFT) {
  Node *node;
  TF_ASSERT_OK(MakeArg(&node, DT_INT32));
  AddArgFullType(node, TFT_SHAPE_TENSOR, TFT_INT32);
  // full type information for the whole node, not for one tensor / one output
  const FullTypeDef &ft = node->def().experimental_type();
  EXPECT_FALSE(full_type::LogMemoryTypeMismatch(true, ft));
}

TEST_F(FullTypeGraphUtilsTest, LogMemoryTypeMismatchWrongFT) {
  Node *node;
  TF_ASSERT_OK(MakeArg(&node, DT_INT32));
  AddArgFullType(node, TFT_SHAPE_TENSOR, TFT_INT32);
  const FullTypeDef &ft = node->def().experimental_type().args()[0];
  // use_host_memory=false does not match TFT_SHAPE_TENSOR
  EXPECT_FALSE(full_type::LogMemoryTypeMismatch(false, ft));
}

}  // namespace tensorflow
