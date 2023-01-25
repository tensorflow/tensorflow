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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {

class FullTypeGraphUtilsTest : public ::testing::Test {
 protected:
  FullTypeGraphUtilsTest()
      : graph_(OpRegistry::Global()),
        root_(Scope::NewRootScope().ExitOnError()) {}

  Status MakeArg(Node **arg, DataType dtype) {
    return NodeBuilder("arg", "_Arg", &root_.graph()->flib_def())
        .Attr("T", dtype)
        .Attr("index", 0)
        .Finalize(root_.graph(), arg);
  }

  Status MakeRet(Node *src, Node **ret, DataType dtype) {
    return NodeBuilder("ret", "_Retval", &root_.graph()->flib_def())
        .Input(src, 0)
        .Attr("T", dtype)
        .Attr("index", 0)
        .Finalize(root_.graph(), ret);
  }

 public:
  Status MakeArgRet(Node **arg, Node **ret, DataType dtype) {
    TF_RETURN_IF_ERROR(MakeArg(arg, dtype));
    return MakeRet(*arg, ret, dtype);
  }

  void AddArgFullType(Node *arg, FullTypeId out_id, FullTypeId data_id) {
    FullTypeDef *t = arg->mutable_def()->mutable_experimental_type();
    t->set_type_id(TFT_PRODUCT);
    FullTypeDef data_t;
    data_t.set_type_id(data_id);
    FullTypeDef out_t;
    out_t.set_type_id(out_id);
    (*out_t.add_args()) = data_t;
    (*t->add_args()) = out_t;
  }

 private:
  Graph graph_;
  Scope root_;
};

TEST_F(FullTypeGraphUtilsTest, MemoryTypesArgNoFT) {
  gtl::InlinedVector<Node *, 4> nodes;
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
  gtl::InlinedVector<Node *, 4> nodes;
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
  gtl::InlinedVector<Node *, 4> nodes;
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
  gtl::InlinedVector<Node *, 4> nodes;
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
  gtl::InlinedVector<Node *, 4> nodes;
  DataTypeVector dtypes;
  MemoryTypeVector memory_types;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  AddArgFullType(arg, TFT_TENSOR, TFT_INT32);

  nodes.push_back(arg);
  dtypes.push_back(DT_INT32);
  Status status = full_type::SetMemoryTypeForArgs(nodes, dtypes, memory_types);
  EXPECT_FALSE(status.ok());
}

TEST_F(FullTypeGraphUtilsTest, WeakAllocAttrsArgIgnore) {
  gtl::InlinedVector<Node *, 4> nodes;
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
  gtl::InlinedVector<Node *, 4> nodes;
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
  gtl::InlinedVector<Node *, 4> nodes;
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
  gtl::InlinedVector<Node *, 4> nodes;
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
  gtl::InlinedVector<Node *, 4> nodes;
  DataTypeVector dtypes;
  MemoryTypeVector memory_types;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  nodes.push_back(ret);
  dtypes.push_back(DT_INT32);
  Status status = full_type::SetMemoryTypeForRets(nodes, dtypes, memory_types);
  EXPECT_FALSE(status.ok());
}

TEST_F(FullTypeGraphUtilsTest, WeakAllocAttrsRetIgnore) {
  gtl::InlinedVector<Node *, 4> nodes;
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
  gtl::InlinedVector<Node *, 4> nodes;
  DataTypeVector dtypes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  AddArgFullType(arg, TFT_TENSOR, TFT_INT32);  // numeric INT32

  nodes.push_back(arg);
  dtypes.push_back(DT_INT32);
  TF_ASSERT_OK(full_type::SingleDeviceSetAllocAttrsForArgs(
      nodes, dtypes, /*ints_on_device=*/true, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_FALSE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, WeakAllocatorAttrsArgWithFTSingleDevice) {
  gtl::InlinedVector<Node *, 4> nodes;
  DataTypeVector dtypes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  AddArgFullType(arg, TFT_SHAPE_TENSOR, TFT_INT32);

  nodes.push_back(arg);
  dtypes.push_back(DT_INT32);
  TF_ASSERT_OK(full_type::WeakSingleDeviceSetAllocAttrsForArgs(
      nodes, dtypes, /*ints_on_device=*/false, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_TRUE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, SingleDeviceAllocAttrsRetError) {
  gtl::InlinedVector<Node *, 4> nodes;
  DataTypeVector dtypes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  // test TFT_SHAPE_TENSOR and ints_on_device=true mismatch
  AddArgFullType(arg, TFT_SHAPE_TENSOR, TFT_INT32);
  nodes.push_back(ret);
  dtypes.push_back(DT_INT32);
  Status status = full_type::SingleDeviceSetAllocAttrsForRets(
      nodes, dtypes, /*ints_on_device=*/true, alloc_attrs);
  EXPECT_FALSE(status.ok());
}

TEST_F(FullTypeGraphUtilsTest, SingleDeviceAllocAttrsNotInt32) {
  gtl::InlinedVector<Node *, 4> nodes;
  DataTypeVector dtypes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_STRING));
  // If dtype is not DT_UINT32, then OK to not have full type information
  nodes.push_back(ret);
  dtypes.push_back(DT_STRING);
  TF_ASSERT_OK(full_type::SingleDeviceSetAllocAttrsForRets(
      nodes, dtypes, /*ints_on_device=*/false, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_TRUE(alloc_attrs[0].on_host());
}

TEST_F(FullTypeGraphUtilsTest, SingleDeviceWeakAllocAttrsRetIgnore) {
  gtl::InlinedVector<Node *, 4> nodes;
  DataTypeVector dtypes;
  std::vector<AllocatorAttributes> alloc_attrs;

  Node *arg, *ret;
  TF_ASSERT_OK(MakeArgRet(&arg, &ret, DT_INT32));
  nodes.push_back(ret);
  dtypes.push_back(DT_INT32);
  TF_ASSERT_OK(full_type::WeakSingleDeviceSetAllocAttrsForRets(
      nodes, dtypes, /*ints_on_device=*/true, alloc_attrs));
  ASSERT_EQ(alloc_attrs.size(), 1);
  ASSERT_FALSE(alloc_attrs[0].on_host());
}

}  // namespace tensorflow
