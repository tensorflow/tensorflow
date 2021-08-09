/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h"

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/padding.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace tensorflow {
namespace {

std::unique_ptr<tfrt::HostContext> CreateTestHostContext(int num_threads) {
  return std::make_unique<tfrt::HostContext>(
      [](const tfrt::DecodedDiagnostic&) {}, tfrt::CreateMallocAllocator(),
      tfrt::CreateSingleThreadedWorkQueue());
}

TEST(TFRTOpKernelTest, TestGetBoolAttr) {
  tfrt::OpAttrs attrs;
  attrs.Set<bool>("foo", true);
  attrs.Set<bool>("bar", false);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  bool value;
  TF_ASSERT_OK(ctx.GetAttr("foo", &value));
  ASSERT_TRUE(value);
  TF_ASSERT_OK(ctx.GetAttr("bar", &value));
  ASSERT_FALSE(value);
}

TEST(TFRTOpKernelTest, TestGetIntAttr) {
  tfrt::OpAttrs attrs;
  attrs.Set<int32>("foo", -2);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  int32_t value;
  TF_ASSERT_OK(ctx.GetAttr("foo", &value));
  ASSERT_EQ(value, -2);
}

TEST(TFRTOpKernelTest, TestGetIntListAttr) {
  tfrt::OpAttrs attrs;
  attrs.SetArray<int32>("foo", {});
  attrs.SetArray<int32>("bar", {1});
  attrs.SetArray<int32>("baz", {1, 2, 3});
  attrs.SetString("bar", "test");
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  std::vector<int32> v1, v2, v3;
  std::vector<int32> expected_v1;
  std::vector<int32> expected_v2 = {1};
  std::vector<int32> expected_v3 = {1, 2, 3};
  TF_ASSERT_OK(ctx.GetAttr("foo", &v1));
  ASSERT_EQ(v1, expected_v1);
  TF_ASSERT_OK(ctx.GetAttr("bar", &v2));
  ASSERT_EQ(v2, expected_v2);
  TF_ASSERT_OK(ctx.GetAttr("baz", &v3));
  ASSERT_EQ(v3, expected_v3);
}

TEST(TFRTOpKernelTest, TestGetStrAttr) {
  tfrt::OpAttrs attrs;
  attrs.SetString("foo", "");
  attrs.SetString("bar", "test");
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  std::string value;
  TF_ASSERT_OK(ctx.GetAttr("foo", &value));
  ASSERT_EQ(value, "");
  TF_ASSERT_OK(ctx.GetAttr("bar", &value));
  ASSERT_EQ(value, "test");
}

TEST(TFRTOpKernelTest, TestGetPaddingAttr) {
  tfrt::OpAttrs attrs;
  attrs.SetString("foo", "VALID");
  attrs.SetString("bar", "SAME");
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  Padding value;
  TF_ASSERT_OK(ctx.GetAttr("foo", &value));
  ASSERT_EQ(value, Padding::VALID);
  TF_ASSERT_OK(ctx.GetAttr("bar", &value));
  ASSERT_EQ(value, Padding::SAME);
}

TEST(TFRTOpKernelTest, TestMissingAttr) {
  tfrt::OpAttrs attrs;
  attrs.Set<bool>("foo", true);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);

  bool value;
  auto status = ctx.GetAttr("bar", &value);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
}

class TestKernel : public TFRTOpKernel {
 public:
  explicit TestKernel(TFRTOpKernelConstruction* construction)
      : TFRTOpKernel(construction) {}

  void Compute(TFRTOpKernelContext* context) override {}
};

TEST(TFRTOpKernelTest, TestKernelMatchesTypeConstraints) {
  tfrt::OpAttrs attrs;
  attrs.Set<tfrt::OpAttrType>("foo", tfrt::OpAttrType::F32);
  attrs.Set<tfrt::OpAttrType>("bar", tfrt::OpAttrType::I32);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);
  TFRTOpKernelReg reg([](TFRTOpKernelConstruction* construction)
                          -> std::unique_ptr<TFRTOpKernel> {
    return std::make_unique<TestKernel>(construction);
  });
  reg.type_constraints["foo"] = DT_FLOAT;
  reg.type_constraints["bar"] = DT_INT32;
  ::tensorflow::tfrt_forwarding_kernel_factories->RegisterFactory(
      "TestKernelFloatInt", reg);
  std::unique_ptr<TFRTOpKernel> op =
      tfrt_forwarding_kernel_factories->CreateKernel("TestKernelFloatInt",
                                                     &ctx);
  ASSERT_NE(op.get(), nullptr);
}

TEST(TFRTOpKernelTest, TestSecondKernelMatchesTypeConstraints) {
  tfrt::OpAttrs attrs;
  attrs.Set<tfrt::OpAttrType>("foo", tfrt::OpAttrType::I32);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);
  TFRTOpKernelReg reg1([](TFRTOpKernelConstruction* construction)
                           -> std::unique_ptr<TFRTOpKernel> {
    return std::make_unique<TestKernel>(construction);
  });
  TFRTOpKernelReg reg2([](TFRTOpKernelConstruction* construction)
                           -> std::unique_ptr<TFRTOpKernel> {
    return std::make_unique<TestKernel>(construction);
  });
  reg1.type_constraints["foo"] = DT_FLOAT;
  reg2.type_constraints["foo"] = DT_INT32;
  ::tensorflow::tfrt_forwarding_kernel_factories->RegisterFactory(
      "TestKernel2ndConstraint", reg1);
  ::tensorflow::tfrt_forwarding_kernel_factories->RegisterFactory(
      "TestKernel2ndConstraint", reg2);

  std::unique_ptr<TFRTOpKernel> op =
      tfrt_forwarding_kernel_factories->CreateKernel("TestKernel2ndConstraint",
                                                     &ctx);
  ASSERT_NE(op.get(), nullptr);
}

TEST(TFRTOpKernelTest, TestKernelDoesNotMatchTypeConstraints) {
  tfrt::OpAttrs attrs;
  attrs.Set<tfrt::OpAttrType>("foo", tfrt::OpAttrType::I32);
  attrs.Set<tfrt::OpAttrType>("bar", tfrt::OpAttrType::I32);
  tfrt::OpAttrsRef attrsref(attrs);

  TFRTOpKernelConstruction ctx(attrsref);
  TFRTOpKernelReg reg([](TFRTOpKernelConstruction* construction)
                          -> std::unique_ptr<TFRTOpKernel> {
    return std::make_unique<TestKernel>(construction);
  });
  reg.type_constraints["foo"] = DT_FLOAT;
  reg.type_constraints["bar"] = DT_INT32;
  ::tensorflow::tfrt_forwarding_kernel_factories->RegisterFactory(
      "TestKernelIntInt", reg);
  std::unique_ptr<TFRTOpKernel> op =
      tfrt_forwarding_kernel_factories->CreateKernel("TestKernelIntInt", &ctx);
  ASSERT_EQ(op.get(), nullptr);
}

TEST(TFRTOpKernelTest, TestAllocateTemp) {
  auto host_context = CreateTestHostContext(1);
  int num_outputs = 1;
  llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> inputs;
  TFRTOpMeta op_meta({DT_INT32});
  TFRTOpKernelContext ctx(inputs, num_outputs, &op_meta, host_context.get());

  Tensor out;
  ASSERT_EQ(out.AllocatedBytes(), 0);
  TF_EXPECT_OK(ctx.allocate_temp(DT_INT32, {}, &out));
  ASSERT_GT(out.AllocatedBytes(), 0);
  out.scalar<int32>()() = 123;
  ASSERT_EQ(out.dtype(), DT_INT32);
  ASSERT_EQ(out.shape().dims(), 0);
}

}  // namespace
}  // namespace tensorflow
