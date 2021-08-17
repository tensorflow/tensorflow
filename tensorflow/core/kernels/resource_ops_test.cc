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

#define EIGEN_USE_THREADS

#include <string>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class MockResource : public ResourceBase {
 public:
  MockResource(bool* alive, int payload) : alive_(alive), payload_(payload) {
    if (alive_ != nullptr) {
      *alive_ = true;
    }
  }
  ~MockResource() override {
    if (alive_ != nullptr) {
      *alive_ = false;
    }
  }
  string DebugString() const override { return ""; }
  bool* alive_;
  int payload_;
};

class MockHandleCreationOpKernel : public OpKernel {
 public:
  explicit MockHandleCreationOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    bool* alive = reinterpret_cast<bool*>(ctx->input(0).scalar<int64_t>()());
    int payload = ctx->input(1).scalar<int>()();
    AllocatorAttributes attr;
    Tensor handle_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                           &handle_tensor, attr));
    handle_tensor.scalar<ResourceHandle>()() =
        ResourceHandle::MakeRefCountingHandle(new MockResource(alive, payload),
                                              ctx->device()->name(), {},
                                              ctx->stack_trace());
    ctx->set_output(0, handle_tensor);
  }
};

REGISTER_OP("MockHandleCreationOp")
    .Input("alive: int64")
    .Input("payload: int32")
    .Output("output: resource");

REGISTER_KERNEL_BUILDER(Name("MockHandleCreationOp").Device(DEVICE_CPU),
                        MockHandleCreationOpKernel);

class MockHandleCreationOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(
        NodeDefBuilder("mock_handle_creation_op", "MockHandleCreationOp")
            .Input(FakeInput(DT_INT64))
            .Input(FakeInput(DT_INT32))
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(MockHandleCreationOpTest, RefCounting) {
  MakeOp();
  bool alive = false;
  int payload = -123;

  // Feed and run
  AddInputFromArray<int64_t>(TensorShape({}),
                             {reinterpret_cast<int64_t>(&alive)});
  AddInputFromArray<int32>(TensorShape({}), {payload});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_TRUE(alive);

  // Check the output.
  Tensor* output = GetOutput(0);
  ResourceHandle& output_handle = output->scalar<ResourceHandle>()();
  ResourceBase* base = output_handle.resource().get();
  EXPECT_TRUE(base);
  EXPECT_EQ(base->RefCount(), 1);
  MockResource* mock = output_handle.GetResource<MockResource>().ValueOrDie();
  EXPECT_TRUE(mock);
  EXPECT_EQ(mock->payload_, payload);
  EXPECT_EQ(base->RefCount(), 1);

  // context_->outputs_ holds the last ref to MockResource
  context_.reset();
  EXPECT_FALSE(alive);
  // For some reason if we don't call context_.reset(), it will trigger a
  // segfault (only in -c fastbuild) when it's called by ~OpsTestBase().
}

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename T>
class MockCopyOpKernel : public OpKernel {
 public:
  explicit MockCopyOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    AllocatorAttributes attr;
    Tensor output_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(input_tensor.dtype(), input_tensor.shape(),
                                &output_tensor, attr));
    // copy_functor will properly call copy constructors on the elements
    functor::DenseUpdate<CPUDevice, T, ASSIGN> copy_functor;
    copy_functor(ctx->eigen_device<CPUDevice>(), output_tensor.flat<T>(),
                 input_tensor.flat<T>());
    ctx->set_output(0, output_tensor);
  }
};

REGISTER_OP("MockCopyOp").Attr("T: type").Input("input: T").Output("output: T");

REGISTER_KERNEL_BUILDER(
    Name("MockCopyOp").Device(DEVICE_CPU).TypeConstraint<ResourceHandle>("T"),
    MockCopyOpKernel<ResourceHandle>);

class MockCopyOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("mock_copy_op", "MockCopyOp")
                     .Input(FakeInput(DT_RESOURCE))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

bool is_equal_handles(const ResourceHandle& a, const ResourceHandle& b) {
  return a.resource() == b.resource() && a.name() == b.name() &&
         a.maybe_type_name() == b.maybe_type_name() &&
         a.hash_code() == b.hash_code() && a.device() == b.device() &&
         a.container() == b.container();
}

TEST_F(MockCopyOpTest, RefCounting) {
  MakeOp();
  int payload = -123;

  // Feed and run
  AddInputFromArray<ResourceHandle>(
      TensorShape({}),
      {ResourceHandle::MakeRefCountingHandle(new MockResource(nullptr, payload),
                                             device_->name(), {}, {})});
  const Tensor* input = inputs_[0].tensor;
  EXPECT_EQ(input->scalar<ResourceHandle>()().resource()->RefCount(), 1);
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor* output = GetOutput(0);
  test::ExpectTensorEqual<ResourceHandle>(*output, *input, is_equal_handles);
  EXPECT_EQ(input->scalar<ResourceHandle>()().resource()->RefCount(), 2);
}

}  // namespace
}  // namespace tensorflow
