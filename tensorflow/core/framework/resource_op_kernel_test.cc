/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/resource_op_kernel.h"

#include <memory>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

// Stub DeviceBase subclass which only returns allocators.
class StubDevice : public DeviceBase {
 public:
  StubDevice() : DeviceBase(nullptr) {}

  Allocator* GetAllocator(AllocatorAttributes) override {
    return cpu_allocator();
  }
};

// Stub resource for testing resource op kernel.
class StubResource : public ResourceBase {
 public:
  string DebugString() const override { return ""; }
  int code;
};

class StubResourceOpKernel : public ResourceOpKernel<StubResource> {
 public:
  using ResourceOpKernel::ResourceOpKernel;

  StubResource* resource() TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    return resource_;
  }

 private:
  Status CreateResource(StubResource** resource) override {
    *resource = CHECK_NOTNULL(new StubResource);
    return GetNodeAttr(def(), "code", &(*resource)->code);
  }

  Status VerifyResource(StubResource* resource) override {
    int code;
    TF_RETURN_IF_ERROR(GetNodeAttr(def(), "code", &code));
    if (code != resource->code) {
      return errors::InvalidArgument("stub has code ", resource->code,
                                     " but requested code ", code);
    }
    return Status::OK();
  }
};

REGISTER_OP("StubResourceOp")
    .Attr("code: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("output: Ref(string)");

REGISTER_KERNEL_BUILDER(Name("StubResourceOp").Device(DEVICE_CPU),
                        StubResourceOpKernel);

class ResourceOpKernelTest : public ::testing::Test {
 protected:
  std::unique_ptr<StubResourceOpKernel> CreateOp(int code,
                                                 const string& shared_name) {
    static std::atomic<int64_t> count(0);
    NodeDef node_def;
    TF_CHECK_OK(NodeDefBuilder(strings::StrCat("test-node", count.fetch_add(1)),
                               "StubResourceOp")
                    .Attr("code", code)
                    .Attr("shared_name", shared_name)
                    .Finalize(&node_def));
    Status status;
    std::unique_ptr<OpKernel> op(CreateOpKernel(
        DEVICE_CPU, &device_, device_.GetAllocator(AllocatorAttributes()),
        node_def, TF_GRAPH_DEF_VERSION, &status));
    TF_EXPECT_OK(status) << status;
    EXPECT_TRUE(op != nullptr);

    // Downcast to StubResourceOpKernel to call resource() later.
    std::unique_ptr<StubResourceOpKernel> resource_op(
        dynamic_cast<StubResourceOpKernel*>(op.get()));
    EXPECT_TRUE(resource_op != nullptr);
    if (resource_op != nullptr) {
      op.release();
    }
    return resource_op;
  }

  Status RunOpKernel(OpKernel* op) {
    OpKernelContext::Params params;

    params.device = &device_;
    params.resource_manager = &mgr_;
    params.op_kernel = op;

    OpKernelContext context(&params);
    op->Compute(&context);
    return context.status();
  }

  StubDevice device_;
  ResourceMgr mgr_;
};

TEST_F(ResourceOpKernelTest, PrivateResource) {
  // Empty shared_name means private resource.
  const int code = -100;
  auto op = CreateOp(code, "");
  ASSERT_TRUE(op != nullptr);
  TF_EXPECT_OK(RunOpKernel(op.get()));

  // Default non-shared name provided from ContainerInfo.
  // TODO(gonnet): This test is brittle since it assumes that the
  // ResourceManager is untouched and thus the private resource name starts
  // with "_0_".
  const string key = "_0_" + op->name();

  StubResource* resource;
  TF_ASSERT_OK(
      mgr_.Lookup<StubResource>(mgr_.default_container(), key, &resource));
  EXPECT_EQ(op->resource(), resource);  // Check resource identity.
  EXPECT_EQ(code, resource->code);      // Check resource stored information.
  resource->Unref();

  // Destroy the op kernel. Expect the resource to be released.
  op = nullptr;
  Status s =
      mgr_.Lookup<StubResource>(mgr_.default_container(), key, &resource);

  EXPECT_FALSE(s.ok());
}

TEST_F(ResourceOpKernelTest, SharedResource) {
  const string shared_name = "shared_stub";
  const int code = -201;
  auto op = CreateOp(code, shared_name);
  ASSERT_TRUE(op != nullptr);
  TF_EXPECT_OK(RunOpKernel(op.get()));

  StubResource* resource;
  TF_ASSERT_OK(mgr_.Lookup<StubResource>(mgr_.default_container(), shared_name,
                                         &resource));
  EXPECT_EQ(op->resource(), resource);  // Check resource identity.
  EXPECT_EQ(code, resource->code);      // Check resource stored information.
  resource->Unref();

  // Destroy the op kernel. Expect the resource not to be released.
  op = nullptr;
  TF_ASSERT_OK(mgr_.Lookup<StubResource>(mgr_.default_container(), shared_name,
                                         &resource));
  resource->Unref();
}

TEST_F(ResourceOpKernelTest, LookupShared) {
  auto op1 = CreateOp(-333, "shared_stub");
  auto op2 = CreateOp(-333, "shared_stub");
  ASSERT_TRUE(op1 != nullptr);
  ASSERT_TRUE(op2 != nullptr);

  TF_EXPECT_OK(RunOpKernel(op1.get()));
  TF_EXPECT_OK(RunOpKernel(op2.get()));
  EXPECT_EQ(op1->resource(), op2->resource());
}

TEST_F(ResourceOpKernelTest, VerifyResource) {
  auto op1 = CreateOp(-444, "shared_stub");
  auto op2 = CreateOp(0, "shared_stub");  // Different resource code.
  ASSERT_TRUE(op1 != nullptr);
  ASSERT_TRUE(op2 != nullptr);

  TF_EXPECT_OK(RunOpKernel(op1.get()));
  EXPECT_FALSE(RunOpKernel(op2.get()).ok());
  EXPECT_TRUE(op1->resource() != nullptr);
  EXPECT_TRUE(op2->resource() == nullptr);
}

}  // namespace
}  // namespace tensorflow
