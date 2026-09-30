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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

class NoMemoryAllocator : public Allocator {
 public:
  std::string Name() override { return "NoMemoryAllocator"; }
  void* AllocateRaw(size_t /*alignment*/, size_t num_bytes) override {
    requested_bytes_ = num_bytes;
    return nullptr;
  }
  void DeallocateRaw(void* /*ptr*/) override {}
  size_t requested_bytes() const { return requested_bytes_; }

 private:
  size_t requested_bytes_ = 0;
};

class AllocationFailureDevice : public DeviceBase {
 public:
  explicit AllocationFailureDevice(Allocator* allocator)
      : DeviceBase(Env::Default()), allocator_(allocator) {}
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override {
    return allocator_;
  }
  const std::string& name() const override { return name_; }

 private:
  Allocator* allocator_;
  const std::string name_ = "/device:CPU:0";
};

TEST(AddManySparseToTensorsMapTest, CheckedHandleAllocation) {
  // Exercise allocation failure without requesting a huge buffer.
  for (int64_t batch_size : {0, 3}) {
    SCOPED_TRACE(batch_size);
    NoMemoryAllocator allocator;
    AllocationFailureDevice device(&allocator);
    ResourceMgr resources;
    NodeDef node;
    ASSERT_TRUE(NodeDefBuilder("add_many", "AddManySparseToTensorsMap")
                    .Input("indices", 0, DT_INT64)
                    .Input("values", 0, DT_FLOAT)
                    .Input("shape", 0, DT_INT64)
                    .Finalize(&node)
                    .ok());
    absl::Status status;
    std::unique_ptr<OpKernel> kernel(CreateOpKernel(
        DEVICE_CPU, &device, cpu_allocator(), node, TF_GRAPH_DEF_VERSION,
        &status));
    ASSERT_TRUE(status.ok()) << status;

    Tensor indices(DT_INT64, TensorShape({0, 2}));
    Tensor values(DT_FLOAT, TensorShape({0}));
    Tensor shape(DT_INT64, TensorShape({2}));
    shape.vec<int64_t>()(0) = batch_size;
    shape.vec<int64_t>()(1) = 5;
    std::vector<TensorValue> inputs{TensorValue(&indices), TensorValue(&values),
                                   TensorValue(&shape)};
    AllocatorAttributes output_attributes;
    OpKernelContext::Params params;
    params.device = &device;
    params.op_kernel = kernel.get();
    params.resource_manager = &resources;
    params.inputs = inputs;
    params.output_attr_array = &output_attributes;
    OpKernelContext context(&params);
    kernel->Compute(&context);

    if (batch_size == 0) {
      ASSERT_TRUE(context.status().ok()) << context.status();
      ASSERT_NE(context.mutable_output(0), nullptr);
      EXPECT_EQ(context.mutable_output(0)->shape(), TensorShape({0}));
      EXPECT_EQ(allocator.requested_bytes(), 0);
    } else {
      EXPECT_TRUE(absl::IsResourceExhausted(context.status())) << context.status();
      EXPECT_EQ(context.mutable_output(0), nullptr);
      EXPECT_EQ(allocator.requested_bytes(), batch_size * sizeof(int64_t));
    }
  }
}

}  // namespace
}  // namespace tensorflow
