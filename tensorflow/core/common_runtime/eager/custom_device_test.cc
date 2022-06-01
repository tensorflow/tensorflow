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

#include "tensorflow/core/common_runtime/eager/custom_device.h"

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/placement_utils.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace eager {
namespace {

using ::testing::ContainsRegex;
using ::testing::HasSubstr;

class TestCustomDevice : public CustomDevice {
 public:
  explicit TestCustomDevice(std::string name) : name_(name) {}
  const std::string& name() override { return name_; }
  Status CopyTensorToDevice(ImmediateExecutionTensorHandle* tensor,
                            ImmediateExecutionTensorHandle** result) override {
    tensor->Ref();
    *result = tensor;
    return OkStatus();
  }
  Status CopyTensorFromDevice(
      ImmediateExecutionTensorHandle* tensor,
      const std::string& target_device_name,
      ImmediateExecutionTensorHandle** result) override {
    tensor->Ref();
    *result = tensor;
    return OkStatus();
  }
  Status Execute(const ImmediateExecutionOperation* op,
                 ImmediateExecutionTensorHandle** retvals,
                 int* num_retvals) override {
    return errors::Unimplemented("Not implemented");
  }

  Status Pack(absl::Span<ImmediateExecutionTensorHandle*> handles,
              ImmediateExecutionTensorHandle** result) override {
    return errors::Unimplemented("Packing is not implemented");
  }

 private:
  std::string name_;
};

class TestCustomDeviceTensorHandle : public CustomDeviceTensorHandle {
 public:
  TestCustomDeviceTensorHandle(ImmediateExecutionContext* context,
                               TestCustomDevice* device,
                               tensorflow::DataType dtype, int64_t length)
      : CustomDeviceTensorHandle(context, device, dtype), length_(length) {}

  void* DevicePointer() const override { return nullptr; }
  Status NumDims(int* num_dims) const override {
    *num_dims = 1;
    return OkStatus();
  }
  Status Dim(int dim_index, int64_t* dim) const override {
    if (dim_index == 0) {
      *dim = length_;
      return OkStatus();
    } else {
      return errors::Internal("Dim out of bounds");
    }
  }

  Status SummarizeValue(std::string& summary) const override {
    summary = std::string("TestValue");
    return OkStatus();
  }

 private:
  const int64_t length_;
};

TEST(CustomDevice, TestTensorHandle) {
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  core::RefCountPtr<EagerContext> ctx(new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr));
  std::string device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:15";
  TestCustomDevice device(device_name);
  core::RefCountPtr<TestCustomDeviceTensorHandle> tensor(
      new TestCustomDeviceTensorHandle(ctx.get(), &device, DT_FLOAT,
                                       /*length=*/3));
  Status s;
  std::string device_type = tensor->DeviceType(&s);
  ASSERT_TRUE(s.ok()) << s.error_message();
  EXPECT_EQ("CUSTOM", device_type);
  int device_index = tensor->DeviceId(&s);
  ASSERT_TRUE(s.ok()) << s.error_message();
  EXPECT_EQ(15, device_index);
  int64_t num_elements = 0;
  s = tensor->NumElements(&num_elements);
  ASSERT_TRUE(s.ok()) << s.error_message();
  EXPECT_EQ(3, num_elements);
  EXPECT_THAT(
      tensor->DebugString(),
      ContainsRegex(
          R"re(TensorHandle\(TestValue, shape=\[3\], dtype=DT_FLOAT, device=.*\))re"));
}

TEST(CustomDevice, TestTensorHandleUnknownDimNumElements) {
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  core::RefCountPtr<EagerContext> ctx(new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr));
  std::string device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:15";
  TestCustomDevice device(device_name);
  core::RefCountPtr<TestCustomDeviceTensorHandle> tensor(
      new TestCustomDeviceTensorHandle(ctx.get(), &device, DT_FLOAT,
                                       /*length=*/-1));
  int64_t num_elements;
  Status s = tensor->NumElements(&num_elements);
  EXPECT_FALSE(s.ok());
  EXPECT_THAT(s.error_message(), HasSubstr("representing varying shapes"));
}

TEST(CustomDevice, TestResourcePlacement) {
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  core::RefCountPtr<EagerContext> ctx(new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr));
  std::string custom_device_name =
      "/job:localhost/replica:0/task:0/device:CUSTOM:15";
  TestCustomDevice custom_device(custom_device_name);
  core::RefCountPtr<TestCustomDeviceTensorHandle> custom_float_tensor(
      new TestCustomDeviceTensorHandle(ctx.get(), &custom_device, DT_FLOAT,
                                       /*length=*/3));
  core::RefCountPtr<TestCustomDeviceTensorHandle> custom_resource_tensor(
      new TestCustomDeviceTensorHandle(ctx.get(), &custom_device, DT_RESOURCE,
                                       /*length=*/3));

  Tensor resource_tensor(DT_RESOURCE, {});
  Device* physical_device = device_mgr.ListDevices().at(0);
  core::RefCountPtr<TensorHandle> physical_resource_tensor(
      TensorHandle::CreateLocalHandle(std::move(resource_tensor),
                                      physical_device, physical_device,
                                      physical_device, ctx.get()));
  Tensor float_tensor(DT_FLOAT, {});
  core::RefCountPtr<TensorHandle> physical_float_tensor(
      TensorHandle::CreateLocalHandle(std::move(float_tensor), physical_device,
                                      physical_device, physical_device,
                                      ctx.get()));
  EagerOperation op(ctx.get());
  TF_ASSERT_OK(op.Reset("AssignVariableOp", ""));
  TF_ASSERT_OK(op.AddInput(physical_resource_tensor.get()));
  TF_ASSERT_OK(op.AddInput(custom_float_tensor.get()));
  CustomDevice* placed_device = nullptr;
  TF_ASSERT_OK(ctx->GetCustomDeviceOpHandler().MaybePinToCustomDevice(
      &placed_device, op));
  // MaybePinToCustomDevice has no opinion about ops which have physical
  // resource-dtype inputs. They'll get placed on physical devices.
  EXPECT_EQ(nullptr, placed_device);

  op.Clear();
  TF_ASSERT_OK(op.Reset("AssignVariableOp", custom_device_name.c_str()));
  TF_ASSERT_OK(op.AddInput(physical_resource_tensor.get()));
  TF_ASSERT_OK(op.AddInput(custom_float_tensor.get()));
  placed_device = nullptr;
  TF_ASSERT_OK(ctx->GetCustomDeviceOpHandler().MaybePinToCustomDevice(
      &placed_device, op));
  // Explicit placement onto a custom device also doesn't trigger custom device
  // placement if there's a physical device resource input.
  EXPECT_EQ(nullptr, placed_device);

  op.Clear();
  TF_ASSERT_OK(
      op.Reset("Identity", "/job:localhost/replica:0/task:0/device:CPU:0"));
  TF_ASSERT_OK(op.AddInput(physical_float_tensor.get()));
  placed_device = nullptr;
  TF_ASSERT_OK(ctx->GetCustomDeviceOpHandler().MaybePinToCustomDevice(
      &placed_device, op));
  // Explicit placements typically override input-based placement onto a custom
  // device.
  EXPECT_EQ(nullptr, placed_device);

  op.Clear();
  TF_ASSERT_OK(op.Reset("AssignVariableOp",
                        "/job:localhost/replica:0/task:0/device:CPU:0"));
  TF_ASSERT_OK(op.AddInput(custom_resource_tensor.get()));
  TF_ASSERT_OK(op.AddInput(physical_float_tensor.get()));
  placed_device = nullptr;
  TF_ASSERT_OK(ctx->GetCustomDeviceOpHandler().MaybePinToCustomDevice(
      &placed_device, op));
  // Even with an explicit physical device placement, custom device resource
  // inputs place the op on the custom device.
  ASSERT_NE(placed_device, nullptr);
  EXPECT_EQ(&custom_device, placed_device);
}

}  // namespace
}  // namespace eager
}  // namespace tensorflow
