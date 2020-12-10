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
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class TestCustomDevice : public CustomDevice {
 public:
  explicit TestCustomDevice(std::string name) : name_(name) {}
  const std::string& name() override { return name_; }
  Status CopyTensorToDevice(TensorHandle* tensor,
                            TensorHandle** result) override {
    tensor->Ref();
    *result = tensor;
    return Status::OK();
  }
  Status CopyTensorFromDevice(TensorHandle* tensor,
                              const std::string& target_device_name,
                              TensorHandle** result) override {
    tensor->Ref();
    *result = tensor;
    return Status::OK();
  }
  Status Execute(const EagerOperation* op, TensorHandle** retvals,
                 int* num_retvals) override {
    return errors::Unimplemented("Not implemented");
  }

 private:
  std::string name_;
};

class TestCustomDeviceTensorHandle : public CustomDeviceTensorHandle {
 public:
  TestCustomDeviceTensorHandle(ImmediateExecutionContext* context,
                               TestCustomDevice* device,
                               tensorflow::DataType dtype)
      : CustomDeviceTensorHandle(context, device, dtype) {}

  Status NumDims(int* num_dims) const override {
    *num_dims = 1;
    return Status::OK();
  }
  Status Dim(int dim_index, int64* dim) const override {
    if (dim_index == 0) {
      *dim = 3;
      return Status::OK();
    } else {
      return errors::Internal("Dim out of bounds");
    }
  }
};

TEST(CustomDevice, TestTensorHandle) {
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  core::RefCountPtr<EagerContext> ctx(new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      false, &device_mgr, false, nullptr, nullptr));
  std::string device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:15";
  TestCustomDevice device(device_name);
  core::RefCountPtr<TestCustomDeviceTensorHandle> tensor(
      new TestCustomDeviceTensorHandle(ctx.get(), &device, DT_FLOAT));
  Status s;
  std::string device_type = tensor->DeviceType(&s);
  ASSERT_TRUE(s.ok()) << s.error_message();
  EXPECT_EQ("CUSTOM", device_type);
  int device_index = tensor->DeviceId(&s);
  ASSERT_TRUE(s.ok()) << s.error_message();
  EXPECT_EQ(15, device_index);
  int64 num_elements = 0;
  s = tensor->NumElements(&num_elements);
  ASSERT_TRUE(s.ok()) << s.error_message();
  EXPECT_EQ(3, num_elements);
  EXPECT_EQ("TensorHandle(shape=[3], dtype=DT_FLOAT)", tensor->DebugString());
}

}  // namespace
}  // namespace tensorflow
