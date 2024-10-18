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

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::tensorflow::test::function::NDef;

constexpr char kFullCPU[] = "/job:a/replica:0/task:0/device:CPU:0";
constexpr char kFullGPU[] = "/job:a/replica:0/task:0/device:FakeGPU:0";

////////////////////////////////////////////////////////////////////////////////
//
// Op, kernel to set up the environment.
//
// The Placer uses information about the op (input types),
// kernel (device constraints). To avoid depending on the full runtime, we
// define dummy implementations of these, and register them with the
// runtime.
//
////////////////////////////////////////////////////////////////////////////////

// A dummy OpKernel that is used to register ops on different devices.
class DummyOp : public OpKernel {
 public:
  explicit DummyOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
};

// Register the following ops so they can be added to a Graph, and
// kernels so that they can be placed on particular device types.
REGISTER_OP("InvalidOp").Output("o: Ref(float)");

REGISTER_OP("TestOp").Output("o: Ref(float)");
REGISTER_KERNEL_BUILDER(Name("TestOp").Device(DEVICE_CPU).Priority(1), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestOp").Device("FakeGPU").Priority(2), DummyOp);

static Device* CreateDevice(const char* type, const char* name) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
    absl::Status Sync() override { return absl::OkStatus(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

class PlacementTest : public ::testing::Test {
 public:
  PlacementTest() : device_manager_(nullptr), context_(nullptr) {}

  ~PlacementTest() override {
    delete device_manager_;
    if (context_) {
      context_->Unref();
    }
  }

  EagerContext* context() { return context_; }

  void InitContext(const SessionOptions& opts,
                   ContextDevicePlacementPolicy policy) {
    ASSERT_EQ(context_, nullptr);
    InitDeviceManager();
    context_ = new EagerContext(
        opts, policy,
        /* async */ false, device_manager_,
        /* device_mgr_owned */ false, /* rendezvous */ nullptr,
        /* cluster_flr */ nullptr, /*collective_executor_mgr=*/nullptr,
        /*run_eager_op_as_function=*/true);
  }

 protected:
  void InitDeviceManager() {
    ASSERT_EQ(device_manager_, nullptr);
    device_manager_ = new DynamicDeviceMgr();
    std::vector<std::unique_ptr<Device>> added_devices;
    SessionOptions opts;

    // Have to use real CPU device. Other, ctx->HostCPU() will return invalid
    // device.
    added_devices.emplace_back(CreateDevice(DEVICE_CPU, kFullCPU));
    added_devices.emplace_back(CreateDevice("FakeGPU", kFullGPU));

    TF_CHECK_OK(device_manager_->AddDevices(std::move(added_devices)));
  }

  DynamicDeviceMgr* device_manager_;
  EagerContext* context_;
};

TEST_F(PlacementTest, SelectDeviceExplicitHardPlacement) {
  SessionOptions options;
  options.config.set_log_device_placement(true);
  options.config.set_allow_soft_placement(false);
  InitContext(options, DEVICE_PLACEMENT_EXPLICIT);

  Device* dev;
  DeviceNameUtils::ParsedName requested;

  // No supported devices should result in an error.
  requested.Clear();
  NodeDef invalid_op = NDef("invalid_op", "InvalidOp", {}, {});

  absl::Status status = context()->SelectDevice(requested, invalid_op, &dev);
  LOG(ERROR) << status;
  EXPECT_TRUE(errors::IsNotFound(status));
  EXPECT_TRUE(
      absl::StrContains(status.message(), "Could not find device for node"))
      << "unexpected error message " << status.message();

  // An invalid requested device should also cause an error.
  ASSERT_TRUE(DeviceNameUtils::ParseLocalName("FakeGPU:99", &requested));
  NodeDef node = NDef("x", "TestOp", {}, {});
  status = context()->SelectDevice(requested, node, &dev);

  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_TRUE(absl::StrContains(status.message(),
                                "Could not satisfy device specification"))
      << "unexpected error message " << status.message();

  // Should pick the device with higher priority if given no constraints.
  requested.Clear();
  TF_ASSERT_OK(context()->SelectDevice(requested, node, &dev));
  EXPECT_EQ(dev->device_type(), "FakeGPU");

  // Should pick a CPU if asked to.
  ASSERT_TRUE(DeviceNameUtils::ParseLocalName("CPU:0", &requested));
  TF_ASSERT_OK(context()->SelectDevice(requested, node, &dev));
  EXPECT_EQ(dev->device_type(), DEVICE_CPU);
}

TEST_F(PlacementTest, SelectDeviceExplicitSoftPlacement) {
  SessionOptions options;
  options.config.set_log_device_placement(true);
  options.config.set_allow_soft_placement(true);
  InitContext(options, DEVICE_PLACEMENT_EXPLICIT);

  Device* dev;
  DeviceNameUtils::ParsedName requested;

  // No supported devices should result in an error.
  requested.Clear();
  NodeDef invalid_op = NDef("invalid_op", "InvalidOp", {}, {});

  absl::Status status = context()->SelectDevice(requested, invalid_op, &dev);
  LOG(ERROR) << status;
  EXPECT_TRUE(errors::IsNotFound(status));
  EXPECT_TRUE(
      absl::StrContains(status.message(), "Could not find device for node"))
      << "unexpected error message " << status.message();

  // An invalid requested device should be replaced by the "best" one.
  ASSERT_TRUE(DeviceNameUtils::ParseLocalName("FakeGPU:99", &requested));
  NodeDef node = NDef("x", "TestOp", {}, {});
  status = context()->SelectDevice(requested, node, &dev);
  EXPECT_EQ(dev->device_type(), "FakeGPU");
}

}  // namespace
}  // namespace tensorflow
