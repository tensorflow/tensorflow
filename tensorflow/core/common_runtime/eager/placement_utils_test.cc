/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/eager/placement_utils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/execute_node.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/lib/core/status_test_util.h"

#define DEVICE_CPU0 "/job:localhost/replica:0/task:0/device:CPU:0"
#define DEVICE_CPU0_TASK1 "/job:localhost/replica:0/task:1/device:CPU:0"
#define DEVICE_GPU0 "/job:localhost/replica:0/task:0/device:GPU:0"

namespace tensorflow {
namespace {

TEST(PlacementUtilsTest, IsColocationExemptFalse) {
  ASSERT_FALSE(eager::IsColocationExempt("Identity"));
}

TEST(PlacementUtilsTest, IsColocationExemptTrue) {
  ASSERT_TRUE(eager::IsColocationExempt("IdentityN"));
}

TEST(PlacementUtilsTest, IsFunctionTrue) {
  ASSERT_TRUE(eager::IsFunction("MyFunction"));
}

TEST(PlacementUtilsTest, IsFunctionFalse) {
  ASSERT_FALSE(eager::IsFunction("Identity"));
}

// Return a fake (local or remote device) with the specified type and name.
static Device* CreateDevice(const char* type, const char* name,
                            bool is_local = true) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr, bool is_local)
        : Device(nullptr, attr), is_local_(is_local) {}
    Status Sync() override { return absl::OkStatus(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
    bool IsLocal() const override { return is_local_; }

   private:
    const bool is_local_;
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  int64_t incarnation = random::New64();
  while (incarnation == 0) {
    incarnation = random::New64();
  }
  attr.set_incarnation(incarnation);
  return new FakeDevice(attr, is_local);
}

static void CreateLocalDeviceVector(
    std::vector<std::unique_ptr<Device>>& devices) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", DEVICE_CPU0));
  devices.emplace_back(std::move(d0));
  std::unique_ptr<Device> d1(CreateDevice("GPU", DEVICE_GPU0));
  devices.emplace_back(std::move(d1));
}

static Device* CreateRemoteDeviceVector(
    std::vector<std::unique_ptr<Device>>& devices) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", DEVICE_CPU0_TASK1, false));
  devices.emplace_back(std::move(d0));
  return devices.back().get();
}

struct MaybePinSmallOpsToCpuTestCase {
  std::string test_name;
  DataType dtype;
  TensorShape shape;
  string op_name;
  const char* device;
  bool expect;
};

class PlacementUtilsSmallOpsTest
    : public ::testing::TestWithParam<MaybePinSmallOpsToCpuTestCase> {};

TEST_P(PlacementUtilsSmallOpsTest, TestMaybePinSmallOpsToCpu) {
  const MaybePinSmallOpsToCpuTestCase& test_case = GetParam();

  bool result;

  std::vector<std::unique_ptr<Device>> devices;
  CreateLocalDeviceVector(devices);
  StaticDeviceMgr device_mgr(std::move(devices));
  core::RefCountPtr<EagerContext> context;
  context = core::RefCountPtr<EagerContext>(new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_EXPLICIT,
      false, &device_mgr, false, nullptr, nullptr, nullptr,
      /*run_eager_op_as_function=*/true));
  auto ctx = context.get();
  ctx->SetRunEagerOpAsFunction(true);
  std::vector<ImmediateExecutionTensorHandle*> arg;
  Tensor input_tensor(test_case.dtype, test_case.shape);
  auto input = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      ctx->CreateLocalHandleFromTFTensor(input_tensor, test_case.device));
  if (test_case.op_name != "RefIdentity") {
    arg.push_back(input.get());
  }

  TF_ASSERT_OK(eager::MaybePinSmallOpsToCpu(&result, test_case.op_name, arg,
                                            test_case.device));
  ASSERT_EQ(result, test_case.expect);
}

INSTANTIATE_TEST_SUITE_P(
    MaybePinSmallOpsToCpuTests, PlacementUtilsSmallOpsTest,
    ::testing::ValuesIn<MaybePinSmallOpsToCpuTestCase>({
        {"OkToPin", DT_INT64, {}, "Identity", DEVICE_CPU0, true},
        {"NotOkToPin_Float", DT_FLOAT, {}, "Identity", DEVICE_CPU0, false},
        {"NotOkToPin_Function", DT_INT64, {}, "MyFunction", DEVICE_CPU0, false},
        {"NotOkToPin_NoInputs",
         DT_INT64,
         {},
         "RefIdentity",
         DEVICE_CPU0,
         false},
        {"NotOkToPin_NotCpu", DT_INT64, {}, "Identity", DEVICE_GPU0, false},
        {"NotOkToPin_TooBig", DT_INT64, {65}, "Identity", DEVICE_CPU0, false},
    }),
    [](const ::testing::TestParamInfo<PlacementUtilsSmallOpsTest::ParamType>&
           info) { return info.param.test_name; });

struct MaybePinToResourceDeviceTestCase {
  std::string test_name;
  DataType dtype;
  string op_name;
  const char* device;
  bool expect;
};

class PlacementUtilsResourceDeviceTest
    : public ::testing::TestWithParam<MaybePinToResourceDeviceTestCase> {};

TEST_P(PlacementUtilsResourceDeviceTest, TestMaybePinToResourceDevice) {
  const MaybePinToResourceDeviceTestCase& test_case = GetParam();
  Device* device = nullptr;

  std::vector<std::unique_ptr<Device>> local_devices;
  CreateLocalDeviceVector(local_devices);
  StaticDeviceMgr local_device_mgr(std::move(local_devices));

  core::RefCountPtr<EagerContext> context;
  context = core::RefCountPtr<EagerContext>(new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_EXPLICIT,
      false, &local_device_mgr, false, nullptr, nullptr, nullptr,
      /*run_eager_op_as_function=*/true));
  auto ctx = context.get();
  auto op = EagerOperation(ctx);
  TF_ASSERT_OK(op.Reset(test_case.op_name.c_str(), DEVICE_CPU0));

  Tensor input_tensor(test_case.dtype, {});
  auto input = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      ctx->CreateLocalHandleFromTFTensor(input_tensor, test_case.device));
  TF_ASSERT_OK(op.AddInput(input.get()));
  ASSERT_TRUE(device == nullptr);
  TF_ASSERT_OK(eager::MaybePinToResourceDevice(&device, op));
  ASSERT_EQ(device != nullptr, test_case.expect);
}

INSTANTIATE_TEST_SUITE_P(
    MaybePinToResourceDeviceTestCase, PlacementUtilsResourceDeviceTest,
    ::testing::ValuesIn<MaybePinToResourceDeviceTestCase>({
        {"OkToPin", DT_RESOURCE, "Identity", DEVICE_CPU0, true},
        {"NotOkToPin_NotResource", DT_FLOAT, "Identity", DEVICE_CPU0, false},
        {"NotOkToPin_ColocationExempt", DT_RESOURCE, "IdentityN", DEVICE_CPU0,
         false},
    }),
    [](const ::testing::TestParamInfo<
        PlacementUtilsResourceDeviceTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(PlacementUtilsTest, MaybePinToResourceDevice_OtherDevice) {
  StaticDeviceMgr device_mgr(
      DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0"));
  Device* device0 = device_mgr.ListDevices().at(0);
  auto remote_device_mgr = std::make_unique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> remote_devices;
  CreateRemoteDeviceVector(remote_devices);
  TF_ASSERT_OK(remote_device_mgr->AddDevices(std::move(remote_devices)));

  Device* device1 = remote_device_mgr->ListDevices().at(0);

  Status s;
  std::unique_ptr<CompositeDevice> composite_device =
      CompositeDevice::MakeDevice({device0->name(), device1->name()},
                                  /*unique_device_id=*/0,
                                  device_mgr.HostCPU()->parsed_name(), &s);
  TF_ASSERT_OK(s);

  auto ctx = new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr, nullptr,
      /*run_eager_op_as_function=*/true);

  // Set a RemoteMgr to the EagerContext.
  auto remote_mgr = std::make_unique<eager::RemoteMgr>(
      /*is_master=*/true, ctx);
  TF_ASSERT_OK(ctx->InitializeRemoteMaster(
      /*server=*/nullptr, /*worker_env=*/nullptr,
      /*worker_session=*/nullptr, /*remote_eager_workers=*/nullptr,
      std::move(remote_device_mgr), /*remote_contexts=*/{},
      EagerContext::NewContextId(),
      /*r=*/nullptr, &device_mgr, /*keep_alive_secs*/ 600,
      /*cluster_flr=*/nullptr, std::move(remote_mgr)));
  ASSERT_NE(ctx->remote_device_mgr(), nullptr);

  auto op = EagerOperation(ctx);
  TF_ASSERT_OK(op.Reset("Identity", DEVICE_CPU0));
  TensorHandle* input = TensorHandle::CreateLazyRemoteHandle(
      /*op_id=*/2, /*output_num=*/1, DT_RESOURCE, device1, /*is_ready=*/true,
      ctx);
  TF_ASSERT_OK(op.AddInput(input));
  ASSERT_NE(input->resource_remote_device_incarnation(), 0);

  Device* device = nullptr;
  TF_ASSERT_OK(eager::MaybePinToResourceDevice(&device, op));
  ASSERT_TRUE(device != nullptr);
  input->Unref();
  ctx->Unref();
}

}  // namespace
}  // namespace tensorflow
