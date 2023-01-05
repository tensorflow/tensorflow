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

#include "tensorflow/core/common_runtime/eager/context.h"

#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/eager/context_distributed_manager.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::testing::HasSubstr;

typedef FunctionDefHelper FDH;

// Return a fake device.
static Device* CreateDevice(const string& type, int n) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
    Status Sync() override { return OkStatus(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name("/job:localhost/replica:0/task:0/device:" + type + ":" +
                std::to_string(n));
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

class EagerContextTest : public ::testing::Test {
 public:
  EagerContext* context() { return context_.get(); }

  void InitContext(const SessionOptions& opts,
                   ContextDevicePlacementPolicy policy, bool async = false) {
    ASSERT_EQ(context_, nullptr);
    InitDeviceManager();
    context_ = core::RefCountPtr<EagerContext>(new EagerContext(
        opts, policy, async, device_manager_.get(),
        /*device_mgr_owned=*/false, /*rendezvous=*/nullptr,
        /*cluster_flr=*/nullptr, /*collective_executor_mgr=*/nullptr,
        /*run_eager_op_as_function=*/true));
  }

 protected:
  void InitDeviceManager() {
    ASSERT_EQ(device_manager_, nullptr);
    device_manager_ = std::make_unique<DynamicDeviceMgr>();
    std::vector<std::unique_ptr<Device>> added_devices;
    added_devices.emplace_back(CreateDevice(DEVICE_CPU, 0));
    added_devices.emplace_back(CreateDevice(DEVICE_CPU, 1));
    added_devices.emplace_back(CreateDevice(DEVICE_GPU, 0));
    added_devices.emplace_back(CreateDevice(DEVICE_GPU, 1));
    added_devices.emplace_back(CreateDevice(DEVICE_TPU, 0));

    TF_CHECK_OK(device_manager_->AddDevices(std::move(added_devices)));
  }

  std::unique_ptr<DynamicDeviceMgr> device_manager_;
  core::RefCountPtr<EagerContext> context_;
};

TEST_F(EagerContextTest, CompositeDevice) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  std::vector<string> underlying_devices = {
      "/job:worker/replica:0/task:0/device:CPU:0",
      "/job:worker/replica:0/task:0/device:CPU:1"};
  CompositeDevice* composite_device_0 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_0));
  EXPECT_EQ(composite_device_0->name(),
            "/job:localhost/replica:0/task:0/device:COMPOSITE:0");
  CompositeDevice* device = nullptr;
  TF_EXPECT_OK(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:0", &device));
  EXPECT_EQ(device, composite_device_0);
  CompositeDevice* composite_device_1 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_1));
  EXPECT_EQ(composite_device_1, composite_device_0);
  underlying_devices.push_back("/job:worker/replica:0/task:0/device:CPU:2");
  CompositeDevice* composite_device_2 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_2));
  EXPECT_EQ(composite_device_2->name(),
            "/job:localhost/replica:0/task:0/device:COMPOSITE:1");
  TF_EXPECT_OK(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:1", &device));
  EXPECT_EQ(device, composite_device_2);

  EXPECT_TRUE(errors::IsNotFound(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:2", &device)));
}

TEST_F(EagerContextTest, CompositeDeviceWithGivenName) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const std::vector<string> underlying_devices_0 = {
      "/job:worker/replica:0/task:0/device:CPU:0",
      "/job:worker/replica:0/task:0/device:CPU:1"};
  const string composite_device_name =
      "/job:worker1/replica:0/task:0/device:COMPOSITE:5";
  // Create a CompositeDevice with the given name.
  CompositeDevice* composite_device_0 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(
      underlying_devices_0, composite_device_name, &composite_device_0));
  EXPECT_EQ(composite_device_0->name(), composite_device_name);

  CompositeDevice* device = nullptr;
  TF_EXPECT_OK(
      context()->FindCompositeDeviceFromName(composite_device_name, &device));
  EXPECT_EQ(device, composite_device_0);

  std::vector<string> underlying_devices_1 = {
      "/job:worker/replica:0/task:0/device:CPU:1",
      "/job:worker/replica:0/task:0/device:CPU:2"};
  // Find a CompositeDevice with the given name.
  CompositeDevice* composite_device_1 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(
      underlying_devices_1, composite_device_0->name(), &composite_device_1));
  EXPECT_EQ(composite_device_1, composite_device_0);
}

TEST_F(EagerContextTest, AddFunctionDef) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
}

TEST_F(EagerContextTest, AddFunctionDefRepeatSame) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
  const FunctionDef x_times_two_copy = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two_copy));
}

TEST_F(EagerContextTest, AddFunctionDefRepeatDifferent) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
  const Tensor kThree = test::AsScalar<int64_t>(3);
  // Same function name but body is different. This should error out.
  const FunctionDef x_times_two_copy = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kThree}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  Status s = context()->AddFunctionDef(x_times_two_copy);
  EXPECT_FALSE(s.ok());
}

TEST_F(EagerContextTest, FunctionErrorRecovery) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT, /*async=*/true);
  context()->SetReuseRendezvousForFunctions(true);
  const FunctionDef assert_and_identity = FDH::Define(
      // Name
      "AssertAndIdentity",
      // Args
      {"x: float", "condition: bool"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"assert"},
           "Assert",
           {"condition", "x"},
           {{"T", std::vector<DataType>{DT_FLOAT}}}},
          {{"y"},
           "Identity",
           {"x"},
           {{"T", DT_FLOAT}},
           /*dep=*/{"assert"}},
      });
  Status s = context()->AddFunctionDef(assert_and_identity);
  auto fail_op = ImmediateOpPtr(context()->CreateOperation());
  TF_ASSERT_OK(fail_op->Reset("AssertAndIdentity",
                              "/job:localhost/replica:0/task:0/device:CPU:0"));
  Tensor float_tensor = test::AsScalar<float>(3.0);
  auto input_float = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          float_tensor, context()->HostCPUName().c_str()));
  Tensor bool_tensor_false = test::AsScalar<bool>(false);
  auto input_bool_false = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          bool_tensor_false, context()->HostCPUName().c_str()));
  TF_ASSERT_OK(fail_op->AddInput(input_float.get()));
  TF_ASSERT_OK(fail_op->AddInput(input_bool_false.get()));
  std::vector<AbstractTensorHandle*> retvals(1);
  int num_retvals = retvals.size();
  StatusGroup op_and_sync_status;
  op_and_sync_status.Update(
      fail_op->Execute(absl::MakeSpan(retvals), &num_retvals));
  op_and_sync_status.Update(context()->SyncExecutors());
  ASSERT_THAT(op_and_sync_status.as_summary_status().error_message(),
              HasSubstr("assertion failed"));
  if (retvals[0] != nullptr) {
    retvals[0]->Unref();
    retvals[0] = nullptr;
  }

  Tensor bool_tensor_true = test::AsScalar<bool>(true);
  auto input_bool_true = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          bool_tensor_true, context()->HostCPUName().c_str()));
  auto success_op = ImmediateOpPtr(context()->CreateOperation());
  TF_ASSERT_OK(success_op->Reset(
      "AssertAndIdentity", "/job:localhost/replica:0/task:0/device:CPU:0"));
  TF_ASSERT_OK(success_op->AddInput(input_float.get()));
  TF_ASSERT_OK(success_op->AddInput(input_bool_true.get()));
  // A second run of the function should work, despite the previous failure.
  TF_ASSERT_OK(success_op->Execute(absl::MakeSpan(retvals), &num_retvals));
  TF_ASSERT_OK(context()->SyncExecutors());
  retvals[0]->Unref();
  retvals[0] = nullptr;
}

TEST_F(EagerContextTest, XlaCompileDeviceType) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT, /*async=*/true);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: int64"},
      // Return values
      {"y: int64"}, {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"y"}, "Mul", {"x", "two"}, {{"T", DT_INT64}}},
      });

  Status s = context()->AddFunctionDef(x_times_two);
  context()->SetJitCompileRewrite(true);
  auto op = ImmediateOpPtr(context()->CreateOperation());
  TF_ASSERT_OK(
      op->Reset("XTimesTwo", "/job:localhost/replica:0/task:0/device:CPU:0"));
  Tensor int_tensor = test::AsScalar<int64_t>(3);
  auto input_int = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          int_tensor, context()->HostCPUName().c_str()));
  TF_ASSERT_OK(op->AddInput(input_int.get()));
  std::vector<AbstractTensorHandle*> retvals(1);
  int num_retvals = retvals.size();
  TF_ASSERT_OK(op->Execute(absl::MakeSpan(retvals), &num_retvals));
  retvals[0]->Unref();
  retvals[0] = nullptr;
}

TEST_F(EagerContextTest, LocalRendezvousCreation) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  auto rendezvous_creator = context()->RendezvousFactory();

  // Create a new rendezvous instance.
  // Initially its ref-count is 2:
  // one added upon rendezvous creation, the other one added by EagerContext.
  Rendezvous* rendezvous_1;
  TF_ASSERT_OK(rendezvous_creator(1, nullptr, &rendezvous_1));
  EXPECT_EQ(rendezvous_1->RefCount(), 2);

  // Create another rendezvous instance with the same step-id.
  // This would add one more ref-count to the existing rendezvous insteance
  // insted of creating a new instance.
  Rendezvous* rendezvous_2;
  TF_ASSERT_OK(rendezvous_creator(1, nullptr, &rendezvous_2));
  EXPECT_EQ(rendezvous_2->RefCount(), 3);

  // Caller releases rendezvous-1.
  rendezvous_1->Unref();
  EXPECT_EQ(rendezvous_1->RefCount(), 2);

  // Caller releases rendezvous-2.
  rendezvous_2->Unref();
  EXPECT_EQ(rendezvous_2->RefCount(), 1);
}

void TestGlobalRendezvous(EagerContext* context, bool reuse_global_rendezvous) {
  context->SetReuseRendezvousForFunctions(reuse_global_rendezvous);
  EXPECT_EQ(context->GetReuseRendezvousForFunctions(), reuse_global_rendezvous);

  auto rendezvous_creator = context->RendezvousFactory();
  Rendezvous* rendezvous_1;
  TF_ASSERT_OK(rendezvous_creator(-1, nullptr, &rendezvous_1));
  EXPECT_EQ(rendezvous_1->RefCount(), 2);
  Rendezvous* rendezvous_2;
  TF_ASSERT_OK(rendezvous_creator(-1, nullptr, &rendezvous_2));
  EXPECT_EQ(rendezvous_2->RefCount(), 3);

  // Global rendezvous's ref-count should be back to 1 after resetting.
  context->ResetGlobalRendezvousForFunction();

  Rendezvous* rendezvous_3;
  TF_ASSERT_OK(rendezvous_creator(-1, nullptr, &rendezvous_3));
  EXPECT_EQ(rendezvous_3->RefCount(), 2);

  // Callers release rendezvous.
  rendezvous_1->Unref();
  rendezvous_2->Unref();
  rendezvous_3->Unref();
}

TEST_F(EagerContextTest, GlobalRendezvousCreation) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);

  TestGlobalRendezvous(context(), false);
}

TEST_F(EagerContextTest, ReuseGlobalRendezvous) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  EXPECT_FALSE(context()->GetReuseRendezvousForFunctions());

  TestGlobalRendezvous(context(), true);
}

}  // namespace
}  // namespace tensorflow
