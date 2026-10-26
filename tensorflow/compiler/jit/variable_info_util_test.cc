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

#include "tensorflow/compiler/jit/variable_info_util.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

// GetVariableInfosFromInputs reads this once per process, so the enabled and
// disabled cases run in separate test targets that set it differently.
constexpr char kAllowHostResidentVarsEnv[] = "TF_XLA_ALLOW_HOST_RESIDENT_VARS";

constexpr char kHostCpu[] = "/job:localhost/replica:0/task:0/device:CPU:0";
constexpr char kExecutingDevice[] =
    "/job:localhost/replica:0/task:0/device:CPU:1";
constexpr char kAccelerator[] = "/job:localhost/replica:0/task:0/device:GPU:0";
constexpr char kContainer[] = "container";

class FakeDevice : public Device {
 public:
  FakeDevice(absl::string_view name, absl::string_view type)
      : Device(Env::Default(), MakeAttributes(name, type)) {}

  absl::Status Sync() override { return absl::OkStatus(); }
  Allocator* GetAllocator(AllocatorAttributes) override {
    return cpu_allocator();
  }

 private:
  static DeviceAttributes MakeAttributes(absl::string_view name,
                                         absl::string_view type) {
    DeviceAttributes attributes;
    attributes.set_name(name);
    attributes.set_device_type(type);
    return attributes;
  }
};

// Stands in for a host-to-device DMA: all fake devices use host memory.
class MemcpyDeviceContext : public DeviceContext {
 public:
  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override {
    std::memcpy(device_tensor->data(), cpu_tensor->data(),
                cpu_tensor->TotalBytes());
    ++num_copies_;
    done(absl::OkStatus());
  }

  int num_copies() const { return num_copies_; }

 private:
  mutable int num_copies_ = 0;
};

class VariableInfoUtilTest : public ::testing::Test {
 protected:
  VariableInfoUtilTest() {
    auto host_cpu = std::make_unique<FakeDevice>(kHostCpu, DEVICE_CPU);
    auto executing_device =
        std::make_unique<FakeDevice>(kExecutingDevice, DEVICE_CPU);
    auto accelerator = std::make_unique<FakeDevice>(kAccelerator, DEVICE_GPU);
    host_cpu_ = host_cpu.get();
    executing_device_ = executing_device.get();
    accelerator_ = accelerator.get();
    std::vector<std::unique_ptr<Device>> devices;
    devices.push_back(std::move(host_cpu));
    devices.push_back(std::move(executing_device));
    devices.push_back(std::move(accelerator));
    device_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(devices));

    lib_def_ = std::make_unique<FunctionLibraryDefinition>(
        OpRegistry::Global(), FunctionDefLibrary());
    pflr_ = std::make_unique<ProcessFunctionLibraryRuntime>(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, lib_def_.get(), OptimizerOptions());

    params_.device = executing_device_;
    params_.function_library = pflr_->GetFLR(kExecutingDevice);
    params_.op_device_context = device_context_.get();
    ctx_ = std::make_unique<OpKernelContext>(&params_, /*num_outputs=*/0);
  }

  // Creates a float variable named `name` on `device`, initialized to `value`
  // if given, and returns a resource tensor holding its handle.
  Tensor CreateVariable(Device* device, absl::string_view name,
                        std::optional<Tensor> value) {
    Var* variable = new Var(DT_FLOAT);
    if (value.has_value()) {
      *variable->tensor() = *value;
      variable->is_initialized = true;
    }
    TF_CHECK_OK(device->resource_manager()->Create(
        kContainer, std::string(name), variable));
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = MakeResourceHandle(
        kContainer, std::string(name), *device, TypeIndex::Make<Var>());
    return handle;
  }

  // Looks up the variable behind `handle` from the executing device.
  absl::Status GetVariableInfos(const Tensor& handle,
                                const std::set<int>* variables_updated,
                                std::vector<VariableInfo>* variable_infos) {
    std::vector<const Tensor*> inputs = {&handle};
    std::vector<int> variable_indices = {0};
    return GetVariableInfosFromInputs(
        executing_device_->resource_manager(), executing_device_, inputs,
        variable_indices, variables_updated, ctx_.get(), variable_infos);
  }

  Device* host_cpu_;
  Device* executing_device_;
  Device* accelerator_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  core::RefCountPtr<MemcpyDeviceContext> device_context_{
      new MemcpyDeviceContext()};
  OpKernelContext::Params params_;
  std::unique_ptr<OpKernelContext> ctx_;
};

class HostResidentVarsDisabledTest : public VariableInfoUtilTest {
 protected:
  void SetUp() override {
    if (std::getenv(kAllowHostResidentVarsEnv) != nullptr) {
      GTEST_SKIP() << "Runs in variable_info_util_test.";
    }
  }
};

class HostResidentVarsEnabledTest : public VariableInfoUtilTest {
 protected:
  void SetUp() override {
    if (std::getenv(kAllowHostResidentVarsEnv) == nullptr) {
      GTEST_SKIP() << "Runs in variable_info_util_host_resident_vars_test.";
    }
  }
};

TEST_F(HostResidentVarsDisabledTest, HostVariableIsRejected) {
  Tensor handle = CreateVariable(host_cpu_, "v", test::AsScalar<float>(1.0f));
  std::set<int> variables_updated;
  std::vector<VariableInfo> variable_infos;

  EXPECT_THAT(
      GetVariableInfos(handle, &variables_updated, &variable_infos),
      StatusIs(absl::StatusCode::kInvalidArgument,
               absl::StrCat("Trying to access resource v located in device ",
                            kHostCpu, " from device ", kExecutingDevice,
                            "\n Cf. https://www.tensorflow.org/xla/"
                            "known_issues#tfvariable_on_a_different_device")));
  EXPECT_EQ(device_context_->num_copies(), 0);
}

TEST_F(HostResidentVarsEnabledTest, ReadOnlyHostVariableIsStagedOnce) {
  Tensor handle =
      CreateVariable(host_cpu_, "v", test::AsTensor<float>({1.0f, 2.0f}));
  std::set<int> variables_updated;
  std::vector<VariableInfo> first_infos;
  std::vector<VariableInfo> second_infos;

  ASSERT_THAT(GetVariableInfos(handle, &variables_updated, &first_infos),
              IsOk());
  ASSERT_THAT(GetVariableInfos(handle, &variables_updated, &second_infos),
              IsOk());

  ASSERT_EQ(first_infos.size(), 1);
  ASSERT_EQ(second_infos.size(), 1);
  EXPECT_EQ(second_infos[0].var(), first_infos[0].var());
  EXPECT_TRUE(second_infos[0].read_only());
  test::ExpectTensorEqual<float>(*second_infos[0].var()->tensor(),
                                 test::AsTensor<float>({1.0f, 2.0f}));
  EXPECT_EQ(device_context_->num_copies(), 1);
}

TEST_F(HostResidentVarsEnabledTest, UninitializedHostVariableFails) {
  Tensor handle = CreateVariable(host_cpu_, "v", /*value=*/std::nullopt);
  std::set<int> variables_updated;
  std::vector<VariableInfo> variable_infos;

  EXPECT_THAT(GetVariableInfos(handle, &variables_updated, &variable_infos),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("is not initialized")));
  EXPECT_EQ(device_context_->num_copies(), 0);
}

TEST_F(HostResidentVarsEnabledTest, UpdatedHostVariableIsRejected) {
  Tensor handle = CreateVariable(host_cpu_, "v", test::AsScalar<float>(1.0f));
  std::set<int> variables_updated = {0};
  std::vector<VariableInfo> variable_infos;

  EXPECT_THAT(GetVariableInfos(handle, &variables_updated, &variable_infos),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Trying to access resource v located "
                                       "in device"),
                             HasSubstr("updated by the cluster cannot be "
                                       "staged"))));
  EXPECT_EQ(device_context_->num_copies(), 0);
}

TEST_F(HostResidentVarsEnabledTest, AcceleratorVariableIsRejected) {
  Tensor handle =
      CreateVariable(accelerator_, "v", test::AsScalar<float>(1.0f));
  std::set<int> variables_updated;
  std::vector<VariableInfo> variable_infos;

  EXPECT_THAT(GetVariableInfos(handle, &variables_updated, &variable_infos),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("only host-resident variables can be "
                                 "staged")));
  EXPECT_EQ(device_context_->num_copies(), 0);
}

}  // namespace
}  // namespace tensorflow
