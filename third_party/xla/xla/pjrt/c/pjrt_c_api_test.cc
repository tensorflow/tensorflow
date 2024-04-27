/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/c/pjrt_c_api_test.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_test_base.h"
#include "xla/pjrt/compile_options.pb.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"

namespace pjrt {
namespace {

// Serialized `ModuleOp` that does add 1.
constexpr absl::string_view module_add_one =
    R"(module {
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.copy"(%arg0) : (tensor<f32>) -> tensor<f32>
  %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = mhlo.add %0, %1 : tensor<f32>
  return %2 : tensor<f32>
}})";

// HLO sample code from go/hlo-text
constexpr absl::string_view kHloString =
    R"(
HloModule TupleCreate_module:
ENTRY %TupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3)
}
)";

class TestCApiFactory {
 public:
  void Register(std::function<const PJRT_Api*()> factory,
                absl::string_view platform_name) {
    absl::MutexLock lock(&mu_);
    CHECK(!factory_);
    factory_ = std::move(factory);
    CHECK(platform_name_.empty()) << "Platform name already provided";
    CHECK(!platform_name.empty()) << "Provided platform name is empty";
    platform_name_ = platform_name;
  }

  std::function<const PJRT_Api*()> Get() const {
    absl::MutexLock lock(&mu_);
    CHECK(factory_) << "Test didn't call RegisterPjRtCApiTestFactory()";
    return factory_;
  }

  std::string GetPlatformName() const {
    absl::MutexLock lock(&mu_);
    CHECK(!platform_name_.empty())
        << "Test didn't call RegisterPjRtCApiTestFactory()";
    return platform_name_;
  }

 private:
  mutable absl::Mutex mu_;
  std::function<const PJRT_Api*()> factory_ ABSL_GUARDED_BY(mu_);
  std::string platform_name_;
};

TestCApiFactory& GetGlobalTestCApiFactory() {
  static auto* const factory = new TestCApiFactory;
  return *factory;
}

const PJRT_Api* GetCApi() { return GetGlobalTestCApiFactory().Get()(); }

std::string GetPlatformName() {
  return GetGlobalTestCApiFactory().GetPlatformName();
}

}  // namespace

void RegisterPjRtCApiTestFactory(std::function<const PJRT_Api*()> factory,
                                 absl::string_view platform_name) {
  GetGlobalTestCApiFactory().Register(std::move(factory), platform_name);
}

namespace {

class PjrtCApiTest : public PjrtCApiTestBase {
 protected:
  PjrtCApiTest() : PjrtCApiTestBase(GetCApi()) {}
  std::string platform_name_ = GetPlatformName();
};

// -------------------------------- API Version --------------------------------

TEST_F(PjrtCApiTest, ApiVersion) {
  CHECK_EQ(api_->pjrt_api_version.major_version, PJRT_API_MAJOR);
  CHECK_EQ(api_->pjrt_api_version.minor_version, PJRT_API_MINOR);
}

// ---------------------------------- Client -----------------------------------

TEST_F(PjrtCApiTest, PlatformName) {
  PJRT_Client_PlatformName_Args args;
  args.client = client_;
  args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  PJRT_Error* error = api_->PJRT_Client_PlatformName(&args);
  ASSERT_EQ(error, nullptr);
  absl::string_view platform_name(args.platform_name, args.platform_name_size);
  ASSERT_EQ(platform_name_, platform_name);
}

TEST_F(PjrtCApiTest, ClientProcessIndex) {
  PJRT_Client_ProcessIndex_Args process_index_args =
      PJRT_Client_ProcessIndex_Args{
          .struct_size = PJRT_Client_ProcessIndex_Args_STRUCT_SIZE,
          .extension_start = nullptr,
          .client = client_,
          .process_index = -1,
      };
  PJRT_Error* error = api_->PJRT_Client_ProcessIndex(&process_index_args);
  CHECK_EQ(error, nullptr);

  // Single-process test should return 0
  CHECK_EQ(process_index_args.process_index, 0);
}

TEST_F(PjrtCApiTest, ClientDevices) {
  absl::Span<PJRT_Device* const> devices = GetClientDevices();

  ASSERT_FALSE(devices.empty());
  for (auto& device : devices) {
    ASSERT_TRUE(this->IsValidDeviceId(device));
  }
}

TEST_F(PjrtCApiTest, ClientAddressableDevices) {
  absl::Span<PJRT_Device* const> addressable_devices =
      GetClientAddressableDevices();

  ASSERT_FALSE(addressable_devices.empty());
  for (auto& device : addressable_devices) {
    ASSERT_TRUE(this->IsValidDeviceId(device));
  }

  absl::Span<PJRT_Device* const> client_devices = GetClientDevices();
  for (auto& addressable_device : addressable_devices) {
    ASSERT_THAT(client_devices, ::testing::Contains(addressable_device));
  }
}

TEST_F(PjrtCApiTest, LookupDevice) {
  PJRT_Client_LookupDevice_Args lookup_device_args =
      PJRT_Client_LookupDevice_Args{
          .struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE,
          .extension_start = nullptr,
          .client = client_,
          .id = 0,
          .device = nullptr,
      };

  PJRT_Error* lookup_device_error =
      api_->PJRT_Client_LookupDevice(&lookup_device_args);

  ASSERT_EQ(lookup_device_error, nullptr);
  int id = GetDeviceId(lookup_device_args.device);
  ASSERT_EQ(id, 0);
}

TEST_F(PjrtCApiTest, LookupAddressableDevice) {
  PJRT_Client_LookupAddressableDevice_Args lookup_addressable_device_args =
      PJRT_Client_LookupAddressableDevice_Args{
          .struct_size = PJRT_Client_LookupAddressableDevice_Args_STRUCT_SIZE,
          .extension_start = nullptr,
          .client = client_,
          .local_hardware_id = 0,
          .addressable_device = nullptr,
      };

  PJRT_Error* lookup_addressable_device_error =
      api_->PJRT_Client_LookupAddressableDevice(
          &lookup_addressable_device_args);

  ASSERT_EQ(lookup_addressable_device_error, nullptr);
  int local_hardware_id =
      GetLocalHardwareId(lookup_addressable_device_args.addressable_device);
  ASSERT_EQ(local_hardware_id, 0);
}

TEST_F(PjrtCApiTest, GetDefaultDeviceAssignmentNominal) {
  constexpr int kNumReplicas = 2;
  constexpr int kNumPartitions = 1;
  std::vector<int> assignment_buffer(kNumReplicas * kNumPartitions);
  PJRT_Client_DefaultDeviceAssignment_Args args{
      .struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client_,
      .num_replicas = kNumReplicas,
      .num_partitions = kNumPartitions,
      .default_assignment_size = assignment_buffer.size(),
      .default_assignment = assignment_buffer.data(),  // in-out
  };
  auto error = ToUniquePtr(api_->PJRT_Client_DefaultDeviceAssignment(&args));
  EXPECT_EQ(error, nullptr);
}

TEST_F(PjrtCApiTest, GetDefaultDeviceAssignmentBufferTooSmall) {
  constexpr int kNumReplicas = 4;
  constexpr int kNumPartitions = 2;
  constexpr size_t kBufferSize = 7;
  std::vector<int> assignment_buffer(kBufferSize);
  PJRT_Client_DefaultDeviceAssignment_Args args{
      .struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client_,
      .num_replicas = kNumReplicas,
      .num_partitions = kNumPartitions,
      .default_assignment_size = assignment_buffer.size(),
      .default_assignment = assignment_buffer.data(),  // in-out
  };
  auto error = ToUniquePtr(api_->PJRT_Client_DefaultDeviceAssignment(&args));
  ASSERT_NE(error, nullptr);
  absl::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(status.message(),
            "PJRT_Client_DefaultDeviceAssignment: `default_assignment_size` 7"
            " < `num_replicas * num_partitions`, 4 * 2 = 8");
}

TEST_F(PjrtCApiTest, LookupDeviceNegativeId) {
  PJRT_Client_LookupDevice_Args args = PJRT_Client_LookupDevice_Args{
      .struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client_,
      .id = -1,
      .device = nullptr,
  };
  absl::Status expected =
      absl::Status(absl::StatusCode::kInvalidArgument,
                   "No matching device found for device_id -1");

  auto error = ToUniquePtr(api_->PJRT_Client_LookupDevice(&args));

  ASSERT_NE(error, nullptr);
  absl::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
  ASSERT_EQ(status, expected);
}

TEST_F(PjrtCApiTest, LookupDeviceOutOfRangeId) {
  int out_of_range_id = GetNumDevices();
  PJRT_Client_LookupDevice_Args args = PJRT_Client_LookupDevice_Args{
      .struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client_,
      .id = out_of_range_id,
      .device = nullptr,
  };
  absl::Status expected = absl::Status(
      absl::StatusCode::kInvalidArgument,
      absl::StrCat("No matching device found for device_id ", out_of_range_id));

  auto error = ToUniquePtr(api_->PJRT_Client_LookupDevice(&args));

  ASSERT_NE(error, nullptr);
  absl::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
  ASSERT_EQ(status, expected);
}

static constexpr std::string_view kExecutableName = "operation";

void destroy_executable(PJRT_LoadedExecutable* executable,
                        const PJRT_Api* api) {
  PJRT_LoadedExecutable_Destroy_Args args{
      .struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .executable = executable,
  };
  PJRT_Error* error = api->PJRT_LoadedExecutable_Destroy(&args);
  CHECK_EQ(error, nullptr);
}

TEST_F(PjrtCApiTest, BufferTransferImmutableUntilTransferCompletes) {
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithType<float>({4});
  std::vector<float> float_data(4);
  std::iota(float_data.begin(), float_data.end(), 41.0f);

  PJRT_Client_BufferFromHostBuffer_Args args = CreateBufferFromHostBufferArgs(
      float_data, shape,
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes);

  PJRT_Error* error = api_->PJRT_Client_BufferFromHostBuffer(&args);
  CHECK_EQ(error, nullptr);

  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer(
      args.buffer, ::pjrt::MakeBufferDeleter(api_));

  std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> event(
      args.done_with_host_buffer, ::pjrt::MakeEventDeleter(api_));

  PJRT_Event_Await_Args await_args;
  await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  await_args.extension_start = nullptr;
  await_args.event = event.get();
  PJRT_Error* event_error = api_->PJRT_Event_Await(&await_args);
  ASSERT_EQ(event_error, nullptr);
}

TEST_F(PjrtCApiTest, Compile) {
  PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
      .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client_,
  };
  std::string options_str = BuildSingleDeviceCompileOptionStr();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  std::string format(::pjrt::kMlirFormat);
  std::string program_code{module_add_one};
  PJRT_Program program = PJRT_Program{
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .extension_start = nullptr,
      .code = program_code.data(),
      .code_size = program_code.length(),
      .format = format.c_str(),
      .format_size = format.size(),
  };
  args.program = &program;

  PJRT_Error* error = api_->PJRT_Client_Compile(&args);
  ::pjrt::LogFatalIfPjrtError(error, api_);

  ASSERT_EQ(error, nullptr);
  destroy_executable(args.executable, api_);
}

TEST_F(PjrtCApiTest, CompileXlaComputation) {
  PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
      .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client_,
  };
  xla::DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = 0;
  xla::DeviceAssignmentProto proto;
  ASSERT_TRUE(device_assignment.Serialize(&proto).ok());
  std::string device_assignment_str = proto.SerializeAsString();
  std::string options_str = BuildSingleDeviceCompileOptionStr();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  absl::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module =
      xla::ParseAndReturnUnverifiedModule(kHloString);
  ASSERT_EQ(hlo_module.ok(), true);
  std::string module_str = hlo_module->get()->ToProto().SerializeAsString();

  std::string format(::pjrt::kHloFormat);
  PJRT_Program program = PJRT_Program{
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .extension_start = nullptr,
      .code = module_str.data(),
      .code_size = module_str.size(),
      .format = format.c_str(),
      .format_size = format.size(),
  };
  args.program = &program;

  PJRT_Error* error = api_->PJRT_Client_Compile(&args);
  ::pjrt::LogFatalIfPjrtError(error, api_);

  ASSERT_EQ(error, nullptr);
  destroy_executable(args.executable, api_);
}

TEST_F(PjrtCApiTest, CompileInvalidOption) {
  PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
      .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client_,
  };
  std::string options_str = "invalid compile options";
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  std::string format(::pjrt::kMlirFormat);
  std::string program_code{module_add_one};
  PJRT_Program program = PJRT_Program{
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .extension_start = nullptr,
      .code = program_code.data(),
      .code_size = program_code.length(),
      .format = format.c_str(),
      .format_size = format.size(),
  };
  args.program = &program;

  PJRT_Error* error = api_->PJRT_Client_Compile(&args);

  absl::Status status = ::pjrt::PjrtErrorToStatus(error, api_);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "PJRT_Client_Compile: failed to deserialize CompileOptionsProto");
  destroy_executable(args.executable, api_);
  ::pjrt::MakeErrorDeleter(api_)(error);
}

TEST_F(PjrtCApiTest, CompileInvalidProgramFormat) {
  PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
      .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client_,
  };
  xla::DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = 0;
  xla::DeviceAssignmentProto proto;
  ASSERT_TRUE(device_assignment.Serialize(&proto).ok());
  std::string device_assignment_str = proto.SerializeAsString();
  std::string options_str = BuildSingleDeviceCompileOptionStr();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  std::string format("invalid");
  PJRT_Program program = PJRT_Program{
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .extension_start = nullptr,
      .code = nullptr,
      .code_size = 0,
      .format = format.c_str(),
      .format_size = format.size(),
  };
  args.program = &program;

  PJRT_Error* error = api_->PJRT_Client_Compile(&args);
  absl::Status status = ::pjrt::PjrtErrorToStatus(error, api_);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "Unknown program format 'invalid'.");
  destroy_executable(args.executable, api_);
  ::pjrt::MakeErrorDeleter(api_)(error);
}

// --------------------------------- Devices -----------------------------------

TEST_F(PjrtCApiTest, DeviceId) {
  auto* device = GetClientDevices()[0];

  int id = GetDeviceId(device);

  CHECK_EQ(id, 0);
}

TEST_F(PjrtCApiTest, DeviceProcessIndex) {
  PJRT_DeviceDescription_ProcessIndex_Args args =
      PJRT_DeviceDescription_ProcessIndex_Args{
          .struct_size = PJRT_DeviceDescription_ProcessIndex_Args_STRUCT_SIZE,
          .extension_start = nullptr,
          .device_description =
              ::pjrt::GetDeviceDescription(api_, GetClientDevices()[0]),
          .process_index = -1,
      };
  PJRT_Error* error = api_->PJRT_DeviceDescription_ProcessIndex(&args);
  ASSERT_EQ(error, nullptr);
  // For single process, it should match client process index
  CHECK_EQ(args.process_index, 0);
}

TEST_F(PjrtCApiTest, DeviceIsAddressable) {
  PJRT_Device_IsAddressable_Args args = PJRT_Device_IsAddressable_Args{
      .struct_size = PJRT_Device_IsAddressable_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .device = GetClientDevices()[0],
      .is_addressable = false,
  };
  PJRT_Error* error = api_->PJRT_Device_IsAddressable(&args);
  ASSERT_EQ(error, nullptr);
  // All devices are addressable in single-process test
  CHECK_EQ(args.is_addressable, true);
}

TEST_F(PjrtCApiTest, DeviceLocalHardwareId) {
  PJRT_Device_LocalHardwareId_Args args = PJRT_Device_LocalHardwareId_Args{
      .struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .device = GetClientDevices()[0],
      .local_hardware_id = -1,
  };
  PJRT_Error* error = api_->PJRT_Device_LocalHardwareId(&args);
  ASSERT_EQ(error, nullptr);
  CHECK_EQ(args.local_hardware_id, 0);
}

// ---------------------------------- Buffers ----------------------------------

class PjrtCApiBufferTest : public PjrtCApiTest {
 protected:
  void SetUp() override {
    PjrtCApiTest::SetUp();
    auto buffer_and_event = create_buffer();
    buffer_ = std::move(buffer_and_event.first);
    event_ = buffer_and_event.second;
  }

  void TearDown() override {
    // event_ need to complete before the client is destroyed; otherwise there
    // is a data race between destroying the client and trying to access the
    // host context in the client for the callback after host to device transfer
    // is completed.
    TF_CHECK_OK(event_.Await());
    // buffer_ must be destroyed before the client is destroyed or else the
    // unique_ptr for buffer_ will go out of scope causing heap-use-after-free
    // error.
    buffer_.reset(nullptr);
    PjrtCApiTest::TearDown();
  }

  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer_;
  xla::PjRtFuture<> event_;
};

TEST_F(PjrtCApiBufferTest, IsDeleted) {
  PJRT_Buffer_IsDeleted_Args is_deleted_args;
  is_deleted_args.struct_size = PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
  is_deleted_args.extension_start = nullptr;
  is_deleted_args.buffer = buffer_.get();
  PJRT_Error* is_deleted_error = api_->PJRT_Buffer_IsDeleted(&is_deleted_args);
  ASSERT_EQ(is_deleted_error, nullptr);
  ASSERT_FALSE(is_deleted_args.is_deleted);

  PJRT_Buffer_Delete_Args delete_args;
  delete_args.struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE;
  delete_args.extension_start = nullptr;
  delete_args.buffer = buffer_.get();
  PJRT_Error* delete_error = api_->PJRT_Buffer_Delete(&delete_args);
  ASSERT_EQ(delete_error, nullptr);

  is_deleted_error = api_->PJRT_Buffer_IsDeleted(&is_deleted_args);
  ASSERT_EQ(is_deleted_error, nullptr);
  ASSERT_TRUE(is_deleted_args.is_deleted);
}

TEST_F(PjrtCApiBufferTest, GetOnDeviceSizeInBytes) {
  PJRT_Buffer_OnDeviceSizeInBytes_Args args;
  args.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  PJRT_Error* on_device_size_bytes_error =
      api_->PJRT_Buffer_OnDeviceSizeInBytes(&args);

  ASSERT_EQ(on_device_size_bytes_error, nullptr);
  ASSERT_GT(args.on_device_size_in_bytes, 0);
}

TEST_F(PjrtCApiBufferTest, ReadyEvent) {
  PJRT_Buffer_ReadyEvent_Args get_event_args;
  get_event_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
  get_event_args.extension_start = nullptr;
  get_event_args.buffer = buffer_.get();
  auto error = ToUniquePtr(api_->PJRT_Buffer_ReadyEvent(&get_event_args));
  ASSERT_EQ(error, nullptr);

  PJRT_Event* event = get_event_args.event;
  ASSERT_NE(event, nullptr);

  // Wait for `buffer_`'s data transfer to complete (if it hasn't already)
  PJRT_Event_Await_Args await_args;
  await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  await_args.extension_start = nullptr;
  await_args.event = event;
  error.reset(api_->PJRT_Event_Await(&await_args));
  ASSERT_EQ(error, nullptr);

  // Must be ready when `PJRT_Event_Await` completes
  PJRT_Event_IsReady_Args ready_args;
  ready_args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
  ready_args.extension_start = nullptr;
  ready_args.event = event;
  error.reset(api_->PJRT_Event_IsReady(&ready_args));
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(ready_args.is_ready);

  // Clean up
  PJRT_Event_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.event = event;
  error.reset(api_->PJRT_Event_Destroy(&destroy_args));
  EXPECT_EQ(error, nullptr);
}

TEST_F(PjrtCApiBufferTest, ToHostBufferNoHostLayout) {
  PJRT_Buffer_ToHostBuffer_Args args;
  args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.src = buffer_.get();
  xla::Shape host_shape = xla::ShapeUtil::MakeShape(xla::F32, {4});
  auto literal = std::make_shared<xla::Literal>(host_shape);
  args.host_layout = nullptr;
  args.dst = literal->untyped_data();
  args.dst_size = xla::ShapeUtil::ByteSizeOfElements(host_shape);
  args.event = nullptr;

  PJRT_Error* error = api_->PJRT_Buffer_ToHostBuffer(&args);
  xla::PjRtFuture<> transfer_to_host =
      ::pjrt::ConvertCEventToCppFuture(args.event, api_);
  TF_CHECK_OK(transfer_to_host.Await());

  EXPECT_EQ(error, nullptr);
  ASSERT_EQ(literal->data<float>().size(), 4);
  std::vector<float> float_data(4);
  std::iota(float_data.begin(), float_data.end(), 41.0f);
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      xla::LiteralUtil::CreateR1<float>(float_data), *literal));
}

TEST_F(PjrtCApiBufferTest, IncreaseAndDecreaseReferenceCount) {
  PJRT_Buffer_IncreaseExternalReferenceCount_Args increase_reference_count_args;
  increase_reference_count_args.struct_size =
      PJRT_Buffer_IncreaseExternalReferenceCount_Args_STRUCT_SIZE;
  increase_reference_count_args.extension_start = nullptr;
  increase_reference_count_args.buffer = buffer_.get();
  PJRT_Error* increase_reference_count_error =
      api_->PJRT_Buffer_IncreaseExternalReferenceCount(
          &increase_reference_count_args);
  EXPECT_EQ(increase_reference_count_error, nullptr);

  PJRT_Buffer_DecreaseExternalReferenceCount_Args decrease_reference_count_args;
  decrease_reference_count_args.struct_size =
      PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE;
  decrease_reference_count_args.extension_start = nullptr;
  decrease_reference_count_args.buffer = buffer_.get();
  PJRT_Error* decrease_reference_error =
      api_->PJRT_Buffer_DecreaseExternalReferenceCount(
          &decrease_reference_count_args);
  EXPECT_EQ(decrease_reference_error, nullptr);
}

TEST_F(PjrtCApiBufferTest, DecreaseReferenceCountReturnsError) {
  PJRT_Buffer_DecreaseExternalReferenceCount_Args args;
  args.struct_size =
      PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  auto error =
      ToUniquePtr(api_->PJRT_Buffer_DecreaseExternalReferenceCount(&args));
  ASSERT_NE(error, nullptr);
  absl::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "Attempting to decrease reference on a buffer with zero reference "
            "count.");
}

TEST_F(PjrtCApiBufferTest, OpaqueDeviceMemoryDataPointer) {
  PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args args;
  args.struct_size = PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  PJRT_Error* error = api_->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&args);
  EXPECT_EQ(error, nullptr);
  EXPECT_NE(args.device_memory_ptr, nullptr);
}

// --------------------------------- Helpers -----------------------------------

class PjrtCommonCApiHelpersTest : public PjrtCApiTest {};

TEST_F(PjrtCommonCApiHelpersTest, PjrtErrorToStatus) {
  // Return success if nullptr
  EXPECT_TRUE(::pjrt::PjrtErrorToStatus(nullptr, api_).ok());
}

}  // namespace
}  // namespace pjrt
