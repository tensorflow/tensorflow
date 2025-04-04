// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/server/ifrt_backend.h"

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/program_serdes.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/common/array_util.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/host_callback.h"
#include "xla/python/ifrt_proxy/server/version.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/status_to_from_proto.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::_;
using ::testing::ByMove;
using ::testing::DoAll;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::Optional;
using ::testing::Pointee;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::tsl::protobuf::TextFormat;
using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

#if defined(PLATFORM_GOOGLE)
using ::testing::EquivToProto;
using ::testing::proto::IgnoringRepeatedFieldOrdering;
using ::testing::proto::Partially;
#endif

constexpr uint64_t kSessionId = 12345;

class IfrtBackendTest
    : public ::testing::TestWithParam</*protocol_version=*/int> {
 protected:
  IfrtProxyVersion Version() {
    IfrtProxyVersion version;
    version.set_protocol_version(GetParam());
    return version;
  }
};

// Makes an empty request with the given op_id. Does not fail.
std::unique_ptr<IfrtRequest> NewIfrtRequest(uint64_t op_id) {
  auto ifrt_request = std::make_unique<IfrtRequest>();
  auto* request_metadata = ifrt_request->mutable_request_metadata();
  request_metadata->set_op_id(op_id);
  return ifrt_request;
}

TEST_P(IfrtBackendTest, CreationFailsWithNullIfrtClient) {
  EXPECT_THAT(IfrtBackend::Create(Version(), kSessionId, nullptr, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(IfrtBackendTest, SuccessfulCreation) {
  auto ifrt_client = std::make_unique<MockClient>();
  ASSERT_THAT(IfrtBackend::Create(Version(), kSessionId, std::move(ifrt_client),
                                  std::make_shared<HostBufferStore>()),
              IsOk());
}

TEST_P(IfrtBackendTest, ShutdownSucceeds) {
  auto ifrt_client = std::make_unique<MockClient>();
  TF_ASSERT_OK_AND_ASSIGN(
      auto ifrt_backend,
      IfrtBackend::Create(Version(), kSessionId, std::move(ifrt_client),
                          std::make_shared<HostBufferStore>()));
}

TEST_P(IfrtBackendTest, ProcessFailsWithNoRequestSet) {
  auto ifrt_client = std::make_unique<MockClient>();
  TF_ASSERT_OK_AND_ASSIGN(
      auto ifrt_backend,
      IfrtBackend::Create(Version(), kSessionId, std::move(ifrt_client),
                          std::make_shared<HostBufferStore>()));

  // Make a new request but leave the `OneOf` `request` field unset. And, that
  // should fail the Process call.
  auto request = std::make_unique<IfrtRequest>();
  auto process_status = ifrt_backend->Process(std::move(request)).Await();
  ASSERT_THAT(process_status, Not(IsOk()));
}

INSTANTIATE_TEST_SUITE_P(
    IfrtBackendTestWithAllVersions, IfrtBackendTest,
    testing::Range(kServerMinVersion, kServerMaxVersion + 1),
    [](const testing::TestParamInfo<IfrtBackendTest::ParamType>& info) {
      return absl::StrCat(info.param);
    });

struct TestProgram : llvm::RTTIExtends<TestProgram, Program> {
  static char ID;  // NOLINT
};

[[maybe_unused]] char TestProgram::ID = 0;  // NOLINT

class TestProgramSerDes : public llvm::RTTIExtends<TestProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::proxy::TestProgram";
  }

  absl::StatusOr<std::string> Serialize(
      Serializable& serializable, std::unique_ptr<SerializeOptions>) override {
    CHECK(llvm::isa<TestProgram>(serializable));
    return "";
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    const auto* deserialize_program_options =
        llvm::cast<DeserializeProgramOptions>(options.get());
    CHECK_OK(deserialize_program_options->client->LookupDevice(DeviceId(0)));

    return std::make_unique<TestProgram>();
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char TestProgramSerDes::ID = 0;  // NOLINT

struct TestCompileOptions
    : llvm::RTTIExtends<TestCompileOptions, CompileOptions> {
  static char ID;  // NOLINT
};

[[maybe_unused]] char TestCompileOptions::ID = 0;  // NOLINT

class TestCompileOptionsSerDes
    : public llvm::RTTIExtends<TestCompileOptionsSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::proxy::TestCompileOptions";
  }

  absl::StatusOr<std::string> Serialize(
      Serializable& serializable, std::unique_ptr<SerializeOptions>) override {
    CHECK(llvm::isa<TestCompileOptions>(serializable));
    return "";
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    return std::make_unique<TestCompileOptions>();
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char TestCompileOptionsSerDes::ID = 0;  // NOLINT

class IfrtBackendHandlerTest : public IfrtBackendTest {
 protected:
  static void SetUpTestSuite() {
    RegisterSerDes<TestProgram>(std::make_unique<TestProgramSerDes>());
    RegisterSerDes<TestCompileOptions>(
        std::make_unique<TestCompileOptionsSerDes>());
  }

  void SetUp() override {
    auto mock_client = std::make_unique<xla::ifrt::MockClient>();

    std::vector<xla::ifrt::Device*> raw_device_ptrs;
    for (int i = 0; i < 2; ++i) {
      auto mock_device = std::make_unique<xla::ifrt::MockDevice>();
      ON_CALL(*mock_device, client()).WillByDefault(Return(mock_client.get()));
      ON_CALL(*mock_device, Id()).WillByDefault(Return(DeviceId(i)));
      ON_CALL(*mock_device, IsAddressable()).WillByDefault(Return(true));
      raw_device_ptrs.push_back(mock_device.get());
      mock_devices_.push_back(std::move(mock_device));
    }

    ON_CALL(*mock_client, devices()).WillByDefault(Return(raw_device_ptrs));
    ON_CALL(*mock_client, GetAllDevices())
        .WillByDefault(Return(raw_device_ptrs));
    ON_CALL(*mock_client, LookupDevice(_))
        .WillByDefault(
            Invoke([this](DeviceId id) -> absl::StatusOr<xla::ifrt::Device*> {
              if (id.value() < 0 || id.value() >= mock_devices_.size()) {
                return absl::NotFoundError(
                    absl::StrCat("Unknown device id: ", id.value()));
              }
              return mock_devices_[id.value()].get();
            }));
    ON_CALL(*mock_client, MakeDeviceList(_))
        .WillByDefault([](absl::Span<xla::ifrt::Device* const> devices) {
          return xla::ifrt::BasicDeviceList::Create(devices);
        });

    // Remembering a raw pointer to the mock client here is OK, since most tests
    // anyway have to make the basic and tacit assumption that the backend will
    // call into the mock client --and thus keep it alive-- for the duration of
    // the test.
    mock_client_ = mock_client.get();

    EXPECT_CALL(*mock_client_, GetDefaultCompiler)
        .WillRepeatedly(Return(&mock_compiler_));

    host_buffer_store_ = std::make_shared<HostBufferStore>();
    TF_ASSERT_OK_AND_ASSIGN(
        backend_,
        IfrtBackend::Create(Version(), kSessionId, std::move(mock_client),
                            host_buffer_store_));
  }

  absl::StatusOr<std::shared_ptr<IfrtResponse>> CallBackend(
      std::unique_ptr<IfrtRequest> request) {
    auto response_future = backend_->Process(std::move(request));
    return std::move(response_future).Await();
  }

  uint64_t NewOpId() {
    absl::MutexLock lock(&mu_);
    return current_op_id_++;
  }

  uint64_t NewHostBufferHandle() { return current_host_buffer_handle_++; }

  // Utility method to set up a given MockArray (in the backend) that can then
  // be the target of the other Array-specific methods. Returns the array
  // handle.
  absl::StatusOr<uint64_t> MakeTestArray(tsl::RCReference<Array> mock_array) {
    EXPECT_CALL(*mock_client_, MakeArrayFromHostBuffer(_, _, _, _, _, _, _, _))
        .WillOnce(Return(std::move(mock_array)));

    auto ifrt_request = NewIfrtRequest(NewOpId());
    {
      const uint64_t host_buffer_handle = NewHostBufferHandle();
      TF_RETURN_IF_ERROR(
          host_buffer_store_->Store(host_buffer_handle, "01234567"));

      auto* make_array =
          ifrt_request->mutable_make_array_from_host_buffer_request();
      make_array->mutable_dtype()->set_kind(DTypeProto::KIND_S32);
      make_array->mutable_shape()->add_dims(2);
      make_array->set_host_buffer_handle(host_buffer_handle);

      TF_ASSIGN_OR_RETURN(auto* device,
                          mock_client_->LookupDevice(DeviceId(1)));
      TF_ASSIGN_OR_RETURN(
          *make_array->mutable_sharding(),
          SingleDeviceSharding::Create(device, MemoryKind())->ToProto());
    }
    TF_ASSIGN_OR_RETURN(auto make_array_response,
                        CallBackend(std::move(ifrt_request)));

    TF_RETURN_IF_ERROR(tsl::StatusFromProto(
        make_array_response->response_metadata().status()));
    return make_array_response->make_array_from_host_buffer_response()
        .array_handle();
  }

  absl::StatusOr<CompileResponse> CompileTestLoadedExecutable(
      absl::StatusOr<std::unique_ptr<LoadedExecutable>> loaded_executable) {
    auto request = NewIfrtRequest(NewOpId());
    CompileRequest* compile_request = request->mutable_compile_request();
    TestProgram program;
    TF_ASSIGN_OR_RETURN(*compile_request->mutable_program(),
                        Serialize(program, /*options=*/nullptr));
    TestCompileOptions compile_options;
    TF_ASSIGN_OR_RETURN(*compile_request->mutable_compile_options(),
                        Serialize(compile_options, /*options=*/nullptr));

    EXPECT_CALL(mock_compiler_, Compile(_, _))
        .WillOnce(Return(ByMove(std::move(loaded_executable))));

    TF_ASSIGN_OR_RETURN(std::shared_ptr<IfrtResponse> response,
                        CallBackend(std::move(request)));

    TF_RET_CHECK(response->has_compile_response());
    return response->compile_response();
  }

  absl::Status CheckFuture(uint64_t handle) {
    if (handle == 0) {
      return absl::InternalError("Test error, future handle is 0");
    }
    auto request = NewIfrtRequest(NewOpId());
    request->mutable_check_future_request()->set_future_handle(handle);
    TF_ASSIGN_OR_RETURN(std::shared_ptr<IfrtResponse> response,
                        CallBackend(std::move(request)));
    return tsl::StatusFromProto(response->response_metadata().status());
  }

  absl::Status CheckValueReady(uint64_t handle) {
    if (handle == 0) {
      return absl::InternalError("Test error, future handle is 0");
    }
    auto request = NewIfrtRequest(NewOpId());
    request->mutable_check_value_ready_request()->add_value_handles(handle);
    TF_ASSIGN_OR_RETURN(std::shared_ptr<IfrtResponse> response,
                        CallBackend(std::move(request)));
    return tsl::StatusFromProto(response->response_metadata().status());
  }

  xla::ifrt::MockClient* mock_client_;
  xla::ifrt::MockCompiler mock_compiler_;
  std::vector<std::unique_ptr<xla::ifrt::MockDevice>> mock_devices_;
  std::shared_ptr<HostBufferStore> host_buffer_store_;

 private:
  absl::Mutex mu_;
  uint64_t current_op_id_ ABSL_GUARDED_BY(mu_) = 1;
  uint64_t current_host_buffer_handle_ = 1;

  std::unique_ptr<IfrtBackend> backend_;
};

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_P(IfrtBackendHandlerTest, Init) {
  EXPECT_CALL(*mock_client_, platform_name())
      .WillRepeatedly(Return("ifrt_backend"));
  EXPECT_CALL(*mock_client_, platform_version()).WillRepeatedly(Return("n/a"));
  EXPECT_CALL(*mock_client_, platform_id()).WillRepeatedly(Return(42));
  EXPECT_CALL(*mock_client_, process_index()).WillRepeatedly(Return(1));
  EXPECT_CALL(*mock_client_, runtime_type())
      .WillRepeatedly(Return("ifrt-service"));

  std::vector<std::vector<xla::ifrt::Device*>> mock_memory_devices;
  mock_memory_devices.reserve(mock_devices_.size());
  for (const auto& mock_device : mock_devices_) {
    mock_memory_devices.push_back({mock_device.get()});
  }

  std::vector<MockMemory> mock_memories(mock_devices_.size());
  MemoryKind kind("mock");
  for (int i = 0; i < mock_memories.size(); ++i) {
    MockMemory& memory = mock_memories[i];
    EXPECT_CALL(memory, Devices())
        .WillRepeatedly(Return(mock_memory_devices[i]));
    EXPECT_CALL(memory, Id()).WillRepeatedly(Return(MemoryId(i)));
    EXPECT_CALL(memory, Kind()).WillRepeatedly(ReturnRef(kind));
  }

  std::vector<std::vector<Memory*>> device_memories;
  device_memories.reserve(mock_devices_.size());
  for (int i = 0; i < mock_devices_.size(); ++i) {
    device_memories.push_back({&mock_memories[i]});
  }

  std::vector<AttributeMap> device_attributes;
  device_attributes.reserve(mock_devices_.size());

  for (int i = 0; i < mock_devices_.size(); ++i) {
    AttributeMap::Map map;
    map.insert({"name", AttributeMap::StringValue(absl::StrCat("device", i))});
    device_attributes.push_back(AttributeMap(std::move(map)));

    MockDevice& mock_device = *mock_devices_[i];
    // TODO(b/314368788): Clean up PJRT device ID APIs.
    EXPECT_CALL(mock_device, Kind()).WillRepeatedly(Return("mock"));
    EXPECT_CALL(mock_device, Memories())
        .WillRepeatedly(Return(device_memories[i]));
    EXPECT_CALL(mock_device, DefaultMemory())
        .WillRepeatedly(Return(&mock_memories[i]));
    EXPECT_CALL(mock_device, Attributes())
        .WillRepeatedly(ReturnRef(device_attributes[i]));
  }

  auto request = NewIfrtRequest(NewOpId());
  request->mutable_init_request();

  if (Version().protocol_version() <= 3) {
    EXPECT_THAT(CallBackend(std::move(request)),
                IsOkAndHolds(Pointee(
                    Partially(IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                      init_response {
                        session_id: 12345
                        platform_name: "ifrt_backend"
                        platform_version: "n/a"
                        platform_id: 42
                        process_index: 1
                        runtime_type: "ifrt-service"
                        all_devices {
                          id: 0
                          device_kind: "mock"
                          default_memory_id: 0
                          memory_ids: [ 0 ]
                          deprecated_attributes {
                            key: "name"
                            value { string_value: "device0" }
                          }
                        }
                        all_devices {
                          id: 1
                          device_kind: "mock"
                          default_memory_id: 1
                          memory_ids: [ 1 ]
                          deprecated_attributes {
                            key: "name"
                            value { string_value: "device1" }
                          }
                        }
                        memories {
                          id: 0
                          memory_space_kind: "mock"
                          device_ids: [ 0 ]
                        }
                        memories {
                          id: 1
                          memory_space_kind: "mock"
                          device_ids: [ 1 ]
                        }
                      }
                    )pb"))))));
  } else if (Version().protocol_version() < 7) {
    EXPECT_THAT(CallBackend(std::move(request)),
                IsOkAndHolds(Pointee(
                    Partially(IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                      init_response {
                        session_id: 12345
                        platform_name: "ifrt_backend"
                        platform_version: "n/a"
                        platform_id: 42
                        process_index: 1
                        runtime_type: "ifrt-service"
                        all_devices {
                          id: 0
                          device_kind: "mock"
                          default_memory_id: 0
                          memory_ids: [ 0 ]
                          attributes {
                            attributes {
                              key: "name"
                              value { string_value: "device0" }
                            }
                          }
                        }
                        all_devices {
                          id: 1
                          device_kind: "mock"
                          default_memory_id: 1
                          memory_ids: [ 1 ]
                          attributes {
                            attributes {
                              key: "name"
                              value { string_value: "device1" }
                            }
                          }
                        }
                        memories {
                          id: 0
                          memory_space_kind: "mock"
                          device_ids: [ 0 ]
                        }
                        memories {
                          id: 1
                          memory_space_kind: "mock"
                          device_ids: [ 1 ]
                        }
                      }
                    )pb"))))));
  } else {
    EXPECT_THAT(CallBackend(std::move(request)),
                IsOkAndHolds(Pointee(
                    Partially(IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                      init_response {
                        session_id: 12345
                        platform_name: "ifrt_backend"
                        platform_version: "n/a"
                        platform_id: 42
                        process_index: 1
                        runtime_type: "ifrt-service"
                        all_devices {
                          id: 0
                          device_kind: "mock"
                          default_memory_id: 0
                          memory_ids: [ 0 ]
                          attributes {
                            attributes {
                              key: "name"
                              value { string_value: "device0" }
                            }
                          }
                        }
                        all_devices {
                          id: 1
                          device_kind: "mock"
                          default_memory_id: 1
                          memory_ids: [ 1 ]
                          attributes {
                            attributes {
                              key: "name"
                              value { string_value: "device1" }
                            }
                          }
                        }
                        primary_device_ids: [ 0, 1 ]
                        memories {
                          id: 0
                          memory_space_kind: "mock"
                          device_ids: [ 0 ]
                        }
                        memories {
                          id: 1
                          memory_space_kind: "mock"
                          device_ids: [ 1 ]
                        }
                      }
                    )pb"))))));
  }
}
#endif

// TODO(b/282757875): Use the MockRuntime fixture to cover the error cases for
// MakeArrayFromHostBuffer and CopyToHostBuffer methods as well.

// Consider redoing the happy-path test below with PjRt CPU-only backend for
// non-SingleDeviceSharding.
TEST_P(IfrtBackendHandlerTest, DisassembleIntoSingleDeviceArraysSucceeds) {
  // Set up a mock source array that returns two single device arrays on
  // disassembly.
  std::vector<tsl::RCReference<xla::ifrt::Array>> single_device_arrays;
  single_device_arrays.push_back(tsl::MakeRef<xla::ifrt::MockArray>());
  single_device_arrays.push_back(tsl::MakeRef<xla::ifrt::MockArray>());
  tsl::RCReference<xla::ifrt::MockArray> source_mock_array =
      tsl::MakeRef<xla::ifrt::MockArray>();
  EXPECT_CALL(*source_mock_array, DisassembleIntoSingleDeviceArrays(_, _))
      .WillOnce(Return(std::move(single_device_arrays)));

  // Inject the mock_array.
  TF_ASSERT_OK_AND_ASSIGN(auto array_handle,
                          MakeTestArray(std::move(source_mock_array)));

  // Disassemble.
  auto disassemble_request = NewIfrtRequest(NewOpId());
  auto* disassemble_into_single_device_arrays =
      disassemble_request
          ->mutable_disassemble_into_single_device_arrays_request();
  disassemble_into_single_device_arrays->set_array_handle(array_handle);
  if (Version().protocol_version() >= 8) {
    disassemble_into_single_device_arrays->set_single_device_shard_semantics(
        proto::SingleDeviceShardSemantics::
            SINGLE_DEVICE_SHARD_SEMANTICS_ALL_SHARDS);
  }
  if (Version().protocol_version() >=
      protocol_version::kClientHandlesOptimization2) {
    disassemble_into_single_device_arrays->add_result_handles(1);
    disassemble_into_single_device_arrays->add_result_handles(2);
  }
  TF_ASSERT_OK_AND_ASSIGN(auto disassemble_response,
                          CallBackend(std::move(disassemble_request)));

  // We must have gotten back two handles corresponding to the two single device
  // arrays we injected.
  EXPECT_THAT(
      disassemble_response->disassemble_into_single_device_arrays_response()
          .array_handles(),
      SizeIs(2));
}

TEST_P(IfrtBackendHandlerTest, MakeArrayFromHostBufferSuccess) {
  // Given the below shape, dtype, and compact byte_strides, the size of the
  // array data needs to be 480 bytes.
  const uint64_t kHostBufferHandle = 1234;
  ASSERT_THAT(
      host_buffer_store_->Store(kHostBufferHandle, std::string(480, 'a')),
      IsOk());

  auto ifrt_request = NewIfrtRequest(NewOpId());
  {
    auto* make_array =
        ifrt_request->mutable_make_array_from_host_buffer_request();
    ASSERT_TRUE(
        TextFormat::ParseFromString(R"pb(
                                      dtype { kind: KIND_F64 }
                                      shape { dims: [ 5, 3, 4 ] }
                                      byte_strides { strides: [ 8, 40, 120 ] }
                                    )pb",
                                    make_array));
    make_array->set_host_buffer_handle(kHostBufferHandle);
    TF_ASSERT_OK_AND_ASSIGN(auto* device,
                            mock_client_->LookupDevice(DeviceId(1)));
    TF_ASSERT_OK_AND_ASSIGN(
        *make_array->mutable_sharding(),
        SingleDeviceSharding::Create(device, MemoryKind())->ToProto());
  }

  const Shape expected_shape({5, 3, 4});
  const std::vector<int64_t> expected_byte_strides_vec = {8, 40, 120};
  const std::optional<absl::Span<const int64_t>> expected_byte_strides =
      absl::Span<const int64_t>(expected_byte_strides_vec);

  tsl::RCReference<xla::ifrt::MockArray> mock_array =
      tsl::MakeRef<xla::ifrt::MockArray>();

  EXPECT_CALL(*mock_client_,
              MakeArrayFromHostBuffer(_, DType(DType::kF64), expected_shape,
                                      expected_byte_strides, _, _, _, _))
      .WillOnce(Return(std::move(mock_array)));

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(ifrt_request)));
  EXPECT_NE(response->make_array_from_host_buffer_response().array_handle(), 0);
}

TEST_P(IfrtBackendHandlerTest, MakeStringArrayFromHostBufferSuccess) {
  // Make a string host buffer.
  const std::vector<absl::Cord> input_strings = {absl::Cord("ab"),
                                                 absl::Cord("cd")};
  TF_ASSERT_OK_AND_ASSIGN(auto serialized_string_buffer,
                          SerializeStringHostBuffer(input_strings));

  const uint64_t kHostBufferHandle = 1234;
  ASSERT_THAT(
      host_buffer_store_->Store(kHostBufferHandle, *serialized_string_buffer),
      IsOk());

  auto ifrt_request = NewIfrtRequest(NewOpId());
  auto* make_array =
      ifrt_request->mutable_make_array_from_host_buffer_request();
  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            dtype { kind: KIND_STRING }
                                            shape { dims: [ 2 ] }
                                          )pb",
                                          make_array));
  make_array->set_host_buffer_handle(kHostBufferHandle);
  TF_ASSERT_OK_AND_ASSIGN(auto* device,
                          mock_client_->LookupDevice(DeviceId(1)));
  TF_ASSERT_OK_AND_ASSIGN(
      *make_array->mutable_sharding(),
      SingleDeviceSharding::Create(device, MemoryKind())->ToProto());

  const DType expected_dtype = DType(DType::kString);
  const Shape expected_shape({2});
  const std::optional<absl::Span<const int64_t>> expected_byte_strides =
      std::nullopt;

  tsl::RCReference<xla::ifrt::MockArray> mock_array =
      tsl::MakeRef<xla::ifrt::MockArray>();

  EXPECT_CALL(*mock_client_,
              MakeArrayFromHostBuffer(_, expected_dtype, expected_shape,
                                      expected_byte_strides, _, _, _, _))
      .WillOnce(Return(std::move(mock_array)));

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(ifrt_request)));

  EXPECT_NE(response->make_array_from_host_buffer_response().array_handle(), 0);
}

TEST_P(IfrtBackendHandlerTest, AssembleArrayFromSingleDeviceArrays) {
  auto ifrt_request = NewIfrtRequest(NewOpId());
  DType dtype = DType(DType::kF32);
  {
    AssembleArrayFromSingleDeviceArraysRequest* req =
        ifrt_request
            ->mutable_assemble_array_from_single_device_arrays_request();
    req->mutable_shape()->add_dims(2);
    req->mutable_shape()->add_dims(2);
    req->set_copy_semantics(proto::ARRAY_COPY_SEMANTICS_ALWAYS_COPY);
    if (Version().protocol_version() > 8) {
      req->set_single_device_shard_semantics(
          proto::SINGLE_DEVICE_SHARD_SEMANTICS_ALL_SHARDS);
    }
    if (Version().protocol_version() >=
        protocol_version::kClientHandlesOptimization2) {
      req->set_result_handle(1);
    }
    if (Version().protocol_version() >=
        protocol_version::kAssembleArrayFromSingleDeviceArraysWithDType) {
      *req->mutable_dtype() = dtype.ToProto();
    }
    TF_ASSERT_OK_AND_ASSIGN(auto* device,
                            mock_client_->LookupDevice(DeviceId(1)));
    TF_ASSERT_OK_AND_ASSIGN(
        *req->mutable_sharding(),
        SingleDeviceSharding::Create(device, MemoryKind())->ToProto());
  }

  std::vector<tsl::RCReference<xla::ifrt::MockArray>> single_device_arrays;
  for (int i = 0; i < 2; ++i) {
    auto array = tsl::MakeRef<xla::ifrt::MockArray>();
    ON_CALL(*array, dtype()).WillByDefault(Return(dtype));
    single_device_arrays.push_back(array);

    TF_ASSERT_OK_AND_ASSIGN(uint64_t array_handle, MakeTestArray(array));
    auto* assemble_array_from_single_device_arrays =
        ifrt_request
            ->mutable_assemble_array_from_single_device_arrays_request();
    assemble_array_from_single_device_arrays->add_single_device_array_handles(
        array_handle);
    if (Version().protocol_version() >= 8) {
      assemble_array_from_single_device_arrays
          ->set_single_device_shard_semantics(
              proto::SingleDeviceShardSemantics::
                  SINGLE_DEVICE_SHARD_SEMANTICS_ALL_SHARDS);
    }
    if (Version().protocol_version() >=
        protocol_version::kAssembleArrayFromSingleDeviceArraysWithDType) {
      *assemble_array_from_single_device_arrays->mutable_dtype() =
          dtype.ToProto();
    }
  }

  tsl::RCReference<xla::ifrt::MockArray> result =
      tsl::MakeRef<xla::ifrt::MockArray>();
  const Shape expected_shape({2, 2});

  EXPECT_CALL(*mock_client_, AssembleArrayFromSingleDeviceArrays(
                                 dtype, expected_shape, _,
                                 ElementsAreArray(single_device_arrays), _, _))
      .WillOnce(Return(std::move(result)));

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(ifrt_request)));
  EXPECT_NE(response->assemble_array_from_single_device_arrays_response()
                .array_handle(),
            0);
}

TEST_P(IfrtBackendHandlerTest, CopyToHostSuccess) {
  Shape shape({5, 3, 4});
  tsl::RCReference<xla::ifrt::MockArray> array =
      tsl::MakeRef<xla::ifrt::MockArray>();
  ON_CALL(*array, shape()).WillByDefault(ReturnRef(shape));
  ON_CALL(*array, dtype()).WillByDefault(Return(DType(DType::kF64)));

  TF_ASSERT_OK_AND_ASSIGN(auto array_handle, MakeTestArray(array));

  auto ifrt_request = NewIfrtRequest(NewOpId());
  auto* copy_to_host = ifrt_request->mutable_copy_to_host_buffer_request();
  ASSERT_TRUE(
      TextFormat::ParseFromString(R"pb(
                                    byte_strides { strides: [ 8, 40, 120 ] }
                                  )pb",
                                  copy_to_host));
  copy_to_host->set_array_handle(array_handle);
  const uint64_t host_buffer_handle = NewHostBufferHandle();
  copy_to_host->set_host_buffer_handle(host_buffer_handle);

  const std::vector<int64_t> expected_byte_strides_vec = {8, 40, 120};
  const std::optional<absl::Span<const int64_t>> expected_byte_strides =
      absl::Span<const int64_t>(expected_byte_strides_vec);
  EXPECT_CALL(*array, CopyToHostBuffer(_, expected_byte_strides, _))
      .WillOnce(Return(Future<>(absl::OkStatus())));

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(ifrt_request)));
  // Given the above shape, dtype, and compact byte_strides, the size of the
  // array data needs to be 480 bytes.
  EXPECT_THAT(host_buffer_store_->Lookup(host_buffer_handle),
              IsOkAndHolds(Pointee(SizeIs(480))));
}

TEST_P(IfrtBackendHandlerTest, CopyToHostSuccessWithStringArray) {
  // Make a string host buffer.
  const std::vector<absl::Cord> input_strings = {absl::Cord("ab"),
                                                 absl::Cord("cd")};
  TF_ASSERT_OK_AND_ASSIGN(auto serialized_string_buffer,
                          SerializeStringHostBuffer(input_strings));

  const uint64_t kHostBufferHandle = 1234;
  ASSERT_THAT(
      host_buffer_store_->Store(kHostBufferHandle, *serialized_string_buffer),
      IsOk());

  auto ifrt_request = NewIfrtRequest(NewOpId());
  auto* make_array =
      ifrt_request->mutable_make_array_from_host_buffer_request();
  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            dtype { kind: KIND_STRING }
                                            shape { dims: [ 2 ] }
                                          )pb",
                                          make_array));
  make_array->set_host_buffer_handle(kHostBufferHandle);
  TF_ASSERT_OK_AND_ASSIGN(auto* device,
                          mock_client_->LookupDevice(DeviceId(1)));
  TF_ASSERT_OK_AND_ASSIGN(
      *make_array->mutable_sharding(),
      SingleDeviceSharding::Create(device, MemoryKind())->ToProto());

  const DType expected_dtype = DType(DType::kString);
  const Shape expected_shape({2});
  const std::optional<absl::Span<const int64_t>> expected_byte_strides =
      std::nullopt;

  tsl::RCReference<xla::ifrt::MockArray> mock_array =
      tsl::MakeRef<xla::ifrt::MockArray>();
  ON_CALL(*mock_array, shape()).WillByDefault(ReturnRef(expected_shape));
  ON_CALL(*mock_array, dtype()).WillByDefault(Return(expected_dtype));

  ON_CALL(*mock_array, CopyToHostBuffer(_, _, _))
      .WillByDefault(Invoke(
          [input_strings = input_strings](
              void* data, std::optional<absl::Span<const int64_t>> byte_strides,
              xla::ifrt::ArrayCopySemantics semantics) {
            auto dst = static_cast<absl::Cord*>(data);
            for (int i = 0; i < input_strings.size(); ++i) {
              dst[i] = input_strings[i];
            }
            return Future<>(absl::OkStatus());
          }));

  EXPECT_CALL(*mock_client_,
              MakeArrayFromHostBuffer(_, expected_dtype, expected_shape,
                                      expected_byte_strides, _, _, _, _))
      .WillOnce(Return(std::move(mock_array)));

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(ifrt_request)));
  ASSERT_NE(response->make_array_from_host_buffer_response().array_handle(), 0);
  auto array_handle =
      response->make_array_from_host_buffer_response().array_handle();

  // Copy the contents of the array to a host buffer.
  ifrt_request = NewIfrtRequest(NewOpId());
  auto* copy_to_host = ifrt_request->mutable_copy_to_host_buffer_request();
  copy_to_host->set_array_handle(array_handle);
  const uint64_t host_buffer_handle = NewHostBufferHandle();
  copy_to_host->set_host_buffer_handle(host_buffer_handle);

  // Retrieve the serialized string buffer that when deserialized must match the
  // input strings.
  ASSERT_THAT(CallBackend(std::move(ifrt_request)), IsOk());
  TF_ASSERT_OK_AND_ASSIGN(auto serialized_string_buffer_got,
                          host_buffer_store_->Lookup(host_buffer_handle));
  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized_string_buffer_got,
      DeserializeStringHostBufferFromString(*serialized_string_buffer_got));

  EXPECT_THAT(deserialized_string_buffer_got, ElementsAreArray(input_strings));
}

TEST_P(IfrtBackendHandlerTest, CopyToHostFailsWithNonExistentArrays) {
  auto ifrt_request = NewIfrtRequest(NewOpId());
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        byte_strides { strides: [ 8, 40, 120 ] }
      )pb",
      ifrt_request->mutable_copy_to_host_buffer_request()));
  ifrt_request->mutable_copy_to_host_buffer_request()->set_array_handle(0);

  EXPECT_THAT(CallBackend(std::move(ifrt_request)),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(IfrtBackendHandlerTest,
       DisassembleIntoSingleArrayFailsWhenBackendRuntimeFails) {
  // Set up a mock source array that fails the disassembly.
  constexpr absl::string_view kDisassembleErrorMessage =
      "Some test-injected error message that is unlikely to match other error "
      "messages - 1234";
  tsl::RCReference<xla::ifrt::MockArray> source_mock_array =
      tsl::MakeRef<xla::ifrt::MockArray>();
  EXPECT_CALL(*source_mock_array, DisassembleIntoSingleDeviceArrays(_, _))
      .WillOnce(Return(absl::UnknownError(kDisassembleErrorMessage)));

  // Set up the mock client to return the source_mock_array when the test tries
  // to MakeArrayFromHostBuffer.
  TF_ASSERT_OK_AND_ASSIGN(auto array_handle,
                          MakeTestArray(std::move(source_mock_array)));

  // Disassembly must fail with the error we injected.
  auto disassemble_request = NewIfrtRequest(NewOpId());
  auto* disassemble_into_single_device_arrays =
      disassemble_request
          ->mutable_disassemble_into_single_device_arrays_request();
  disassemble_into_single_device_arrays->set_array_handle(array_handle);
  if (Version().protocol_version() >= 8) {
    disassemble_into_single_device_arrays->set_single_device_shard_semantics(
        proto::SingleDeviceShardSemantics::
            SINGLE_DEVICE_SHARD_SEMANTICS_ALL_SHARDS);
  }
  ASSERT_THAT(
      CallBackend(std::move(disassemble_request)),
      StatusIs(absl::StatusCode::kUnknown, StrEq(kDisassembleErrorMessage)));
}

// Matcher for matching the `device_list` argument of `Client::CopyArrays()`.
MATCHER_P(EqualsDeviceList, device_list, "") { return *arg == *device_list; }

TEST_P(IfrtBackendHandlerTest, CopyArrays) {
  std::vector<tsl::RCReference<xla::ifrt::Array>> src_arrays;
  src_arrays.push_back(tsl::MakeRef<xla::ifrt::MockArray>());

  std::vector<tsl::RCReference<xla::ifrt::Array>> copied_arrays;
  copied_arrays.push_back(tsl::MakeRef<xla::ifrt::MockArray>());

  BasicDeviceList::Devices ds;
  TF_ASSERT_OK_AND_ASSIGN(ds.emplace_back(),
                          mock_client_->LookupDevice(DeviceId(1)));
  DeviceListRef devices = BasicDeviceList::Create(std::move(ds));
  MemoryKind memory_kind("device");

  EXPECT_CALL(*mock_client_, CopyArrays(ElementsAreArray(src_arrays),
                                        Optional(EqualsDeviceList(devices)),
                                        Optional(memory_kind),
                                        ArrayCopySemantics::kAlwaysCopy))
      .WillOnce(Return(
          std::vector<tsl::RCReference<xla::ifrt::Array>>(copied_arrays)));

  auto ifrt_request = NewIfrtRequest(NewOpId());
  CopyArraysRequest* copy_arrays_request =
      ifrt_request->mutable_copy_arrays_request();
  for (const auto& src_array : src_arrays) {
    TF_ASSERT_OK_AND_ASSIGN(auto src_array_handle, MakeTestArray(src_array));
    copy_arrays_request->add_array_handles(src_array_handle);
  }
  for (const auto& device : devices->devices()) {
    copy_arrays_request->add_device_ids(device->Id().value());
  }
  copy_arrays_request->set_memory_kind(std::string(*memory_kind.memory_kind()));
  copy_arrays_request->set_copy_semantics(
      proto::ARRAY_COPY_SEMANTICS_ALWAYS_COPY);
  if (Version().protocol_version() >=
      protocol_version::kClientHandlesOptimization2) {
    copy_arrays_request->add_result_handles(1);
  }

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(ifrt_request)));

  EXPECT_THAT(tsl::StatusFromProto(response->response_metadata().status()),
              IsOk());
  EXPECT_THAT(response->copy_arrays_response().array_handles(),
              SizeIs(copied_arrays.size()));
}

TEST_P(IfrtBackendHandlerTest, FullyReplicatedShardSuccess) {
  auto fully_replicated_mock_array = tsl::MakeRef<xla::ifrt::MockArray>();
  auto resultant_array = tsl::MakeRef<xla::ifrt::MockArray>();
  EXPECT_CALL(*fully_replicated_mock_array, FullyReplicatedShard(_))
      .WillOnce(Return(std::move(resultant_array)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto fully_replicated_array_handle,
      MakeTestArray(std::move(fully_replicated_mock_array)));

  auto ifrt_request = NewIfrtRequest(NewOpId());
  auto* fully_replicated_shard_request =
      ifrt_request->mutable_fully_replicated_shard_request();
  fully_replicated_shard_request->set_array_handle(
      fully_replicated_array_handle);
  if (Version().protocol_version() >=
      protocol_version::kClientHandlesOptimization2) {
    fully_replicated_shard_request->set_result_handle(1234);
  }
  fully_replicated_shard_request->set_copy_semantics(
      proto::ARRAY_COPY_SEMANTICS_ALWAYS_COPY);

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(ifrt_request)));
  EXPECT_NE(response->fully_replicated_shard_response().array_handle(), 0);
}

TEST_P(IfrtBackendHandlerTest, FullyReplicatedShardFailure) {
  auto fully_replicated_mock_array = tsl::MakeRef<xla::ifrt::MockArray>();
  EXPECT_CALL(*fully_replicated_mock_array, FullyReplicatedShard(_))
      .WillOnce(Return(absl::UnknownError("injected error")));
  TF_ASSERT_OK_AND_ASSIGN(
      auto fully_replicated_array_handle,
      MakeTestArray(std::move(fully_replicated_mock_array)));

  auto ifrt_request = NewIfrtRequest(NewOpId());
  auto* fully_replicated_shard_request =
      ifrt_request->mutable_fully_replicated_shard_request();
  fully_replicated_shard_request->set_array_handle(
      fully_replicated_array_handle);
  fully_replicated_shard_request->set_copy_semantics(
      proto::ARRAY_COPY_SEMANTICS_ALWAYS_COPY);

  EXPECT_THAT(CallBackend(std::move(ifrt_request)),
              StatusIs(absl::StatusCode::kUnknown, StrEq("injected error")));
}

TEST_P(IfrtBackendHandlerTest,
       FullyReplicatedShardFailsWithNonExistentArrayHandle) {
  auto ifrt_request = NewIfrtRequest(NewOpId());
  auto* fully_replicated_shard_request =
      ifrt_request->mutable_fully_replicated_shard_request();
  fully_replicated_shard_request->set_array_handle(0);
  fully_replicated_shard_request->set_copy_semantics(
      proto::ARRAY_COPY_SEMANTICS_ALWAYS_COPY);

  EXPECT_THAT(CallBackend(std::move(ifrt_request)),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(IfrtBackendHandlerTest,
       CheckArrayReadyRequestRelaysTheResultFromBackend) {
  auto mock_array = tsl::MakeRef<xla::ifrt::MockArray>();
  TF_ASSERT_OK_AND_ASSIGN(auto array_handle,
                          MakeTestArray(std::move(mock_array)));
  EXPECT_CALL(*mock_client_, GetReadyFuture(_))
      .WillOnce(Return(Future<>(absl::OkStatus())))
      .WillOnce(Return(Future<>(absl::UnknownError("injected error"))));

  {
    auto ifrt_request = NewIfrtRequest(NewOpId());
    ifrt_request->mutable_check_value_ready_request()->add_value_handles(
        array_handle);
    TF_ASSERT_OK_AND_ASSIGN(auto ifrt_response,
                            CallBackend(std::move(ifrt_request)));

    EXPECT_THAT(ifrt_response->response_metadata().status().code(),
                tensorflow::error::OK);
    EXPECT_TRUE(ifrt_response->has_check_value_ready_response());
  }

  {
    auto ifrt_request = NewIfrtRequest(NewOpId());
    ifrt_request->mutable_check_value_ready_request()->add_value_handles(
        array_handle);
    EXPECT_THAT(CallBackend(std::move(ifrt_request)),
                StatusIs(absl::StatusCode::kUnknown, StrEq("injected error")));
  }
}

TEST_P(IfrtBackendHandlerTest,
       CheckArrayReadyRequestFailsWithNonExistentArrayHandle) {
  auto ifrt_request = NewIfrtRequest(NewOpId());
  ifrt_request->mutable_check_value_ready_request()->add_value_handles(0);
  EXPECT_THAT(CallBackend(std::move(ifrt_request)),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(IfrtBackendHandlerTest, DeleteArraySuccess) {
  auto mock_array1 = tsl::MakeRef<xla::ifrt::MockArray>();
  EXPECT_CALL(*mock_array1, Delete())
      .WillOnce(Return(Future<>(absl::OkStatus())));
  auto mock_array2 = tsl::MakeRef<xla::ifrt::MockArray>();
  EXPECT_CALL(*mock_array2, Delete())
      .WillOnce(Return(Future<>(absl::OkStatus())));

  TF_ASSERT_OK_AND_ASSIGN(auto array_handle1,
                          MakeTestArray(std::move(mock_array1)));
  TF_ASSERT_OK_AND_ASSIGN(auto array_handle2,
                          MakeTestArray(std::move(mock_array2)));

  uint64_t op_id = NewOpId();
  auto ifrt_request = NewIfrtRequest(op_id);
  ifrt_request->mutable_delete_array_request()->add_array_handle(array_handle1);
  ifrt_request->mutable_delete_array_request()->add_array_handle(array_handle2);
  TF_ASSERT_OK_AND_ASSIGN(auto resp, CallBackend(std::move(ifrt_request)));
  EXPECT_THAT(tsl::StatusFromProto(resp->response_metadata().status()), IsOk());
  TF_EXPECT_OK(
      CheckFuture(resp->delete_array_response().deletion_future_handle()));
}

TEST_P(IfrtBackendHandlerTest,
       DeleteArrayReturnsFutureWithNonExistentArrayHandle) {
  // Create one existing array.
  auto mock_array1 = tsl::MakeRef<xla::ifrt::MockArray>();
  EXPECT_CALL(*mock_array1, Delete())
      .WillOnce(Return(Future<>(absl::OkStatus())));
  TF_ASSERT_OK_AND_ASSIGN(auto real_handle,
                          MakeTestArray(std::move(mock_array1)));

  constexpr int kBadHandle = 400;
  auto ifrt_request = NewIfrtRequest(NewOpId());
  ifrt_request->mutable_delete_array_request()->add_array_handle(real_handle);
  ifrt_request->mutable_delete_array_request()->add_array_handle(kBadHandle);
  TF_ASSERT_OK_AND_ASSIGN(auto resp, CallBackend(std::move(ifrt_request)));

  EXPECT_THAT(
      CheckFuture(resp->delete_array_response().deletion_future_handle()),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(IfrtBackendHandlerTest,
       IsDeleteRelaysBackTheReturnValueFromBackendRuntime) {
  tsl::RCReference<xla::ifrt::MockArray> mock_array =
      tsl::MakeRef<xla::ifrt::MockArray>();

  EXPECT_CALL(*mock_array, IsDeleted())
      .WillOnce(Return(true))
      .WillOnce(Return(false));

  TF_ASSERT_OK_AND_ASSIGN(auto array_handle,
                          MakeTestArray(std::move(mock_array)));

  auto ifrt_request = NewIfrtRequest(NewOpId());
  ifrt_request->mutable_is_array_deleted_request()->set_array_handle(
      array_handle);
  TF_ASSERT_OK_AND_ASSIGN(auto resp, CallBackend(std::move(ifrt_request)));
  EXPECT_TRUE(resp->is_array_deleted_response().deleted());

  ifrt_request = NewIfrtRequest(NewOpId());
  ifrt_request->mutable_is_array_deleted_request()->set_array_handle(
      array_handle);
  TF_ASSERT_OK_AND_ASSIGN(resp, CallBackend(std::move(ifrt_request)));
  EXPECT_FALSE(resp->is_array_deleted_response().deleted());
}

TEST_P(IfrtBackendHandlerTest, IsDeleteFailsForNonExistentArrays) {
  auto ifrt_request = NewIfrtRequest(NewOpId());
  ifrt_request->mutable_is_array_deleted_request()->set_array_handle(0);
  EXPECT_THAT(CallBackend(std::move(ifrt_request)),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(IfrtBackendHandlerTest, DestructArrayTest) {
  tsl::RCReference<xla::ifrt::MockArray> mock_array1 =
      tsl::MakeRef<xla::ifrt::MockArray>();
  TF_ASSERT_OK_AND_ASSIGN(auto array_handle1,
                          MakeTestArray(std::move(mock_array1)));
  tsl::RCReference<xla::ifrt::MockArray> mock_array2 =
      tsl::MakeRef<xla::ifrt::MockArray>();
  TF_ASSERT_OK_AND_ASSIGN(auto array_handle2,
                          MakeTestArray(std::move(mock_array2)));

  auto ifrt_request = NewIfrtRequest(NewOpId());
  ifrt_request->mutable_destruct_array_request()->add_array_handle(
      array_handle1);
  ifrt_request->mutable_destruct_array_request()->add_array_handle(
      array_handle2);
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_resp, CallBackend(std::move(ifrt_request)));
  EXPECT_TRUE(ifrt_resp->has_destruct_array_response());

  // Retrying DestructArray should fail. And, this establishes that: (1) the
  // handle no longer exists on the server, (2) DestructArray fails for
  // non-existent arrays and (3) DestructArray is not idempotent.
  ifrt_request = NewIfrtRequest(NewOpId());
  ifrt_request->mutable_destruct_array_request()->add_array_handle(
      array_handle1);
  EXPECT_THAT(CallBackend(std::move(ifrt_request)),
              StatusIs(absl::StatusCode::kNotFound));
}

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_P(IfrtBackendHandlerTest, CompileSuccess) {
  std::vector<MockDevice> devices(4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_CALL(devices[i], Id()).WillOnce(Return(DeviceId(i)));
  }

  std::vector<xla::ifrt::Device*> addressable_devices;
  for (int i = 0; i < 4; ++i) {
    addressable_devices.push_back(&devices[i]);
  }

  auto executable = std::make_unique<MockLoadedExecutable>();
  EXPECT_CALL(*executable, name()).WillOnce(Return("executable_name"));
  EXPECT_CALL(*executable, num_devices()).WillOnce(Return(4));
  EXPECT_CALL(*executable, addressable_devices())
      .WillOnce(Return(absl::MakeSpan(addressable_devices)));
  EXPECT_CALL(*executable, Fingerprint()).WillOnce(Return("fingerprint"));
  EXPECT_CALL(*executable, GetReadyFuture())
      .WillOnce(Return(Future<>(absl::OkStatus())));

  ASSERT_OK_AND_ASSIGN(CompileResponse response,
                       CompileTestLoadedExecutable(std::move(executable)));
  EXPECT_THAT(response, Partially(EquivToProto(R"pb(
                name: "executable_name"
                num_devices: 4
                addressable_device_ids: [ 0, 1, 2, 3 ]
                fingerprint_value: "fingerprint"
              )pb")));
  TF_EXPECT_OK(CheckFuture(response.ready_future_handle()));
}
#endif

TEST_P(IfrtBackendHandlerTest, CompileFailure) {
  ASSERT_THAT(
      CompileTestLoadedExecutable(absl::InternalError("injected error")),
      StatusIs(absl::StatusCode::kInternal, StrEq("injected error")));
}

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_P(IfrtBackendHandlerTest, LoadedExecutableMetadata) {
  MockLoadedExecutable* executable;
  uint64_t handle;
  {
    auto e = std::make_unique<MockLoadedExecutable>();
    executable = e.get();
    TF_ASSERT_OK_AND_ASSIGN(CompileResponse response,
                            CompileTestLoadedExecutable(std::move(e)));
    handle = response.loaded_executable_handle();
  }

  {
    OpSharding op_sharding1;
    ASSERT_TRUE(
        TextFormat::ParseFromString(R"pb(type: REPLICATED)pb", &op_sharding1));

    OpSharding op_sharding2;
    ASSERT_TRUE(TextFormat::ParseFromString(
        R"pb(type: OTHER
             tile_shape {
               element_type: BF16
               dimensions: [ 2, 2 ]
             }
             tile_assignment_dimensions: [ 0, 1 ])pb",
        &op_sharding2));

    EXPECT_CALL(*executable, GetParameterShardings())
        .WillOnce(Return(std::vector<OpSharding>{op_sharding1, op_sharding2}));

    EXPECT_CALL(*executable, GetOutputShardings())
        .WillOnce(Return(std::vector<OpSharding>{op_sharding1}));

    std::vector<std::shared_ptr<const xla::PjRtLayout>> parameter_layouts;
    parameter_layouts.push_back(std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::MakeDescendingLayout(/*rank=*/1)));
    parameter_layouts.push_back(std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::MakeDescendingLayout(/*rank=*/2)));
    EXPECT_CALL(*executable, GetParameterLayouts())
        .WillOnce(Return(std::move(parameter_layouts)));

    std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts;
    output_layouts.push_back(std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::MakeDescendingLayout(/*rank=*/2)));
    EXPECT_CALL(*executable, GetOutputLayouts())
        .WillOnce(Return(std::move(output_layouts)));
    EXPECT_CALL(*executable, GetOutputMemoryKinds())
        .WillOnce(Return(std::vector<std::vector<absl::string_view>>{{"foo"}}));

    auto request = NewIfrtRequest(NewOpId());
    LoadedExecutableMetadataRequest* metadata_request =
        request->mutable_loaded_executable_metadata_request();
    metadata_request->set_loaded_executable_handle(handle);

    EXPECT_THAT(CallBackend(std::move(request)),
                IsOkAndHolds(Pointee(Partially(EquivToProto(R"pb(
                  loaded_executable_metadata_response {
                    parameter_shardings {
                      shardings { type: REPLICATED }
                      shardings {
                        type: OTHER
                        tile_shape {
                          element_type: BF16
                          dimensions: [ 2, 2 ]
                        }
                        tile_assignment_dimensions: [ 0, 1 ]
                      }
                    }
                    output_shardings { shardings { type: REPLICATED } }
                    parameter_layouts_list {
                      layouts { minor_to_major: 0 }
                      layouts { minor_to_major: [ 1, 0 ] }
                    }
                    output_layouts_list { layouts { minor_to_major: [ 1, 0 ] } }
                    output_memory_kinds {
                      memory_kind_lists { memory_kinds: [ "foo" ] }
                    }
                  }
                )pb")))));
  }

  {
    EXPECT_CALL(*executable, GetParameterShardings())
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*executable, GetOutputShardings())
        .WillOnce(Return(std::nullopt));
    EXPECT_CALL(*executable, GetParameterLayouts())
        .WillOnce(Return(absl::UnimplementedError("unimplemented")));
    EXPECT_CALL(*executable, GetOutputLayouts())
        .WillOnce(Return(absl::UnimplementedError("unimplemented")));
    EXPECT_CALL(*executable, GetOutputMemoryKinds())
        .WillOnce(Return(std::vector<std::vector<absl::string_view>>{}));

    auto request = NewIfrtRequest(NewOpId());
    LoadedExecutableMetadataRequest* metadata_request =
        request->mutable_loaded_executable_metadata_request();
    metadata_request->set_loaded_executable_handle(handle);

    TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                            CallBackend(std::move(request)));
    const auto& metadata_response =
        response->loaded_executable_metadata_response();
    EXPECT_FALSE(metadata_response.has_parameter_shardings());
    EXPECT_FALSE(metadata_response.has_output_shardings());
    EXPECT_TRUE(metadata_response.has_parameter_layouts_error());
    EXPECT_TRUE(metadata_response.has_output_layouts_error());
  }
}
#endif

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_P(IfrtBackendHandlerTest, LoadedExecutableExecute) {
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::Device* const device,
                          mock_client_->LookupDevice(DeviceId(0)));

  MockLoadedExecutable* executable;
  uint64_t handle;
  {
    auto e = std::make_unique<MockLoadedExecutable>();
    executable = e.get();
    TF_ASSERT_OK_AND_ASSIGN(CompileResponse response,
                            CompileTestLoadedExecutable(std::move(e)));
    handle = response.loaded_executable_handle();
  }

  constexpr int kNumArgs = 3;
  constexpr int kNumOutputs = 2;

  Shape shape({2, 2});
  auto sharding = SingleDeviceSharding::Create(device, MemoryKind());

  auto make_array = [&]() {
    auto array = tsl::MakeRef<MockArray>();
    ON_CALL(*array, dtype()).WillByDefault(Return(DType(DType::kF32)));
    ON_CALL(*array, shape()).WillByDefault(ReturnRef(shape));
    ON_CALL(*array, sharding()).WillByDefault(ReturnRef(*sharding));
    return array;
  };

  std::vector<tsl::RCReference<Array>> outputs;
  outputs.reserve(kNumOutputs);
  for (int i = 0; i < kNumOutputs; ++i) {
    outputs.push_back(make_array());
  }

  EXPECT_CALL(*executable, Execute(SizeIs(kNumArgs), _, _))
      .WillOnce(
          Invoke([&](absl::Span<tsl::RCReference<Array>> args,
                     const xla::ifrt::LoadedExecutable::ExecuteOptions& options,
                     std::optional<DeviceListRef> devices)
                     -> absl::StatusOr<LoadedExecutable::ExecuteResult> {
            return LoadedExecutable::ExecuteResult{
                .status = Future<>(absl::InternalError("injected error")),
                .outputs = outputs,
            };
          }));

  auto request = NewIfrtRequest(NewOpId());
  LoadedExecutableExecuteRequest* execute_request =
      request->mutable_loaded_executable_execute_request();
  for (int i = 0; i < kNumArgs; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(uint64_t arg_handle, MakeTestArray(make_array()));
    execute_request->add_args_handles(arg_handle);
  }
  execute_request->set_loaded_executable_handle(handle);
  xla::ifrt::LoadedExecutable::ExecuteOptions execute_options;
  execute_options.fill_status = true;
  TF_ASSERT_OK_AND_ASSIGN(*execute_request->mutable_execute_options(),
                          execute_options.ToProto());

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                          CallBackend(std::move(request)));
  EXPECT_THAT(response, Pointee(Partially(EquivToProto(R"pb(
                loaded_executable_execute_response {
                  outputs {
                    dtype { kind: KIND_F32 }
                    shape { dims: [ 2, 2 ] }
                  }
                  outputs {
                    dtype { kind: KIND_F32 }
                    shape { dims: [ 2, 2 ] }
                  }
                }
              )pb"))));
  TF_ASSERT_OK_AND_ASSIGN(
      auto sharding_proto,
      SingleDeviceSharding::Create(device, MemoryKind())->ToProto());
  for (const auto& output :
       response->loaded_executable_execute_response().outputs()) {
    EXPECT_THAT(output.sharding(), EquivToProto(sharding_proto));
    EXPECT_NE(output.array_handle(), 0);
  }

  EXPECT_THAT(
      CheckFuture(
          response->loaded_executable_execute_response().status_handle()),
      StatusIs(absl::StatusCode::kInternal, StrEq("injected error")));

  // The second call to `CheckFuture` fails since `CheckFuture` above performs a
  // destructive read.
  EXPECT_THAT(
      CheckFuture(
          response->loaded_executable_execute_response().status_handle()),
      StatusIs(absl::StatusCode::kNotFound,
               HasSubstr("Unknown future handle")));
}
#endif

TEST_P(IfrtBackendHandlerTest, LoadedExecutableExecuteErrorWithClientHandles) {
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::Device* const device,
                          mock_client_->LookupDevice(DeviceId(0)));

  MockLoadedExecutable* executable;
  uint64_t handle;
  {
    auto e = std::make_unique<MockLoadedExecutable>();
    executable = e.get();
    TF_ASSERT_OK_AND_ASSIGN(CompileResponse response,
                            CompileTestLoadedExecutable(std::move(e)));
    handle = response.loaded_executable_handle();
  }

  constexpr int kNumArgs = 3;
  constexpr int kNumOutputs = 2;

  Shape shape({2, 2});
  auto sharding = SingleDeviceSharding::Create(device, MemoryKind());

  auto make_array = [&]() {
    auto array = tsl::MakeRef<MockArray>();
    ON_CALL(*array, dtype()).WillByDefault(Return(DType(DType::kF32)));
    ON_CALL(*array, shape()).WillByDefault(ReturnRef(shape));
    ON_CALL(*array, sharding()).WillByDefault(ReturnRef(*sharding));
    return array;
  };

  EXPECT_CALL(*executable, Execute(SizeIs(kNumArgs), _, _))
      .WillOnce(
          Invoke([&](absl::Span<tsl::RCReference<Array>> args,
                     const xla::ifrt::LoadedExecutable::ExecuteOptions& options,
                     std::optional<DeviceListRef> devices)
                     -> absl::StatusOr<LoadedExecutable::ExecuteResult> {
            return absl::InternalError("injected error");
          }));

  auto request = NewIfrtRequest(NewOpId());
  LoadedExecutableExecuteRequest* execute_request =
      request->mutable_loaded_executable_execute_request();
  for (int i = 0; i < kNumArgs; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(uint64_t arg_handle, MakeTestArray(make_array()));
    execute_request->add_args_handles(arg_handle);
  }
  execute_request->set_loaded_executable_handle(handle);
  constexpr uint64_t kFirstResultHandle = 1000;
  for (int i = 0; i < kNumOutputs; ++i) {
    execute_request->add_result_array_handle(kFirstResultHandle + i);
  }
  execute_request->set_result_status_handle(kFirstResultHandle + kNumOutputs);

  xla::ifrt::LoadedExecutable::ExecuteOptions execute_options;
  execute_options.fill_status = true;
  TF_ASSERT_OK_AND_ASSIGN(*execute_request->mutable_execute_options(),
                          execute_options.ToProto());

  auto status_is_err =
      StatusIs(absl::StatusCode::kInternal, StrEq("injected error"));

  EXPECT_THAT(CallBackend(std::move(request)), status_is_err);

  EXPECT_THAT(CheckFuture(kFirstResultHandle + kNumOutputs), status_is_err);

  for (int i = 0; i < kNumOutputs; ++i) {
    EXPECT_THAT(CheckValueReady(kFirstResultHandle + i), status_is_err);
  }
}

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_P(IfrtBackendHandlerTest, LoadedExecutableDelete) {
  MockLoadedExecutable* executable;
  uint64_t handle;
  {
    auto e = std::make_unique<MockLoadedExecutable>();
    executable = e.get();
    TF_ASSERT_OK_AND_ASSIGN(CompileResponse response,
                            CompileTestLoadedExecutable(std::move(e)));
    handle = response.loaded_executable_handle();
  }

  {
    EXPECT_CALL(*executable, Delete())
        .WillOnce(Return(Future<>(absl::OkStatus())));

    auto request = NewIfrtRequest(NewOpId());
    LoadedExecutableDeleteRequest* delete_request =
        request->mutable_loaded_executable_delete_request();
    delete_request->set_loaded_executable_handle(handle);

    TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                            CallBackend(std::move(request)));
    ASSERT_TRUE(response->has_loaded_executable_delete_response());

    EXPECT_THAT(
        CheckFuture(
            response->loaded_executable_delete_response().future_handle()),
        IsOk());
  }

  {
    EXPECT_CALL(*executable, IsDeleted()).WillOnce(Return(true));

    auto request = NewIfrtRequest(NewOpId());
    LoadedExecutableIsDeletedRequest* is_deleted_request =
        request->mutable_loaded_executable_is_deleted_request();
    is_deleted_request->set_loaded_executable_handle(handle);

    EXPECT_THAT(CallBackend(std::move(request)),
                IsOkAndHolds(Pointee(Partially(EquivToProto(R"pb(
                  loaded_executable_is_deleted_response { is_deleted: true }
                )pb")))));
  }
}
#endif

TEST_P(IfrtBackendHandlerTest, LoadedExecutableDestruct) {
  MockLoadedExecutable* executable;
  uint64_t handle;
  {
    auto e = std::make_unique<MockLoadedExecutable>();
    executable = e.get();
    TF_ASSERT_OK_AND_ASSIGN(CompileResponse response,
                            CompileTestLoadedExecutable(std::move(e)));
    handle = response.loaded_executable_handle();
  }

  {
    auto request = NewIfrtRequest(NewOpId());
    LoadedExecutableDestructRequest* destruct_request =
        request->mutable_loaded_executable_destruct_request();
    destruct_request->set_loaded_executable_handle(handle);

    TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                            CallBackend(std::move(request)));
    ASSERT_TRUE(response->has_loaded_executable_destruct_response());
  }

  // Any attempt to access the loaded executable handle should now return an
  // error.
  {
    auto request = NewIfrtRequest(NewOpId());
    LoadedExecutableDestructRequest* destruct_request =
        request->mutable_loaded_executable_destruct_request();
    destruct_request->set_loaded_executable_handle(handle);

    EXPECT_THAT(CallBackend(std::move(request)),
                StatusIs(absl::StatusCode::kNotFound,
                         HasSubstr("Unknown loaded executable handle")));
  }
}

TEST_P(IfrtBackendHandlerTest, LoadedHostCallbackExecute) {
  // Build a remote host callback with one F32 argument and one F32 result.
  std::vector<xla::HostCallbackArgInfo> hcb_args = {{
      .channel_id = 1,
      .shape = xla::ShapeUtil::MakeShape(xla::F32, {}),
  }};
  std::vector<xla::HostCallbackArgInfo> hcb_results = {{
      .channel_id = 2,
      .shape = xla::ShapeUtil::MakeShape(xla::F32, {}),
  }};
  auto hcb = tsl::MakeRef<RemoteLoadedHostCallback>(
      mock_client_, std::move(hcb_args), std::move(hcb_results),
      /*queue=*/nullptr);

  // Compile an executable with the above host callback. The resulting loaded
  // host callback handle and `xla::HostCallback` are kept for triggering host
  // callback execution.
  //
  // The setup code must use `xla::ifrt::XlaCompileOptions` for now since this
  // is the only allowed compile options type that is currently recognized as
  // supporting host callbacks.
  MockLoadedExecutable* executable;
  tsl::RCReference<xla::ifrt::LoadedHostCallback> loaded_host_callback;
  uint64_t loaded_host_callback_handle;
  {
    auto request = NewIfrtRequest(NewOpId());
    CompileRequest* compile_request = request->mutable_compile_request();

    TestProgram program;
    TF_ASSERT_OK_AND_ASSIGN(*compile_request->mutable_program(),
                            Serialize(program, /*options=*/nullptr));
    xla::ifrt::XlaCompileOptions compile_options;
    TF_ASSERT_OK_AND_ASSIGN(*compile_request->mutable_compile_options(),
                            Serialize(compile_options, /*options=*/nullptr));

    TF_ASSERT_OK_AND_ASSIGN(std::string host_callback_serialized,
                            hcb->Serialize());
    compile_request->add_host_callbacks(std::move(host_callback_serialized));

    auto e = std::make_unique<MockLoadedExecutable>();
    executable = e.get();

    EXPECT_CALL(mock_compiler_, Compile(_, _))
        .WillOnce(DoAll(
            Invoke(
                [&](const std::unique_ptr<xla::ifrt::Program>& program,
                    const std::unique_ptr<xla::ifrt::CompileOptions>& options) {
                  auto* xla_compile_options =
                      llvm::cast<xla::ifrt::XlaCompileOptions>(options.get());
                  auto& loaded_host_callbacks =
                      xla_compile_options->loaded_host_callbacks;
                  ASSERT_EQ(loaded_host_callbacks.size(), 1);
                  loaded_host_callback = loaded_host_callbacks.front();
                }),
            Return(ByMove(std::move(e)))));

    TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                            CallBackend(std::move(request)));

    ASSERT_TRUE(response->has_compile_response());
    CompileResponse compile_response = response->compile_response();

    loaded_host_callback_handle =
        compile_response.loaded_host_callback_handles(0);
    ASSERT_THAT(loaded_host_callback, NotNull());
  }

  // Enqueue a host callback execution. This is done on a separate thread since
  // `LoadedHostCallbackPollRequest` blocks until there is a pending execution.
  auto host_callback_thread = absl::WrapUnique(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "HostCallback", [&]() {
        xla::Literal x = xla::LiteralUtil::CreateR0(1.0f);

        std::vector<void*> operands;
        operands.push_back(x.untyped_data());

        xla::Literal out = xla::LiteralUtil::CreateR0(0.0f);
        std::vector<void*> results;
        results.push_back(out.untyped_data());

        const xla::HostCallback* xla_host_callback =
            &llvm::cast<RemoteLoadedHostCallback>(loaded_host_callback.get())
                 ->host_callback();
        ASSERT_THAT(
            xla_host_callback->callback(results.data(), operands.data()),
            IsOk());
        EXPECT_EQ(out, xla::LiteralUtil::CreateR0(2.0f));
      }));

  // Poll for a host callback execution and verify its argument against the one
  // passed by the execution thread above.
  uint64_t host_callback_execution_handle;
  {
    const uint64_t operand_host_buffer_handle = NewHostBufferHandle();

    auto request = NewIfrtRequest(NewOpId());
    LoadedHostCallbackPollRequest* poll_request =
        request->mutable_loaded_host_callback_poll_request();
    poll_request->set_loaded_host_callback_handle(loaded_host_callback_handle);
    poll_request->set_operand_host_buffer_handle(operand_host_buffer_handle);

    TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                            CallBackend(std::move(request)));

    ASSERT_TRUE(response->has_loaded_host_callback_poll_response());
    const LoadedHostCallbackPollResponse& poll_response =
        response->loaded_host_callback_poll_response();
    host_callback_execution_handle =
        poll_response.host_callback_execution_handle();

    TF_ASSERT_OK_AND_ASSIGN(
        const std::shared_ptr<const std::string> operands,
        host_buffer_store_->Lookup(operand_host_buffer_handle));
    EXPECT_EQ(xla::BorrowingLiteral(operands->data(),
                                    xla::ShapeUtil::MakeShape(xla::F32, {})),
              xla::LiteralUtil::CreateR0(1.0f));
  }

  // Return the execution result. This will unblock the execution thread above,
  // which also verifies the result.
  {
    auto result = xla::LiteralUtil::CreateR0(2.0f);
    std::string result_buffer(absl::string_view(
        static_cast<const char*>(result.untyped_data()), result.size_bytes()));

    const uint64_t result_host_buffer_handle = NewHostBufferHandle();
    ASSERT_THAT(host_buffer_store_->Store(result_host_buffer_handle,
                                          std::move(result_buffer)),
                IsOk());

    auto request = NewIfrtRequest(NewOpId());
    LoadedHostCallbackReturnRequest* ret_request =
        request->mutable_loaded_host_callback_return_request();
    ret_request->set_host_callback_execution_handle(
        host_callback_execution_handle);
    ret_request->set_result_host_buffer_handle(result_host_buffer_handle);

    TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                            CallBackend(std::move(request)));
    ASSERT_TRUE(response->has_loaded_host_callback_return_response());
  }
}

TEST_P(IfrtBackendHandlerTest, GetDefaultDeviceAssignmentSuccess) {
  const int kNumReplicas = 1;
  const int kNumPartitions = 3;

  EXPECT_CALL(*mock_client_,
              GetDefaultDeviceAssignment(kNumReplicas, kNumPartitions))
      .WillOnce(Return(xla::DeviceAssignment(kNumReplicas, kNumPartitions)));

  auto request = NewIfrtRequest(NewOpId());
  auto* default_device_assignment_request =
      request->mutable_get_default_device_assignment_request();
  default_device_assignment_request->set_num_replicas(kNumReplicas);
  default_device_assignment_request->set_num_partitions(kNumPartitions);

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(request)));
  TF_ASSERT_OK_AND_ASSIGN(auto assignment_got,
                          xla::DeviceAssignment::Deserialize(
                              response->get_default_device_assignment_response()
                                  .device_assignment()));
  EXPECT_EQ(assignment_got->replica_count(), kNumReplicas);
  EXPECT_EQ(assignment_got->computation_count(), kNumPartitions);
}

TEST_P(IfrtBackendHandlerTest,
       GetDefaultDeviceAssignmentFailsIfTheBackendFails) {
  const int kNumReplicas = 1;
  const int kNumPartitions = 3;

  EXPECT_CALL(*mock_client_,
              GetDefaultDeviceAssignment(kNumReplicas, kNumPartitions))
      .WillOnce(Return(absl::UnknownError("injected error")));

  auto request = NewIfrtRequest(NewOpId());
  auto* default_device_assignment_request =
      request->mutable_get_default_device_assignment_request();
  default_device_assignment_request->set_num_replicas(kNumReplicas);
  default_device_assignment_request->set_num_partitions(kNumPartitions);

  EXPECT_THAT(CallBackend(std::move(request)),
              StatusIs(absl::StatusCode::kUnknown, StrEq("injected error")));
}

TEST_P(IfrtBackendHandlerTest, GetDefaultLayoutSuccess) {
  const auto kDefaultLayout = std::make_shared<xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(1));
  const xla::ifrt::DType kDType = xla::ifrt::DType(xla::ifrt::DType::kF32);
  const std::vector<int64_t> kDims = {1, 2, 3};
  const int64_t kDeviceId = 42;
  const auto mock_device = std::make_unique<xla::ifrt::MockDevice>();
  const std::string kMemoryKindStr = "xla::ifrt::MemoryKind()";
  const xla::ifrt::MemoryKind kMemoryKind(kMemoryKindStr);

  ON_CALL(*mock_client_, LookupDevice(DeviceId(kDeviceId)))
      .WillByDefault(Return(mock_device.get()));

  EXPECT_CALL(*mock_client_,
              GetDefaultLayout(kDType, absl::MakeConstSpan(kDims),
                               mock_device.get(), kMemoryKind))
      .WillOnce(Return(std::shared_ptr<const xla::PjRtLayout>(kDefaultLayout)));

  auto request = NewIfrtRequest(NewOpId());
  auto* default_layout_request = request->mutable_get_default_layout_request();
  *default_layout_request->mutable_dtype() = kDType.ToProto();
  default_layout_request->mutable_dims()->Reserve(kDims.size());
  for (int64_t dim : kDims) {
    default_layout_request->add_dims(dim);
  }
  default_layout_request->set_device_id(kDeviceId);
  default_layout_request->set_memory_kind(kMemoryKindStr);

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(request)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto layout_got,
      xla::PjRtLayout::Deserialize(
          response->get_default_layout_response().serialized_pjrt_layout()));
  EXPECT_EQ(*layout_got, *kDefaultLayout);
}

INSTANTIATE_TEST_SUITE_P(
    IfrtBackendHandlerTestWithAllVersions, IfrtBackendHandlerTest,
    testing::Range(kServerMinVersion, kServerMaxVersion + 1),
    [](const testing::TestParamInfo<IfrtBackendHandlerTest::ParamType>& info) {
      return absl::StrCat(info.param);
    });

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
