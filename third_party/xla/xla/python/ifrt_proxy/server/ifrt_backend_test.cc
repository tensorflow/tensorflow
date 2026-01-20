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
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "google/protobuf/text_format.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/profiling/device_time_measurement.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
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
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/common/array_util.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/host_callback.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_to_from_proto.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::_;
using ::testing::ByMove;
using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::MatchesRegex;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::Optional;
using ::testing::Pointee;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::tsl::protobuf::TextFormat;

using ::tsl::proto_testing::EquivToProto;
using ::tsl::proto_testing::Partially;

constexpr uint64_t kSessionId = 12345;

class IfrtBackendTest
    : public ::testing::TestWithParam</*protocol_version=*/int> {
 protected:
  IfrtProxyVersion Version() {
    IfrtProxyVersion version;
    version.set_protocol_version(GetParam());
    // TODO(hyeontaek): For a more realistic test setup, the IFRT SerDes version
    // should vary by the IFRT Proxy protocol version.
    version.set_ifrt_serdes_version_number(
        SerDesVersion::current().version_number().value());
    return version;
  }
  SerDesVersion ifrt_serdes_version() {
    return SerDesAnyVersionAccessor::Get(
        SerDesVersionNumber(Version().ifrt_serdes_version_number()));
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
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(IfrtBackendTest, SuccessfulCreation) {
  auto ifrt_client = std::make_unique<MockClient>();
  ASSERT_THAT(IfrtBackend::Create(Version(), kSessionId, std::move(ifrt_client),
                                  std::make_shared<HostBufferStore>()),
              absl_testing::IsOk());
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
  ASSERT_THAT(process_status, Not(absl_testing::IsOk()));
}

INSTANTIATE_TEST_SUITE_P(
    IfrtBackendTestWithAllVersions, IfrtBackendTest,
    testing::Range(protocol_version::kServerMin,
                   protocol_version::kServerMax + 1),
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
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions>) override {
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
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions>) override {
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

    ON_CALL(*mock_client, Attributes())
        .WillByDefault(ReturnRef(client_attributes_));

    std::vector<xla::ifrt::Device*> raw_device_ptrs;
    for (int i = 0; i < 2; ++i) {
      auto mock_device = std::make_unique<xla::ifrt::MockDevice>();
      ON_CALL(*mock_device, client()).WillByDefault(Return(mock_client.get()));
      ON_CALL(*mock_device, Id()).WillByDefault(Return(DeviceId(i)));
      ON_CALL(*mock_device, IsAddressable()).WillByDefault(Return(true));
      raw_device_ptrs.push_back(mock_device.get());
      mock_devices_.push_back(std::move(mock_device));
    }

    xla::DeviceAssignment device_assignment(1, 1);
    device_assignment(0, 0) = mock_devices_[0]->Id().value();
    ON_CALL(*mock_client, GetDefaultDeviceAssignment(_, _))
        .WillByDefault(Return(device_assignment));
    ON_CALL(*mock_client, addressable_devices())
        .WillByDefault(Return(raw_device_ptrs));
    ON_CALL(*mock_client, devices()).WillByDefault(Return(raw_device_ptrs));
    ON_CALL(*mock_client, GetAllDevices())
        .WillByDefault(Return(raw_device_ptrs));
    ON_CALL(*mock_client, LookupDevice(_))
        .WillByDefault(
            [this](DeviceId id) -> absl::StatusOr<xla::ifrt::Device*> {
              if (id.value() < 0 || id.value() >= mock_devices_.size()) {
                return absl::NotFoundError(
                    absl::StrCat("Unknown device id: ", id.value()));
              }
              return mock_devices_[id.value()].get();
            });
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
    absl::MutexLock lock(mu_);
    return current_op_id_++;
  }

  uint64_t NewHostBufferHandle() { return current_host_buffer_handle_++; }

  // Utility method to set up a given MockArray (in the backend) that can then
  // be the target of the other Array-specific methods. Returns the array
  // handle.
  absl::StatusOr<uint64_t> MakeTestArray(ArrayRef mock_array) {
    EXPECT_CALL(*mock_client_, MakeArrayFromHostBuffer(_, _, _, _, _, _, _))
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
      TF_RETURN_IF_ERROR(SingleDeviceSharding::Create(device, MemoryKind())
                             ->ToProto(*make_array->mutable_sharding(),
                                       ifrt_serdes_version()));
    }
    TF_ASSIGN_OR_RETURN(auto make_array_response,
                        CallBackend(std::move(ifrt_request)));

    TF_RETURN_IF_ERROR(tsl::StatusFromProto(
        make_array_response->response_metadata().status()));
    return make_array_response->make_array_from_host_buffer_response()
        .array_handle();
  }

  absl::StatusOr<CompileResponse> CompileTestLoadedExecutable(
      absl::StatusOr<LoadedExecutableRef> loaded_executable) {
    auto request = NewIfrtRequest(NewOpId());
    CompileRequest* compile_request = request->mutable_compile_request();
    TestProgram program;
    {
      auto serialize_options =
          std::make_unique<SerializeOptions>(ifrt_serdes_version());
      TF_ASSIGN_OR_RETURN(*compile_request->mutable_program(),
                          Serialize(program, std::move(serialize_options)));
    }
    {
      TestCompileOptions compile_options;
      auto serialize_options =
          std::make_unique<SerializeOptions>(ifrt_serdes_version());
      TF_ASSIGN_OR_RETURN(
          *compile_request->mutable_compile_options(),
          Serialize(compile_options, std::move(serialize_options)));
    }

    EXPECT_CALL(mock_compiler_, CompileAndLoad(_, _))
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
  xla::ifrt::AttributeMap client_attributes_{xla::ifrt::AttributeMap::Map(
      {{"test_key", xla::ifrt::AttributeMap::StringValue("test_value")}})};
  std::unique_ptr<IfrtBackend> backend_;
};

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
    EXPECT_CALL(mock_device, PlatformName()).WillRepeatedly(Return("mock"));
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

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                          CallBackend(std::move(request)));
  ASSERT_TRUE(response->has_init_response()) << response->DebugString();

  InitResponse init_response = std::move(response->init_response());
  LOG(INFO) << "init_response: " << init_response.DebugString();

  EXPECT_EQ(init_response.session_id(), 12345);
  EXPECT_EQ(init_response.platform_name(), "ifrt_backend");
  EXPECT_EQ(init_response.platform_version(), "n/a");
  EXPECT_EQ(init_response.platform_id(), 42);
  EXPECT_EQ(init_response.process_index(), 1);
  EXPECT_EQ(init_response.runtime_type(), "ifrt-service");

  EXPECT_EQ(init_response.all_devices().size(), 2);
  for (auto device : init_response.all_devices()) {
    int device_canonical_num = device.id();
    EXPECT_EQ(device.platform_name(), "mock");
    EXPECT_EQ(device.device_kind(), "mock");
    EXPECT_EQ(device.default_memory_id(), device_canonical_num);
    EXPECT_EQ(device.memory_ids().size(), 1);
    EXPECT_EQ(device.memory_ids(0), device_canonical_num);
    std::string expected_name = absl::StrCat("device", device_canonical_num);
    EXPECT_EQ(device.attributes().attributes().size(), 1);
    EXPECT_EQ(device.attributes().attributes().at("name").string_value(),
              expected_name);
  }

  EXPECT_EQ(init_response.memories().size(), 2);
  for (auto memory : init_response.memories()) {
    int memory_canonical_num = memory.id();
    EXPECT_EQ(memory.memory_space_kind(), "mock");
    EXPECT_EQ(memory.device_ids().size(), 1);
    EXPECT_EQ(memory.device_ids(0), memory_canonical_num);
  }

  EXPECT_THAT(init_response.primary_device_ids(), ElementsAre(0, 1));

  EXPECT_EQ(init_response.client_attributes().attributes().size(), 1);
  EXPECT_EQ(init_response.client_attributes()
                .attributes()
                .at("test_key")
                .string_value(),
            "test_value");
}

// TODO(b/282757875): Use the MockRuntime fixture to cover the error cases for
// MakeArrayFromHostBuffer and CopyToHostBuffer methods as well.

// Consider redoing the happy-path test below with PjRt CPU-only backend for
// non-SingleDeviceSharding.
TEST_P(IfrtBackendHandlerTest, DisassembleIntoSingleDeviceArraysSucceeds) {
  // Set up a mock source array that returns two single device arrays on
  // disassembly.
  std::vector<xla::ifrt::ArrayRef> single_device_arrays;
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
  disassemble_into_single_device_arrays->set_single_device_shard_semantics(
      proto::SingleDeviceShardSemantics::
          SINGLE_DEVICE_SHARD_SEMANTICS_ALL_SHARDS);
  disassemble_into_single_device_arrays->add_result_handles(1);
  disassemble_into_single_device_arrays->add_result_handles(2);
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
      absl_testing::IsOk());

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
    TF_ASSERT_OK(
        SingleDeviceSharding::Create(device, MemoryKind())
            ->ToProto(*make_array->mutable_sharding(), ifrt_serdes_version()));
  }

  const Shape expected_shape({5, 3, 4});
  const std::vector<int64_t> expected_byte_strides_vec = {8, 40, 120};
  const std::optional<absl::Span<const int64_t>> expected_byte_strides =
      absl::Span<const int64_t>(expected_byte_strides_vec);

  tsl::RCReference<xla::ifrt::MockArray> mock_array =
      tsl::MakeRef<xla::ifrt::MockArray>();

  EXPECT_CALL(*mock_client_,
              MakeArrayFromHostBuffer(_, DType(DType::kF64), expected_shape,
                                      expected_byte_strides, _, _, _))
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
      absl_testing::IsOk());

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
  TF_ASSERT_OK(
      SingleDeviceSharding::Create(device, MemoryKind())
          ->ToProto(*make_array->mutable_sharding(), ifrt_serdes_version()));

  const DType expected_dtype = DType(DType::kString);
  const Shape expected_shape({2});
  const std::optional<absl::Span<const int64_t>> expected_byte_strides =
      std::nullopt;

  tsl::RCReference<xla::ifrt::MockArray> mock_array =
      tsl::MakeRef<xla::ifrt::MockArray>();

  EXPECT_CALL(*mock_client_,
              MakeArrayFromHostBuffer(_, expected_dtype, expected_shape,
                                      expected_byte_strides, _, _, _))
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
    req->set_single_device_shard_semantics(
        proto::SINGLE_DEVICE_SHARD_SEMANTICS_ALL_SHARDS);
    req->set_result_handle(1);
    dtype.ToProto(*req->mutable_dtype(), ifrt_serdes_version());
    TF_ASSERT_OK_AND_ASSIGN(auto* device,
                            mock_client_->LookupDevice(DeviceId(1)));
    TF_ASSERT_OK(
        SingleDeviceSharding::Create(device, MemoryKind())
            ->ToProto(*req->mutable_sharding(), ifrt_serdes_version()));
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
    assemble_array_from_single_device_arrays->set_single_device_shard_semantics(
        proto::SingleDeviceShardSemantics::
            SINGLE_DEVICE_SHARD_SEMANTICS_ALL_SHARDS);
    dtype.ToProto(*assemble_array_from_single_device_arrays->mutable_dtype(),
                  ifrt_serdes_version());
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
      .WillOnce(Return(tsl::Future<>(absl::OkStatus())));

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(ifrt_request)));
  // Given the above shape, dtype, and compact byte_strides, the size of the
  // array data needs to be 480 bytes.
  EXPECT_THAT(host_buffer_store_->Lookup(host_buffer_handle),
              absl_testing::IsOkAndHolds(Pointee(SizeIs(480))));
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
      absl_testing::IsOk());

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
  TF_ASSERT_OK(
      SingleDeviceSharding::Create(device, MemoryKind())
          ->ToProto(*make_array->mutable_sharding(), ifrt_serdes_version()));

  const DType expected_dtype = DType(DType::kString);
  const Shape expected_shape({2});
  const std::optional<absl::Span<const int64_t>> expected_byte_strides =
      std::nullopt;

  tsl::RCReference<xla::ifrt::MockArray> mock_array =
      tsl::MakeRef<xla::ifrt::MockArray>();
  ON_CALL(*mock_array, shape()).WillByDefault(ReturnRef(expected_shape));
  ON_CALL(*mock_array, dtype()).WillByDefault(Return(expected_dtype));

  ON_CALL(*mock_array, CopyToHostBuffer(_, _, _))
      .WillByDefault([input_strings = input_strings](
                         void* data,
                         std::optional<absl::Span<const int64_t>> byte_strides,
                         xla::ifrt::ArrayCopySemantics semantics) {
        auto dst = static_cast<absl::Cord*>(data);
        for (int i = 0; i < input_strings.size(); ++i) {
          dst[i] = input_strings[i];
        }
        return tsl::Future<>(absl::OkStatus());
      });

  EXPECT_CALL(*mock_client_,
              MakeArrayFromHostBuffer(_, expected_dtype, expected_shape,
                                      expected_byte_strides, _, _, _))
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
  ASSERT_THAT(CallBackend(std::move(ifrt_request)), absl_testing::IsOk());
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
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
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
  disassemble_into_single_device_arrays->set_single_device_shard_semantics(
      proto::SingleDeviceShardSemantics::
          SINGLE_DEVICE_SHARD_SEMANTICS_ALL_SHARDS);
  ASSERT_THAT(CallBackend(std::move(disassemble_request)),
              absl_testing::StatusIs(absl::StatusCode::kUnknown,
                                     StrEq(kDisassembleErrorMessage)));
}

// Matcher for matching the `device_list` argument of `Client::CopyArrays()`.
MATCHER_P(EqualsDeviceList, device_list, "") { return *arg == *device_list; }

TEST_P(IfrtBackendHandlerTest, CopyArrays) {
  std::vector<xla::ifrt::ArrayRef> src_arrays;
  src_arrays.push_back(tsl::MakeRef<xla::ifrt::MockArray>());

  std::vector<xla::ifrt::ArrayRef> copied_arrays;
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
      .WillOnce(Return(std::vector<xla::ifrt::ArrayRef>(copied_arrays)));

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
  copy_arrays_request->add_result_handles(1);

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(ifrt_request)));

  EXPECT_THAT(tsl::StatusFromProto(response->response_metadata().status()),
              absl_testing::IsOk());
  EXPECT_THAT(response->copy_arrays_response().array_handles(),
              SizeIs(copied_arrays.size()));
}

TEST_P(IfrtBackendHandlerTest, ReshardArrays) {
  auto layout1 = std::make_shared<const xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(1));
  auto layout2 = std::make_shared<const xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(2));

  auto mock_array = tsl::MakeRef<xla::ifrt::MockArray>();
  ON_CALL(*mock_array, dtype()).WillByDefault(Return(DType(DType::kF32)));
  Shape shape({2, 2});
  ON_CALL(*mock_array, shape()).WillByDefault(ReturnRef(shape));
  ON_CALL(*mock_array, pjrt_layout()).WillByDefault(Return(layout1));

  const std::vector<xla::ifrt::ArrayRef> src_arrays{{mock_array}};

  auto reshared_array = tsl::MakeRef<xla::ifrt::MockArray>();
  ON_CALL(*reshared_array, pjrt_layout()).WillByDefault(Return(layout2));
  std::vector<xla::ifrt::ArrayRef> result_arrays;
  result_arrays.push_back(reshared_array);

  TF_ASSERT_OK_AND_ASSIGN(Device * device,
                          mock_client_->LookupDevice(DeviceId(0)));
  ShardingRef sharding(SingleDeviceSharding::Create(device, MemoryKind()));

  std::vector<ArraySpec> specs{{DType(DType::kF32), shape, sharding, layout2}};

  EXPECT_CALL(*mock_client_, ReshardArrays(ElementsAreArray(src_arrays), _,
                                           ArrayCopySemantics::kAlwaysCopy))
      .WillOnce(Return(result_arrays));

  auto ifrt_request = NewIfrtRequest(NewOpId());
  ReshardArraysRequest* reshard_arrays_request =
      ifrt_request->mutable_reshard_arrays_request();

  TF_ASSERT_OK_AND_ASSIGN(auto src_array_handle, MakeTestArray(mock_array));
  reshard_arrays_request->add_array_handles(src_array_handle);

  for (const auto& spec : specs) {
    TF_ASSERT_OK(spec.ToProto(*reshard_arrays_request->add_array_specs(),
                              ifrt_serdes_version()));
  }
  reshard_arrays_request->set_copy_semantics(
      proto::ARRAY_COPY_SEMANTICS_ALWAYS_COPY);
  reshard_arrays_request->add_result_handles(1);

  TF_ASSERT_OK_AND_ASSIGN(auto response, CallBackend(std::move(ifrt_request)));

  ASSERT_THAT(tsl::StatusFromProto(response->response_metadata().status()),
              absl_testing::IsOk());
  EXPECT_THAT(response->reshard_arrays_response().array_handles(),
              SizeIs(result_arrays.size()));
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
  fully_replicated_shard_request->set_result_handle(1234);
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
              absl_testing::StatusIs(absl::StatusCode::kUnknown,
                                     StrEq("injected error")));
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
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(IfrtBackendHandlerTest,
       CheckArrayReadyRequestRelaysTheResultFromBackend) {
  auto mock_array = tsl::MakeRef<xla::ifrt::MockArray>();
  TF_ASSERT_OK_AND_ASSIGN(auto array_handle,
                          MakeTestArray(std::move(mock_array)));
  EXPECT_CALL(*mock_client_, GetReadyFuture(_))
      .WillOnce(Return(tsl::Future<>(absl::OkStatus())))
      .WillOnce(Return(tsl::Future<>(absl::UnknownError("injected error"))));

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
                absl_testing::StatusIs(absl::StatusCode::kUnknown,
                                       StrEq("injected error")));
  }
}

TEST_P(IfrtBackendHandlerTest,
       CheckArrayReadyRequestFailsWithNonExistentArrayHandle) {
  auto ifrt_request = NewIfrtRequest(NewOpId());
  ifrt_request->mutable_check_value_ready_request()->add_value_handles(0);
  EXPECT_THAT(CallBackend(std::move(ifrt_request)),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(IfrtBackendHandlerTest, DeleteArraySuccess) {
  auto mock_array1 = tsl::MakeRef<xla::ifrt::MockArray>();
  EXPECT_CALL(*mock_array1, Delete())
      .WillOnce(Return(tsl::Future<>(absl::OkStatus())));
  auto mock_array2 = tsl::MakeRef<xla::ifrt::MockArray>();
  EXPECT_CALL(*mock_array2, Delete())
      .WillOnce(Return(tsl::Future<>(absl::OkStatus())));

  TF_ASSERT_OK_AND_ASSIGN(auto array_handle1,
                          MakeTestArray(std::move(mock_array1)));
  TF_ASSERT_OK_AND_ASSIGN(auto array_handle2,
                          MakeTestArray(std::move(mock_array2)));

  uint64_t op_id = NewOpId();
  auto ifrt_request = NewIfrtRequest(op_id);
  ifrt_request->mutable_delete_array_request()->add_array_handle(array_handle1);
  ifrt_request->mutable_delete_array_request()->add_array_handle(array_handle2);
  TF_ASSERT_OK_AND_ASSIGN(auto resp, CallBackend(std::move(ifrt_request)));
  EXPECT_THAT(tsl::StatusFromProto(resp->response_metadata().status()),
              absl_testing::IsOk());
  TF_EXPECT_OK(
      CheckFuture(resp->delete_array_response().deletion_future_handle()));
}

TEST_P(IfrtBackendHandlerTest,
       DeleteArrayReturnsFutureWithNonExistentArrayHandle) {
  // Create one existing array.
  auto mock_array1 = tsl::MakeRef<xla::ifrt::MockArray>();
  EXPECT_CALL(*mock_array1, Delete())
      .WillOnce(Return(tsl::Future<>(absl::OkStatus())));
  TF_ASSERT_OK_AND_ASSIGN(auto real_handle,
                          MakeTestArray(std::move(mock_array1)));

  constexpr int kBadHandle = 400;
  auto ifrt_request = NewIfrtRequest(NewOpId());
  ifrt_request->mutable_delete_array_request()->add_array_handle(real_handle);
  ifrt_request->mutable_delete_array_request()->add_array_handle(kBadHandle);
  TF_ASSERT_OK_AND_ASSIGN(auto resp, CallBackend(std::move(ifrt_request)));

  EXPECT_THAT(
      CheckFuture(resp->delete_array_response().deletion_future_handle()),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
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
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
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
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_P(IfrtBackendHandlerTest, CompileSuccess) {
  std::vector<MockDevice> devices(4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_CALL(devices[i], Id()).WillRepeatedly(Return(DeviceId(i)));
  }

  std::vector<xla::ifrt::Device*> addressable_devices;
  for (int i = 0; i < 4; ++i) {
    addressable_devices.push_back(&devices[i]);
  }

  auto device_list = BasicDeviceList::Create(addressable_devices);

  auto executable = std::make_unique<MockLoadedExecutable>();
  EXPECT_CALL(*executable, name()).WillOnce(Return("executable_name"));
  EXPECT_CALL(*executable, num_devices()).WillOnce(Return(4));
  EXPECT_CALL(*executable, devices())
      .WillOnce(Return(std::make_optional(device_list)));
  EXPECT_CALL(*executable, addressable_devices())
      .WillOnce(Return(absl::MakeSpan(addressable_devices)));
  EXPECT_CALL(*executable, Fingerprint()).WillOnce(Return("fingerprint"));
  EXPECT_CALL(*executable, GetReadyFuture())
      .WillOnce(Return(tsl::Future<>(absl::OkStatus())));

  TF_ASSERT_OK_AND_ASSIGN(CompileResponse response,
                          CompileTestLoadedExecutable(std::move(executable)));
  EXPECT_THAT(response, Partially(EquivToProto(R"pb(
                name: "executable_name"
                num_devices: 4
                addressable_device_ids: [ 0, 1, 2, 3 ]
                device_ids: [ 0, 1, 2, 3 ]
                fingerprint_value: "fingerprint"
              )pb")));
  TF_EXPECT_OK(CheckFuture(response.ready_future_handle()));
}

TEST_P(IfrtBackendHandlerTest, CompileFailure) {
  ASSERT_THAT(
      CompileTestLoadedExecutable(absl::InternalError("injected error")),
      absl_testing::StatusIs(absl::StatusCode::kInternal,
                             StrEq("injected error")));
}

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
        xla::LayoutUtil::MakeDescendingLayout(/*num_dims=*/1)));
    parameter_layouts.push_back(std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::MakeDescendingLayout(/*num_dims=*/2)));
    EXPECT_CALL(*executable, GetParameterLayouts())
        .WillOnce(Return(std::move(parameter_layouts)));

    std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts;
    output_layouts.push_back(std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::MakeDescendingLayout(/*num_dims=*/2)));
    EXPECT_CALL(*executable, GetOutputLayouts())
        .WillOnce(Return(std::move(output_layouts)));
    EXPECT_CALL(*executable, GetOutputMemoryKinds())
        .WillOnce(Return(std::vector<std::vector<absl::string_view>>{{"foo"}}));

    auto request = NewIfrtRequest(NewOpId());
    LoadedExecutableMetadataRequest* metadata_request =
        request->mutable_loaded_executable_metadata_request();
    metadata_request->set_loaded_executable_handle(handle);

    EXPECT_THAT(CallBackend(std::move(request)),
                absl_testing::IsOkAndHolds(Pointee(Partially(EquivToProto(R"pb(
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

  std::vector<ArrayRef> outputs;
  outputs.reserve(kNumOutputs);
  for (int i = 0; i < kNumOutputs; ++i) {
    outputs.push_back(make_array());
  }

  EXPECT_CALL(*executable, Execute(SizeIs(kNumArgs), _, _))
      .WillOnce([&](absl::Span<ArrayRef> args,
                    const xla::ifrt::LoadedExecutable::ExecuteOptions& options,
                    std::optional<DeviceListRef> devices)
                    -> absl::StatusOr<LoadedExecutable::ExecuteResult> {
        return LoadedExecutable::ExecuteResult{
            .status = tsl::Future<>(absl::InternalError("injected error")),
            .outputs = outputs,
        };
      });

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
  TF_ASSERT_OK(execute_options.ToProto(
      *execute_request->mutable_execute_options(), ifrt_serdes_version()));

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
  TF_ASSERT_OK_AND_ASSIGN(auto sharding_proto,
                          SingleDeviceSharding::Create(device, MemoryKind())
                              ->ToProto(ifrt_serdes_version()));
  for (const auto& output :
       response->loaded_executable_execute_response().outputs()) {
    EXPECT_THAT(output.sharding(), EquivToProto(sharding_proto));
    EXPECT_NE(output.array_handle(), 0);
  }

  auto check_execution_result = [&](uint64_t handle) -> absl::Status {
    if (handle == 0) {
      return absl::InternalError("Test error, future handle is 0");
    }
    if (Version().protocol_version() >= protocol_version::kExecuteResult) {
      auto request = NewIfrtRequest(NewOpId());
      request->mutable_loaded_executable_fetch_execute_result_request()
          ->set_result_status_handle(handle);
      TF_ASSIGN_OR_RETURN(std::shared_ptr<IfrtResponse> response,
                          CallBackend(std::move(request)));
      return tsl::StatusFromProto(response->response_metadata().status());
    } else {
      return CheckFuture(handle);
    }
  };

  EXPECT_THAT(
      check_execution_result(
          response->loaded_executable_execute_response().status_handle()),
      absl_testing::StatusIs(absl::StatusCode::kInternal,
                             StrEq("injected error")));

  // The second call to `check_execution_result` fails since
  // `check_execution_result` above performs a destructive read.
  EXPECT_THAT(
      check_execution_result(
          response->loaded_executable_execute_response().status_handle()),
      absl_testing::StatusIs(
          absl::StatusCode::kNotFound,
          MatchesRegex("Unknown (future|result status) handle.*")));
}

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
      .WillOnce([&](absl::Span<ArrayRef> args,
                    const xla::ifrt::LoadedExecutable::ExecuteOptions& options,
                    std::optional<DeviceListRef> devices)
                    -> absl::StatusOr<LoadedExecutable::ExecuteResult> {
        return absl::InternalError("injected error");
      });

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
  TF_ASSERT_OK(execute_options.ToProto(
      *execute_request->mutable_execute_options(), ifrt_serdes_version()));

  auto status_is_err = absl_testing::StatusIs(absl::StatusCode::kInternal,
                                              StrEq("injected error"));

  EXPECT_THAT(CallBackend(std::move(request)), status_is_err);

  {
    const uint64_t handle = kFirstResultHandle + kNumOutputs;
    if (Version().protocol_version() >= protocol_version::kExecuteResult) {
      auto request = NewIfrtRequest(NewOpId());
      request->mutable_loaded_executable_fetch_execute_result_request()
          ->set_result_status_handle(handle);
      EXPECT_THAT(CallBackend(std::move(request)), status_is_err);
    } else {
      EXPECT_THAT(CheckFuture(handle), status_is_err);
    }
  }

  for (int i = 0; i < kNumOutputs; ++i) {
    EXPECT_THAT(CheckValueReady(kFirstResultHandle + i), status_is_err);
  }
}

TEST_P(IfrtBackendHandlerTest, LoadedExecutableDeviceTime) {
  if (tsl::kIsOpenSource) {
    GTEST_SKIP()
        << "DeviceTimeMeasurement implementation isn't available in OSS.";
  }
  if (Version().protocol_version() < protocol_version::kExecuteResult) {
    GTEST_SKIP()
        << "Device time measurement is not supported in this protocol version";
  }

  MockLoadedExecutable* executable;
  uint64_t handle;
  {
    auto e = std::make_unique<MockLoadedExecutable>();
    executable = e.get();
    TF_ASSERT_OK_AND_ASSIGN(CompileResponse response,
                            CompileTestLoadedExecutable(std::move(e)));
    handle = response.loaded_executable_handle();
  }

  EXPECT_CALL(*executable, Execute(_, _, _))
      .WillOnce([&](absl::Span<ArrayRef> args,
                    const xla::ifrt::LoadedExecutable::ExecuteOptions& options,
                    std::optional<DeviceListRef> devices)
                    -> absl::StatusOr<LoadedExecutable::ExecuteResult> {
        std::optional<uint64_t> device_time_key =
            xla::GetDeviceTimeMeasurementKey();
        if (device_time_key.has_value()) {
          xla::RecordDeviceTimeMeasurement(
              *device_time_key, absl::Microseconds(1234),
              xla::DeviceTimeMeasurement::DeviceType::kTpu);
        }
        LoadedExecutable::ExecuteResult result;
        result.status = tsl::Future<>(absl::OkStatus());
        return result;
      });

  constexpr uint64_t kResultStatusHandle = 1000;
  {
    auto request = NewIfrtRequest(NewOpId());
    LoadedExecutableExecuteRequest* execute_request =
        request->mutable_loaded_executable_execute_request();
    execute_request->set_loaded_executable_handle(handle);
    execute_request->set_result_status_handle(kResultStatusHandle);

    xla::ifrt::LoadedExecutable::ExecuteOptions execute_options;
    execute_options.fill_status = true;
    TF_ASSERT_OK(execute_options.ToProto(
        *execute_request->mutable_execute_options(), ifrt_serdes_version()));

    EXPECT_OK(CallBackend(std::move(request)));
  }

  {
    auto request = NewIfrtRequest(NewOpId());
    request->mutable_loaded_executable_fetch_execute_result_request()
        ->set_result_status_handle(kResultStatusHandle);
    TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                            CallBackend(std::move(request)));
    EXPECT_THAT(response, Pointee(Partially(EquivToProto(R"pb(
                  loaded_executable_fetch_execute_result_response {
                    device_time { key: "tpu" value: 1234.0 }
                    device_time { key: "gpu" value: 0 }
                  }
                )pb"))));
  }
}

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

    EXPECT_THAT(
        CallBackend(std::move(request)),
        absl_testing::StatusIs(absl::StatusCode::kNotFound,
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
    {
      auto serialize_options =
          std::make_unique<SerializeOptions>(ifrt_serdes_version());
      TF_ASSERT_OK_AND_ASSIGN(*compile_request->mutable_program(),
                              Serialize(program, std::move(serialize_options)));
    }
    {
      xla::ifrt::XlaCompileOptions compile_options;
      auto serialize_options =
          std::make_unique<SerializeOptions>(ifrt_serdes_version());
      TF_ASSERT_OK_AND_ASSIGN(
          *compile_request->mutable_compile_options(),
          Serialize(compile_options, std::move(serialize_options)));
    }

    TF_ASSERT_OK_AND_ASSIGN(std::string host_callback_serialized,
                            hcb->Serialize());
    compile_request->add_host_callbacks(std::move(host_callback_serialized));

    auto e = std::make_unique<MockLoadedExecutable>();
    executable = e.get();

    EXPECT_CALL(mock_compiler_, CompileAndLoad(_, _))
        .WillOnce(DoAll(
            [&](const std::unique_ptr<xla::ifrt::Program>& program,
                const std::unique_ptr<xla::ifrt::CompileOptions>& options) {
              auto* xla_compile_options =
                  llvm::cast<xla::ifrt::XlaCompileOptions>(options.get());
              auto& loaded_host_callbacks =
                  xla_compile_options->loaded_host_callbacks;
              ASSERT_EQ(loaded_host_callbacks.size(), 1);
              loaded_host_callback = loaded_host_callbacks.front();
            },
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
            absl_testing::IsOk());
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
        const HostBufferStore::MemRegion operands,
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
                absl_testing::IsOk());

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
              absl_testing::StatusIs(absl::StatusCode::kUnknown,
                                     StrEq("injected error")));
}

TEST_P(IfrtBackendHandlerTest, GetDefaultPjRtLayoutSuccess) {
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
              GetDefaultPjRtLayout(kDType, absl::MakeConstSpan(kDims),
                                   mock_device.get(), kMemoryKind))
      .WillOnce(Return(std::shared_ptr<const xla::PjRtLayout>(kDefaultLayout)));

  auto request = NewIfrtRequest(NewOpId());
  auto* default_layout_request = request->mutable_get_default_layout_request();
  kDType.ToProto(*default_layout_request->mutable_dtype(),
                 ifrt_serdes_version());
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

TEST_P(IfrtBackendHandlerTest, LoadedExecutableMetadataWithMpmd) {
  uint64_t handle;
  std::vector<xla::ifrt::Device*> mesh1_devices_backing = {
      mock_devices_[0].get()};
  {
    auto e = std::make_unique<MockMpmdLoadedExecutable>();
    MockMpmdLoadedExecutable* executable = e.get();

    ON_CALL(*executable, name()).WillByDefault(Return("mpmd_exec"));
    ON_CALL(*executable, num_devices()).WillByDefault(Return(1));
    auto device_list = BasicDeviceList::Create({});
    ON_CALL(*executable, devices()).WillByDefault(Return(device_list));
    ON_CALL(*executable, addressable_devices())
        .WillByDefault(Return(absl::Span<xla::ifrt::Device* const>({})));
    ON_CALL(*executable, Fingerprint())
        .WillByDefault(Return("mpmd_fingerprint"));
    ON_CALL(*executable, GetReadyFuture())
        .WillByDefault(Return(tsl::Future<>(absl::OkStatus())));

    ON_CALL(*executable, GetParameterShardings())
        .WillByDefault(Return(std::nullopt));
    ON_CALL(*executable, GetOutputShardings())
        .WillByDefault(Return(std::nullopt));
    ON_CALL(*executable, GetParameterLayouts())
        .WillByDefault(
            Return(std::vector<std::shared_ptr<const xla::PjRtLayout>>()));
    ON_CALL(*executable, GetOutputLayouts())
        .WillByDefault(
            Return(std::vector<std::shared_ptr<const xla::PjRtLayout>>()));
    ON_CALL(*executable, GetOutputMemoryKinds())
        .WillByDefault(Return(std::vector<std::vector<absl::string_view>>()));
    ON_CALL(*executable, GetDonatableInputIndices())
        .WillByDefault(Return(absl::Span<const int>()));
    ON_CALL(*executable, GetCompiledMemoryStats())
        .WillByDefault(Return(CompiledMemoryStats()));
    ON_CALL(*executable, SizeOfGeneratedCodeInBytes()).WillByDefault(Return(0));

    absl::flat_hash_map<std::string, CompiledMemoryStats> stats;
    stats["mesh1"] = CompiledMemoryStats();
    EXPECT_CALL(*executable, GetMpmdCompiledMemoryStats())
        .WillOnce(Return(stats));

    TF_ASSERT_OK_AND_ASSIGN(CompileResponse response,
                            CompileTestLoadedExecutable(std::move(e)));
    handle = response.loaded_executable_handle();
  }

  auto request = NewIfrtRequest(NewOpId());
  LoadedExecutableMpmdMetadataRequest* mpmd_metadata_request =
      request->mutable_loaded_executable_mpmd_metadata_request();
  mpmd_metadata_request->set_mpmd_loaded_executable_handle(handle);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                          CallBackend(std::move(request)));

  EXPECT_THAT(response, Pointee(Partially(EquivToProto(R"pb(
                loaded_executable_mpmd_metadata_response {
                  mpmd_compiled_memory_stats {
                    compiled_memory_stats {
                      key: "mesh1"
                      value {}
                    }
                  }
                }
              )pb"))));
}

TEST_P(IfrtBackendHandlerTest, LoadedExecutableMpmdCostAnalysis) {
  uint64_t handle;
  {
    auto e = std::make_unique<MockMpmdLoadedExecutable>();
    MockMpmdLoadedExecutable* executable = e.get();

    ON_CALL(*executable, name()).WillByDefault(Return("mpmd_exec"));
    ON_CALL(*executable, num_devices()).WillByDefault(Return(1));
    auto device_list = BasicDeviceList::Create({});
    ON_CALL(*executable, devices()).WillByDefault(Return(device_list));
    ON_CALL(*executable, addressable_devices())
        .WillByDefault(Return(absl::Span<xla::ifrt::Device* const>()));
    ON_CALL(*executable, Fingerprint())
        .WillByDefault(Return("mpmd_fingerprint"));
    ON_CALL(*executable, GetReadyFuture())
        .WillByDefault(Return(tsl::Future<>(absl::OkStatus())));

    absl::flat_hash_map<std::string, xla::ifrt::AttributeMap> cost_analysis;
    xla::ifrt::AttributeMap mesh1_attrs(xla::ifrt::AttributeMap::Map{
        {"cost", xla::ifrt::AttributeMap::FloatValue{1.0f}}});
    cost_analysis.insert({"mesh1", std::move(mesh1_attrs)});

    EXPECT_CALL(*executable, GetMpmdCostAnalysis())
        .WillOnce(Return(cost_analysis));

    TF_ASSERT_OK_AND_ASSIGN(CompileResponse response,
                            CompileTestLoadedExecutable(std::move(e)));
    handle = response.loaded_executable_handle();
  }

  auto request = NewIfrtRequest(NewOpId());
  request->mutable_loaded_executable_mpmd_cost_analysis_request()
      ->set_loaded_executable_handle(handle);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtResponse> response,
                          CallBackend(std::move(request)));

  EXPECT_THAT(response, Pointee(Partially(EquivToProto(R"pb(
                loaded_executable_mpmd_cost_analysis_response {
                  attributes {
                    attributes {
                      key: "mesh1"
                      value {
                        attributes {
                          key: "cost"
                          value { float_value: 1.0 }
                        }
                      }
                    }
                  }
                }
              )pb"))));
}

TEST_P(IfrtBackendHandlerTest, CompileSuccessWithMpmdAddressableDevices) {
  auto executable = std::make_unique<MockMpmdLoadedExecutable>();

  ON_CALL(*executable, name()).WillByDefault(Return("mpmd_exec"));
  ON_CALL(*executable, num_devices()).WillByDefault(Return(1));
  auto empty_device_list = BasicDeviceList::Create({});
  ON_CALL(*executable, devices()).WillByDefault(Return(empty_device_list));
  ON_CALL(*executable, addressable_devices())
      .WillByDefault(Return(absl::Span<xla::ifrt::Device* const>()));
  ON_CALL(*executable, Fingerprint()).WillByDefault(Return("mpmd_fingerprint"));
  ON_CALL(*executable, GetReadyFuture())
      .WillByDefault(Return(tsl::Future<>(absl::OkStatus())));

  std::vector<xla::ifrt::Device*> mesh1_devices = {mock_devices_[0].get()};
  std::vector<xla::ifrt::Device*> mesh2_devices = {mock_devices_[1].get()};

  absl::flat_hash_map<std::string, absl::Span<xla::ifrt::Device* const>>
      mpmd_addressable_devices_map;
  mpmd_addressable_devices_map["mesh1"] = absl::MakeConstSpan(mesh1_devices);
  mpmd_addressable_devices_map["mesh2"] = absl::MakeConstSpan(mesh2_devices);

  if (Version().protocol_version() >=
      protocol_version::kMpmdLoadedExecutableMethods) {
    EXPECT_CALL(*executable, GetMpmdAddressableDevices())
        .WillOnce(Return(mpmd_addressable_devices_map));
  }

  TF_ASSERT_OK_AND_ASSIGN(CompileResponse response,
                          CompileTestLoadedExecutable(std::move(executable)));

  if (Version().protocol_version() >=
      protocol_version::kMpmdLoadedExecutableMethods) {
    EXPECT_THAT(response, Partially(EquivToProto(R"pb(
                  mpmd_addressable_devices {
                    mpmd_addressable_devices {
                      key: "mesh1"
                      value { mpmd_addressable_device_ids: 0 }
                    }
                    mpmd_addressable_devices {
                      key: "mesh2"
                      value { mpmd_addressable_device_ids: 1 }
                    }
                  }
                )pb")));
  } else {
    EXPECT_FALSE(response.has_mpmd_addressable_devices());
  }
}

INSTANTIATE_TEST_SUITE_P(
    IfrtBackendHandlerTestWithAllVersions, IfrtBackendHandlerTest,
    testing::Range(protocol_version::kServerMin,
                   protocol_version::kServerMax + 1),
    [](const testing::TestParamInfo<IfrtBackendHandlerTest::ParamType>& info) {
      return absl::StrCat(info.param);
    });

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
