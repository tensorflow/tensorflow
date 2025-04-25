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

#include "xla/pjrt/c/pjrt_c_api_gpu.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/client/client_library.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/ffi_api.h"
#include "xla/ffi/type_id_registry.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_internal.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_test.h"
#include "xla/pjrt/c/pjrt_c_api_test_base.h"
#include "xla/pjrt/c/pjrt_c_api_triton_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/mem.h"

namespace pjrt {
namespace {

using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::tsl::testing::StatusIs;

#ifdef TENSORFLOW_USE_ROCM
const bool kUnused = (RegisterPjRtCApiTestFactory([]() { return GetPjrtApi(); },
                                                  /*platform_name=*/"rocm"),
                      true);
#else   // TENSORFLOW_USE_ROCM
const bool kUnused = (RegisterPjRtCApiTestFactory([]() { return GetPjrtApi(); },
                                                  /*platform_name=*/"cuda"),
                      true);
#endif  // TENSORFLOW_USE_ROCM

class PjrtCApiGpuTest : public PjrtCApiTestBase {
 public:
  PjrtCApiGpuTest() : PjrtCApiTestBase(GetPjrtApi()) {}
};

TEST_F(PjrtCApiGpuTest, CreateViewOfDeviceBuffer) {
  // Prepares a device memory ptr on GPU.
  auto [buffer, buffer_future] = create_buffer();
  TF_CHECK_OK(buffer_future.Await());
  PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args device_buffer_ptr_args;
  device_buffer_ptr_args.struct_size =
      PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE;
  device_buffer_ptr_args.extension_start = nullptr;
  device_buffer_ptr_args.buffer = buffer.get();
  PJRT_Error* device_buffer_ptr_error =
      api_->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&device_buffer_ptr_args);
  ASSERT_EQ(device_buffer_ptr_error, nullptr);
  // Looks up a device.
  PJRT_Buffer_Device_Args device_args = PJRT_Buffer_Device_Args{
      /*struct_size=*/PJRT_Buffer_Device_Args_STRUCT_SIZE,
      /*extension_start=*/nullptr,
      /*buffer=*/buffer.get(),
  };
  PJRT_Error* device_error = api_->PJRT_Buffer_Device(&device_args);
  ASSERT_EQ(device_error, nullptr);

  // Prepares PJRT_Client_CreateViewOfDeviceBuffer_Args.
  PJRT_Client_CreateViewOfDeviceBuffer_Args create_view_args;
  create_view_args.struct_size =
      PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
  create_view_args.extension_start = nullptr;
  create_view_args.client = client_;
  create_view_args.device_buffer_ptr = device_buffer_ptr_args.device_memory_ptr;
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::S32, {4});
  create_view_args.dims = shape.dimensions().data();
  create_view_args.num_dims = shape.dimensions().size();
  create_view_args.element_type =
      pjrt::ConvertToPjRtBufferType(shape.element_type());
  pjrt::BufferMemoryLayoutData c_layout_data;
  TF_ASSERT_OK_AND_ASSIGN(
      c_layout_data, pjrt::ConvertToBufferMemoryLayoutData(shape.layout()));
  create_view_args.layout = &(c_layout_data.c_layout);
  create_view_args.device = device_args.device;
  create_view_args.memory = nullptr;
  std::function<void()> on_delete_callback = []() mutable {};
  create_view_args.on_delete_callback_arg =
      new std::function(on_delete_callback);
  create_view_args.on_delete_callback = [](void* device_buffer_ptr,
                                           void* user_arg) {
    auto c_func = reinterpret_cast<std::function<void()>*>(user_arg);
    (*c_func)();
    delete c_func;
  };
  create_view_args.stream = reinterpret_cast<intptr_t>(nullptr);

  PJRT_Error* error =
      api_->PJRT_Client_CreateViewOfDeviceBuffer(&create_view_args);

  ASSERT_EQ(error, nullptr);
  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> view_buffer(
      create_view_args.buffer, ::pjrt::MakeBufferDeleter(api_));

  // Transfers view_buffer to host to verify.
  PJRT_Buffer_ToHostBuffer_Args to_host_args;
  to_host_args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  to_host_args.extension_start = nullptr;
  to_host_args.src = view_buffer.get();
  xla::Shape host_shape = xla::ShapeUtil::MakeShape(xla::F32, {4});
  auto literal = std::make_shared<xla::Literal>(host_shape);
  to_host_args.host_layout = nullptr;
  to_host_args.dst = literal->untyped_data();
  to_host_args.dst_size = xla::ShapeUtil::ByteSizeOfElements(host_shape);
  to_host_args.event = nullptr;

  PJRT_Error* to_host_error = api_->PJRT_Buffer_ToHostBuffer(&to_host_args);

  ASSERT_EQ(to_host_error, nullptr);
  xla::PjRtFuture<> transfer_to_host =
      ::pjrt::ConvertCEventToCppFuture(to_host_args.event, api_);
  TF_CHECK_OK(transfer_to_host.Await());
  ASSERT_EQ(literal->data<float>().size(), 4);
  std::vector<float> float_data(4);
  std::iota(float_data.begin(), float_data.end(), 41.0f);
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      xla::LiteralUtil::CreateR1<float>(float_data), *literal));
}
class PjrtCApiGpuTransferManagerTest : public PjrtCApiGpuTest {
 public:
  ~PjrtCApiGpuTransferManagerTest() override {
    transfer_manager_.reset(nullptr);
  }
  void CreateTransferManager(const xla::Shape& host_shape) {
    transfer_manager_ = create_transfer_manager(host_shape);
  }

  std::unique_ptr<PJRT_AsyncHostToDeviceTransferManager,
                  PJRT_AsyncHostToDeviceTransferManagerDeleter>
      transfer_manager_;
};

class PjrtCApiGpuBufferTest : public PjrtCApiGpuTest {
 public:
  PjrtCApiGpuBufferTest() : PjrtCApiGpuTest() {
    auto buffer_and_event = create_buffer();
    buffer_ = std::move(buffer_and_event.first);
    event_ = buffer_and_event.second;
  }

  ~PjrtCApiGpuBufferTest() override {
    // event_ needs to complete before the client is destroyed; otherwise there
    // is a data race between destroying the client and trying to access the
    // host context in the client for the callback after host to device transfer
    // is completed.
    TF_EXPECT_OK(event_.Await());
    // buffer_ must be destroyed before the client is destroyed or else the
    // unique_ptr for buffer_ will go out of scope causing heap-use-after-free
    // error.
    buffer_.reset(nullptr);
  }

  std::unique_ptr<PJRT_Buffer, PJRT_BufferDeleter> buffer_;
  xla::PjRtFuture<> event_;
};

TEST_F(PjrtCApiGpuBufferTest, CopyRawToHost) {
  size_t size = buffer_->buffer->GetOnDeviceSizeInBytes().value();
  PJRT_Buffer_CopyRawToHost_Args args;
  args.struct_size = PJRT_Buffer_CopyRawToHost_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  args.dst =
      tsl::port::AlignedMalloc(size, tsl::Allocator::kAllocatorAlignment);
  args.offset = 0;
  args.transfer_size = size;
  PJRT_Error* error = api_->PJRT_Buffer_CopyRawToHost(&args);
  ASSERT_THAT(error, IsNull());
  xla::PjRtFuture<> copy_to_host_event =
      ConvertCEventToCppFuture(args.event, api_);
  TF_EXPECT_OK(copy_to_host_event.Await());
  EXPECT_EQ(*(static_cast<float*>(args.dst)), 41);
  tsl::port::AlignedSizedFree(args.dst, tsl::Allocator::kAllocatorAlignment,
                              size);
}

TEST_F(PjrtCApiGpuBufferTest, CopyRawToHostWithInvalidOffset) {
  size_t size = buffer_->buffer->GetOnDeviceSizeInBytes().value();
  PJRT_Buffer_CopyRawToHost_Args args;
  args.struct_size = PJRT_Buffer_CopyRawToHost_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer_.get();
  args.dst =
      tsl::port::AlignedMalloc(size, tsl::Allocator::kAllocatorAlignment);
  args.offset = size + 1;  // offset is invalid
  args.transfer_size = size;
  PJRT_Error* error = api_->PJRT_Buffer_CopyRawToHost(&args);
  ASSERT_EQ(error, nullptr);
  xla::PjRtFuture<> copy_to_host_event =
      ConvertCEventToCppFuture(args.event, api_);
  absl::Status status = copy_to_host_event.Await();
  std::string expected_message = absl::StrFormat(
      "Copy raw buffer called on buffer size %lld with "
      "invalid offset %lld, transfer size %lld",
      size, args.offset, args.transfer_size);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr(expected_message)));
  free(args.dst);
}

// TODO(b/399495406): Add tests for other GPU Executable behaviors.
class PjrtCApiGpuExecutableTest : public PjrtCApiGpuTest {
 protected:
  std::unique_ptr<PJRT_LoadedExecutable, PJRT_LoadedExecutableDeleter>
      executable_;

  PjrtCApiGpuExecutableTest() {
    executable_ = create_executable(api_, client_);
  }

  ~PjrtCApiGpuExecutableTest() override { executable_.reset(); }
};

TEST_F(PjrtCApiGpuExecutableTest, GetCompiledMemoryStats) {
  auto executable = PjrtCApiTestBase::GetExecutable(executable_.get(), api_);
  TF_ASSERT_OK_AND_ASSIGN(auto stats,
                          pjrt::GetCompiledMemoryStats(api_, executable.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref_stats,
                          executable.get()->get()->GetCompiledMemoryStats());
  EXPECT_EQ(ref_stats.generated_code_size_in_bytes,
            stats.generated_code_size_in_bytes);
  EXPECT_EQ(ref_stats.argument_size_in_bytes, stats.argument_size_in_bytes);
  EXPECT_EQ(ref_stats.output_size_in_bytes, stats.output_size_in_bytes);
  EXPECT_EQ(ref_stats.alias_size_in_bytes, stats.alias_size_in_bytes);
  EXPECT_EQ(ref_stats.temp_size_in_bytes, stats.temp_size_in_bytes);
  EXPECT_EQ(ref_stats.host_generated_code_size_in_bytes,
            stats.host_generated_code_size_in_bytes);
  EXPECT_EQ(ref_stats.host_argument_size_in_bytes,
            stats.host_argument_size_in_bytes);
  EXPECT_EQ(ref_stats.host_output_size_in_bytes,
            stats.host_output_size_in_bytes);
  EXPECT_EQ(ref_stats.host_alias_size_in_bytes, stats.host_alias_size_in_bytes);
  EXPECT_EQ(ref_stats.host_temp_size_in_bytes, stats.host_temp_size_in_bytes);
}

TEST_F(PjrtCApiGpuTest, CreateAndDestroyExecuteContext) {
  PJRT_ExecuteContext_Create_Args create_arg;
  create_arg.struct_size = PJRT_ExecuteContext_Create_Args_STRUCT_SIZE;
  create_arg.extension_start = nullptr;
  create_arg.context = nullptr;

  EXPECT_EQ(api_->PJRT_ExecuteContext_Create(&create_arg), nullptr);
  EXPECT_NE(create_arg.context, nullptr);

  const PJRT_FFI_Extension* ffi_extension =
      pjrt::FindExtension<PJRT_FFI_Extension>(
          api_, PJRT_Extension_Type::PJRT_Extension_Type_FFI);
  ASSERT_NE(ffi_extension, nullptr);

  std::string string_data = "string_data";

  PJRT_FFI_UserData_Add_Args add_args;
  add_args.struct_size = PJRT_FFI_UserData_Add_Args_STRUCT_SIZE;
  add_args.extension_start = nullptr;
  add_args.user_data.type_id = 42;
  add_args.user_data.data = &string_data;
  add_args.user_data.deleter = nullptr;
  add_args.context = create_arg.context;
  EXPECT_EQ(ffi_extension->user_data_add(&add_args), nullptr);

  TF_ASSERT_OK_AND_ASSIGN(
      auto lookup_user_data,
      create_arg.context->execute_context->ffi_context().Lookup(
          xla::ffi::TypeIdRegistry::TypeId(42)));
  EXPECT_EQ(lookup_user_data, &string_data);

  PJRT_ExecuteContext_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.context = create_arg.context;

  api_->PJRT_ExecuteContext_Destroy(&destroy_args);
}

TEST_F(PjrtCApiGpuTest, DmaMapAndUnmap) {
  size_t dma_size = 1024 * 1024;
  size_t alignment = 1024 * 1024;
  auto host_dma_ptr = xla::AlignedAlloc(alignment, dma_size);

  PJRT_Client_DmaMap_Args dma_args;
  dma_args.struct_size = PJRT_Client_DmaMap_Args_STRUCT_SIZE;
  dma_args.extension_start = nullptr;
  dma_args.client = client_;
  dma_args.data = host_dma_ptr.get();
  dma_args.size = dma_size;
  PJRT_Error* dma_error = api_->PJRT_Client_DmaMap(&dma_args);
  ASSERT_EQ(dma_error, nullptr);
  MakeErrorDeleter(api_)(dma_error);

  PJRT_Client_DmaUnmap_Args unmap_args;
  unmap_args.struct_size = PJRT_Client_DmaUnmap_Args_STRUCT_SIZE;
  unmap_args.extension_start = nullptr;
  unmap_args.client = client_;
  unmap_args.data = host_dma_ptr.get();
  PJRT_Error* unmap_error = api_->PJRT_Client_DmaUnmap(&unmap_args);
  ASSERT_EQ(unmap_error, nullptr);
  MakeErrorDeleter(api_)(unmap_error);
}

TEST_F(PjrtCApiGpuTransferManagerTest, SetBufferError) {
  xla::Shape host_shape =
      xla::ShapeUtil::MakeShape(xla::F32, /*dimensions=*/{8});
  std::vector<float> float_data = {1, 2, 3, 4, 5, 6, 7, 8};

  CreateTransferManager(host_shape);

  PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args add_metadata_args;
  add_metadata_args.struct_size =
      PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args_STRUCT_SIZE;
  add_metadata_args.extension_start = nullptr;
  add_metadata_args.transfer_manager = transfer_manager_.get();
  std::vector<PJRT_NamedValue> transfer_metadata;
  transfer_metadata.reserve(1);
  std::string test_key = "test_key";
  std::string test_value = "test_value";
  PJRT_NamedValue test_named_value;
  test_named_value.name = test_key.c_str();
  test_named_value.name_size = test_key.size();
  test_named_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kString;
  test_named_value.string_value = test_value.c_str();
  test_named_value.value_size = test_value.size();
  transfer_metadata.push_back(test_named_value);
  add_metadata_args.transfer_metadata = transfer_metadata.data();
  add_metadata_args.num_metadata = transfer_metadata.size();
  PJRT_Error* add_metadata_error =
      PJRT_AsyncHostToDeviceTransferManager_AddMetadata(&add_metadata_args);
  ASSERT_EQ(add_metadata_error, nullptr);

  PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args buffer_count_args;
  buffer_count_args.struct_size =
      PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args_STRUCT_SIZE;
  buffer_count_args.extension_start = nullptr;
  buffer_count_args.transfer_manager = transfer_manager_.get();
  PJRT_Error* buffer_count_error =
      PJRT_AsyncHostToDeviceTransferManager_BufferCount(&buffer_count_args);
  ASSERT_EQ(buffer_count_error, nullptr);
  EXPECT_EQ(buffer_count_args.buffer_count, 1);

  PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args retrieve_args;
  retrieve_args.struct_size =
      PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args_STRUCT_SIZE;
  retrieve_args.extension_start = nullptr;
  retrieve_args.transfer_manager = transfer_manager_.get();
  retrieve_args.buffer_index = 0;
  PJRT_Error* retrieve_error =
      PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(&retrieve_args);
  ASSERT_EQ(retrieve_error, nullptr);
  PJRT_Buffer* buffer_out = retrieve_args.buffer_out;

  PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args
      set_buffer_error_args;
  set_buffer_error_args.struct_size =
      PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args_STRUCT_SIZE;
  set_buffer_error_args.extension_start = nullptr;
  set_buffer_error_args.transfer_manager = transfer_manager_.get();
  set_buffer_error_args.buffer_index = 0;
  set_buffer_error_args.error_code = PJRT_Error_Code_INTERNAL;
  std::string error_message = "test error";
  set_buffer_error_args.error_message = error_message.data();
  set_buffer_error_args.error_message_size = error_message.size();
  PJRT_Error* set_buffer_error_error =
      PJRT_AsyncHostToDeviceTransferManager_SetBufferError(
          &set_buffer_error_args);
  ASSERT_EQ(set_buffer_error_error, nullptr);

  EXPECT_THAT(buffer_out->buffer->ToLiteralSync(),
              StatusIs(absl::StatusCode::kInternal, HasSubstr(error_message)));

  PJRT_BufferDeleter buffer_deleter = MakeBufferDeleter(api_);
  buffer_deleter(buffer_out);
}

TEST_F(PjrtCApiGpuTransferManagerTest, TransferRawDataToBufferIsSuccessful) {
  xla::Shape host_shape =
      xla::ShapeUtil::MakeShape(xla::U32, /*dimensions=*/{8});
  std::vector<uint32_t> data = {1, 2, 3, 4, 5, 6, 7, 8};
  absl::Span<const char> raw_data_view = GetRawView(data);
  CreateTransferManager(host_shape);

  PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args retrieve_args;
  retrieve_args.struct_size =
      PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args_STRUCT_SIZE;
  retrieve_args.extension_start = nullptr;
  retrieve_args.transfer_manager = transfer_manager_.get();
  retrieve_args.buffer_index = 0;
  PJRT_Error* retrieve_error =
      PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer(&retrieve_args);
  ASSERT_EQ(retrieve_error, nullptr);
  PJRT_Buffer* buffer_out = retrieve_args.buffer_out;
  EXPECT_FALSE(buffer_out->buffer->GetReadyFuture().IsReady());

  TF_ASSERT_OK_AND_ASSIGN(xla::Shape result_shape,
                          buffer_out->buffer->HostShape());
  EXPECT_EQ(result_shape, host_shape);

  PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args buffer_size_args;
  buffer_size_args.struct_size =
      PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args_STRUCT_SIZE;
  buffer_size_args.extension_start = nullptr;
  buffer_size_args.transfer_manager = transfer_manager_.get();
  buffer_size_args.buffer_index = 0;
  PJRT_Error* buffer_size_error =
      PJRT_AsyncHostToDeviceTransferManager_BufferSize(&buffer_size_args);
  ASSERT_EQ(buffer_size_error, nullptr);
  EXPECT_EQ(buffer_size_args.buffer_size,
            buffer_out->buffer->GetOnDeviceSizeInBytes().value());

  PJRT_AsyncHostToDeviceTransferManager_Device_Args device_args;
  device_args.struct_size =
      PJRT_AsyncHostToDeviceTransferManager_Device_Args_STRUCT_SIZE;
  device_args.extension_start = nullptr;
  device_args.transfer_manager = transfer_manager_.get();
  PJRT_Error* device_error =
      PJRT_AsyncHostToDeviceTransferManager_Device(&device_args);
  ASSERT_EQ(device_error, nullptr);
  EXPECT_EQ(device_args.device_out, GetClientDevices()[0]);

  PJRT_AsyncHostToDeviceTransferManager_TransferData_Args transfer_args;
  transfer_args.struct_size =
      PJRT_AsyncHostToDeviceTransferManager_TransferData_Args_STRUCT_SIZE;
  transfer_args.extension_start = nullptr;
  transfer_args.transfer_manager = transfer_manager_.get();
  transfer_args.buffer_index = 0;
  transfer_args.data = raw_data_view.data();
  transfer_args.offset = 0;
  transfer_args.transfer_size = raw_data_view.size();
  transfer_args.is_last_transfer = true;
  PJRT_Error* transfer_error =
      PJRT_AsyncHostToDeviceTransferManager_TransferData(&transfer_args);
  ASSERT_EQ(transfer_error, nullptr);
  std::unique_ptr<PJRT_Event, PJRT_EventDeleter> done_with_h2d_transfer_event(
      transfer_args.done_with_h2d_transfer, MakeEventDeleter(api_));

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> literal,
                          buffer_out->buffer->ToLiteralSync());
  EXPECT_EQ(literal->element_count(), 8);
  EXPECT_THAT(literal->data<uint32_t>(), ElementsAreArray(data));

  PJRT_BufferDeleter buffer_deleter = MakeBufferDeleter(api_);
  buffer_deleter(buffer_out);
}

absl::StatusOr<PJRT_Client_Create_Args> BuildCreateArg(
    ::pjrt::PJRT_KeyValueCallbackData* kv_callback_data,
    std::vector<PJRT_NamedValue>& c_options) {
  PJRT_Client_Create_Args args;
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.create_options = c_options.data();
  args.num_options = c_options.size();
  args.kv_get_callback = kv_callback_data->c_kv_get;
  args.kv_get_user_arg = &kv_callback_data->kv_get_c_func;
  args.kv_put_callback = kv_callback_data->c_kv_put;
  args.kv_put_user_arg = &kv_callback_data->kv_put_c_func;
  args.kv_try_get_user_arg = &kv_callback_data->kv_try_get_c_func;
  args.kv_try_get_callback = kv_callback_data->c_kv_try_get;
  args.client = nullptr;
  return args;
}

TEST(PjrtCApiGpuKVStoreTest, CreateClientWithKVCallback) {
  auto api = GetPjrtApi();
  auto kv_store = std::make_shared<xla::InMemoryKeyValueStore>();
  std::shared_ptr<::pjrt::PJRT_KeyValueCallbackData> kv_callback_data =
      ::pjrt::ConvertToCKeyValueCallbacks(kv_store);
  xla::ClientLibrary::DestroyLocalInstances();

  int num_nodes = 2;
  std::vector<std::thread> threads;
  // `num_nodes` clients will be created on the same GPU.
  for (int i = 0; i < num_nodes; i++) {
    threads.emplace_back([api, i, num_nodes,
                          kv_callback_data = kv_callback_data,
                          kv_store = kv_store] {
      absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
          {"num_nodes", static_cast<int64_t>(num_nodes)},
          {"node_id", static_cast<int64_t>(i)},
          {"visible_devices", std::vector<int64_t>({0})}};
      TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                              ::pjrt::ConvertToPjRtNamedValueList(options));
      TF_ASSERT_OK_AND_ASSIGN(
          PJRT_Client_Create_Args create_arg,
          BuildCreateArg(kv_callback_data.get(), c_options));
      PJRT_Error* error = api->PJRT_Client_Create(&create_arg);
      EXPECT_EQ(error, nullptr) << error->status.message();

      PJRT_Client_Devices_Args device_args;
      device_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
      device_args.extension_start = nullptr;
      device_args.client = create_arg.client;

      PJRT_Error* device_error = api->PJRT_Client_Devices(&device_args);
      EXPECT_EQ(device_error, nullptr);
      EXPECT_EQ(device_args.num_devices, 2);

      PJRT_Client_AddressableDevices_Args addressable_device_args;
      addressable_device_args.struct_size =
          PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
      addressable_device_args.extension_start = nullptr;
      addressable_device_args.client = create_arg.client;

      PJRT_Error* addressable_device_error =
          api->PJRT_Client_AddressableDevices(&addressable_device_args);
      EXPECT_EQ(addressable_device_error, nullptr);
      EXPECT_EQ(addressable_device_args.num_addressable_devices, 1);

      PJRT_Client_Destroy_Args destroy_args;
      destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
      destroy_args.extension_start = nullptr;
      destroy_args.client = create_arg.client;

      PJRT_Error* destroy_error = api->PJRT_Client_Destroy(&destroy_args);
      CHECK_EQ(destroy_error, nullptr);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

TEST(PjrtCApiGpuAllocatorTest, ValidOptionsParsing) {
  auto api = GetPjrtApi();
  std::vector<std::string> allocator_options = {"default", "platform", "bfc",
                                                "cuda_async"};
  for (const std::string& allocator_option : allocator_options) {
#ifdef TENSORFLOW_USE_ROCM
    if (allocator_option == "cuda_async") {
      VLOG(1) << "cuda_async allocator not available on ROCm!";
      continue;
    }
#endif
    absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
        {"allocator", allocator_option},
        {"visible_devices", xla::PjRtValueType(std::vector<int64_t>{0, 1})},
    };
    if (allocator_option == "bfc" || allocator_option == "cuda_async") {
      options["memory_fraction"] = 0.5f;
    }
    if (allocator_option == "cuda_async") {
      options["preallocate"] = true;
    }
    TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                            ::pjrt::ConvertToPjRtNamedValueList(options));
    PJRT_Client_Create_Args create_arg;
    create_arg.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    create_arg.extension_start = nullptr;
    create_arg.client = nullptr;
    create_arg.create_options = c_options.data();
    create_arg.num_options = c_options.size();
    PJRT_Error* error = api->PJRT_Client_Create(&create_arg);
    EXPECT_EQ(error, nullptr) << error->status.message();

    PJRT_Client_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.client = create_arg.client;

    PJRT_Error* destroy_error = api->PJRT_Client_Destroy(&destroy_args);
    CHECK_EQ(destroy_error, nullptr);

    PJRT_Error_Destroy_Args error_destroy_args;
    error_destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    error_destroy_args.extension_start = nullptr;
    error_destroy_args.error = error;
    api->PJRT_Error_Destroy(&error_destroy_args);
  }
}

TEST(PjrtCApiGpuAllocatorTest, InvalidAllocatorOptionsParsing) {
  auto api = GetPjrtApi();
  absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
      {"allocator", static_cast<std::string>("invalid_allocator")},
      {"memory_fraction", 0.5f},
      {"preallocate", true},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                          ::pjrt::ConvertToPjRtNamedValueList(options));
  PJRT_Client_Create_Args create_arg;
  create_arg.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create_arg.extension_start = nullptr;
  create_arg.client = nullptr;
  create_arg.create_options = c_options.data();
  create_arg.num_options = c_options.size();
  PJRT_Error* error = api->PJRT_Client_Create(&create_arg);
  EXPECT_NE(error, nullptr);
  EXPECT_THAT(error->status,
              ::tsl::testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  "Allocator invalid_allocator not supported for PJRT GPU "
                  "plugin. Supported allocator options are: 'default', "
                  "'platform', 'bfc' and 'cuda_async'."));

  PJRT_Error_Destroy_Args error_destroy_args;
  error_destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  error_destroy_args.extension_start = nullptr;
  error_destroy_args.error = error;

  api->PJRT_Error_Destroy(&error_destroy_args);
}

TEST(PjrtCApiPlatformNameTest, AvailablePlatformName) {
  auto api = GetPjrtApi();
  std::string expected_platform_name_for_cuda = "cuda";
  std::string expected_platform_name_for_rocm = "rocm";
  absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
      {"platform_name", static_cast<std::string>("gpu")},
      {"allocator", static_cast<std::string>("default")},
      {"visible_devices", xla::PjRtValueType(std::vector<int64_t>{0, 1})},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                          ::pjrt::ConvertToPjRtNamedValueList(options));
  PJRT_Client_Create_Args create_arg;
  create_arg.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create_arg.extension_start = nullptr;
  create_arg.client = nullptr;
  create_arg.create_options = c_options.data();
  create_arg.num_options = c_options.size();
  PJRT_Error* error = api->PJRT_Client_Create(&create_arg);
  EXPECT_EQ(error, nullptr) << error->status.message();

  PJRT_Client_PlatformName_Args platform_name_args;
  platform_name_args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  platform_name_args.extension_start = nullptr;
  platform_name_args.client = create_arg.client;

  PJRT_Error* platform_name_error =
      api->PJRT_Client_PlatformName(&platform_name_args);
  EXPECT_EQ(platform_name_error, nullptr);
  EXPECT_THAT(platform_name_args.platform_name,
              testing::AnyOf(expected_platform_name_for_cuda,
                             expected_platform_name_for_rocm));

  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.client = create_arg.client;

  PJRT_Error* destroy_error = api->PJRT_Client_Destroy(&destroy_args);
  CHECK_EQ(destroy_error, nullptr);
}

TEST(PjrtCApiPlatformNameTest, UnavailablePlatformName) {
  auto api = GetPjrtApi();
  absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
      {"platform_name", static_cast<std::string>("invalid_platform_name")},
      {"allocator", static_cast<std::string>("default")},
      {"visible_devices", xla::PjRtValueType(std::vector<int64_t>{0, 1})},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                          ::pjrt::ConvertToPjRtNamedValueList(options));
  PJRT_Client_Create_Args create_arg;
  create_arg.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create_arg.extension_start = nullptr;
  create_arg.client = nullptr;
  create_arg.create_options = c_options.data();
  create_arg.num_options = c_options.size();
  PJRT_Error* error = api->PJRT_Client_Create(&create_arg);
  EXPECT_NE(error, nullptr);
  EXPECT_THAT(error->status,
              ::tsl::testing::StatusIs(
                  absl::StatusCode::kNotFound,
                  testing::StartsWith("Could not find registered platform with "
                                      "name: \"invalid_platform_name\". "
                                      "Available platform names are:")));

  PJRT_Error_Destroy_Args error_destroy_args;
  error_destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  error_destroy_args.extension_start = nullptr;
  error_destroy_args.error = error;

  api->PJRT_Error_Destroy(&error_destroy_args);
}

TEST(PjrtCApiGpuExtensionTest,
     ShouldStageHostToDeviceTransferWithOptionSetToTrue) {
  auto api = GetPjrtApi();

  absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
      {"should_stage_host_to_device_transfers", true},
      {"visible_devices", xla::PjRtValueType(std::vector<int64_t>{0})},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                          ::pjrt::ConvertToPjRtNamedValueList(options));
  PJRT_Client_Create_Args create_arg;
  create_arg.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create_arg.extension_start = nullptr;
  create_arg.client = nullptr;
  create_arg.create_options = c_options.data();
  create_arg.num_options = c_options.size();
  PJRT_Error* error = api->PJRT_Client_Create(&create_arg);
  EXPECT_EQ(error, nullptr) << error->status.message();

  xla::PjRtClient* cpp_client = create_arg.client->client.get();
  auto* gpu_client =
      tensorflow::down_cast<xla::StreamExecutorGpuClient*>(cpp_client);
  EXPECT_TRUE(gpu_client->should_stage_host_to_device_transfers());

  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.client = create_arg.client;
  PJRT_Error* destroy_error = api->PJRT_Client_Destroy(&destroy_args);
  CHECK_EQ(destroy_error, nullptr);
}

TEST(PjrtCApiGpuExtensionTest,
     ShouldStageHostToDeviceTransferWithOptionSetToFalse) {
  auto api = GetPjrtApi();

  absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
      {"should_stage_host_to_device_transfers", false},
      {"visible_devices", xla::PjRtValueType(std::vector<int64_t>{0})},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                          ::pjrt::ConvertToPjRtNamedValueList(options));
  PJRT_Client_Create_Args create_arg;
  create_arg.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create_arg.extension_start = nullptr;
  create_arg.client = nullptr;
  create_arg.create_options = c_options.data();
  create_arg.num_options = c_options.size();
  PJRT_Error* error = api->PJRT_Client_Create(&create_arg);
  EXPECT_EQ(error, nullptr) << error->status.message();

  xla::PjRtClient* cpp_client = create_arg.client->client.get();
  auto* gpu_client =
      tensorflow::down_cast<xla::StreamExecutorGpuClient*>(cpp_client);
  EXPECT_FALSE(gpu_client->should_stage_host_to_device_transfers());

  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.client = create_arg.client;
  PJRT_Error* destroy_error = api->PJRT_Client_Destroy(&destroy_args);
  CHECK_EQ(destroy_error, nullptr);
}

TEST(PJRTGpuDeviceTopologyTest, CreateGpuTopology) {
  auto pjrt_api = gpu_plugin::GetGpuPjrtApi();

  PJRT_TopologyDescription_Create_Args args;
  args.struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology = nullptr;
  args.num_options = 0;
  args.create_options = nullptr;

  PJRT_Error* error = pjrt_api->PJRT_TopologyDescription_Create(&args);
  EXPECT_EQ(error, nullptr) << error->status.message();

  auto pjrt_topology =
      reinterpret_cast<const PJRT_TopologyDescription*>(args.topology);
  ASSERT_NE(pjrt_topology, nullptr);

  EXPECT_TRUE((pjrt_topology->topology->platform_id() == xla::CudaId() &&
               pjrt_topology->topology->platform_name() == xla::CudaName()) ||
              (pjrt_topology->topology->platform_id() == xla::RocmId() &&
               pjrt_topology->topology->platform_name() == xla::RocmName()));

  PJRT_TopologyDescription_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_TopologyDescription_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.topology = const_cast<PJRT_TopologyDescription*>(pjrt_topology);
  PJRT_Error* destroy_error =
      pjrt_api->PJRT_TopologyDescription_Destroy(&destroy_args);
  EXPECT_EQ(destroy_error, nullptr) << destroy_error->status.message();
}

constexpr char const* kTargetConfigString = R"(gpu_device_info {
  threads_per_block_limit: 1024
  threads_per_warp: 32
  shared_memory_per_block: 49152
  shared_memory_per_core: 98304
  threads_per_core_limit: 2048
  core_count: 80
  fpus_per_core: 64
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65535
  block_dim_limit_z: 65535
  memory_bandwidth: 898048000000
  l2_cache_size: 6291456
  clock_rate_ghz: 1.53
  device_memory_size: 34072559616
  shared_memory_per_block_optin: 98304
  cuda_compute_capability {
    major: 7
  }
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}
platform_name: "CUDA"
dnn_version_info {
  major: 9
  minor: 3
}
device_description_str: "Tesla V100-SXM2-32GB"
)";

TEST(PJRTGpuDeviceTopologyTest, CreateExplicitGpuTopologyAndTargetConfig) {
  auto pjrt_api = gpu_plugin::GetGpuPjrtApi();

  absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
      {"topology", static_cast<std::string>("16 x 2 x 4")},
      {"target_config", static_cast<std::string>(kTargetConfigString)}};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                          ::pjrt::ConvertToPjRtNamedValueList(options));

  PJRT_TopologyDescription_Create_Args args;
  args.struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology = nullptr;
  args.num_options = c_options.size();
  args.create_options = c_options.data();

  PJRT_Error* error = pjrt_api->PJRT_TopologyDescription_Create(&args);
  EXPECT_EQ(error, nullptr) << error->status.message();

  auto pjrt_topology =
      reinterpret_cast<const PJRT_TopologyDescription*>(args.topology);
  ASSERT_NE(pjrt_topology, nullptr);

  EXPECT_TRUE((pjrt_topology->topology->platform_id() == xla::CudaId() &&
               pjrt_topology->topology->platform_name() == xla::CudaName()) ||
              (pjrt_topology->topology->platform_id() == xla::RocmId() &&
               pjrt_topology->topology->platform_name() == xla::RocmName()));

  EXPECT_EQ(pjrt_topology->topology->ProcessCount().value(), 16 * 2);
  EXPECT_EQ(pjrt_topology->topology->DeviceDescriptions().size(), 16 * 2 * 4);
  EXPECT_EQ(pjrt_topology->topology->DeviceDescriptions()[0]->device_kind(),
            "Tesla V100-SXM2-32GB");
  for (int i = 0; i < pjrt_topology->topology->DeviceDescriptions().size() - 1;
       ++i) {
    EXPECT_EQ(pjrt_topology->topology->DeviceDescriptions()[i]->id(), i);
  }

  PJRT_TopologyDescription_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_TopologyDescription_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.topology = const_cast<PJRT_TopologyDescription*>(pjrt_topology);
  PJRT_Error* destroy_error =
      pjrt_api->PJRT_TopologyDescription_Destroy(&destroy_args);
  EXPECT_EQ(destroy_error, nullptr) << destroy_error->status.message();
}

TEST(PJRTGpuDeviceTopologyTest, CreateExplicitGpuTopology) {
  auto pjrt_api = gpu_plugin::GetGpuPjrtApi();

  absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
      {"topology", static_cast<std::string>("16 x 2 x 4")}};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                          ::pjrt::ConvertToPjRtNamedValueList(options));

  PJRT_TopologyDescription_Create_Args args;
  args.struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology = nullptr;
  args.num_options = c_options.size();
  args.create_options = c_options.data();

  PJRT_Error* error = pjrt_api->PJRT_TopologyDescription_Create(&args);
  EXPECT_EQ(error, nullptr) << error->status.message();

  auto pjrt_topology =
      reinterpret_cast<const PJRT_TopologyDescription*>(args.topology);
  ASSERT_NE(pjrt_topology, nullptr);

  EXPECT_EQ(pjrt_topology->topology->ProcessCount().value(), 16 * 2);
  EXPECT_EQ(pjrt_topology->topology->DeviceDescriptions().size(), 16 * 2 * 4);

  PJRT_TopologyDescription_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_TopologyDescription_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.topology = const_cast<PJRT_TopologyDescription*>(pjrt_topology);
  PJRT_Error* destroy_error =
      pjrt_api->PJRT_TopologyDescription_Destroy(&destroy_args);
  EXPECT_EQ(destroy_error, nullptr) << destroy_error->status.message();
}

void TestCustomCallV2() {}

TEST(PjrtCApiGpuExtensionTest, CustomCallUntyped) {
  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  std::string function_name = "untyped_function_name";
  args.function_name = function_name.c_str();
  args.function_name_size = function_name.size();
  args.api_version = 0;
  args.handler_instantiate = nullptr;
  args.handler_prepare = nullptr;
  args.handler_initialize = nullptr;
  args.handler_execute = reinterpret_cast<void*>(&TestCustomCallV2);
  auto api = GetPjrtApi();
  const PJRT_Extension_Base* next = api->extension_start;
  while (next != nullptr &&
         next->type !=
             PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call) {
    next = next->next;
  }
  ASSERT_NE(next, nullptr);

  PJRT_Error* error =
      reinterpret_cast<const PJRT_Gpu_Custom_Call*>(next)->custom_call(&args);

  CHECK_EQ(error, nullptr);
  void* custom_call = xla::CustomCallTargetRegistry::Global()->Lookup(
      function_name, stream_executor::GpuPlatformName());
  EXPECT_EQ(custom_call, reinterpret_cast<void*>(&TestCustomCallV2));
}

TEST(PjrtCApiGpuExtensionTest, CustomCallTyped) {
  static constexpr auto* noop = +[] { return xla::ffi::Error::Success(); };
  XLA_FFI_DEFINE_HANDLER(kNoop, noop, xla::ffi::Ffi::Bind());

  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  std::string function_name = "typed_function_name";
  args.function_name = function_name.c_str();
  args.function_name_size = function_name.size();
  args.api_version = 1;
  args.handler_instantiate = nullptr;
  args.handler_prepare = nullptr;
  args.handler_initialize = nullptr;
  args.handler_execute = reinterpret_cast<void*>(kNoop);
  auto api = GetPjrtApi();
  const PJRT_Extension_Base* next = api->extension_start;
  while (next != nullptr &&
         next->type !=
             PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call) {
    next = next->next;
  }
  ASSERT_NE(next, nullptr);

  PJRT_Error* error =
      reinterpret_cast<const PJRT_Gpu_Custom_Call*>(next)->custom_call(&args);

  CHECK_EQ(error, nullptr);
  auto registration =
      xla::ffi::FindHandler(function_name, stream_executor::GpuPlatformName())
          .value();
  EXPECT_EQ(reinterpret_cast<void*>(registration.bundle.execute), kNoop);
}

constexpr absl::string_view kAddOneTTIR = R"(
module {
  tt.func public @add_one(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}) {
    %0 = tt.get_program_id x : i32
    %1 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    %2 = tt.load %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    %cst = arith.constant 1.000000e+00 : f32
    %3 = arith.addf %1, %cst : f32
    %4 = tt.load %arg2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    tt.store %arg2, %3 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
    %5 = tt.load %arg3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
    tt.store %arg3, %2 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
    tt.return
  }
}
)";

TEST(PjrtCAPIGpuExtensionTest, TritonCompile) {
  constexpr absl::string_view kArchName = "7.0";
  PJRT_Triton_Compile_Args args;
  args.struct_size = PJRT_Triton_Compile_Args_STRUCT_SIZE;
  args.module = kAddOneTTIR.data();
  args.module_size = kAddOneTTIR.size();
  args.arch_name = kArchName.data();
  args.arch_name_size = kArchName.size();
  args.num_stages = 1;
  args.num_ctas = 1;
  args.num_warps = 1;
  auto api = GetPjrtApi();
  const auto* triton_ext = pjrt::FindExtension<PJRT_Triton_Extension>(
      api, PJRT_Extension_Type::PJRT_Extension_Type_Triton);
  ASSERT_NE(triton_ext, nullptr);

  PJRT_Error* error = triton_ext->compile(&args);
  CHECK_EQ(error, nullptr) << error->status.message();
  delete[] args.out_asm;
}

}  // namespace
}  // namespace pjrt
