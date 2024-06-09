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

#include <cstdint>
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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/ffi_api.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_test.h"
#include "xla/pjrt/c/pjrt_c_api_test_base.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/tests/literal_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace pjrt {
namespace {

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
  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer =
      create_buffer().first;
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
          xla::ffi::ExecutionContext::TypeId(42)));
  EXPECT_EQ(lookup_user_data, &string_data);

  PJRT_ExecuteContext_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.context = create_arg.context;

  api_->PJRT_ExecuteContext_Destroy(&destroy_args);
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
  args.client = nullptr;
  return args;
}

TEST(PjrtCApiGpuKVStoreTest, CreateClientWithKVCallback) {
  auto api = GetPjrtApi();
  auto kv_store = std::make_shared<xla::InMemoryKeyValueStore>();
  std::shared_ptr<::pjrt::PJRT_KeyValueCallbackData> kv_callback_data =
      ::pjrt::ConvertToCKeyValueCallbacks(kv_store);

  int num_nodes = 2;
  std::vector<std::thread> threads;
  // `num_nodes` clients will be created on the same GPU.
  for (int i = 0; i < num_nodes; i++) {
    threads.emplace_back([api, i, num_nodes,
                          kv_callback_data = kv_callback_data,
                          kv_store = kv_store] {
      absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
          {"num_nodes", static_cast<int64_t>(num_nodes)},
          {"node_id", static_cast<int64_t>(i)}};
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
#if TENSORFLOW_USE_ROCM
  EXPECT_EQ(platform_name_args.platform_name, expected_platform_name_for_rocm);
#else
  EXPECT_EQ(platform_name_args.platform_name, expected_platform_name_for_cuda);
#endif

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

void TestCustomCallV2() {}

TEST(PjrtCApiGpuExtensionTest, CustomCallUntyped) {
  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  std::string function_name = "untyped_function_name";
  args.function_name = function_name.c_str();
  args.function_name_size = function_name.size();
  args.api_version = 0;
  args.custom_call_function = reinterpret_cast<void*>(&TestCustomCallV2);
  auto api = GetPjrtApi();
  const PJRT_Extension_Base* next =
      reinterpret_cast<const PJRT_Extension_Base*>(api->extension_start);
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

static void* kNoop = xla::ffi::Ffi::Bind()
                         .To([]() { return xla::ffi::Error::Success(); })
                         .release();

TEST(PjrtCApiGpuExtensionTest, CustomCallTyped) {
  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  std::string function_name = "typed_function_name";
  args.function_name = function_name.c_str();
  args.function_name_size = function_name.size();
  args.api_version = 1;
  args.custom_call_function = kNoop;
  auto api = GetPjrtApi();
  const PJRT_Extension_Base* next =
      reinterpret_cast<const PJRT_Extension_Base*>(api->extension_start);
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

}  // namespace
}  // namespace pjrt
