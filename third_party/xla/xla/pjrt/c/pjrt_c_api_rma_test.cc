/* Copyright 2026 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_device_event.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_rma_extension.h"
#include "xla/pjrt/c/pjrt_c_api_rma_internal.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"  // IWYU pragma: keep

namespace pjrt {
namespace {

TEST(PjRtCApiRmaTest, CreateRmaExtensionTest) {
  PJRT_Rma_Extension rma_ext = CreateRmaExtension(nullptr);
  EXPECT_EQ(rma_ext.base.type, PJRT_Extension_Type_Rma);
  EXPECT_EQ(rma_ext.base.next, nullptr);

  EXPECT_NE(rma_ext.PJRT_Rma_ExportWindow, nullptr);
  EXPECT_NE(rma_ext.PJRT_Rma_ImportWindow, nullptr);
  EXPECT_NE(rma_ext.PJRT_Rma_DestroyRemoteWindow, nullptr);
  EXPECT_NE(rma_ext.PJRT_Rma_Put, nullptr);
  EXPECT_NE(rma_ext.PJRT_Rma_Signal, nullptr);
  EXPECT_NE(rma_ext.PJRT_Rma_WaitSignal, nullptr);
}

TEST(PjRtCApiRmaTest, ExportImportDestroyWindowTest) {
  PJRT_Rma_Extension rma_ext = CreateRmaExtension(nullptr);

  // 1. Export window from a buffer pointer
  PJRT_RawBuffer_FunctionTable vtable;
  std::memset(&vtable, 0, sizeof(vtable));
  vtable.get_on_device_size_in_bytes = [](const PJRT_RawBuffer*) -> size_t {
    return 1024;
  };
  PJRT_RawBuffer mock_buffer{&vtable};
  PJRT_RawBuffer* raw_buf = &mock_buffer;

  PJRT_Rma_ExportWindow_Args export_args;
  export_args.struct_size = PJRT_Rma_ExportWindow_Args_STRUCT_SIZE;
  export_args.extension_start = nullptr;
  export_args.buffer = raw_buf;
  export_args.serialized_descriptor = nullptr;
  export_args.serialized_descriptor_size = 0;

  PJRT_Error* error = rma_ext.PJRT_Rma_ExportWindow(&export_args);
  EXPECT_EQ(error, nullptr);
  EXPECT_NE(export_args.serialized_descriptor, nullptr);
  EXPECT_GT(export_args.serialized_descriptor_size, 0);

  std::string exported_desc(export_args.serialized_descriptor,
                            export_args.serialized_descriptor_size);

  PJRT_Rma_DestroyDescriptor_Args destroy_desc_args;
  destroy_desc_args.struct_size = PJRT_Rma_DestroyDescriptor_Args_STRUCT_SIZE;
  destroy_desc_args.extension_start = nullptr;
  destroy_desc_args.serialized_descriptor = export_args.serialized_descriptor;
  error = rma_ext.PJRT_Rma_DestroyDescriptor(&destroy_desc_args);
  EXPECT_EQ(error, nullptr);

  // 2. Import window
  PJRT_Rma_ImportWindow_Args import_args;
  import_args.struct_size = PJRT_Rma_ImportWindow_Args_STRUCT_SIZE;
  import_args.extension_start = nullptr;
  import_args.serialized_descriptor = exported_desc.data();
  import_args.serialized_descriptor_size = exported_desc.size();
  import_args.remote_window = nullptr;

  error = rma_ext.PJRT_Rma_ImportWindow(&import_args);
  EXPECT_EQ(error, nullptr);
  ASSERT_NE(import_args.remote_window, nullptr);

  // 3. Signal and WaitSignal
  PJRT_Rma_Signal_Args signal_args;
  signal_args.struct_size = PJRT_Rma_Signal_Args_STRUCT_SIZE;
  signal_args.extension_start = nullptr;
  signal_args.dst_remote_window = import_args.remote_window;
  signal_args.signal_id = 0;
  signal_args.event = nullptr;

  error = rma_ext.PJRT_Rma_Signal(&signal_args);
  EXPECT_EQ(error, nullptr);
  PJRT_Event* sig_event = signal_args.event;
  ASSERT_NE(sig_event, nullptr);
  EXPECT_TRUE(sig_event->future.IsReady());
  delete sig_event;

  PJRT_Rma_WaitSignal_Args wait_args = {};
  wait_args.struct_size = PJRT_Rma_WaitSignal_Args_STRUCT_SIZE;
  wait_args.extension_start = nullptr;
  wait_args.local_buffer = raw_buf;
  wait_args.signal_id = 0;
  wait_args.event = nullptr;

  error = rma_ext.PJRT_Rma_WaitSignal(&wait_args);
  EXPECT_EQ(error, nullptr);
  PJRT_Event* wait_event = wait_args.event;
  ASSERT_NE(wait_event, nullptr);
  EXPECT_TRUE(wait_event->future.IsReady());
  delete wait_event;

  // 4. Destroy remote window
  PJRT_Rma_DestroyRemoteWindow_Args destroy_args;
  destroy_args.struct_size = PJRT_Rma_DestroyRemoteWindow_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.remote_window = import_args.remote_window;

  error = rma_ext.PJRT_Rma_DestroyRemoteWindow(&destroy_args);
  EXPECT_EQ(error, nullptr);
}

TEST(PjRtCApiRmaTest, PutTest) {
  PJRT_Rma_Extension rma_ext = CreateRmaExtension(nullptr);

  PJRT_RawBuffer_FunctionTable vtable;
  std::memset(&vtable, 0, sizeof(vtable));
  vtable.get_on_device_size_in_bytes = [](const PJRT_RawBuffer*) -> size_t {
    return 1024;
  };
  vtable.slice = [](PJRT_RawBuffer* raw_buffer, int64_t offset,
                    int64_t slice_size,
                    PJRT_RawBuffer** sliced_buffer) -> PJRT_Error* {
    *sliced_buffer = raw_buffer;
    return nullptr;
  };
  vtable.schedule_copy_to =
      [](PJRT_RawBuffer* src_buffer,
         PJRT_DeviceEventVector* transfer_dependency_events,
         PJRT_RawBuffer* dst_buffer,
         PJRT_DeviceEventPromise* definition_event_promise,
         PJRT_DeviceEventPromise* src_usage_event_promise,
         void (*allocation_event_callback)(PJRT_Error* status, void* user_data),
         void* allocation_event_user_data) {
        if (allocation_event_callback) {
          allocation_event_callback(nullptr, allocation_event_user_data);
        }
      };

  PJRT_RawBuffer src_buf{&vtable};
  PJRT_RawBuffer dst_buf{&vtable};

  PJRT_Rma_ExportWindow_Args export_args;
  export_args.struct_size = PJRT_Rma_ExportWindow_Args_STRUCT_SIZE;
  export_args.extension_start = nullptr;
  export_args.buffer = &dst_buf;
  export_args.serialized_descriptor = nullptr;
  export_args.serialized_descriptor_size = 0;
  PJRT_Error* error = rma_ext.PJRT_Rma_ExportWindow(&export_args);
  EXPECT_EQ(error, nullptr);

  PJRT_Rma_ImportWindow_Args import_args;
  import_args.struct_size = PJRT_Rma_ImportWindow_Args_STRUCT_SIZE;
  import_args.extension_start = nullptr;
  import_args.serialized_descriptor = export_args.serialized_descriptor;
  import_args.serialized_descriptor_size =
      export_args.serialized_descriptor_size;
  import_args.remote_window = nullptr;
  error = rma_ext.PJRT_Rma_ImportWindow(&import_args);
  EXPECT_EQ(error, nullptr);

  PJRT_Rma_DestroyDescriptor_Args destroy_desc_args;
  destroy_desc_args.struct_size = PJRT_Rma_DestroyDescriptor_Args_STRUCT_SIZE;
  destroy_desc_args.extension_start = nullptr;
  destroy_desc_args.serialized_descriptor = export_args.serialized_descriptor;
  error = rma_ext.PJRT_Rma_DestroyDescriptor(&destroy_desc_args);
  EXPECT_EQ(error, nullptr);

  PJRT_Rma_Put_Args put_args;
  put_args.struct_size = PJRT_Rma_Put_Args_STRUCT_SIZE;
  put_args.extension_start = nullptr;
  put_args.src_buffer = &src_buf;
  put_args.dst_remote_window = import_args.remote_window;
  put_args.src_offset_bytes = 0;
  put_args.dst_offset_bytes = 0;
  put_args.transfer_size_bytes = 256;
  put_args.event = nullptr;

  error = rma_ext.PJRT_Rma_Put(&put_args);
  EXPECT_EQ(error, nullptr);
  PJRT_Event* put_event = put_args.event;
  ASSERT_NE(put_event, nullptr);
  EXPECT_TRUE(put_event->future.IsReady());
  delete put_event;

  PJRT_Rma_DestroyRemoteWindow_Args destroy_args;
  destroy_args.struct_size = PJRT_Rma_DestroyRemoteWindow_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.remote_window = import_args.remote_window;
  error = rma_ext.PJRT_Rma_DestroyRemoteWindow(&destroy_args);
  EXPECT_EQ(error, nullptr);
}

}  // namespace
}  // namespace pjrt
