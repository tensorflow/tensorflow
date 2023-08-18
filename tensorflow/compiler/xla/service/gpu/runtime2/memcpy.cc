/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime2/memcpy.h"

#include "tensorflow/compiler/xla/service/gpu/runtime2/hal.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/vm.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla::gpu {

//===-----------------------------------------------------------------------===/
// XLA:GPU memcpy API
//===-----------------------------------------------------------------------===/

Status DispatchMemcpyD2D(const vm::ExecutionContext& ctx,
                         iree_hal_allocator_t* device_allocator,
                         iree_hal_buffer_view_t* dst,
                         iree_hal_buffer_view_t* src) {
  se::Stream* stream = ctx.run_options->stream();

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase dst_data,
                      GetDeviceMemory(device_allocator, dst));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase src_data,
                      GetDeviceMemory(device_allocator, src));

  stream->ThenMemcpy(&dst_data, src_data, src_data.size());
  return OkStatus();
}

StatusOr<bool> DispatchLoadI1(const vm::ExecutionContext& ctx,
                              iree_hal_allocator_t* device_allocator,
                              iree_hal_buffer_view_t* view, int32_t offset) {
  se::Stream* stream = ctx.run_options->stream();

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase data,
                      GetDeviceMemory(device_allocator, view));

  if (offset >= data.size())
    return absl::InvalidArgumentError("offset out of bounds");

  std::byte* ptr = static_cast<std::byte*>(data.opaque());
  se::DeviceMemoryBase value(ptr + offset, 1);

  bool dst;
  stream->ThenMemcpy(&dst, value, 1);
  return dst;
}

//===-----------------------------------------------------------------------===/
// XLA:GPU custom module memcpy API
//===-----------------------------------------------------------------------===/

namespace vm {

MemcpyAPI::MemcpyAPI(iree_hal_allocator_t* device_allocator)
    : device_allocator_(device_allocator) {}

iree::Status MemcpyAPI::MemcpyD2D(iree::vm::ref<ExecutionContext> ctx,
                                  iree::vm::ref<iree_hal_buffer_view_t> dst,
                                  iree::vm::ref<iree_hal_buffer_view_t> src) {
  return FromStatus(
      DispatchMemcpyD2D(*ctx, device_allocator_, dst.get(), src.get()));
}

iree::StatusOr<int32_t> MemcpyAPI::LoadI1(
    iree::vm::ref<ExecutionContext> ctx,
    iree::vm::ref<iree_hal_buffer_view_t> view, int32_t offset) {
  return FromStatusOr(
      DispatchLoadI1(*ctx, device_allocator_, view.get(), offset));
}

}  // namespace vm
}  // namespace xla::gpu
