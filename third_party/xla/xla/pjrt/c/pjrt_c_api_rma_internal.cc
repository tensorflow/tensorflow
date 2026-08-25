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

#include "xla/pjrt/c/pjrt_c_api_rma_internal.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_rma_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

struct PJRT_Rma_RemoteWindow {
  PJRT_RawBuffer* local_buffer{nullptr};
  uint64_t window_id{0};
  size_t size_in_bytes{0};
  std::string descriptor;
};

namespace pjrt {
namespace {

struct WindowRegistry {
  absl::Mutex mu;
  uint64_t next_id ABSL_GUARDED_BY(mu){1};
  absl::flat_hash_map<uint64_t, std::pair<PJRT_RawBuffer*, size_t>> windows
      ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<const PJRT_RawBuffer*, uint64_t> buffer_to_id
      ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<const void*, uint64_t> dev_ptr_to_id ABSL_GUARDED_BY(mu);
};

WindowRegistry& GetWindowRegistry() {
  static auto* registry = new WindowRegistry();
  return *registry;
}

// Signal and WaitSignal event synchronization registry for local/mock
// transfers. Supports multiple queued signals and waiters per key.
struct SignalRegistry {
  absl::Mutex mu;
  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, std::deque<xla::Promise<>>>
      waiters ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, int64_t> signal_counts
      ABSL_GUARDED_BY(mu);
};

SignalRegistry& GetSignalRegistry() {
  static auto* registry = new SignalRegistry();
  return *registry;
}

}  // namespace

PJRT_Error* PJRT_Rma_ExportWindow(PJRT_Rma_ExportWindow_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Rma_ExportWindow_Args", PJRT_Rma_ExportWindow_Args_STRUCT_SIZE,
      args->struct_size));
  if (!args->buffer) {
    return StatusToPjRtError(
        absl::InvalidArgumentError("Null buffer in ExportWindow"));
  }
  size_t size = 0;
  if (args->buffer->vtable != nullptr &&
      args->buffer->vtable->get_on_device_size_in_bytes != nullptr) {
    size = args->buffer->vtable->get_on_device_size_in_bytes(args->buffer);
  }

  auto& registry = GetWindowRegistry();
  uint64_t win_id = 0;
  {
    absl::MutexLock lock(&registry.mu);
    win_id = registry.next_id++;
    registry.windows[win_id] = {args->buffer, size};
    registry.buffer_to_id[args->buffer] = win_id;
    if (args->buffer->vtable != nullptr &&
        args->buffer->vtable->opaque_device_memory_data_pointer != nullptr) {
      void* dev_ptr =
          args->buffer->vtable->opaque_device_memory_data_pointer(args->buffer);
      if (dev_ptr != nullptr) {
        registry.dev_ptr_to_id[dev_ptr] = win_id;
      }
    }
  }

  // Serialize window descriptor as an opaque token:
  // "pjrt_rma_win:<win_id>:<size>"
  std::string desc =
      absl::StrFormat("pjrt_rma_win:%016" PRIx64 ":%zu", win_id, size);

  char* out_str = new char[desc.size() + 1];
  std::memcpy(out_str, desc.data(), desc.size() + 1);
  args->serialized_descriptor = out_str;
  args->serialized_descriptor_size = desc.size();
  return nullptr;
}

PJRT_Error* PJRT_Rma_ImportWindow(PJRT_Rma_ImportWindow_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Rma_ImportWindow_Args", PJRT_Rma_ImportWindow_Args_STRUCT_SIZE,
      args->struct_size));
  if (!args->serialized_descriptor || args->serialized_descriptor_size == 0) {
    return StatusToPjRtError(
        absl::InvalidArgumentError("Invalid serialized_descriptor"));
  }
  std::string desc(args->serialized_descriptor,
                   args->serialized_descriptor_size);
  uint64_t win_id = 0;
  size_t size = 0;
  if (std::sscanf(desc.c_str(), "pjrt_rma_win:%" SCNx64 ":%zu", &win_id,
                  &size) < 2) {
    return StatusToPjRtError(
        absl::InvalidArgumentError("Malformed RMA descriptor"));
  }

  auto& registry = GetWindowRegistry();
  PJRT_RawBuffer* local_buffer = nullptr;
  {
    absl::MutexLock lock(&registry.mu);
    auto it = registry.windows.find(win_id);
    if (it == registry.windows.end()) {
      return StatusToPjRtError(
          absl::InvalidArgumentError("Unknown or expired RMA window token"));
    }
    local_buffer = it->second.first;
    if (size == 0) {
      size = it->second.second;
    }
  }

  if (local_buffer != nullptr && local_buffer->vtable != nullptr &&
      local_buffer->vtable->inc_ref != nullptr) {
    local_buffer->vtable->inc_ref(local_buffer);
  }

  auto* remote_win = new PJRT_Rma_RemoteWindow();
  remote_win->window_id = win_id;
  remote_win->local_buffer = local_buffer;
  remote_win->size_in_bytes = size;
  remote_win->descriptor = desc;
  args->remote_window = remote_win;
  return nullptr;
}

PJRT_Error* PJRT_Rma_DestroyRemoteWindow(
    PJRT_Rma_DestroyRemoteWindow_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Rma_DestroyRemoteWindow_Args",
      PJRT_Rma_DestroyRemoteWindow_Args_STRUCT_SIZE, args->struct_size));
  if (args->remote_window) {
    if (args->remote_window->local_buffer) {
      if (args->remote_window->local_buffer->vtable &&
          args->remote_window->local_buffer->vtable->dec_ref) {
        args->remote_window->local_buffer->vtable->dec_ref(
            args->remote_window->local_buffer);
      }
    }
    delete args->remote_window;
  }
  return nullptr;
}

PJRT_Error* PJRT_Rma_Put(PJRT_Rma_Put_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Rma_Put_Args", PJRT_Rma_Put_Args_STRUCT_SIZE, args->struct_size));
  if (!args->src_buffer || !args->dst_remote_window ||
      !args->dst_remote_window->local_buffer) {
    return StatusToPjRtError(
        absl::InvalidArgumentError("Null or invalid arguments to Rma_Put"));
  }
  if (args->transfer_size_bytes == 0) {
    args->event = new PJRT_Event{xla::Future<>(absl::OkStatus())};
    return nullptr;
  }

  PJRT_RawBuffer* src_buf = args->src_buffer;
  PJRT_RawBuffer* dst_buf = args->dst_remote_window->local_buffer;
  if (!src_buf->vtable || !dst_buf->vtable ||
      !src_buf->vtable->schedule_copy_to) {
    return StatusToPjRtError(
        absl::InvalidArgumentError("Invalid raw buffer vtable in Rma_Put"));
  }

  PJRT_RawBuffer* sliced_src = nullptr;
  PJRT_RawBuffer* sliced_dst = nullptr;

  if (args->src_offset_bytes > 0 ||
      (src_buf->vtable->get_on_device_size_in_bytes &&
       args->transfer_size_bytes <
           src_buf->vtable->get_on_device_size_in_bytes(src_buf))) {
    if (!src_buf->vtable->slice) {
      return StatusToPjRtError(
          absl::UnimplementedError("Slicing not supported on src buffer"));
    }
    PJRT_Error* err =
        src_buf->vtable->slice(src_buf, args->src_offset_bytes,
                               args->transfer_size_bytes, &sliced_src);
    if (err) {
      return err;
    }
    src_buf = sliced_src;
  }

  if (args->dst_offset_bytes > 0 ||
      (dst_buf->vtable->get_on_device_size_in_bytes &&
       args->transfer_size_bytes <
           dst_buf->vtable->get_on_device_size_in_bytes(dst_buf))) {
    if (!dst_buf->vtable->slice) {
      if (sliced_src && sliced_src->vtable && sliced_src->vtable->dec_ref) {
        sliced_src->vtable->dec_ref(sliced_src);
      }
      return StatusToPjRtError(
          absl::UnimplementedError("Slicing not supported on dst buffer"));
    }
    PJRT_Error* err =
        dst_buf->vtable->slice(dst_buf, args->dst_offset_bytes,
                               args->transfer_size_bytes, &sliced_dst);
    if (err) {
      if (sliced_src && sliced_src->vtable && sliced_src->vtable->dec_ref) {
        sliced_src->vtable->dec_ref(sliced_src);
      }
      return err;
    }
    dst_buf = sliced_dst;
  }

  if (!src_buf || !src_buf->vtable || !dst_buf || !dst_buf->vtable ||
      !src_buf->vtable->schedule_copy_to) {
    if (sliced_src && sliced_src->vtable && sliced_src->vtable->dec_ref) {
      sliced_src->vtable->dec_ref(sliced_src);
    }
    if (sliced_dst && sliced_dst->vtable && sliced_dst->vtable->dec_ref) {
      sliced_dst->vtable->dec_ref(sliced_dst);
    }
    return StatusToPjRtError(
        absl::InvalidArgumentError("Invalid raw buffer or vtable in Rma_Put"));
  }

  if (!sliced_src && src_buf->vtable->inc_ref != nullptr) {
    src_buf->vtable->inc_ref(src_buf);
  }
  if (!sliced_dst && dst_buf->vtable->inc_ref != nullptr) {
    dst_buf->vtable->inc_ref(dst_buf);
  }

  auto [promise, future] = xla::MakePromise();
  struct CallbackState {
    xla::Promise<> promise;
    PJRT_RawBuffer* src;
    PJRT_RawBuffer* dst;
  };
  auto* cb_state = new CallbackState{std::move(promise), src_buf, dst_buf};

  src_buf->vtable->schedule_copy_to(
      src_buf,
      /*transfer_dependency_events=*/nullptr, dst_buf,
      /*definition_event_promise=*/nullptr,
      /*src_usage_event_promise=*/nullptr,
      [](PJRT_Error* error, void* user_data) {
        auto* state = static_cast<CallbackState*>(user_data);
        if (error) {
          state->promise.Set(absl::InternalError("schedule_copy_to failed"));
        } else {
          state->promise.Set(absl::OkStatus());
        }
        if (state->src && state->src->vtable && state->src->vtable->dec_ref) {
          state->src->vtable->dec_ref(state->src);
        }
        if (state->dst && state->dst->vtable && state->dst->vtable->dec_ref) {
          state->dst->vtable->dec_ref(state->dst);
        }
        delete state;
      },
      cb_state);

  args->event = new PJRT_Event{std::move(future)};
  return nullptr;
}

// 5. Remote hardware signaling / synchronization.
PJRT_Error* PJRT_Rma_Signal(PJRT_Rma_Signal_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Rma_Signal_Args", PJRT_Rma_Signal_Args_STRUCT_SIZE,
      args->struct_size));
  if (!args->dst_remote_window) {
    return StatusToPjRtError(
        absl::InvalidArgumentError("Null dst_remote_window in Rma_Signal"));
  }
  uint64_t win_id = args->dst_remote_window->window_id;
  auto key = std::make_pair(win_id, args->signal_id);

  auto& registry = GetSignalRegistry();
  absl::MutexLock lock(&registry.mu);
  auto it = registry.waiters.find(key);
  if (it != registry.waiters.end() && !it->second.empty()) {
    auto promise = std::move(it->second.front());
    it->second.pop_front();
    if (it->second.empty()) {
      registry.waiters.erase(it);
    }
    promise.Set(absl::OkStatus());
  } else {
    registry.signal_counts[key]++;
  }
  args->event = new PJRT_Event{xla::Future<>(absl::OkStatus())};
  return nullptr;
}

PJRT_Error* PJRT_Rma_WaitSignal(PJRT_Rma_WaitSignal_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Rma_WaitSignal_Args", PJRT_Rma_WaitSignal_Args_STRUCT_SIZE,
      args->struct_size));
  if (!args->local_buffer && (!args->local_window_descriptor ||
                              args->local_window_descriptor_size == 0)) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "Null local_buffer or descriptor in Rma_WaitSignal"));
  }

  uint64_t win_id = 0;
  if (args->local_window_descriptor != nullptr &&
      args->local_window_descriptor_size > 0 &&
      args->local_window_descriptor_size < 1024) {
    std::string desc(args->local_window_descriptor,
                     args->local_window_descriptor_size);
    size_t sz = 0;
    std::sscanf(desc.c_str(), "pjrt_rma_win:%" SCNx64 ":%zu", &win_id, &sz);
  }

  if (win_id == 0 && args->local_buffer) {
    auto& win_reg = GetWindowRegistry();
    absl::MutexLock lock(&win_reg.mu);
    auto it = win_reg.buffer_to_id.find(args->local_buffer);
    if (it != win_reg.buffer_to_id.end()) {
      win_id = it->second;
    } else {
      if (args->local_buffer->vtable != nullptr &&
          args->local_buffer->vtable->opaque_device_memory_data_pointer !=
              nullptr) {
        void* dev_ptr =
            args->local_buffer->vtable->opaque_device_memory_data_pointer(
                args->local_buffer);
        if (dev_ptr != nullptr) {
          auto dev_it = win_reg.dev_ptr_to_id.find(dev_ptr);
          if (dev_it != win_reg.dev_ptr_to_id.end()) {
            win_id = dev_it->second;
          }
        }
      }
    }
  }

  auto key = std::make_pair(win_id, args->signal_id);

  auto& registry = GetSignalRegistry();
  absl::MutexLock lock(&registry.mu);
  auto sig_it = registry.signal_counts.find(key);
  if (sig_it != registry.signal_counts.end() && sig_it->second > 0) {
    sig_it->second--;
    if (sig_it->second == 0) {
      registry.signal_counts.erase(sig_it);
    }
    args->event = new PJRT_Event{xla::Future<>(absl::OkStatus())};
  } else {
    auto [promise, future] = xla::MakePromise();
    registry.waiters[key].push_back(std::move(promise));
    args->event = new PJRT_Event{std::move(future)};
  }
  return nullptr;
}

PJRT_Error* PJRT_Rma_DestroyDescriptor(PJRT_Rma_DestroyDescriptor_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Rma_DestroyDescriptor_Args",
      PJRT_Rma_DestroyDescriptor_Args_STRUCT_SIZE, args->struct_size));
  delete[] args->serialized_descriptor;
  return nullptr;
}

PJRT_Rma_Extension CreateRmaExtension(PJRT_Extension_Base* next) {
  return {
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_Rma_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Rma,
          /*next=*/next,
      },
      /*PJRT_Rma_ExportWindow=*/pjrt::PJRT_Rma_ExportWindow,
      /*PJRT_Rma_ImportWindow=*/pjrt::PJRT_Rma_ImportWindow,
      /*PJRT_Rma_DestroyRemoteWindow=*/pjrt::PJRT_Rma_DestroyRemoteWindow,
      /*PJRT_Rma_Put=*/pjrt::PJRT_Rma_Put,
      /*PJRT_Rma_Signal=*/pjrt::PJRT_Rma_Signal,
      /*PJRT_Rma_WaitSignal=*/pjrt::PJRT_Rma_WaitSignal,
      /*PJRT_Rma_DestroyDescriptor=*/pjrt::PJRT_Rma_DestroyDescriptor,
  };
}

}  // namespace pjrt
