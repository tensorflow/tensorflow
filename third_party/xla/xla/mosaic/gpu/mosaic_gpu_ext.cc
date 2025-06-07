/* Copyright 2021 The JAX Authors.

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
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "third_party/py/jax/jaxlib/gpu/vendor.h"
#include "third_party/py/jax/jaxlib/kernel_nanobind_helpers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace jax::cuda {
namespace {

namespace ffi = xla::ffi;
namespace nb = nanobind;

static std::string ToString(CUresult result) {
  const char* error_name;
  if (cuGetErrorName(result, &error_name)) {
    return absl::StrCat("UNKNOWN ERROR (", static_cast<int>(result), ")");
  }
  const char* error_string;
  if (cuGetErrorString(result, &error_string)) {
    return error_name;
  }
  return absl::StrCat(error_name, ": ", error_string);
}

// Ensure it is safe to store gpuEvent_t in a uint64_t buffer.
static_assert(sizeof(gpuEvent_t) <= sizeof(uint64_t));

static const auto* kEventRecord =
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<gpuStream_t>>()
        .Attr<bool>("copy_before")
        .RemainingArgs()
        .Ret<ffi::BufferR0<ffi::U64>>()  // event
        .RemainingRets()
        .To([](gpuStream_t stream, bool copy_before, auto remaining_args,
               auto ret, auto remaining_rets) {
          static auto* event = new gpuEvent_t;
          if (auto res = gpuEventCreate(event, GPU_EVENT_DEFAULT); res) {
            return ffi::Error::Internal(
                absl::StrCat("Failed to create event: ", ToString(res)));
          }
          auto do_copy = [&]() {
            gpuMemcpyAsync(ret->untyped_data(), event, sizeof(gpuEvent_t),
                           gpuMemcpyHostToDevice, stream);
          };
          if (copy_before) {
            do_copy();
          }
          if (auto res = gpuEventRecord(*event, stream); res) {
            return ffi::Error::Internal(
                absl::StrCat("Failed to record event: ", ToString(res)));
          }
          if (!copy_before) {
            do_copy();
          }
          return ffi::Error::Success();
        })
        .release();

XLA_FFI_Error* EventRecord(XLA_FFI_CallFrame* call_frame) {
  return kEventRecord->Call(call_frame);
}

static const auto* kEventElapsed =
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<gpuStream_t>>()
        .Arg<ffi::BufferR0<ffi::U64>>()  // start_event
        .Arg<ffi::BufferR0<ffi::U64>>()  // end_event
        .Ret<ffi::BufferR0<ffi::F32>>()  // elapsed_ms
        .To([](gpuStream_t stream, auto start, auto end, auto out) {
          gpuStreamSynchronize(stream);
          gpuEvent_t start_event = nullptr;
          gpuEvent_t end_event = nullptr;

          absl::Cleanup cleanup = [&]() {
            gpuEventDestroy(start_event);
            gpuEventDestroy(end_event);
          };

          gpuMemcpy(&start_event, start.untyped_data(), sizeof(gpuEvent_t),
                    gpuMemcpyDeviceToHost);
          gpuMemcpy(&end_event, end.untyped_data(), sizeof(gpuEvent_t),
                    gpuMemcpyDeviceToHost);

          float elapsed;
          if (auto res = gpuEventElapsedTime(&elapsed, start_event, end_event);
              res) {
            return ffi::Error::Internal(absl::StrCat(
                "Failed to get elapsed time between events: ", ToString(res)));
          }
          gpuMemcpy(out->untyped_data(), &elapsed, sizeof(float),
                    gpuMemcpyHostToDevice);
          return ffi::Error::Success();
        })
        .release();

XLA_FFI_Error* EventElapsed(XLA_FFI_CallFrame* call_frame) {
  return kEventElapsed->Call(call_frame);
}

#define THROW(...)                                                 \
  do {                                                             \
    throw std::runtime_error(                                      \
        absl::StrCat("Mosaic GPU profiler error: ", __VA_ARGS__)); \
  } while (0)

#define THROW_IF(expr, ...)       \
  do {                            \
    if (expr) THROW(__VA_ARGS__); \
  } while (0)

#define THROW_IF_CUPTI_ERROR(expr, ...)          \
  do {                                           \
    CUptiResult _result = (expr);                \
    if (_result != CUPTI_SUCCESS) {              \
      const char* s;                             \
      cuptiGetErrorMessage(_result, &s);         \
      THROW(s, ": " __VA_OPT__(, ) __VA_ARGS__); \
    }                                            \
  } while (0)

// CUPTI can only have one subscriber per process, so it's ok to make the
// profiler state global.
struct {
  CUpti_SubscriberHandle subscriber;
  std::vector<std::tuple<const char* /*kernel_name*/, double /*ms*/>> timings;
} profiler_state;

void callback_request(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  // 10 MiB buffer size is generous but somewhat arbitrary, it's at the upper
  // bound of what's recommended in CUPTI documentation:
  // https://docs.nvidia.com/cupti/main/main.html#cupti-callback-api:~:text=For%20typical%20workloads%2C%20it%E2%80%99s%20suggested%20to%20choose%20a%20size%20between%201%20and%2010%20MB.
  const int buffer_size = 10 * (1 << 20);
  // 8 byte alignment is specified in the official CUPTI code samples, see
  // extras/CUPTI/samples/common/helper_cupti_activity.h in your CUDA
  // installation.
  *buffer = new (std::align_val_t(8)) uint8_t[buffer_size];
  *size = buffer_size;
  *maxNumRecords = 0;
}

void callback_complete(CUcontext context, uint32_t streamId, uint8_t* buffer,
                       size_t size, size_t validSize) {
  // take ownership of the buffer once CUPTI is done using it
  absl::Cleanup cleanup = [buffer]() {
    operator delete[](buffer, std::align_val_t(8));
  };
  CUpti_Activity* record = nullptr;
  while (true) {
    CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        // TODO(andportnoy) handle multi-GPU
        CUpti_ActivityKernel9* kernel = (CUpti_ActivityKernel9*)record;
        // Convert integer nanoseconds to floating point milliseconds to match
        // the interface of the events-based profiler.
        double duration_ms = (kernel->end - kernel->start) / 1e6;
        const char* kernel_name = kernel->name;
        profiler_state.timings.push_back(
            std::make_tuple(kernel_name, duration_ms));
      }
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      // no more records available
      break;
    } else {
      THROW_IF_CUPTI_ERROR(status);
    }
  }

  size_t num_dropped;
  THROW_IF_CUPTI_ERROR(
      cuptiActivityGetNumDroppedRecords(context, streamId, &num_dropped),
      "failed to get number of dropped activity records");
  THROW_IF(num_dropped > 0, "activity records were dropped");
}

NB_MODULE(_mosaic_gpu_ext, m) {
  m.def("registrations", []() {
    return nb::make_tuple(
        nb::make_tuple("mgpu_event_record", EncapsulateFunction(EventRecord)),
        nb::make_tuple("mgpu_event_elapsed",
                       EncapsulateFunction(EventElapsed)));
  });
  m.def("_sync_all_devices", []() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != gpuSuccess) {
      throw std::runtime_error("Failed to get device count");
    }
    for (int i = 0; i < devices; ++i) {
      if (cudaSetDevice(i) != gpuSuccess) {
        throw std::runtime_error("Failed to set device");
      }
      if (cudaDeviceSynchronize() != gpuSuccess) {
        throw std::runtime_error("Failed to synchronize device");
      }
    }
  });
  m.def("_cupti_init", []() {
    profiler_state.timings.clear();
    // Ok to pass nullptr for the callback here because we don't register any
    // callbacks through cuptiEnableCallback.
    auto subscribe_result = cuptiSubscribe(
        &profiler_state.subscriber, /*callback=*/nullptr, /*userdata=*/nullptr);
    if (subscribe_result == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
      THROW(
          "Attempted to subscribe to CUPTI while another subscriber, such as "
          "Nsight Systems or Nsight Compute, is active. CUPTI backend of the "
          "Mosaic GPU profiler cannot be used in that mode since CUPTI does "
          "not support multiple subscribers.");
    }
    THROW_IF_CUPTI_ERROR(subscribe_result, "failed to subscribe to CUPTI");
    THROW_IF_CUPTI_ERROR(
        cuptiActivityRegisterCallbacks(callback_request, callback_complete),
        "failed to register CUPTI activity callbacks");
    THROW_IF_CUPTI_ERROR(
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
        "failed to enable tracking of kernel activity by CUPTI");
  });
  m.def(
      "_cupti_get_timings",
      [](bool finalize) {
        THROW_IF_CUPTI_ERROR(
            cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
            "failed to disable tracking of kernel activity by CUPTI");
        THROW_IF_CUPTI_ERROR(
            cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
            "failed to flush CUPTI activity buffers");
        if (finalize) {
          THROW_IF_CUPTI_ERROR(cuptiFinalize(), "failed to detach CUPTI");
        }
        THROW_IF_CUPTI_ERROR(cuptiUnsubscribe(profiler_state.subscriber),
                             "failed to unsubscribe from CUPTI");
        return profiler_state.timings;
      },
      nb::arg("finalize") = true);
}

}  // namespace
}  // namespace jax::cuda
