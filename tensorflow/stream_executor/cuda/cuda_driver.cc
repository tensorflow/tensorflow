/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/cuda/cuda_driver.h"

#include <stdint.h>
#include <stdlib.h>
#include <map>
#include <set>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/human_readable.h"
#include "tensorflow/stream_executor/lib/notification.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/stacktrace.h"
#include "tensorflow/stream_executor/lib/static_threadlocal.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"

bool FLAGS_gpuexec_cuda_driver_inject_init_error = false;
bool FLAGS_gpuexec_cuda_sync_around_driver_calls = false;
bool FLAGS_gpuexec_cuda_device_0_only = false;

// Debugging: on each push and pop of a cuda context, verify the current context
// matches the expected one.
constexpr bool kVerifyCudaContext = false;

namespace stream_executor {
namespace cuda {

namespace {

// Manages the singleton map of contexts that we've created, mapping
// from the CUcontext to the CudaContext* that we pass around internally.
// This also manages assignment of unique ids to CudaContexts, to allow
// for fast comparison of a context against the current context.
//
// CUDA-runtime-created contexts are avoided, if triple angle
// brace launches are required, by using the scoped activations in
// cuda_activation.h.
class CreatedContexts {
 public:
  // Returns whether context is a member of the live set.
  static bool Has(CUcontext context) {
    tf_shared_lock lock(mu_);
    return Live()->find(context) != Live()->end();
  }

  // Adds context to the live set, or returns it if it's already present.
  static CudaContext* Add(CUcontext context) {
    CHECK(context != nullptr);
    mutex_lock lock(mu_);
    auto insert_result = Live()->insert(std::make_pair(context, nullptr));
    auto it = insert_result.first;
    if (insert_result.second) {
      // context was not present in the map.  Add it.
      it->second = MakeUnique<CudaContext>(context, next_id_++);
    }
    return it->second.get();
  }

  // Removes context from the live set.
  static void Remove(CUcontext context) {
    CHECK(context != nullptr);
    mutex_lock lock(mu_);
    auto it = Live()->find(context);
    CHECK(it != Live()->end()) << context;
    Live()->erase(it);
  }

 private:
  // Returns the live map singleton.
  static std::map<CUcontext, std::unique_ptr<CudaContext>> *Live() {
    static auto singleton =
        new std::map<CUcontext, std::unique_ptr<CudaContext>>;
    return singleton;
  }

  // Lock that guards access-to/mutation-of the live set.
  static mutex mu_;
  static int64 next_id_;
};

/* static */ mutex CreatedContexts::mu_{LINKER_INITIALIZED};
/* static */ int64 CreatedContexts::next_id_ = 1;  // 0 means "no context"

// Formats CUresult to output prettified values into a log stream.
string ToString(CUresult result) {
  const char *error_name;
  if (cuGetErrorName(result, &error_name)) {
    return absl::StrCat("UNKNOWN ERROR (", static_cast<int>(result), ")");
  }
  const char *error_string;
  if (cuGetErrorString(result, &error_string)) {
    return error_name;
  }
  return absl::StrCat(error_name, ": ", error_string);
}

// Returns the current context and checks that it is in the set of CUDA contexts
// created by StreamExecutor (to ensure that the CUDA runtime didn't create a
// context behind our backs).
CUcontext CurrentContext() {
  CUcontext current = CUDADriver::CurrentContextOrDie();
  if (current != nullptr && !CreatedContexts::Has(current)) {
    LOG(FATAL) << "current context was not created by the StreamExecutor "
                  "cuda_driver API: "
               << current
               << "; a CUDA runtime call "
                  "was likely performed without using a StreamExecutor context";
  }
  return current;
}

// CUDA driver routines may require a large amount of stack (particularly
// cuModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
static mutex driver_executor_threadpool_mu(LINKER_INITIALIZED);
static port::ThreadPool *InitializeDriverExecutor() {
  return new port::ThreadPool(port::Env::Default(), port::ThreadOptions(),
                              "cuda_driver", 1);
}

port::ThreadPool *GetDriverExecutor() {
  mutex_lock lock(driver_executor_threadpool_mu);
  static port::ThreadPool *thread_pool = InitializeDriverExecutor();
  return thread_pool;
}

}  // namespace

string MemorySpaceString(MemorySpace memory_space) {
  switch (memory_space) {
    case MemorySpace::kHost:
      return "host";
    case MemorySpace::kDevice:
      return "device";
    default:
      LOG(FATAL) << "impossible memory space";
  }
}

namespace {

// Call cuCtxtSynchronize and crash if it doesn't succeed.
void SynchronizeOrDie() {
  auto res = cuCtxSynchronize();
  if (res != CUDA_SUCCESS) {
    LOG(FATAL) << "Synchronize found "
               << ToString(res) << " :: " << port::CurrentStackTrace();
  }
}

struct ThreadLocalData {
  int64 id;
  CudaContext* context;  // Only valid if id == a known good context.
  int depth;
};

SE_STATIC_THREAD_LOCAL_POD(ThreadLocalData, tls_data);

}  // namespace

ScopedActivateContext::ScopedActivateContext(CudaContext* cuda_context) {
  if (FLAGS_gpuexec_cuda_sync_around_driver_calls) SynchronizeOrDie();

  auto* tls = &tls_data.get();
  tls->depth++;
  if (tls->id == cuda_context->id()) {
    if (kVerifyCudaContext) {
      CHECK_EQ(CurrentContext(), cuda_context->context());
    }
    DCHECK_EQ(CurrentContext(), cuda_context->context());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching context from " << tls->id
          << " to " << cuda_context->id();

  to_restore_ = (tls->depth == 1 ? nullptr : tls->context);

  // Set the context and update thread local.
  CHECK_EQ(CUDA_SUCCESS, cuCtxSetCurrent(cuda_context->context()));
  tls->id = cuda_context->id();
  tls->context = cuda_context;
}

ScopedActivateContext::~ScopedActivateContext() {
  if (FLAGS_gpuexec_cuda_sync_around_driver_calls) SynchronizeOrDie();

  auto* tls = &tls_data.get();

  if (kVerifyCudaContext) {
    // Note that if kVerifyCudaContext is used, and contexts are deleted, it's
    // possible this could fail in the CurrentContext() call.
    CHECK_EQ(CurrentContext(),
             tls->context == nullptr ? nullptr : tls->context->context());
  }

  tls->depth--;
  DCHECK_GE(tls->depth, 0);
  if (to_restore_ == nullptr) {
    // Leave context, tls->id, and tls->context set.
    return;
  }

  // Set context and update thread local.
  CHECK_EQ(CUDA_SUCCESS, cuCtxSetCurrent(to_restore_->context()));
  tls->id = to_restore_->id();
  tls->context = to_restore_;
}

namespace {

// Returns a stringified device number associated with pointer, primarily for
// logging purposes. Returns "?" if the device could not be successfully
// queried.
string CUDAPointerToDeviceString(CUdeviceptr pointer) {
  auto value = CUDADriver::GetPointerDevice(pointer);
  if (value.ok()) {
    return absl::StrCat(value.ValueOrDie());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified memory space associated with pointer, primarily for
// logging purposes. Returns "?" if the memory space could not be successfully
// queried.
string CUDAPointerToMemorySpaceString(CUdeviceptr pointer) {
  auto value = CUDADriver::GetPointerMemorySpace(pointer);
  if (value.ok()) {
    return MemorySpaceString(value.ValueOrDie());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified representation of whether or not peer access is
// permitted between the "from" and "to" pointers' associated contexts,
// primarily for logging purposes. Returns "error" if an error is encountered
// in the process of querying.
string CUDAPointersToCanAccessString(CUdeviceptr from, CUdeviceptr to) {
  auto from_context = CUDADriver::GetPointerContext(from);
  if (!from_context.ok()) {
    LOG(ERROR) << "could not retrieve source pointer's context: "
               << from_context.status();
    return "error";
  }
  auto to_context = CUDADriver::GetPointerContext(to);
  if (!to_context.ok()) {
    LOG(ERROR) << "could not retrieve destination pointer's context: "
               << to_context.status();
    return "error";
  }
  return CUDADriver::CanEnablePeerAccess(from_context.ValueOrDie(),
                                         to_context.ValueOrDie())
             ? "true"
             : "false";
}


// Actually performs the work of CUDA initialization. Wrapped up in one-time
// execution guard.
static port::Status InternalInit() {
  CUresult res = CUDA_ERROR_NO_DEVICE;
  if (FLAGS_gpuexec_cuda_driver_inject_init_error) {
    LOG(ERROR) << "injecting CUDA init error; initialization will fail";
  } else {
    res = cuInit(0 /* = flags */);
  }

  if (res == CUDA_SUCCESS) {
    return port::Status::OK();
  }

  LOG(ERROR) << "failed call to cuInit: " << ToString(res);
  Diagnostician::LogDiagnosticInformation();
  return port::Status(port::error::ABORTED,
                      absl::StrCat("failed call to cuInit: ", ToString(res)));
}

}  // namespace

/* static */ port::Status CUDADriver::Init() {
  // Cached return value from calling InternalInit(), as cuInit need only be
  // called once, but CUDADriver::Init may be called many times.
  static port::Status init_retval;
  static bool set = false;
  static mutex *init_mu = new mutex;

  mutex_lock lock(*init_mu);
  if (!set) {
    init_retval = InternalInit();
    set = true;
  }

  return init_retval;
}

/* static */ port::Status CUDADriver::GetDevice(int device_ordinal,
                                                CUdevice *device) {
  CUresult res = cuDeviceGet(device, device_ordinal);
  if (res == CUDA_SUCCESS) {
    return port::Status::OK();
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed call to cuDeviceGet: ", ToString(res)));
}

/* static */ bool CUDADriver::GetDeviceName(CUdevice device,
                                            string *device_name) {
  static const size_t kCharLimit = 64;
  absl::InlinedVector<char, 4> chars(kCharLimit);
  CUresult res = cuDeviceGetName(chars.begin(), kCharLimit - 1, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to get device name for " << device << ": "
               << ToString(res);
    return false;
  }
  chars[kCharLimit - 1] = '\0';
  *device_name = chars.begin();
  return true;
}

bool DeviceOptionsToContextFlags(const DeviceOptions &device_options,
                                 int *flags) {
  static_assert(DeviceOptions::kMask == 0xf,
                "needs update for new device options");

  if (device_options.flags() & DeviceOptions::kDoNotReclaimStackAllocation) {
    *flags |= CU_CTX_LMEM_RESIZE_TO_MAX;
  }

  // If no flags are set the default is CU_CTX_SCHED_AUTO, which
  // in Google environments is very likely to mean SPIN.
  if (device_options.flags() & DeviceOptions::kScheduleSpin) {
    *flags |= CU_CTX_SCHED_SPIN;
  }
  if (device_options.flags() & DeviceOptions::kScheduleYield) {
    *flags |= CU_CTX_SCHED_YIELD;
  }
  if (device_options.flags() & DeviceOptions::kScheduleBlockingSync) {
    *flags |= CU_CTX_SCHED_BLOCKING_SYNC;
  }

  return true;
}

/* static */ port::Status CUDADriver::CreateContext(
    CUdevice device, const DeviceOptions &device_options,
    CudaContext **context) {
  *context = nullptr;

  int flags = 0;
  if (!DeviceOptionsToContextFlags(device_options, &flags)) {
    LOG(WARNING) << "could not convert all device options into context flags";
  }

  CUresult res;
  CUcontext former_context;
  CUcontext new_context;

  unsigned int former_primary_context_flags;
  int former_primary_context_is_active;
  CHECK_EQ(CUDA_SUCCESS,
           cuDevicePrimaryCtxGetState(device, &former_primary_context_flags,
                                      &former_primary_context_is_active));
  if (former_primary_context_flags != flags) {
    if (former_primary_context_is_active) {
      LOG(ERROR)
          << "The primary context is active and has a different flag set ("
          << former_primary_context_flags << ") than the desired flag set ("
          << flags << ").";
    } else {
      CHECK_EQ(CUDA_SUCCESS, cuDevicePrimaryCtxSetFlags(device, flags));
    }
  }

  former_context = CUDADriver::CurrentContextOrDie();
  res = cuDevicePrimaryCtxRetain(&new_context, device);
  if (former_context != nullptr) {
    CUdevice former_device;
    if (cuCtxGetDevice(&former_device) == CUDA_SUCCESS) {
      if (former_device == device) {
        if (former_context == new_context) {
          VLOG(2) << "The primary context " << former_context << " for device "
                  << device
                  << " exists before initializing the StreamExecutor.";
        } else {
          LOG(WARNING) << "A non-primary context " << former_context
                       << " for device " << device
                       << " exists before initializing the StreamExecutor. The "
                       << "primary context is now " << new_context << ". We "
                       << "haven't verified StreamExecutor works with that.";
        }
      }
    } else {
      LOG(ERROR) << "Failed to get the device of the current context "
                 << former_context;
    }
  }
  CHECK_EQ(CUDA_SUCCESS, cuCtxSetCurrent(former_context));

  if (res == CUDA_SUCCESS) {
    *context = CreatedContexts::Add(new_context);
    CHECK(*context != nullptr)
        << "success in this call must entail non-null result";
    VLOG(2) << "created or reused context " << new_context
            << " for this thread";
    return port::Status::OK();
  }

  string message = "failed call to cuDevicePrimaryCtxRetain: " + ToString(res);
  if (res == CUDA_ERROR_OUT_OF_MEMORY) {
    uint64 total_memory;
    if (GetDeviceTotalMemory(device, &total_memory)) {
      absl::StrAppend(&message, "; total memory reported: ", total_memory);
    } else {
      absl::StrAppend(&message, "; could not query total memory");
    }
  }

  return port::Status(port::error::INTERNAL, message);
}

/* static */ void CUDADriver::DestroyContext(CudaContext* context) {
  if (context == nullptr) {
    return;
  }
  CUcontext former_context = CurrentContext();
  CUresult res = cuCtxSetCurrent(context->context());
  CUdevice device;
  cuCtxGetDevice(&device);
  cuCtxSetCurrent(former_context);

  res = cuDevicePrimaryCtxRelease(device);

  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to release CUDA context; leaking: " << ToString(res);
  }

  CreatedContexts::Remove(context->context());
}

/* static */ bool CUDADriver::FuncGetAttribute(CUfunction_attribute attribute,
                                               CUfunction func,
                                               int *attribute_value) {
  CUresult res = cuFuncGetAttribute(attribute_value, attribute, func);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query kernel attribute. kernel: " << func
               << ", attribute: " << attribute;
    return false;
  }
  return true;
}

/* static */ bool CUDADriver::FuncSetCacheConfig(CUfunction function,
                                                 CUfunc_cache cache_config) {
  CUresult res = cuFuncSetCacheConfig(function, cache_config);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to set CUDA kernel cache config. kernel: " << function
               << ", config: " << cache_config << ", result: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ port::StatusOr<CUsharedconfig>
CUDADriver::ContextGetSharedMemConfig(CudaContext* context) {
  CUsharedconfig shared_mem_config;
  ScopedActivateContext activation(context);
  CUresult result = cuCtxGetSharedMemConfig(&shared_mem_config);
  if (result != CUDA_SUCCESS) {
    CUdevice device;
    cuCtxGetDevice(&device);
    LOG(ERROR) << "failed to get CUDA device shared memory config. "
               << "Context device ID: " << device
               << ", result: " << ToString(result);
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("failed to get shared memory config: ", ToString(result)));
  }
  return shared_mem_config;
}

/* static */ port::Status CUDADriver::ContextSetSharedMemConfig(
    CudaContext* context, CUsharedconfig shared_mem_config) {
  ScopedActivateContext activation(context);
  CUresult result = cuCtxSetSharedMemConfig(shared_mem_config);
  if (result != CUDA_SUCCESS) {
    CUdevice device;
    cuCtxGetDevice(&device);
    LOG(ERROR) << "failed to set CUDA device shared memory config. "
               << "Context device ID: " << device
               << ", config: " << shared_mem_config
               << ", result: " << ToString(result);
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("failed to set shared memory config: ", ToString(result)));
  }
  return port::Status::OK();
}

/* static */ bool CUDADriver::LaunchKernel(
    CudaContext* context, CUfunction function, unsigned int grid_dim_x,
    unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, CUstream stream, void **kernel_params,
    void **extra) {
  ScopedActivateContext activation(context);
  VLOG(2) << "launching kernel: " << function << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z;
  CUresult res = cuLaunchKernel(function, grid_dim_x, grid_dim_y, grid_dim_z,
                                block_dim_x, block_dim_y, block_dim_z,
                                shared_mem_bytes, stream, kernel_params, extra);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to launch CUDA kernel: " << function
               << "; result: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully launched kernel";
  return true;
}

/* static */ port::Status CUDADriver::LoadCubin(CudaContext* context,
                                                const char *cubin_bytes,
                                                CUmodule *module) {
  ScopedActivateContext activation(context);
  CUresult result = cuModuleLoadFatBinary(module, cubin_bytes);
  if (result != CUDA_SUCCESS) {
    return port::Status(port::error::INTERNAL,
                        "failed to load in-memory CUBIN: " + ToString(result));
  }

  return port::Status::OK();
}

/* static */ bool CUDADriver::LoadPtx(CudaContext* context,
                                      const char *ptx_contents,
                                      CUmodule *module) {
  port::Notification notification;
  bool ret = true;
  GetDriverExecutor()->Schedule([context, ptx_contents, module, &ret,
                                 &notification]() {
    ScopedActivateContext activation(context);
    void *ptx_data = const_cast<char *>(ptx_contents);
    static const unsigned int kLogBufferBytesLimit = 1024;
    unsigned int error_log_buffer_bytes = kLogBufferBytesLimit;
    unsigned int info_log_buffer_bytes = kLogBufferBytesLimit;
    absl::InlinedVector<char, 4> error_log_buffer(error_log_buffer_bytes);
    absl::InlinedVector<char, 4> info_log_buffer(info_log_buffer_bytes);
    bool log_verbose = true;
    CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                              CU_JIT_ERROR_LOG_BUFFER,
                              CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                              CU_JIT_INFO_LOG_BUFFER, CU_JIT_LOG_VERBOSE};
    // Note that the driver API wants the contents of this values to be stored
    // in an array of void*s, so we coerce them accordingly.
    void *option_values[] = {
        absl::bit_cast<void *>(uintptr_t(error_log_buffer_bytes)),
        absl::bit_cast<void *>(error_log_buffer.data()),
        absl::bit_cast<void *>(uintptr_t(info_log_buffer_bytes)),
        absl::bit_cast<void *>(info_log_buffer.data()),
        absl::bit_cast<void *>(uintptr_t(log_verbose))};
    CHECK(TF_ARRAYSIZE(options) == TF_ARRAYSIZE(option_values));

    CUresult res;
    {
      // TODO(leary) Need to see if NVIDIA can expunge the leakiness in their
      // module loading: see http://b/13248943

      res = cuModuleLoadDataEx(module, ptx_data, TF_ARRAYSIZE(options),
                               options, option_values);
    }

    // The PTX JIT mutates the values in the option values array to reflect the
    // size of the logs it output; now that we've made the call, read the values
    // back out.
    error_log_buffer_bytes = reinterpret_cast<uintptr_t>(option_values[0]);
    info_log_buffer_bytes = reinterpret_cast<uintptr_t>(option_values[2]);
    CHECK_LE(error_log_buffer_bytes, kLogBufferBytesLimit);
    CHECK_LE(info_log_buffer_bytes, kLogBufferBytesLimit);

    if (res != CUDA_SUCCESS) {
      LOG(ERROR) << "failed to load PTX text as a module: " << ToString(res);
      // As a precaution for null termination of the API-provided value, ensure
      // that at least the last byte is null.
      error_log_buffer[error_log_buffer_bytes ?
                       error_log_buffer_bytes - 1 : 0] = '\0';
      LOG(ERROR) << "error log buffer (" << error_log_buffer_bytes
                 << " bytes): " << error_log_buffer.data();
      ret = false;
      notification.Notify();
    }

    VLOG(3) << "PTX compilation info log (" << info_log_buffer_bytes
            << " bytes): " << info_log_buffer.data();
    VLOG(3) << "PTX compilation error log (" << error_log_buffer_bytes
            << " bytes): " << error_log_buffer.data();
    CHECK(module != nullptr);
    notification.Notify();
  });
  notification.WaitForNotification();

  return ret;
}

/* static */ bool CUDADriver::SynchronousMemsetUint8(CudaContext* context,
                                                     CUdeviceptr location,
                                                     uint8 value, size_t size) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemsetD8(location, value, size);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to memset memory: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool CUDADriver::SynchronousMemsetUint32(CudaContext* context,
                                                      CUdeviceptr location,
                                                      uint32 value,
                                                      size_t uint32_count) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemsetD32(location, value, uint32_count);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to memset memory: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool CUDADriver::AsynchronousMemsetUint8(CudaContext* context,
                                                      CUdeviceptr location,
                                                      uint8 value,
                                                      size_t uint32_count,
                                                      CUstream stream) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemsetD8Async(location, value, uint32_count, stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to enqueue async memset operation: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully enqueued async memset operation";
  return true;
}

/* static */ bool CUDADriver::AsynchronousMemsetUint32(CudaContext* context,
                                                       CUdeviceptr location,
                                                       uint32 value,
                                                       size_t uint32_count,
                                                       CUstream stream) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemsetD32Async(location, value, uint32_count, stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to enqueue async memset operation: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully enqueued async memset operation";
  return true;
}

/* static */ bool CUDADriver::AddStreamCallback(CudaContext* context,
                                                CUstream stream,
                                                StreamCallback callback,
                                                void *data) {
  // Note: flags param is required to be zero according to CUDA 6.0.
  CUresult res = cuStreamAddCallback(stream, callback, data, 0 /* = flags */);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "unable to add host callback: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool CUDADriver::GetModuleFunction(CudaContext *context,
                                                CUmodule module,
                                                const char *kernel_name,
                                                CUfunction *function) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  CUresult res = cuModuleGetFunction(function, module, kernel_name);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to get PTX kernel \"" << kernel_name
               << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool CUDADriver::GetModuleSymbol(CudaContext* context,
                                              CUmodule module,
                                              const char *symbol_name,
                                              CUdeviceptr *dptr,
                                              size_t *bytes) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  CUresult res = cuModuleGetGlobal(dptr, bytes, module, symbol_name);
  if (res != CUDA_SUCCESS) {
    // symbol may not be found in the current module, but it may reside in
    // another module.
    VLOG(2) << "failed to get symbol \"" << symbol_name
            << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ void CUDADriver::UnloadModule(CudaContext *context,
                                           CUmodule module) {
  ScopedActivateContext activated{context};
  CUresult res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << ToString(res);
  }
}

/* static */ port::StatusOr<CUdevice> CUDADriver::DeviceFromContext(
    CudaContext* context) {
  ScopedActivateContext activated{context};
  CUdevice device = -1;
  CUresult result = cuCtxGetDevice(&device);
  if (result == CUDA_SUCCESS) {
    return device;
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed to get device for context: ", ToString(result)));
}

/* static */ bool CUDADriver::CreateStream(CudaContext *context,
                                           CUstream *out) {
  // TODO(leary) can we switch this to CU_STREAM_NON_BLOCKING or will that mess
  // up synchronization with respect to memsets and any other things that have
  // to occur on the default stream?
  ScopedActivateContext activated{context};
  CUresult res = cuStreamCreate(out, 0);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not allocate CUDA stream for context "
               << context->context() << ": " << ToString(res);
    return false;
  }

  VLOG(2) << "successfully created stream " << *out << " for context "
          << context->context() << " on thread";
  return true;
}

/* static */ void CUDADriver::DestroyStream(CudaContext* context,
                                            CUstream *stream) {
  if (*stream == nullptr) {
    return;
  }

  ScopedActivateContext activated{context};
  CUresult res = cuStreamDestroy(*stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to destroy CUDA stream for context "
               << context->context() << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << *stream << " for context "
            << context->context();
    *stream = nullptr;
  }
}

/* static */ void *CUDADriver::DeviceAllocate(CudaContext *context,
                                              uint64 bytes) {
  ScopedActivateContext activated{context};
  CUdeviceptr result = 0;
  CUresult res = cuMemAlloc(&result, bytes);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to allocate "
               << port::HumanReadableNumBytes::ToString(bytes) << " (" << bytes
               << " bytes) from device: " << ToString(res);
    return nullptr;
  }
  void *ptr = reinterpret_cast<void *>(result);
  VLOG(2) << "allocated " << ptr << " for context " << context->context()
          << " of " << bytes << " bytes";
  return ptr;
}

/* static */ void CUDADriver::DeviceDeallocate(CudaContext* context,
                                               void *location) {
  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  CUresult res = cuMemFree(pointer);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated " << location << " for context "
            << context->context();
  }
}

/* static */ void *CUDADriver::UnifiedMemoryAllocate(CudaContext *context,
                                                     uint64 bytes) {
  ScopedActivateContext activation(context);
  CUdeviceptr result = 0;
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  CUresult res = cuMemAllocManaged(&result, bytes, CU_MEM_ATTACH_GLOBAL);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes unified memory; result: " << ToString(res);
    return nullptr;
  }
  void *ptr = reinterpret_cast<void *>(result);
  VLOG(2) << "allocated " << ptr << " for context " << context->context()
          << " of " << bytes << " bytes in unified memory";
  return ptr;
}

/* static */ void CUDADriver::UnifiedMemoryDeallocate(CudaContext *context,
                                                      void *location) {
  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  CUresult res = cuMemFree(pointer);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to free unified memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated unified memory at " << location << " for context "
            << context->context();
  }
}

/* static */ void *CUDADriver::HostAllocate(CudaContext *context,
                                            uint64 bytes) {
  ScopedActivateContext activation(context);
  void *host_mem = nullptr;
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  CUresult res = cuMemHostAlloc(&host_mem, bytes, CU_MEMHOSTALLOC_PORTABLE);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes on host: " << ToString(res);
  }
  return host_mem;
}

/* static */ void CUDADriver::HostDeallocate(CudaContext* context,
                                             void *location) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemFreeHost(location);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "error deallocating host memory at " << location << ": "
               << ToString(res);
  }
}

/* static */ bool CUDADriver::HostRegister(CudaContext* context, void *location,
                                           uint64 bytes) {
  ScopedActivateContext activation(context);
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  CUresult res =
      cuMemHostRegister(location, bytes, CU_MEMHOSTREGISTER_PORTABLE);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool CUDADriver::HostUnregister(CudaContext* context,
                                             void *location) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemHostUnregister(location);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ port::Status CUDADriver::DestroyEvent(CudaContext* context,
                                                   CUevent *event) {
  if (*event == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "input event cannot be null");
  }

  ScopedActivateContext activated{context};
  CUresult res = cuEventDestroy(*event);
  *event = nullptr;

  switch (res) {
    case CUDA_SUCCESS:
      return port::Status::OK();
    case CUDA_ERROR_DEINITIALIZED:
    case CUDA_ERROR_NOT_INITIALIZED:
      return port::Status(
          port::error::FAILED_PRECONDITION,
          port::Printf("error destroying CUDA event in context %p: %s", context,
                       ToString(res).c_str()));
    default:
      return port::Status(
          port::error::INTERNAL,
          port::Printf("error destroying CUDA event in context %p: %s", context,
                       ToString(res).c_str()));
  }
}

/* static */ port::Status CUDADriver::RecordEvent(CudaContext* context,
                                                  CUevent event,
                                                  CUstream stream) {
  ScopedActivateContext activated{context};
  CUresult res = cuEventRecord(event, stream);
  switch (res) {
    case CUDA_SUCCESS:
      return port::Status::OK();
    case CUDA_ERROR_DEINITIALIZED:
    case CUDA_ERROR_NOT_INITIALIZED:
      return port::Status(
          port::error::FAILED_PRECONDITION,
          port::Printf("error recording CUDA event on stream %p: %s", stream,
                       ToString(res).c_str()));
    default:
      return port::Status(
          port::error::INVALID_ARGUMENT,
          port::Printf("error recording CUDA event on stream %p: %s", stream,
                       ToString(res).c_str()));
  }
}

/* static */ port::StatusOr<CUresult> CUDADriver::QueryEvent(
    CudaContext *context, CUevent event) {
  ScopedActivateContext activated{context};
  CUresult res = cuEventQuery(event);
  if (res != CUDA_SUCCESS && res != CUDA_ERROR_NOT_READY) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf("failed to query event: %s", ToString(res).c_str()));
  }

  return res;
}

/* static */ bool CUDADriver::GetEventElapsedTime(CudaContext* context,
                                                  float *elapsed_milliseconds,
                                                  CUevent start, CUevent stop) {
  ScopedActivateContext activated{context};
  // The stop event must have completed in order for cuEventElapsedTime to
  // work.
  CUresult res = cuEventSynchronize(stop);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to synchronize the stop event: " << ToString(res);
    return false;
  }
  res = cuEventElapsedTime(elapsed_milliseconds, start, stop);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to get elapsed time between events: "
               << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool CUDADriver::WaitStreamOnEvent(CudaContext* context,
                                                CUstream stream,
                                                CUevent event) {
  ScopedActivateContext activation(context);
  CUresult res = cuStreamWaitEvent(stream, event, 0 /* = flags */);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not wait stream on event: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool CUDADriver::SynchronizeContext(CudaContext* context) {
  ScopedActivateContext activation(context);
  CUresult res = cuCtxSynchronize();
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not synchronize on CUDA context: " << ToString(res)
               << " :: " << port::CurrentStackTrace();
    return false;
  }

  return true;
}

/* static */ port::Status CUDADriver::SynchronizeStream(CudaContext *context,
                                                        CUstream stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  CUresult res = cuStreamSynchronize(stream);
  if (res != CUDA_SUCCESS) {
    port::Status status = port::InternalError(
        absl::StrCat("could not synchronize on CUDA stream: ", ToString(res)));
    LOG(ERROR) << status << " :: " << port::CurrentStackTrace();
    return status;
  }
  VLOG(2) << "successfully synchronized stream " << stream << " on context "
          << context;
  return port::Status::OK();
}

/* static */ bool CUDADriver::IsStreamIdle(CudaContext *context,
                                           CUstream stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  CUresult res = cuStreamQuery(stream);
  if (res == CUDA_SUCCESS) {
    return true;
  }

  if (res != CUDA_ERROR_NOT_READY) {
    LOG(ERROR) << "stream in bad state on status query: " << ToString(res);
  }
  return false;
}

/* static */ port::Status CUDADriver::SynchronousMemcpyD2H(CudaContext *context,
                                                           void *host_dst,
                                                           CUdeviceptr gpu_src,
                                                           uint64 size) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemcpyDtoH(host_dst, gpu_src, size);
  if (res != CUDA_SUCCESS) {
    return port::InternalError(
        port::Printf("failed to synchronous memcpy from device to host: %s; "
                     "host dst: %p; GPU src: %p; size: %llu=0x%llx",
                     ToString(res).c_str(), host_dst,
                     absl::bit_cast<void *>(gpu_src), size, size));
  }
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return port::Status::OK();
}

/* static */ port::Status CUDADriver::SynchronousMemcpyH2D(CudaContext *context,
                                                           CUdeviceptr gpu_dst,
                                                           const void *host_src,
                                                           uint64 size) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemcpyHtoD(gpu_dst, host_src, size);
  if (res != CUDA_SUCCESS) {
    return port::InternalError(port::Printf(
        "failed to synchronous memcpy from host to device: %s; GPU dst: %p;"
        " host src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), absl::bit_cast<void *>(gpu_dst), host_src, size,
        size));
  }
  VLOG(2) << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return port::Status::OK();
}

/* static */ port::Status CUDADriver::SynchronousMemcpyD2D(CudaContext *context,
                                                           CUdeviceptr gpu_dst,
                                                           CUdeviceptr gpu_src,
                                                           uint64 size) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemcpyDtoD(gpu_dst, gpu_src, size);
  if (res != CUDA_SUCCESS) {
    return port::InternalError(port::Printf(
        "failed to synchronous memcpy from host to device: %s; GPU dst: %p; "
        "GPU src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), absl::bit_cast<void *>(gpu_dst),
        absl::bit_cast<void *>(gpu_src), size, size));
  }
  VLOG(2) << "successfully sync memcpy'd d2d of " << size << " bytes";
  return port::Status::OK();
}

/* static */ bool CUDADriver::AsynchronousMemcpyD2H(CudaContext* context,
                                                    void *host_dst,
                                                    CUdeviceptr gpu_src,
                                                    uint64 size,
                                                    CUstream stream) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemcpyDtoHAsync(host_dst, gpu_src, size, stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << port::Printf(
        "failed to enqueue async memcpy from device to host: %s; host dst: %p; "
        "GPU src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), host_dst, absl::bit_cast<void *>(gpu_src), size,
        size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void *>(gpu_src) << " to "
          << host_dst << " on stream " << stream;
  return true;
}

/* static */ bool CUDADriver::AsynchronousMemcpyH2D(CudaContext* context,
                                                    CUdeviceptr gpu_dst,
                                                    const void *host_src,
                                                    uint64 size,
                                                    CUstream stream) {
  ScopedActivateContext activation(context);
  CUresult res = cuMemcpyHtoDAsync(gpu_dst, host_src, size, stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << port::Printf(
        "failed to enqueue async memcpy from host to device: %s; GPU dst: %p; "
        "host src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), absl::bit_cast<void *>(gpu_dst), host_src, size,
        size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " on stream " << stream;
  return true;
}

/* static */ bool CUDADriver::AsynchronousMemcpyD2D(CudaContext* context,
                                                    CUdeviceptr gpu_dst,
                                                    CUdeviceptr gpu_src,
                                                    uint64 size,
                                                    CUstream stream) {
  ScopedActivateContext activation(context);
  CUresult result = cuMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream);
  if (result != CUDA_SUCCESS) {
    LOG(ERROR) << port::Printf(
        "failed to enqueue async memcpy from device to device: %s"
        "; GPU dst: %p on %s %s"
        "; GPU src: %p on %s %s"
        "; can access? %s; size: %llu=0x%llx",
        ToString(result).c_str(), absl::bit_cast<void *>(gpu_dst),
        CUDAPointerToMemorySpaceString(gpu_dst).c_str(),
        CUDAPointerToDeviceString(gpu_dst).c_str(),
        absl::bit_cast<void *>(gpu_src),
        CUDAPointerToMemorySpaceString(gpu_src).c_str(),
        CUDAPointerToDeviceString(gpu_src).c_str(),
        CUDAPointersToCanAccessString(gpu_src, gpu_dst).c_str(), size, size);

    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes";
  return true;
}

/* static */ port::Status CUDADriver::CreateEvent(CudaContext* context,
                                                  CUevent *result,
                                                  EventFlags flags) {
  int cuflags;
  switch (flags) {
    case EventFlags::kDefault:
      cuflags = CU_EVENT_DEFAULT;
      break;
    case EventFlags::kDisableTiming:
      cuflags = CU_EVENT_DISABLE_TIMING;
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(flags);
  }

  ScopedActivateContext activated{context};
  CUresult res = cuEventCreate(result, cuflags);

  if (res == CUDA_SUCCESS) {
    return port::Status::OK();
  } else if (res == CUDA_ERROR_OUT_OF_MEMORY) {
    return port::Status(port::error::RESOURCE_EXHAUSTED,
                        "could not create CUDA event: out of device memory");
  } else {
    return port::Status(
        port::error::FAILED_PRECONDITION,
        absl::StrCat("could not create CUDA event: ", ToString(res)));
  }
}

/* static */ int CUDADriver::GetDeviceCount() {
  int device_count = 0;
  CUresult res = cuDeviceGetCount(&device_count);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not retrieve CUDA device count: " << ToString(res);
    return 0;
  }

  if (FLAGS_gpuexec_cuda_device_0_only && device_count > 1) {
    device_count = 1;
  }
  return device_count;
}

/* static */ port::StatusOr<CudaContext*> CUDADriver::GetPointerContext(
    CUdeviceptr pointer) {
  CudaContext* context = nullptr;
  CUresult result =
      cuPointerGetAttribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT, pointer);
  if (result == CUDA_SUCCESS) {
    CHECK(context != nullptr) << "success should entail non-null context";
    return context;
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed to query device pointer for context: ",
                   ToString(result)));
}

/* static */ port::StatusOr<MemorySpace> CUDADriver::GetPointerMemorySpace(
    CUdeviceptr pointer) {
  unsigned int value;
  CUresult result =
      cuPointerGetAttribute(&value, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer);
  if (result == CUDA_SUCCESS) {
    switch (value) {
      case CU_MEMORYTYPE_DEVICE:
        return MemorySpace::kDevice;
      case CU_MEMORYTYPE_HOST:
        return MemorySpace::kHost;
      default:
        return port::Status(
            port::error::INTERNAL,
            absl::StrCat("unknown memory space provided by CUDA API: ", value));
    }
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed to query device pointer for memory space: ",
                   ToString(result)));
}

/* static */ port::Status CUDADriver::GetPointerAddressRange(CUdeviceptr dptr,
                                                             CUdeviceptr *base,
                                                             size_t *size) {
  CUresult result = cuMemGetAddressRange(base, size, dptr);
  if (result == CUDA_SUCCESS) {
    return port::Status::OK();
  } else if (result == CUDA_ERROR_NOT_FOUND) {
    // We differentiate between "this pointer is unknown" (return here) and
    // "there was an internal error while performing this operation" (return
    // below).
    return port::Status(
        port::error::NOT_FOUND,
        port::Printf("not a device pointer %p; %s",
                     reinterpret_cast<void *>(dptr), ToString(result).c_str()));
  }

  return port::Status(
      port::error::INTERNAL,
      port::Printf("failed to get pointer into for device pointer %p; %s",
                   reinterpret_cast<void *>(dptr), ToString(result).c_str()));
}

/* static */ port::StatusOr<CUdevice> CUDADriver::GetPointerDevice(
    CUdeviceptr pointer) {
  auto result = GetPointerContext(pointer);
  if (!result.ok()) {
    return result.status();
  }

  return DeviceFromContext(result.ValueOrDie());
}

/* static */ port::Status CUDADriver::GetComputeCapability(int *cc_major,
                                                           int *cc_minor,
                                                           CUdevice device) {
  *cc_major = 0;
  *cc_minor = 0;

  CUresult res = cuDeviceGetAttribute(
      cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (res != CUDA_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf(
            "failed to get compute capability major for device: %s; %d",
            ToString(res).c_str(), device));
  }

  res = cuDeviceGetAttribute(
      cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  if (res != CUDA_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf(
            "failed to get compute capability minor for device: %s; %d",
            ToString(res).c_str(), device));
  }

  return port::Status::OK();
}

// Helper function that turns the integer output of cuDeviceGetAttribute to type
// T and wraps it in a StatusOr.
template <typename T>
static port::StatusOr<T> GetSimpleAttribute(CUdevice device,
                                            CUdevice_attribute attribute) {
  int value = -1;
  CUresult result = cuDeviceGetAttribute(&value, attribute, device);
  if (result != CUDA_SUCCESS) {
    return port::Status(
        port::error::NOT_FOUND,
        absl::StrCat("could not retrieve CUDA device attribute (", attribute,
                     "): ", ToString(result)));
  }
  T converted = value;
  return converted;
}

/* static */ port::StatusOr<int> CUDADriver::GetMultiprocessorCount(
    CUdevice device) {
  return GetSimpleAttribute<int>(device,
                                 CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
}

/* static */ port::StatusOr<int64> CUDADriver::GetMaxSharedMemoryPerCore(
    CUdevice device) {
  return GetSimpleAttribute<int64>(
      device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
}

/* static */ port::StatusOr<int64> CUDADriver::GetMaxSharedMemoryPerBlock(
    CUdevice device) {
  return GetSimpleAttribute<int64>(
      device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
}

/* static */ port::StatusOr<int64> CUDADriver::GetMaxThreadsPerMultiprocessor(
    CUdevice device) {
  return GetSimpleAttribute<int64>(
      device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
}

/* static */ port::StatusOr<int64> CUDADriver::GetMaxThreadsPerBlock(
    CUdevice device) {
  return GetSimpleAttribute<int64>(device,
                                   CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

/* static */ port::StatusOr<int64> CUDADriver::GetMaxRegistersPerBlock(
    CUdevice device) {
  return GetSimpleAttribute<int64>(device,
                                   CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
}

/* static */ port::StatusOr<int64> CUDADriver::GetThreadsPerWarp(
    CUdevice device) {
  return GetSimpleAttribute<int64>(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
}

/* static */ bool CUDADriver::GetGridLimits(int *x, int *y, int *z,
                                            CUdevice device) {
  int value;
  CUresult res =
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim x: " << ToString(res);
    return false;
  }
  *x = value;

  res =
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim y: " << ToString(res);
    return false;
  }
  *y = value;

  res =
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim z: " << ToString(res);
    return false;
  }
  *z = value;
  return true;
}

/* static */ bool CUDADriver::GetDriverVersion(int *driver_version) {
  CUresult res = cuDriverGetVersion(driver_version);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query driver version: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ port::StatusOr<int> CUDADriver::GetDeviceAttribute(
    CUdevice_attribute attribute, CUdevice device) {
  int val;
  CUresult res = cuDeviceGetAttribute(&val, attribute, device);
  if (res != CUDA_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf("failed to get device attribute %d for device %d: %s",
                     attribute, device, ToString(res).c_str()));
  }
  return val;
}

/* static */ bool CUDADriver::IsEccEnabled(CUdevice device, bool *result) {
  int value = -1;
  CUresult res =
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query ECC status: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ bool CUDADriver::GetDeviceMemoryInfo(CudaContext* context,
                                                  int64 *free_out,
                                                  int64 *total_out) {
  ScopedActivateContext activation(context);
  size_t free = 0;
  size_t total = 0;
  CUresult res = cuMemGetInfo(&free, &total);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query device memory info: " << ToString(res);
    return false;
  }

  *free_out = free;
  *total_out = total;
  return true;
}

/* static */ bool CUDADriver::GetDeviceTotalMemory(CUdevice device,
                                                   uint64 *result) {
  size_t value = -1;
  CUresult res = cuDeviceTotalMem(&value, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query total available memory: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ string CUDADriver::GetPCIBusID(CUdevice device) {
  string pci_bus_id;
  static const int kBufferSize = 64;
  absl::InlinedVector<char, 4> chars(kBufferSize);
  chars[kBufferSize - 1] = '\0';
  CUresult res = cuDeviceGetPCIBusId(chars.begin(), kBufferSize - 1, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << ToString(res);
    return pci_bus_id;
  }
  pci_bus_id = chars.begin();
  return pci_bus_id;
}

/* static */ bool CUDADriver::CanEnablePeerAccess(CudaContext* from,
                                                  CudaContext* to) {
  if (from == to) {
    return true;  // A context can always access its own memory.
  }

  int can_access_peer = -1;
  auto from_device = DeviceFromContext(from);
  if (!from_device.ok()) {
    LOG(ERROR) << "failed to resolve 'from' peer access context to a device: "
               << from_device.status();
    return false;
  }
  auto to_device = DeviceFromContext(to);
  if (!to_device.ok()) {
    LOG(ERROR) << "failed to resolve 'to' peer access context to a device: "
               << to_device.status();
    return false;
  }
  CUresult res = cuDeviceCanAccessPeer(
      &can_access_peer, from_device.ValueOrDie(), to_device.ValueOrDie());
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to detect peer access capability: " << ToString(res);
    return false;
  }

  return can_access_peer;
}

/* static */ port::Status CUDADriver::EnablePeerAccess(CudaContext* from,
                                                       CudaContext* to) {
  if (from == to) {
    return port::Status::OK();  // A context can always access its own memory.
  }

  ScopedActivateContext activated{from};
  CUresult result = cuCtxEnablePeerAccess(to->context(), 0 /* = flags */);
  if (result != CUDA_SUCCESS &&
      result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf("failed to enable peer access from %p to %p: %s", from, to,
                     ToString(result).c_str()));
  }

  return port::Status::OK();
}

/* static */ port::StatusOr<int> CUDADriver::GetMaxOccupiedBlocksPerCore(
    CudaContext* context, CUfunction kernel, int threads_per_block,
    size_t dynamic_shared_memory_bytes) {
  ScopedActivateContext activation(context);

  int max_blocks;
  CUresult result = cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks, kernel, threads_per_block, dynamic_shared_memory_bytes);
  if (result != CUDA_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf("failed to calculate occupancy of kernel %p: %s", kernel,
                     ToString(result).c_str()));
  }

  return max_blocks;
}

/* static */ CUcontext CUDADriver::CurrentContextOrDie() {
  CUcontext current = nullptr;
  CUresult result = cuCtxGetCurrent(&current);
  if (result != CUDA_SUCCESS) {
    LOG(FATAL) << "failed to query current context: " << ToString(result);
  }
  return current;
}

}  // namespace cuda
}  // namespace stream_executor
