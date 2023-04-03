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

#include <stdint.h>
#include <stdlib.h>

#include <map>
#include <set>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_diagnostics.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/compiler/xla/stream_executor/platform/logging.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/numbers.h"
#include "tensorflow/tsl/platform/stacktrace.h"
#include "tensorflow/tsl/platform/static_threadlocal.h"
#include "tensorflow/tsl/platform/threadpool.h"

bool FLAGS_gpuexec_rocm_driver_inject_init_error = false;
bool FLAGS_gpuexec_rocm_sync_around_driver_calls = false;
bool FLAGS_gpuexec_rocm_device_0_only = false;

#define RETURN_IF_ROCM_ERROR(expr, ...)                                     \
  do {                                                                      \
    hipError_t _res = (expr);                                               \
    if (TF_PREDICT_FALSE(_res != hipSuccess)) {                             \
      return tsl::errors::Internal(__VA_ARGS__, ": ",                       \
                                   ::stream_executor::gpu::ToString(_res)); \
    }                                                                       \
  } while (0)

// Debugging: on each push and pop of a rocm context, verify the current device
// matches the expected one.
constexpr bool kVerifyGpuContext = false;

namespace stream_executor {
namespace gpu {

// GpuContext wraps the device_ordinal.
// Only reason we need this wrapper class is to make the GpuDriver* API
class GpuContext {
 public:
  GpuContext(const int v) : device_ordinal_(v) {}

  int device_ordinal() const { return device_ordinal_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  const int device_ordinal_;
};

namespace {

// Formats hipError_t to output prettified values into a log stream.
// Error summaries taken from:
string ToString(hipError_t result) {
#define OSTREAM_ROCM_ERROR(__name) \
  case hipError##__name:           \
    return "HIP_ERROR_" #__name;

  switch (result) {
    OSTREAM_ROCM_ERROR(InvalidValue)
    OSTREAM_ROCM_ERROR(OutOfMemory)
    OSTREAM_ROCM_ERROR(NotInitialized)
    OSTREAM_ROCM_ERROR(Deinitialized)
    OSTREAM_ROCM_ERROR(NoDevice)
    OSTREAM_ROCM_ERROR(InvalidDevice)
    OSTREAM_ROCM_ERROR(InvalidImage)
    OSTREAM_ROCM_ERROR(InvalidContext)
    OSTREAM_ROCM_ERROR(InvalidHandle)
    OSTREAM_ROCM_ERROR(NotFound)
    OSTREAM_ROCM_ERROR(NotReady)
    OSTREAM_ROCM_ERROR(NoBinaryForGpu)

    // Encountered an uncorrectable ECC error during execution.
    OSTREAM_ROCM_ERROR(ECCNotCorrectable)

    // Load/store on an invalid address. Must reboot all context.
    case 700:
      return "ROCM_ERROR_ILLEGAL_ADDRESS";
    // Passed too many / wrong arguments, too many threads for register count.
    case 701:
      return "ROCM_ERROR_LAUNCH_OUT_OF_RESOURCES";

      OSTREAM_ROCM_ERROR(ContextAlreadyInUse)
      OSTREAM_ROCM_ERROR(PeerAccessUnsupported)
      OSTREAM_ROCM_ERROR(Unknown)  // Unknown internal error to ROCM.
    default:
      return absl::StrCat("hipError_t(", static_cast<int>(result), ")");
  }
}

// ROCM driver routines may require a large amount of stack (particularly
// hipModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
tsl::thread::ThreadPool* GetDriverExecutor() {
  static tsl::thread::ThreadPool* thread_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), tsl::ThreadOptions(), "rocm_driver", 1);
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

// Returns the current device set in HIP. This is done by calling the
// HIP driver (e.g., this value is not our cached view of the current device).
static int CurrentDeviceOrDie() {
  int current = -1;
  hipError_t result = wrap::hipGetDevice(&current);
  if (result != hipSuccess) {
    LOG(FATAL) << "failed to query current device: " << ToString(result);
  }
  return current;
}

namespace {

// Call hipDeviceSynchronize and crash if it doesn't succeed.
void SynchronizeOrDie() {
  auto res = wrap::hipDeviceSynchronize();
  if (res != hipSuccess) {
    LOG(FATAL) << "Synchronize found " << ToString(res)
               << " :: " << tsl::CurrentStackTrace();
  }
}

struct ThreadLocalData {
  int current_device_ordinal;
  int depth;
};

TSL_STATIC_THREAD_LOCAL_POD(ThreadLocalData, tls_data);

}  // namespace

ScopedActivateContext::ScopedActivateContext(GpuContext* context) {
  if (FLAGS_gpuexec_rocm_sync_around_driver_calls) {
    SynchronizeOrDie();
  }

  auto* tls = &tls_data.get();
  if (tls->depth == 0) {
    tls->current_device_ordinal = CurrentDeviceOrDie();
  }

  if (kVerifyGpuContext) {
    CHECK_EQ(CurrentDeviceOrDie(), tls->current_device_ordinal);
  }

  tls->depth++;

  to_restore_ = context;

  if (context->device_ordinal() == tls->current_device_ordinal) {
    DCHECK_EQ(CurrentDeviceOrDie(), context->device_ordinal());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching device from "
          << tls->current_device_ordinal << " to " << context->device_ordinal();

  // Set the device and update thread local.
  CHECK_EQ(hipSuccess, wrap::hipSetDevice(context->device_ordinal()));
  tls->current_device_ordinal = context->device_ordinal();
}

ScopedActivateContext::~ScopedActivateContext() {
  if (FLAGS_gpuexec_rocm_sync_around_driver_calls) {
    SynchronizeOrDie();
  }

  auto* tls = &tls_data.get();

  if (kVerifyGpuContext) {
    CHECK_EQ(CurrentDeviceOrDie(), tls->current_device_ordinal);
  }

  tls->depth--;
  DCHECK_GE(tls->depth, 0);

  if (to_restore_->device_ordinal() == tls->current_device_ordinal) {
    DCHECK_EQ(CurrentDeviceOrDie(), to_restore_->device_ordinal());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching device from "
          << tls->current_device_ordinal << " to "
          << to_restore_->device_ordinal();

  // Set context and update thread local.
  CHECK_EQ(hipSuccess, wrap::hipSetDevice(to_restore_->device_ordinal()));
  tls->current_device_ordinal = to_restore_->device_ordinal();
}

namespace {

// Returns a stringified device number associated with pointer, primarily for
// logging purposes. Returns "?" if the device could not be successfully
// queried.
string ROCMPointerToDeviceString(hipDeviceptr_t pointer) {
  auto value = GpuDriver::GetPointerDevice(pointer);
  if (value.ok()) {
    return absl::StrCat(value.value());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified memory space associated with pointer, primarily for
// logging purposes. Returns "?" if the memory space could not be successfully
// queried.
string ROCMPointerToMemorySpaceString(hipDeviceptr_t pointer) {
  auto value = GpuDriver::GetPointerMemorySpace(pointer);
  if (value.ok()) {
    return MemorySpaceString(value.value());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified representation of whether or not peer access is
// permitted between the "from" and "to" pointers' associated contexts,
// primarily for logging purposes. Returns "error" if an error is encountered
// in the process of querying.
string ROCMPointersToCanAccessString(hipDeviceptr_t from, hipDeviceptr_t to) {
  hipPointerAttribute_t from_pointerAttributes;
  hipError_t result =
      wrap::hipPointerGetAttributes(&from_pointerAttributes, from);
  if (result != hipSuccess) {
    LOG(ERROR) << "could not retrieve source pointer's device: "
               << ToString(result);
    return "error";
  }

  hipPointerAttribute_t to_pointerAttributes;
  result = wrap::hipPointerGetAttributes(&to_pointerAttributes, to);
  if (result != hipSuccess) {
    LOG(ERROR) << "could not retrieve destination pointer's device: "
               << ToString(result);
    return "error";
  }

  GpuContext fromCtx(from_pointerAttributes.device);
  GpuContext toCtx(to_pointerAttributes.device);

  return GpuDriver::CanEnablePeerAccess(&fromCtx, &toCtx) ? "true" : "false";
}

// Actually performs the work of ROCM initialization. Wrapped up in one-time
// execution guard.
static tsl::Status InternalInit() {
  hipError_t res = hipErrorNoDevice;
  if (FLAGS_gpuexec_rocm_driver_inject_init_error) {
    LOG(ERROR) << "injecting ROCM init error; initialization will fail";
  } else {
    res = wrap::hipInit(0 /* = flags */);
  }

  if (res == hipSuccess) {
    return tsl::OkStatus();
  }

  LOG(ERROR) << "failed call to hipInit: " << ToString(res);
  Diagnostician::LogDiagnosticInformation();
<<<<<<< HEAD
  return tsl::Status(absl::StatusCode::kAborted,
                     absl::StrCat("failed call to hipInit: ", ToString(res)));
=======
  return tsl::Status{absl::StatusCode::kAborted,
                     absl::StrCat("failed call to hipInit: ", ToString(res))};
>>>>>>> upstream/master
}

}  // namespace

/* static */ tsl::Status GpuDriver::Init() {
  // Cached return value from calling InternalInit(), as hipInit need only be
  // called once, but GpuDriver::Init may be called many times.
  static tsl::Status* init_retval = [] {
    return new tsl::Status(InternalInit());
  }();
  return *init_retval;
}

/* static */ tsl::Status GpuDriver::GetDevice(int device_ordinal,
                                              hipDevice_t* device) {
  hipError_t res = wrap::hipDeviceGet(device, device_ordinal);
  if (res == hipSuccess) {
    return tsl::OkStatus();
  }

<<<<<<< HEAD
  return tsl::Status(
      absl::StatusCode::kInternal,
      absl::StrCat("failed call to hipDeviceGet: ", ToString(res)));
=======
  return tsl::Status{
      absl::StatusCode::kInternal,
      absl::StrCat("failed call to hipDeviceGet: ", ToString(res))};
>>>>>>> upstream/master
}

/* static */ tsl::Status GpuDriver::GetDeviceName(hipDevice_t device,
                                                  string* device_name) {
  static const size_t kCharLimit = 64;
  absl::InlinedVector<char, 4> chars(kCharLimit);
  RETURN_IF_ROCM_ERROR(
      wrap::hipDeviceGetName(chars.begin(), kCharLimit - 1, device),
      "Failed to get device name");
  chars[kCharLimit - 1] = '\0';
  *device_name = chars.begin();
  return tsl::OkStatus();
}

bool DeviceOptionsToContextFlags(const DeviceOptions& device_options,
                                 int* flags) {
  static_assert(DeviceOptions::kMask == 0xf,
                "needs update for new device options");
  return true;
}

/* static */ tsl::Status GpuDriver::CreateContext(
    int device_ordinal, hipDevice_t device, const DeviceOptions& device_options,
    GpuContext** context) {
  // TODO(hanbinyoon): Create a real context, i.e., by calling hipCtxCreate().
  *context = new GpuContext(device_ordinal);
  return tsl::OkStatus();
}
/* static */ void GpuDriver::DestroyContext(GpuContext* context) {
  if (context == nullptr) {
    return;
  }
  delete context;
}

/* static */ hipCtx_t GpuDriver::GetContextHandle(GpuContext* context) {
  // TODO(hanbinyoon): Return a real context.
  return nullptr;
}

/* static */ tsl::Status GpuDriver::FuncGetAttribute(
    hipDeviceAttribute_t attribute, hipFunction_t func, int* attribute_value) {
  // TODO(ROCm) properly implement this feature in HIP
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::FuncSetCacheConfig(
    hipFunction_t function, hipFuncCache_t cache_config) {
  RETURN_IF_ROCM_ERROR(wrap::hipFuncSetCacheConfig(function, cache_config),
                       "Failed to set ROCM kernel cache config.");
  return tsl::OkStatus();
}

/* static */ tsl::StatusOr<hipSharedMemConfig>
GpuDriver::ContextGetSharedMemConfig(GpuContext* context) {
  hipSharedMemConfig shared_mem_config;
  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(wrap::hipDeviceGetSharedMemConfig(&shared_mem_config),
                       "Failed to get shared memory config");
  return shared_mem_config;
}

/* static */ tsl::Status GpuDriver::ContextSetSharedMemConfig(
    GpuContext* context, hipSharedMemConfig shared_mem_config) {
  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(wrap::hipDeviceSetSharedMemConfig(shared_mem_config),
                       "Failed to set ROCM device shared memory config");
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::LaunchKernel(
    GpuContext* context, absl::string_view kernel_name, hipFunction_t function,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes,
    GpuStreamHandle stream, void** kernel_params, void** extra) {
  ScopedActivateContext activation{context};
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << " smem: " << shared_mem_bytes;
  RETURN_IF_ROCM_ERROR(wrap::hipModuleLaunchKernel(
                           function, grid_dim_x, grid_dim_y, grid_dim_z,
                           block_dim_x, block_dim_y, block_dim_z,
                           shared_mem_bytes, stream, kernel_params, extra),
                       "Failed to launch ROCm kernel: ", kernel_name,
                       " with block dimensions: ", block_dim_x, "x",
                       block_dim_y, "x", block_dim_z);
  VLOG(2) << "successfully launched kernel";
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::LoadPtx(GpuContext* context,
                                            const char* ptx_contents,
                                            hipModule_t* module) {
  LOG(ERROR) << "Feature not supported on ROCm platform (LoadPtx)";
  return tsl::errors::Internal("Not Implemented");
}

/* static */ tsl::Status GpuDriver::LoadCubin(GpuContext* context,
                                              const char* cubin_bytes,
                                              hipModule_t* module) {
<<<<<<< HEAD
  return tsl::Status(absl::StatusCode::kInternal,
                     "Feature not supported on ROCm platform (LoadCubin)");
=======
  return tsl::Status{absl::StatusCode::kInternal,
                     "Feature not supported on ROCm platform (LoadCubin)"};
>>>>>>> upstream/master
}

/* static */ tsl::Status GpuDriver::LoadHsaco(GpuContext* context,
                                              const char* hsaco_contents,
                                              hipModule_t* module) {
  absl::Notification notification;
  tsl::Status ret = tsl::OkStatus();
  GetDriverExecutor()->Schedule(
      [context, hsaco_contents, module, &ret, &notification]() {
        ScopedActivateContext activation{context};
        void* hsaco_data = const_cast<char*>(hsaco_contents);

        hipError_t res = wrap::hipModuleLoadData(module, hsaco_data);

        if (res != hipSuccess) {
          ret = tsl::errors::Internal("Failed to load HSACO: ", ToString(res));
          notification.Notify();
        }

        CHECK(module != nullptr);
        notification.Notify();
      });
  notification.WaitForNotification();

  return ret;
}

/* static */ tsl::Status GpuDriver::SynchronousMemsetUint8(
    GpuContext* context, hipDeviceptr_t location, uint8 value, size_t size) {
  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(wrap::hipMemsetD8(location, value, size),
                       "Failed to memset memory");
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::SynchronousMemsetUint32(
    GpuContext* context, hipDeviceptr_t location, uint32 value,
    size_t uint32_count) {
  ScopedActivateContext activation{context};
  void* pointer = absl::bit_cast<void*>(location);
  RETURN_IF_ROCM_ERROR(wrap::hipMemsetD32(pointer, value, uint32_count),
                       "Failed to memset memory");
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::AsynchronousMemsetUint8(
    GpuContext* context, hipDeviceptr_t location, uint8 value,
    size_t uint32_count, GpuStreamHandle stream) {
  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(
      wrap::hipMemsetAsync(location, value, uint32_count, stream),
      "Failed to enqueue async memset operation");
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::AsynchronousMemsetUint32(
    GpuContext* context, hipDeviceptr_t location, uint32 value,
    size_t uint32_count, GpuStreamHandle stream) {
  ScopedActivateContext activation{context};
  void* pointer = absl::bit_cast<void*>(location);
  RETURN_IF_ROCM_ERROR(
      wrap::hipMemsetD32Async(pointer, value, uint32_count, stream),
      "Failed to enqueue async memset operation");
  VLOG(2) << "successfully enqueued async memset operation";
  return tsl::OkStatus();
}

/* static */ bool GpuDriver::AddStreamCallback(GpuContext* context,
                                               GpuStreamHandle stream,
                                               StreamCallback callback,
                                               void* data) {
  hipError_t res = wrap::hipStreamAddCallback(
      stream, (hipStreamCallback_t)callback, data, 0 /* = flags */);
  if (res != hipSuccess) {
    LOG(ERROR) << "unable to add host callback: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool GpuDriver::GetModuleFunction(GpuContext* context,
                                               hipModule_t module,
                                               const char* kernel_name,
                                               hipFunction_t* function) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  hipError_t res = wrap::hipModuleGetFunction(function, module, kernel_name);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to get kernel \"" << kernel_name
               << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::GetModuleSymbol(GpuContext* context,
                                             hipModule_t module,
                                             const char* symbol_name,
                                             hipDeviceptr_t* dptr,
                                             size_t* bytes) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  hipError_t res = wrap::hipModuleGetGlobal(dptr, bytes, module, symbol_name);
  if (res != hipSuccess) {
    // symbol may not be found in the current module, but it may reside in
    // another module.
    VLOG(2) << "failed to get symbol \"" << symbol_name
            << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ void GpuDriver::UnloadModule(GpuContext* context,
                                          hipModule_t module) {
  ScopedActivateContext activated{context};
  hipError_t res = wrap::hipModuleUnload(module);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << ToString(res);
  }
}

/* static */ bool GpuDriver::CreateStream(GpuContext* context,
                                          GpuStreamHandle* stream,
                                          int priority) {
  ScopedActivateContext activated{context};
  hipError_t res;
  if (priority == 0) {
    res = wrap::hipStreamCreateWithFlags(
        stream, hipStreamDefault);  // switch to hipStreamNonBlocking?
  } else {
    res = wrap::hipStreamCreateWithPriority(
        stream, hipStreamDefault, priority);  // switch to hipStreamNonBlocking?
  }
  if (res != hipSuccess) {
    LOG(ERROR) << "could not allocate ROCM stream for device "
               << context->device_ordinal() << ": " << ToString(res);
    return false;
  }

  VLOG(2) << "successfully created stream " << *stream << " for device "
          << context->device_ordinal() << " on thread";
  return true;
}

/* static */ void GpuDriver::DestroyStream(GpuContext* context,
                                           GpuStreamHandle* stream) {
  if (*stream == nullptr) {
    return;
  }

  ScopedActivateContext activated{context};
  hipError_t res = wrap::hipStreamDestroy(*stream);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to destroy ROCM stream for device "
               << context->device_ordinal() << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << *stream << " for device "
            << context->device_ordinal();
    *stream = nullptr;
  }
}

/* static */ void* GpuDriver::DeviceAllocate(GpuContext* context,
                                             uint64_t bytes) {
  ScopedActivateContext activated{context};
  hipDeviceptr_t result = 0;
  hipError_t res = wrap::hipMalloc(&result, bytes);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to allocate "
               << tsl::strings::HumanReadableNumBytes(bytes) << " (" << bytes
               << " bytes) from device: " << ToString(res);
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  VLOG(2) << "allocated " << ptr << " for device " << context->device_ordinal()
          << " of " << bytes << " bytes";
  return ptr;
}

/* static */ void GpuDriver::DeviceDeallocate(GpuContext* context,
                                              void* location) {
  ScopedActivateContext activation{context};
  hipDeviceptr_t pointer = absl::bit_cast<hipDeviceptr_t>(location);
  hipError_t res = wrap::hipFree(pointer);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated " << location << " for device "
            << context->device_ordinal();
  }
}

/* static */ void* GpuDriver::UnifiedMemoryAllocate(GpuContext* context,
                                                    uint64_t bytes) {
  ScopedActivateContext activated{context};

  LOG(ERROR)
      << "Feature not supported on ROCm platform (UnifiedMemoryAllocate)";
  return nullptr;
}

/* static */ void GpuDriver::UnifiedMemoryDeallocate(GpuContext* context,
                                                     void* location) {
  LOG(ERROR)
      << "Feature not supported on ROCm platform (UnifiedMemoryDeallocate)";
}

/* static */ void* GpuDriver::HostAllocate(GpuContext* context,
                                           uint64_t bytes) {
  ScopedActivateContext activation{context};
  void* host_mem = nullptr;
  // "Portable" memory is visible to all ROCM contexts. Safe for our use model.
  hipError_t res = wrap::hipHostMalloc(&host_mem, bytes, hipHostMallocPortable);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes on host: " << ToString(res);
  }
  return host_mem;
}

/* static */ void GpuDriver::HostDeallocate(GpuContext* context,
                                            void* location) {
  ScopedActivateContext activation{context};
  hipError_t res = wrap::hipHostFree(location);
  if (res != hipSuccess) {
    LOG(ERROR) << "error deallocating host memory at " << location << ": "
               << ToString(res);
  }
}

/* static */ bool GpuDriver::HostRegister(GpuContext* context, void* location,
                                          uint64_t bytes) {
  ScopedActivateContext activation{context};
  // "Portable" memory is visible to all ROCM contexts. Safe for our use model.
  hipError_t res =
      wrap::hipHostRegister(location, bytes, hipHostRegisterPortable);
  if (res != hipSuccess) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool GpuDriver::HostUnregister(GpuContext* context,
                                            void* location) {
  ScopedActivateContext activation{context};
  hipError_t res = wrap::hipHostUnregister(location);
  if (res != hipSuccess) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ tsl::Status GpuDriver::DestroyEvent(GpuContext* context,
                                                 GpuEventHandle* event) {
  if (*event == nullptr) {
<<<<<<< HEAD
    return tsl::Status(absl::StatusCode::kInvalidArgument,
                       "input event cannot be null");
=======
    return tsl::Status{absl::StatusCode::kInvalidArgument,
                       "input event cannot be null"};
>>>>>>> upstream/master
  }

  ScopedActivateContext activated{context};
  hipError_t res = wrap::hipEventDestroy(*event);
  *event = nullptr;

  switch (res) {
    case hipSuccess:
      return tsl::OkStatus();
    case hipErrorDeinitialized:
    case hipErrorNotInitialized:
<<<<<<< HEAD
      return tsl::Status(
=======
      return tsl::Status{
>>>>>>> upstream/master
          absl::StatusCode::kFailedPrecondition,
          absl::StrFormat("error destroying ROCM event in device %d: %s",
                          context->device_ordinal(), ToString(res).c_str()));
    default:
<<<<<<< HEAD
      return tsl::Status(
=======
      return tsl::Status{
>>>>>>> upstream/master
          absl::StatusCode::kInternal,
          absl::StrFormat("error destroying ROCM event in device %d: %s",
                          context->device_ordinal(), ToString(res).c_str()));
  }
}

/* static */ tsl::Status GpuDriver::RecordEvent(GpuContext* context,
                                                GpuEventHandle event,
                                                GpuStreamHandle stream) {
  ScopedActivateContext activated{context};
  hipError_t res = wrap::hipEventRecord(event, stream);
  switch (res) {
    case hipSuccess:
      return tsl::OkStatus();
    case hipErrorDeinitialized:
    case hipErrorNotInitialized:
<<<<<<< HEAD
      return tsl::Status(
=======
      return tsl::Status{
>>>>>>> upstream/master
          absl::StatusCode::kFailedPrecondition,
          absl::StrFormat("error recording ROCM event on stream %p: %s", stream,
                          ToString(res).c_str()));
    default:
<<<<<<< HEAD
      return tsl::Status(
=======
      return tsl::Status{
>>>>>>> upstream/master
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("error recording ROCM event on stream %p: %s", stream,
                          ToString(res).c_str()));
  }
}

/* static */ tsl::StatusOr<hipError_t> GpuDriver::QueryEvent(
    GpuContext* context, GpuEventHandle event) {
  ScopedActivateContext activated{context};
  hipError_t res = wrap::hipEventQuery(event);
  if (res != hipSuccess && res != hipErrorNotReady) {
<<<<<<< HEAD
    return tsl::Status(
        absl::StatusCode::kInternal,
        absl::StrFormat("failed to query event: %s", ToString(res).c_str()));
=======
    return tsl::Status{
        absl::StatusCode::kInternal,
        absl::StrFormat("failed to query event: %s", ToString(res).c_str())};
>>>>>>> upstream/master
  }

  return res;
}

/* static */ bool GpuDriver::GetEventElapsedTime(GpuContext* context,
                                                 float* elapsed_milliseconds,
                                                 GpuEventHandle start,
                                                 GpuEventHandle stop) {
  ScopedActivateContext activated{context};
  // The stop event must have completed in order for hipEventElapsedTime to
  // work.
  hipError_t res = wrap::hipEventSynchronize(stop);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to synchronize the stop event: " << ToString(res);
    return false;
  }
  res = wrap::hipEventElapsedTime(elapsed_milliseconds, start, stop);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to get elapsed time between events: "
               << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::WaitStreamOnEvent(GpuContext* context,
                                               GpuStreamHandle stream,
                                               GpuEventHandle event) {
  ScopedActivateContext activation{context};
  hipError_t res = wrap::hipStreamWaitEvent(stream, event, 0 /* = flags */);
  if (res != hipSuccess) {
    LOG(ERROR) << "could not wait stream on event: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::SynchronizeContext(GpuContext* context) {
  ScopedActivateContext activation{context};
  hipError_t res = wrap::hipDeviceSynchronize();
  if (res != hipSuccess) {
    LOG(ERROR) << "could not synchronize on ROCM device: " << ToString(res)
               << " :: " << tsl::CurrentStackTrace();
    return false;
  }

  return true;
}

/* static */ tsl::Status GpuDriver::SynchronizeStream(GpuContext* context,
                                                      GpuStreamHandle stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  RETURN_IF_ROCM_ERROR(wrap::hipStreamSynchronize(stream),
                       "Could not synchronize on ROCM stream");
  VLOG(2) << "successfully synchronized stream " << stream << " on device "
          << context->device_ordinal();
  return tsl::OkStatus();
}

/* static */ bool GpuDriver::IsStreamIdle(GpuContext* context,
                                          GpuStreamHandle stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  hipError_t res = wrap::hipStreamQuery(stream);
  if (res == hipSuccess) {
    return true;
  }

  if (res != hipErrorNotReady) {
    LOG(ERROR) << "stream in bad state on status query: " << ToString(res);
  }
  return false;
}

/* static */ tsl::Status GpuDriver::SynchronousMemcpyD2H(GpuContext* context,
                                                         void* host_dst,
                                                         hipDeviceptr_t gpu_src,
                                                         uint64_t size) {
  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(
      wrap::hipMemcpyDtoH(host_dst, gpu_src, size),
      absl::StrFormat("failed to synchronous memcpy from device to host: "
                      "host dst: %p; Gpu src: %p; size: %llu=0x%llx",
                      host_dst, absl::bit_cast<void*>(gpu_src), size, size));
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::SynchronousMemcpyH2D(GpuContext* context,
                                                         hipDeviceptr_t gpu_dst,
                                                         const void* host_src,
                                                         uint64_t size) {
  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(
      wrap::hipMemcpyHtoD(gpu_dst, const_cast<void*>(host_src), size),
      absl::StrFormat(
          "failed to synchronous memcpy from host to device: Gpu dst: %p;"
          " host src: %p; size: %llu=0x%llx",
          absl::bit_cast<void*>(gpu_dst), host_src, size, size));
  VLOG(2) << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::SynchronousMemcpyD2D(GpuContext* context,
                                                         hipDeviceptr_t gpu_dst,
                                                         hipDeviceptr_t gpu_src,
                                                         uint64_t size) {
  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(
      wrap::hipMemcpyDtoD(gpu_dst, gpu_src, size),
      absl::StrFormat(
          "failed to synchronous memcpy from host to device:Gpu dst: %p; "
          "Gpu src: %p; size: %llu=0x%llx",
          absl::bit_cast<void*>(gpu_dst), absl::bit_cast<void*>(gpu_src), size,
          size));
  VLOG(2) << "successfully sync memcpy'd d2d of " << size << " bytes";
  return tsl::OkStatus();
}

/* static */ bool GpuDriver::AsynchronousMemcpyD2H(GpuContext* context,
                                                   void* host_dst,
                                                   hipDeviceptr_t gpu_src,
                                                   uint64_t size,
                                                   GpuStreamHandle stream) {
  ScopedActivateContext activation{context};
  hipError_t res = wrap::hipMemcpyDtoHAsync(host_dst, gpu_src, size, stream);
  if (res != hipSuccess) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from device to host: %s; host dst: %p; "
        "Gpu src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), host_dst, absl::bit_cast<void*>(gpu_src), size,
        size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream;
  return true;
}

/* static */ bool GpuDriver::AsynchronousMemcpyH2D(GpuContext* context,
                                                   hipDeviceptr_t gpu_dst,
                                                   const void* host_src,
                                                   uint64_t size,
                                                   GpuStreamHandle stream) {
  ScopedActivateContext activation{context};
  hipError_t res = wrap::hipMemcpyHtoDAsync(
      gpu_dst, const_cast<void*>(host_src), size, stream);
  if (res != hipSuccess) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from host to device: %s; Gpu dst: %p; "
        "host src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), absl::bit_cast<void*>(gpu_dst), host_src, size,
        size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " on stream " << stream;
  return true;
}

/* static */ bool GpuDriver::AsynchronousMemcpyD2D(GpuContext* context,
                                                   hipDeviceptr_t gpu_dst,
                                                   hipDeviceptr_t gpu_src,
                                                   uint64_t size,
                                                   GpuStreamHandle stream) {
  ScopedActivateContext activation{context};
  hipError_t result = wrap::hipMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream);
  if (result != hipSuccess) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from device to device: %s"
        "; Gpu dst: %p on %s %s"
        "; Gpu src: %p on %s %s"
        "; can access? %s; size: %llu=0x%llx",
        ToString(result).c_str(), absl::bit_cast<void*>(gpu_dst),
        ROCMPointerToMemorySpaceString(gpu_dst).c_str(),
        ROCMPointerToDeviceString(gpu_dst).c_str(),
        absl::bit_cast<void*>(gpu_src),
        ROCMPointerToMemorySpaceString(gpu_src).c_str(),
        ROCMPointerToDeviceString(gpu_src).c_str(),
        ROCMPointersToCanAccessString(gpu_src, gpu_dst).c_str(), size, size);

    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes";
  return true;
}

/* static */ tsl::Status GpuDriver::InitEvent(GpuContext* context,
                                              GpuEventHandle* event,
                                              EventFlags flags) {
  int hipflags;
  switch (flags) {
    case EventFlags::kDefault:
      hipflags = hipEventDefault;
      break;
    case EventFlags::kDisableTiming:
      hipflags = hipEventDisableTiming | hipEventReleaseToSystem;
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(hipflags);
  }

  ScopedActivateContext activated{context};
  hipError_t res = wrap::hipEventCreateWithFlags(event, hipflags);

  if (res == hipSuccess) {
    return tsl::OkStatus();
  } else if (res == hipErrorMemoryAllocation) {
<<<<<<< HEAD
    return tsl::Status(absl::StatusCode::kResourceExhausted,
                       "could not create ROCM event: out of device memory");
  } else {
    return tsl::Status(
        absl::StatusCode::kFailedPrecondition,
        absl::StrCat("could not create ROCM event: ", ToString(res)));
=======
    return tsl::Status{absl::StatusCode::kResourceExhausted,
                       "could not create ROCM event: out of device memory"};
  } else {
    return tsl::Status{
        absl::StatusCode::kFailedPrecondition,
        absl::StrCat("could not create ROCM event: ", ToString(res))};
>>>>>>> upstream/master
  }
}

/* static */ int GpuDriver::GetDeviceCount() {
  int device_count = 0;
  hipError_t res = wrap::hipGetDeviceCount(&device_count);
  if (res != hipSuccess) {
    LOG(ERROR) << "could not retrieve ROCM device count: " << ToString(res);
    return 0;
  }

  if (FLAGS_gpuexec_rocm_device_0_only && device_count > 1) {
    device_count = 1;
  }
  return device_count;
}

/* static */ tsl::Status GpuDriver::GetComputeCapability(int* cc_major,
                                                         int* cc_minor,
                                                         hipDevice_t device) {
  return tsl::Status(
      absl::StatusCode::kInternal,
      absl::StrFormat("failed to get compute capability for device: %d "
                      "(unsupported API on AMD Gpus)",
                      device));
}

/* static */ tsl::Status GpuDriver::GetPointerAddressRange(hipDeviceptr_t dptr,
                                                           hipDeviceptr_t* base,
                                                           size_t* size) {
  hipError_t result = wrap::hipMemGetAddressRange(base, size, dptr);
  if (result == hipSuccess) {
    return tsl::OkStatus();
  } else if (result == hipErrorNotFound) {
    // We differentiate between "this pointer is unknown" (return here) and
    // "there was an internal error while performing this operation" (return
    // below).
<<<<<<< HEAD
    return tsl::Status(absl::StatusCode::kNotFound,
=======
    return tsl::Status{absl::StatusCode::kNotFound,
>>>>>>> upstream/master
                       absl::StrFormat("not a device pointer %p; %s",
                                       reinterpret_cast<void*>(dptr),
                                       ToString(result).c_str()));
  }

<<<<<<< HEAD
  return tsl::Status(
=======
  return tsl::Status{
>>>>>>> upstream/master
      absl::StatusCode::kInternal,
      absl::StrFormat("failed to get pointer into for device pointer %p; %s",
                      reinterpret_cast<void*>(dptr), ToString(result).c_str()));
}

/* static */ tsl::StatusOr<MemorySpace> GpuDriver::GetPointerMemorySpace(
    hipDeviceptr_t pointer) {
  unsigned int value;
  hipError_t result = hipSuccess;
  if (result == hipSuccess) {
    switch (value) {
      case hipMemoryTypeDevice:
        return MemorySpace::kDevice;
      case hipMemoryTypeHost:
        return MemorySpace::kHost;
      default:
<<<<<<< HEAD
        return tsl::Status(
            absl::StatusCode::kInternal,
            absl::StrCat("unknown memory space provided by ROCM API: ", value));
    }
  }

  return tsl::Status(
=======
        return tsl::Status{
            absl::StatusCode::kInternal,
            absl::StrCat("unknown memory space provided by ROCM API: ", value)};
    }
  }

  return tsl::Status{
>>>>>>> upstream/master
      absl::StatusCode::kInternal,
      absl::StrCat("failed to query device pointer for memory space: ",
                   ToString(result)));
}

/* static */ tsl::StatusOr<hipDevice_t> GpuDriver::GetPointerDevice(
    hipDeviceptr_t pointer) {
  hipPointerAttribute_t pointerAttributes;
  hipError_t result =
      wrap::hipPointerGetAttributes(&pointerAttributes, pointer);
  if (result != hipSuccess) {
<<<<<<< HEAD
    return tsl::Status(
        absl::StatusCode::kInternal,
        absl::StrCat("failed to get device for pointer: ", ToString(result)));
=======
    return tsl::Status{
        absl::StatusCode::kInternal,
        absl::StrCat("failed to get device for pointer: ", ToString(result))};
>>>>>>> upstream/master
  }

  hipDevice_t device;
  result = wrap::hipDeviceGet(&device, pointerAttributes.device);
  if (result != hipSuccess) {
<<<<<<< HEAD
    return tsl::Status(
        absl::StatusCode::kInternal,
        absl::StrCat("failed to get device for pointer: ", ToString(result)));
=======
    return tsl::Status{
        absl::StatusCode::kInternal,
        absl::StrCat("failed to get device for pointer: ", ToString(result))};
>>>>>>> upstream/master
  }

  return device;
}

/* static */ tsl::Status GpuDriver::GetGpuISAVersion(int* version,
                                                     hipDevice_t device) {
  hipDeviceProp_t props;
  hipError_t result = wrap::hipGetDeviceProperties(&props, device);
  if (result == hipSuccess) {
    *version = props.gcnArch;
    return tsl::OkStatus();
  }
  *version = 0;
<<<<<<< HEAD
  return tsl::Status(
=======
  return tsl::Status{
>>>>>>> upstream/master
      absl::StatusCode::kInternal,
      absl::StrFormat("failed to determine AMDGpu ISA version for device %d",
                      device));
}

/* static */ tsl::Status GpuDriver::GetGpuGCNArchName(
    hipDevice_t device, std::string* gcnArchName) {
  hipDeviceProp_t props;
  hipError_t result = wrap::hipGetDeviceProperties(&props, device);
  if (result == hipSuccess) {
    *gcnArchName = props.gcnArchName;
    return tsl::OkStatus();
  }
  *gcnArchName = "";
<<<<<<< HEAD
  return tsl::Status(
=======
  return tsl::Status{
>>>>>>> upstream/master
      absl::StatusCode::kInternal,
      absl::StrFormat("failed to determine AMDGpu GCN Arch Name for device %d",
                      device));
}

/* static */ tsl::StatusOr<bool> GpuDriver::GetMFMASupport() {
  hipDeviceProp_t props;
  int dev = 0;
  hipError_t result = hipGetDevice(&dev);
  result = wrap::hipGetDeviceProperties(&props, dev);
  if (result == hipSuccess) {
    std::string gcnArchName = props.gcnArchName;
    VLOG(3) << "GCN arch name " << gcnArchName;
    auto pos = gcnArchName.find(":");
    if (pos != string::npos) gcnArchName = gcnArchName.substr(0, pos);
    pos = gcnArchName.find("gfx");
    if (pos != string::npos) gcnArchName = gcnArchName.substr(pos + 3);
    VLOG(3) << "GCN arch name (stripped) " << gcnArchName;
    return ((gcnArchName == "908") || (gcnArchName == "909") ||
            (gcnArchName == "90a") || (gcnArchName == "940"));
  }
<<<<<<< HEAD
  return tsl::Status(
=======
  return tsl::Status{
>>>>>>> upstream/master
      absl::StatusCode::kInternal,
      absl::StrFormat("failed to determine AMDGpu GCN Arch Name for device %d",
                      dev));
}

// Helper function that turns the integer output of hipDeviceGetAttribute to
// type T and wraps it in a StatusOr.
template <typename T>
static tsl::StatusOr<T> GetSimpleAttribute(hipDevice_t device,
                                           hipDeviceAttribute_t attribute) {
  int value = -1;
  hipError_t result = wrap::hipDeviceGetAttribute(&value, attribute, device);
  if (result != hipSuccess) {
<<<<<<< HEAD
    return tsl::Status(
=======
    return tsl::Status{
>>>>>>> upstream/master
        absl::StatusCode::kNotFound,
        absl::StrCat("could not retrieve ROCM device attribute (", attribute,
                     "): ", ToString(result)));
  }
  T converted = value;
  return converted;
}

/* static */ tsl::StatusOr<int> GpuDriver::GetMultiprocessorCount(
    hipDevice_t device) {
  return GetSimpleAttribute<int>(device, hipDeviceAttributeMultiprocessorCount);
}

/* static */ tsl::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerCore(
    hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(
      device, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor);
}

/* static */ tsl::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerBlock(
    hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(device,
                                     hipDeviceAttributeMaxSharedMemoryPerBlock);
}

/* static */ tsl::StatusOr<int64_t> GpuDriver::GetMaxThreadsPerMultiprocessor(
    hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(
      device, hipDeviceAttributeMaxThreadsPerMultiProcessor);
}

/* static */ tsl::StatusOr<int64_t> GpuDriver::GetMaxThreadsPerBlock(
    hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(device,
                                     hipDeviceAttributeMaxThreadsPerBlock);
}

/* static */ tsl::StatusOr<int64_t> GpuDriver::GetMaxRegistersPerBlock(
    hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(device,
                                     hipDeviceAttributeMaxRegistersPerBlock);
}

/* static */ tsl::StatusOr<int64_t> GpuDriver::GetThreadsPerWarp(
    hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(device, hipDeviceAttributeWarpSize);
}

/* static */ bool GpuDriver::GetGridLimits(int* x, int* y, int* z,
                                           hipDevice_t device) {
  int value;
  hipError_t res = wrap::hipDeviceGetAttribute(
      &value, hipDeviceAttributeMaxGridDimX, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query max grid dim x: " << ToString(res);
    return false;
  }
  *x = value;

  res = wrap::hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimY,
                                    device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query max grid dim y: " << ToString(res);
    return false;
  }
  *y = value;

  res = wrap::hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimZ,
                                    device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query max grid dim z: " << ToString(res);
    return false;
  }
  *z = value;
  return true;
}

/* static */ bool GpuDriver::GetDriverVersion(int* driver_version) {
  hipError_t res = wrap::hipDriverGetVersion(driver_version);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query driver version: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::GetDeviceProperties(
    hipDeviceProp_t* device_properties, int device_ordinal) {
  hipError_t res =
      wrap::hipGetDeviceProperties(device_properties, device_ordinal);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ tsl::StatusOr<int> GpuDriver::GetDeviceAttribute(
    hipDeviceAttribute_t attribute, hipDevice_t device) {
  return GetSimpleAttribute<int>(device, attribute);
}

/* static */ bool GpuDriver::IsEccEnabled(hipDevice_t device, bool* result) {
  int value = -1;
  hipError_t res = hipSuccess;
  // TODO(ROCm) implement this feature in HIP
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query ECC status: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ bool GetReservedMemory(uint64_t* reserve) {
  hipDeviceProp_t props;
  hipDevice_t dev;
  hipError_t res = wrap::hipGetDevice(&dev);

  if (res != hipSuccess) {
    LOG(FATAL) << "failed to query current device: " << ToString(res);
    return false;
  }
  res = wrap::hipGetDeviceProperties(&props, dev);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  std::string gcnArchName = props.gcnArchName;
  // On gfx90a, we hide 1 GB of GPU memory (512MB for gfx908) from TF,
  // to allow for late allocations by internal ROCm libraries
  // (e.g. rocBLAS alone needs~200 MB to put its kernels as of ROCm 4.1)
  const uint64_t RESERVED_GFX908 = 1048576 * 512;
  const uint64_t RESERVED_GFX9_X = 1048576 * 1024;
  if (gcnArchName.substr(0, 6) == "gfx908") {
    *reserve = RESERVED_GFX908;
  } else if (gcnArchName.substr(0, 6) == "gfx90a" ||
             gcnArchName.substr(0, 6) == "gfx940") {
    *reserve = RESERVED_GFX9_X;
  }
  return true;
}

/* static */ bool GpuDriver::GetDeviceMemoryInfo(GpuContext* context,
                                                 int64_t* free_out,
                                                 int64_t* total_out) {
  ScopedActivateContext activation{context};
  size_t free = 0;
  size_t total = 0;
  hipError_t res = wrap::hipMemGetInfo(&free, &total);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device memory info: " << ToString(res);
    return false;
  }

  uint64_t reserve = 0;
  if (!GetReservedMemory(&reserve)) {
    LOG(ERROR) << "failed to reserved device memory for ROCm libraries";
    return false;
  }

  VLOG(1) << "Device memory: " << total / 1048576 << " MB total, "
          << free / 1048576 << " MB free, reserving " << reserve / 1048576
          << " MB";

  // overflow check
  if (free > std::numeric_limits<int64_t>::max()) {
    LOG(ERROR) << "free memory (" << free << ") is overflow int64_t";
    return false;
  }

  *free_out = free >= reserve ? free - reserve : 0;
  *total_out = total - reserve;
  return true;
}

/* static */ bool GpuDriver::GetDeviceTotalMemory(hipDevice_t device,
                                                  uint64_t* result) {
  size_t value = -1;
  hipError_t res = wrap::hipDeviceTotalMem(&value, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query total available memory: " << ToString(res);
    return false;
  }
  uint64_t reserve = 0;
  if (!GetReservedMemory(&reserve)) {
    LOG(ERROR) << "failed to reserved device memory for ROCm libraries";
    return false;
  }
  *result = value - reserve;
  return true;
}

/* static */ string GpuDriver::GetPCIBusID(hipDevice_t device) {
  string pci_bus_id;
  static const int kBufferSize = 64;
  absl::InlinedVector<char, 4> chars(kBufferSize);
  chars[kBufferSize - 1] = '\0';
  hipError_t res =
      wrap::hipDeviceGetPCIBusId(chars.begin(), kBufferSize - 1, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << ToString(res);
    return pci_bus_id;
  }
  pci_bus_id = chars.begin();
  return pci_bus_id;
}

/* static */ bool GpuDriver::CanEnablePeerAccess(GpuContext* from,
                                                 GpuContext* to) {
  if (from->device_ordinal() == to->device_ordinal()) {
    return true;  // A device can always access its own memory.
  }

  int can_access_peer = -1;
  hipError_t res = wrap::hipDeviceCanAccessPeer(
      &can_access_peer, from->device_ordinal(), to->device_ordinal());
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to detect peer access capability: " << ToString(res);
    return false;
  }

  return can_access_peer;
}

/* static */ tsl::Status GpuDriver::EnablePeerAccess(GpuContext* from,
                                                     GpuContext* to) {
  if (from->device_ordinal() == to->device_ordinal()) {
    return tsl::OkStatus();  // A device can always access its own memory.
  }

  ScopedActivateContext activated{from};
  hipError_t result =
      wrap::hipDeviceEnablePeerAccess(to->device_ordinal(), 0 /* = flags */);
  if (result != hipSuccess && result != hipErrorPeerAccessAlreadyEnabled) {
<<<<<<< HEAD
    return tsl::Status(
=======
    return tsl::Status{
>>>>>>> upstream/master
        absl::StatusCode::kInternal,
        absl::StrFormat("failed to enable peer access from %d to %d: %s",
                        from->device_ordinal(), to->device_ordinal(),
                        ToString(result).c_str()));
  }

  return tsl::OkStatus();
}

/* static */ tsl::StatusOr<int> GpuDriver::GetMaxOccupiedBlocksPerCore(
    GpuContext* context, hipFunction_t kernel, int threads_per_block,
    size_t dynamic_shared_memory_bytes) {
  ScopedActivateContext activation{context};

  int max_blocks = 0;
  hipError_t result = hipSuccess;
  // TODO(ROCm) implement this feature in HIP
  if (result != hipSuccess) {
<<<<<<< HEAD
    return tsl::Status(
=======
    return tsl::Status{
>>>>>>> upstream/master
        absl::StatusCode::kInternal,
        absl::StrFormat("failed to calculate occupancy of kernel %p: %s",
                        kernel, ToString(result).c_str()));
  }

  return max_blocks;
}

}  // namespace gpu
}  // namespace stream_executor
