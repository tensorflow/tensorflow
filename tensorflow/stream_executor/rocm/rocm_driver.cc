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

#include "tensorflow/stream_executor/rocm/rocm_driver.h"

#include <stdint.h>
#include <stdlib.h>
#include <map>
#include <set>
#include <utility>

#include "tensorflow/stream_executor/rocm/rocm_diagnostics.h"
#include "tensorflow/stream_executor/lib/casts.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/human_readable.h"
#include "tensorflow/stream_executor/lib/notification.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/lib/stacktrace.h"
#include "tensorflow/stream_executor/lib/static_threadlocal.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/lib/inlined_vector.h"

bool FLAGS_gpuexec_rocm_driver_inject_init_error = false;
bool FLAGS_gpuexec_rocm_sync_around_driver_calls = false;
bool FLAGS_gpuexec_rocm_device_0_only = false;

// Debugging: on each push and pop of a rocm context, verify the current context
// matches the expected one.
constexpr bool kVerifyROCmContext = false;

namespace stream_executor {
namespace rocm {

namespace {

// Manages the singleton map of contexts that we've created, mapping
// from the hipCtx_t to the ROCmContext* that we pass around internally.
// This also manages assignment of unique ids to ROCmContexts, to allow
// for fast comparison of a context against the current context.
//
// ROCM-runtime-created contexts are avoided, if triple angle
// brace launches are required, by using the scoped activations in
// rocm_activation.h.
class CreatedContexts {
 public:
  // Returns whether context is a member of the live set.
  static bool Has(hipCtx_t context) {
    tf_shared_lock lock{mu_};
    return Live()->find(context) != Live()->end();
  }

  // Adds context to the live set.
  static ROCmContext* Add(hipCtx_t context) {
    CHECK(context != nullptr);
    mutex_lock lock{mu_};
    auto rocm_context = new ROCmContext(context, next_id_++);
    Live()->insert(
        std::make_pair(context, std::unique_ptr<ROCmContext>(rocm_context)));
    return rocm_context;
  }

  // Removes context from the live set.
  static void Remove(hipCtx_t context) {
    CHECK(context != nullptr);
    mutex_lock lock{mu_};
    auto it = Live()->find(context);
    CHECK(it != Live()->end()) << context;
    Live()->erase(it);
  }

 private:
  // Returns the live map singleton.
  static std::map<hipCtx_t, std::unique_ptr<ROCmContext>> *Live() {
    static auto singleton =
        new std::map<hipCtx_t, std::unique_ptr<ROCmContext>>;
    return singleton;
  }

  // Lock that guards access-to/mutation-of the live set.
  static mutex mu_;
  static int64 next_id_;
};

/* static */ mutex CreatedContexts::mu_{LINKER_INITIALIZED};
/* static */ int64 CreatedContexts::next_id_ = 1;  // 0 means "no context"

// Formats hipError_t to output prettified values into a log stream.
// Error summaries taken from:
//
// TODO(leary) switch to cuGetErrorName when updated rocm.h is available.
string ToString(hipError_t result) {
#define OSTREAM_ROCM_ERROR(__name) \
  case hipError##__name:        \
    return "HIP_ERROR_" #__name;

///////////////
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
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
      return port::StrCat("hipError_t(", static_cast<int>(result), ")");
  }
#pragma GCC diagnostic pop
}

// Returns the current context and checks that it is in the set of ROCM contexts
// created by StreamExecutor (to ensure that the ROCM runtime didn't create a
// context behind our backs).
hipCtx_t CurrentContext() {
  hipCtx_t current = ROCMDriver::CurrentContextOrDie();
  // XXX TODO FIGURE THIS OUT
#if 0
  if (current != nullptr && !CreatedContexts::Has(current)) {
    LOG(FATAL) << "current context was not created by the StreamExecutor "
                  "rocm_driver API: "
               << current
               << "; a ROCM runtime call "
                  "was likely performed without using a StreamExecutor context";
  }
#endif
  return current;
}

// ROCM driver routines may require a large amount of stack (particularly
// hipModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
static mutex driver_executor_threadpool_mu(LINKER_INITIALIZED);
static port::ThreadPool *InitializeDriverExecutor() {
  return new port::ThreadPool(port::Env::Default(), port::ThreadOptions(),
                              "rocm_driver", 1);
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

// Call hipCtxtSynchronize and crash if it doesn't succeed.
void SynchronizeOrDie() {
  auto res = hipCtxSynchronize();
  if (res != hipSuccess) {
    LOG(FATAL) << "Synchronize found "
               << ToString(res) << " :: " << port::CurrentStackTrace();
  }
}

struct ThreadLocalData {
  int64 id;
  ROCmContext* context;  // Only valid if id == a known good context.
  int depth;
};

SE_STATIC_THREAD_LOCAL_POD(ThreadLocalData, tls_data);

}  // namespace

ScopedActivateContext::ScopedActivateContext(ROCmContext* rocm_context) {
  if (FLAGS_gpuexec_rocm_sync_around_driver_calls) SynchronizeOrDie();

  auto* tls = &tls_data.get();
  tls->depth++;
  if (tls->id == rocm_context->id()) {
    if (kVerifyROCmContext) {
      CHECK_EQ(CurrentContext(), rocm_context->context());
    }
    DCHECK_EQ(CurrentContext(), rocm_context->context());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching context from " << tls->id
          << " to " << rocm_context->id();

  to_restore_ = (tls->depth == 1 ? nullptr : tls->context);

  // Set the context and update thread local.
  CHECK_EQ(hipSuccess, hipCtxSetCurrent(rocm_context->context()));
  tls->id = rocm_context->id();
  tls->context = rocm_context;
}

ScopedActivateContext::~ScopedActivateContext() {
  if (FLAGS_gpuexec_rocm_sync_around_driver_calls) SynchronizeOrDie();

  auto* tls = &tls_data.get();

  if (kVerifyROCmContext) {
    // Note that if kVerifyROCmContext is used, and contexts are deleted, it's
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
  CHECK_EQ(hipSuccess, hipCtxSetCurrent(to_restore_->context()));
  tls->id = to_restore_->id();
  tls->context = to_restore_;
}

namespace {

// Returns a stringified device number associated with pointer, primarily for
// logging purposes. Returns "?" if the device could not be successfully
// queried.
string ROCMPointerToDeviceString(hipDeviceptr_t pointer) {
  auto value = ROCMDriver::GetPointerDevice(pointer);
  if (value.ok()) {
    return port::StrCat(value.ValueOrDie());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified memory space associated with pointer, primarily for
// logging purposes. Returns "?" if the memory space could not be successfully
// queried.
string ROCMPointerToMemorySpaceString(hipDeviceptr_t pointer) {
  auto value = ROCMDriver::GetPointerMemorySpace(pointer);
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
string ROCMPointersToCanAccessString(hipDeviceptr_t from, hipDeviceptr_t to) {
  auto from_context = ROCMDriver::GetPointerContext(from);
  if (!from_context.ok()) {
    LOG(ERROR) << "could not retrieve source pointer's context: "
               << from_context.status();
    return "error";
  }
  auto to_context = ROCMDriver::GetPointerContext(to);
  if (!to_context.ok()) {
    LOG(ERROR) << "could not retrieve destination pointer's context: "
               << to_context.status();
    return "error";
  }
  return ROCMDriver::CanEnablePeerAccess(from_context.ValueOrDie(),
                                         to_context.ValueOrDie())
             ? "true"
             : "false";
}


// Actually performs the work of ROCM initialization. Wrapped up in one-time
// execution guard.
static port::Status InternalInit() {
  hipError_t res = hipErrorNoDevice;
  if (FLAGS_gpuexec_rocm_driver_inject_init_error) {
    LOG(ERROR) << "injecting ROCM init error; initialization will fail";
  } else {
    res = hipInit(0 /* = flags */);
  }

  if (res == hipSuccess) {
    return port::Status::OK();
  }

  LOG(ERROR) << "failed call to hipInit: " << ToString(res);
  Diagnostician::LogDiagnosticInformation();
  return port::Status{port::error::ABORTED,
                      port::StrCat("failed call to hipInit: ", ToString(res))};
}

}  // namespace

/* static */ port::Status ROCMDriver::Init() {
  // Cached return value from calling InternalInit(), as hipInit need only be
  // called once, but ROCMDriver::Init may be called many times.
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

/* static */ port::Status ROCMDriver::GetDevice(int device_ordinal,
                                                hipDevice_t *device) {
  hipError_t res = hipDeviceGet(device, device_ordinal);
  if (res == hipSuccess) {
    return port::Status::OK();
  }

  return port::Status{
      port::error::INTERNAL,
      port::StrCat("failed call to hipDeviceGet: ", ToString(res))};
}

/* static */ bool ROCMDriver::GetDeviceName(hipDevice_t device,
                                            string *device_name) {
  static const size_t kCharLimit = 64;
  port::InlinedVector<char, 4> chars(kCharLimit);
  hipError_t res = hipDeviceGetName(chars.begin(), kCharLimit - 1, device);
  if (res != hipSuccess) {
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
  return true;
}

/* static */ port::Status ROCMDriver::CreateContext(
    hipDevice_t device, DeviceOptions device_options, ROCmContext** context) {
  *context = nullptr;

  int flags = 0;
  if (!DeviceOptionsToContextFlags(device_options, &flags)) {
    LOG(WARNING) << "could not convert all device options into context flags";
  }

  hipError_t res;
  hipCtx_t former_context;
  hipCtx_t new_context;
  {
    former_context = CurrentContext();
    if (former_context != nullptr) {
      LOG(WARNING)
          << "creating context when one is currently active; existing: "
          << former_context;
    }
    res = hipCtxCreate(&new_context, flags, device);
  }
  CHECK_EQ(hipSuccess, hipCtxSetCurrent(former_context));

  if (res == hipSuccess) {
    *context = CreatedContexts::Add(new_context);
    CHECK(*context != nullptr)
        << "success in this call must entail non-null result";
    VLOG(2) << "created context " << context << " for this thread";
    return port::Status::OK();
  }

  string message = "failed call to hipCtxCreate: " + ToString(res);
  if (res == hipErrorMemoryAllocation) {
    uint64 total_memory;
    if (GetDeviceTotalMemory(device, &total_memory)) {
      port::StrAppend(&message, "; total memory reported: ", total_memory);
    } else {
      port::StrAppend(&message, "; could not query total memory");
    }
  }

  return port::Status{port::error::INTERNAL, message};
}

/* static */ void ROCMDriver::DestroyContext(ROCmContext* context) {
  if (context == nullptr) {
    return;
  }
  hipError_t res = hipCtxDestroy(context->context());

  if (res != hipSuccess) {
    LOG(ERROR) << "failed to release ROCM context; leaking: " << ToString(res);
  }

  CreatedContexts::Remove(context->context());
}

/* static */ bool ROCMDriver::FuncGetAttribute(hipDeviceAttribute_t attribute,
                                               hipFunction_t func,
                                               int *attribute_value) {
  // XXX FIXME properly implement this feature in HIP
  hipError_t res = hipSuccess;
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query kernel attribute. kernel: " << func
               << ", attribute: " << attribute;
    return false;
  }
  return true;
}

/* static */ bool ROCMDriver::FuncSetCacheConfig(hipFunction_t function,
                                                 hipFuncCache_t cache_config) {
  hipError_t res = hipFuncSetCacheConfig(function, cache_config);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to set ROCM kernel cache config. kernel: " << function
               << ", config: " << cache_config << ", result: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ port::StatusOr<hipSharedMemConfig>
ROCMDriver::ContextGetSharedMemConfig(ROCmContext* context) {
  hipSharedMemConfig shared_mem_config;
  ScopedActivateContext activation{context};
  hipError_t result = hipCtxGetSharedMemConfig(&shared_mem_config);
  if (result != hipSuccess) {
    hipDevice_t device;
    hipCtxGetDevice(&device);
    LOG(ERROR) << "failed to get ROCM device shared memory config. "
               << "Context device ID: " << device
               << ", result: " << ToString(result);
    return port::Status{
        port::error::INTERNAL,
        port::StrCat("failed to get shared memory config: ", ToString(result))};
  }
  return shared_mem_config;
}

/* static */ port::Status ROCMDriver::ContextSetSharedMemConfig(
    ROCmContext* context, hipSharedMemConfig shared_mem_config) {
  ScopedActivateContext activation{context};
  hipError_t result = hipCtxSetSharedMemConfig(shared_mem_config);
  if (result != hipSuccess) {
    hipDevice_t device;
    hipCtxGetDevice(&device);
    LOG(ERROR) << "failed to set ROCM device shared memory config. "
               << "Context device ID: " << device
               << ", config: " << shared_mem_config
               << ", result: " << ToString(result);
    return port::Status{
        port::error::INTERNAL,
        port::StrCat("failed to set shared memory config: ", ToString(result))};
  }
  return port::Status::OK();
}

/* static */ bool ROCMDriver::LaunchKernel(
    ROCmContext* context, hipFunction_t function, unsigned int grid_dim_x,
    unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, hipStream_t stream, void **kernel_params,
    void **extra) {
  ScopedActivateContext activation{context};
  VLOG(2) << "launching kernel: " << function << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << " smem: " << shared_mem_bytes;
  hipError_t res = hipModuleLaunchKernel(
     function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
     block_dim_z, shared_mem_bytes, stream, kernel_params, extra);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to launch ROCM kernel: " << function
               << "; result: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully launched kernel";
  return true;
}

/* static */ bool ROCMDriver::LoadHsaco(ROCmContext* context,
                                        const char *hsaco_contents,
                                        hipModule_t *module) {
  port::Notification notification;
  bool ret = true;
  GetDriverExecutor()->Schedule([context, hsaco_contents, module, &ret,
                                 &notification]() {
    ScopedActivateContext activation{context};
    void *hsaco_data = const_cast<char *>(hsaco_contents);

    hipError_t res = hipModuleLoadData(module, hsaco_data);

    if (res != hipSuccess) {
      LOG(ERROR) << "failed to load HSACO: " << ToString(res);
      ret = false;
      notification.Notify();
    }

    CHECK(module != nullptr);
    notification.Notify();
  });
  notification.WaitForNotification();

  return ret;
}

/* static */ bool ROCMDriver::SynchronousMemsetUint8(ROCmContext* context,
                                                     hipDeviceptr_t location,
                                                     uint8 value, size_t size) {
  ScopedActivateContext activation{context};
  hipError_t res = hipMemset(location, value, size);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to memset memory: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool ROCMDriver::SynchronousMemsetUint32(ROCmContext* context,
                                                      hipDeviceptr_t location,
                                                      uint32 value,
                                                      size_t uint32_count) {
  ScopedActivateContext activation{context};
  void * pointer = port::bit_cast<void *>(location);
  unsigned char valueC = static_cast<unsigned char>(value);
  uint32_t value32 = (valueC << 24) | (valueC << 16) | (valueC << 8) | (valueC) ;
  assert (value32 == value); // if mismatch this indicates case where hipMemsetAsyc can't emulate hipMemSetD32
  hipError_t res =
      hipMemset(pointer, static_cast<int>(value), uint32_count*4);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to memset memory: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool ROCMDriver::AsynchronousMemsetUint8(ROCmContext* context,
                                                      hipDeviceptr_t location,
                                                      uint8 value,
                                                      size_t uint32_count,
                                                      hipStream_t stream) {
  ScopedActivateContext activation{context};
  hipError_t res = hipMemsetAsync(location, value, uint32_count, stream);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to enqueue async memset operation: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully enqueued async memset operation";
  return true;
}

/* static */ bool ROCMDriver::AsynchronousMemsetUint32(ROCmContext* context,
                                                       hipDeviceptr_t location,
                                                       uint32 value,
                                                       size_t uint32_count,
                                                       hipStream_t stream) {
  ScopedActivateContext activation{context};
  void * pointer = port::bit_cast<void *>(location);

  /// XXX - need to set a 32-bit value here, need hipMemsetD32
  unsigned char valueC = static_cast<unsigned char>(value);
  uint32_t value32 = (valueC << 24) | (valueC << 16) | (valueC << 8) | (valueC) ;
  assert (value32 == value); // if mismatch this indicates case where hipMemsetAsyc can't emulate hipMemSetD32
  hipError_t res =
      hipMemsetAsync(pointer, value, uint32_count*4, stream);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to enqueue async memset operation: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully enqueued async memset operation";
  return true;
}

/* static */ bool ROCMDriver::AddStreamCallback(ROCmContext* context,
                                                hipStream_t stream,
                                                StreamCallback callback,
                                                void *data) {
  hipError_t res = hipStreamAddCallback(stream, (hipStreamCallback_t) callback, data, 0 /* = flags */);
  if (res != hipSuccess) {
    LOG(ERROR) << "unable to add host callback: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool ROCMDriver::GetModuleFunction(ROCmContext *context,
                                                hipModule_t module,
                                                const char *kernel_name,
                                                hipFunction_t *function) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  hipError_t res = hipModuleGetFunction(function, module, kernel_name);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to get kernel \"" << kernel_name
               << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool ROCMDriver::GetModuleSymbol(ROCmContext* context,
                                              hipModule_t module,
                                              const char *symbol_name,
                                              hipDeviceptr_t *dptr,
                                              size_t *bytes) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  hipError_t res = hipModuleGetGlobal(dptr, bytes, module, symbol_name);
  if (res != hipSuccess) {
    // symbol may not be found in the current module, but it may reside in
    // another module.
    VLOG(2) << "failed to get symbol \"" << symbol_name
            << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ void ROCMDriver::UnloadModule(ROCmContext *context,
                                           hipModule_t module) {
  ScopedActivateContext activated{context};
  hipError_t res = hipModuleUnload(module);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << ToString(res);
  }
}

/* static */ port::StatusOr<hipDevice_t> ROCMDriver::DeviceFromContext(
    ROCmContext* context) {
  ScopedActivateContext activated{context};
  hipDevice_t device;
  hipError_t result = hipCtxGetDevice(&device);
  if (result == hipSuccess) {
    return device;
  }

  return port::Status{
      port::error::INTERNAL,
      port::StrCat("failed to get device for context: ", ToString(result))};
}

/* static */ bool ROCMDriver::CreateStream(ROCmContext *context,
                                           hipStream_t *out) {
  ScopedActivateContext activated{context};
  hipError_t res = hipStreamCreateWithFlags(out, 0);
  if (res != hipSuccess) {
    LOG(ERROR) << "could not allocate ROCM stream for context " << context
               << ": " << ToString(res);
    return false;
  }

  VLOG(2) << "successfully created stream " << *out << " for context "
          << context << " on thread";
  return true;
}

/* static */ void ROCMDriver::DestroyStream(ROCmContext* context,
                                            hipStream_t *stream) {
  if (*stream == nullptr) {
    return;
  }

  ScopedActivateContext activated{context};
  hipError_t res = hipStreamDestroy(*stream);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to destroy ROCM stream for context " << context
               << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << *stream << " for context "
            << context;
    *stream = nullptr;
  }
}

/* static */ void *ROCMDriver::DeviceAllocate(ROCmContext *context,
                                              uint64 bytes) {
  ScopedActivateContext activated{context};
  hipDeviceptr_t result = 0;
  hipError_t res = hipMalloc(&result, bytes);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to allocate "
               << port::HumanReadableNumBytes::ToString(bytes) << " (" << bytes
               << " bytes) from device: " << ToString(res);
    return nullptr;
  }
  void *ptr = reinterpret_cast<void *>(result);
  VLOG(2) << "allocated " << ptr << " for context " << context << " of "
          << bytes << " bytes";
  return ptr;
}

/* static */ void ROCMDriver::DeviceDeallocate(ROCmContext* context,
                                               void *location) {
  ScopedActivateContext activation{context};
  hipDeviceptr_t pointer = port::bit_cast<hipDeviceptr_t>(location);
  hipError_t res = hipFree(pointer);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated " << location << " for context " << context;
  }
}

/* static */ void *ROCMDriver::HostAllocate(ROCmContext *context,
                                            uint64 bytes) {
  ScopedActivateContext activation{context};
  void *host_mem = nullptr;
  // "Portable" memory is visible to all ROCM contexts. Safe for our use model.
  hipError_t res = hipHostMalloc(&host_mem, bytes, hipHostMallocPortable);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes on host: " << ToString(res);
  }
  return host_mem;
}

/* static */ void ROCMDriver::HostDeallocate(ROCmContext* context,
                                             void *location) {
  ScopedActivateContext activation{context};
  hipError_t res = hipHostFree(location);
  if (res != hipSuccess) {
    LOG(ERROR) << "error deallocating host memory at " << location << ": "
               << ToString(res);
  }
}

/* static */ bool ROCMDriver::HostRegister(ROCmContext* context, void *location,
                                           uint64 bytes) {
  ScopedActivateContext activation{context};
  // "Portable" memory is visible to all ROCM contexts. Safe for our use model.
  hipError_t res =
      hipHostRegister(location, bytes, hipHostRegisterPortable);
  if (res != hipSuccess) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool ROCMDriver::HostUnregister(ROCmContext* context,
                                             void *location) {
  ScopedActivateContext activation{context};
  hipError_t res = hipHostUnregister(location);
  if (res != hipSuccess) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ port::Status ROCMDriver::DestroyEvent(ROCmContext* context,
                                                   hipEvent_t *event) {
  if (*event == nullptr) {
    return port::Status{port::error::INVALID_ARGUMENT,
                        "input event cannot be null"};
  }

  ScopedActivateContext activated{context};
  hipError_t res = hipEventDestroy(*event);
  *event = nullptr;

  switch (res) {
    case hipSuccess:
      return port::Status::OK();
    case hipErrorDeinitialized:
    case hipErrorNotInitialized:
      return port::Status{
          port::error::FAILED_PRECONDITION,
          port::Printf("error destroying ROCM event in context %p: %s", context,
                       ToString(res).c_str())};
    default:
      return port::Status{
          port::error::INTERNAL,
          port::Printf("error destroying ROCM event in context %p: %s", context,
                       ToString(res).c_str())};
  }
}

/* static */ port::Status ROCMDriver::RecordEvent(ROCmContext* context,
                                                  hipEvent_t event,
                                                  hipStream_t stream) {
  ScopedActivateContext activated{context};
  hipError_t res = hipEventRecord(event, stream);
  switch (res) {
    case hipSuccess:
      return port::Status::OK();
    case hipErrorDeinitialized:
    case hipErrorNotInitialized:
      return port::Status{
          port::error::FAILED_PRECONDITION,
          port::Printf("error recording ROCM event on stream %p: %s", stream,
                       ToString(res).c_str())};
    default:
      return port::Status{
          port::error::INVALID_ARGUMENT,
          port::Printf("error recording ROCM event on stream %p: %s", stream,
                       ToString(res).c_str())};
  }
}

/* static */ port::StatusOr<hipError_t> ROCMDriver::QueryEvent(
    ROCmContext *context, hipEvent_t event) {
  ScopedActivateContext activated{context};
  hipError_t res = hipEventQuery(event);
  if (res != hipSuccess && res != hipErrorNotReady) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf("failed to query event: %s", ToString(res).c_str())};
  }

  return res;
}

/* static */ bool ROCMDriver::GetEventElapsedTime(ROCmContext* context,
                                                  float *elapsed_milliseconds,
                                                  hipEvent_t start, hipEvent_t stop) {
  ScopedActivateContext activated{context};
  // The stop event must have completed in order for hipEventElapsedTime to
  // work.
  hipError_t res = hipEventSynchronize(stop);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to synchronize the stop event: " << ToString(res);
    return false;
  }
  res = hipEventElapsedTime(elapsed_milliseconds, start, stop);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to get elapsed time between events: "
               << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool ROCMDriver::WaitStreamOnEvent(ROCmContext* context,
                                                hipStream_t stream,
                                                hipEvent_t event) {
  ScopedActivateContext activation{context};
  hipError_t res = hipStreamWaitEvent(stream, event, 0 /* = flags */);
  if (res != hipSuccess) {
    LOG(ERROR) << "could not wait stream on event: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool ROCMDriver::SynchronizeContext(ROCmContext* context) {
  ScopedActivateContext activation{context};
  hipError_t res = hipCtxSynchronize();
  if (res != hipSuccess) {
    LOG(ERROR) << "could not synchronize on ROCM context: " << ToString(res)
               << " :: " << port::CurrentStackTrace();
    return false;
  }

  return true;
}

/* static */ port::Status ROCMDriver::SynchronizeStream(ROCmContext* context,
                                                hipStream_t stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  hipError_t res = hipStreamSynchronize(stream);
  if (res != hipSuccess) {
    port::Status status = port::InternalError(
        port::StrCat("could not synchronize on ROCM stream: ", ToString(res)));
    LOG(ERROR) << status << " :: " << port::CurrentStackTrace();
    return status;
  }
  VLOG(2) << "successfully synchronized stream " << stream << " on context "
          << context;
  return port::Status::OK();
}

/* static */ bool ROCMDriver::IsStreamIdle(ROCmContext *context,
                                           hipStream_t stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  hipError_t res = hipStreamQuery(stream);
  if (res == hipSuccess) {
    return true;
  }

  if (res != hipErrorNotReady) {
    LOG(ERROR) << "stream in bad state on status query: " << ToString(res);
  }
  return false;
}

/* static */ port::Status ROCMDriver::SynchronousMemcpyD2H(ROCmContext *context,
                                                           void *host_dst,
                                                           hipDeviceptr_t gpu_src,
                                                           uint64 size) {
  ScopedActivateContext activation{context};
  hipError_t res = hipMemcpyDtoH(host_dst, gpu_src, size);
  if (res != hipSuccess) {
    return port::InternalError(
        port::Printf("failed to synchronous memcpy from device to host: %s; "
                     "host dst: %p; GPU src: %p; size: %llu=0x%llx",
                     ToString(res).c_str(), host_dst,
                     port::bit_cast<void *>(gpu_src), size, size));
  }
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return port::Status::OK();
}

/* static */ port::Status ROCMDriver::SynchronousMemcpyH2D(ROCmContext *context,
                                                           hipDeviceptr_t gpu_dst,
                                                           const void *host_src,
                                                           uint64 size) {
  ScopedActivateContext activation{context};
  hipError_t res = hipMemcpyHtoD(gpu_dst, const_cast<void*>(host_src), size);
  if (res != hipSuccess) {
    return port::InternalError(port::Printf(
        "failed to synchronous memcpy from host to device: %s; GPU dst: %p;"
        " host src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), port::bit_cast<void *>(gpu_dst), host_src, size,
        size));
  }
  VLOG(2) << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return port::Status::OK();
}

/* static */ port::Status ROCMDriver::SynchronousMemcpyD2D(ROCmContext *context,
                                                           hipDeviceptr_t gpu_dst,
                                                           hipDeviceptr_t gpu_src,
                                                           uint64 size) {
  ScopedActivateContext activation{context};
  hipError_t res = hipMemcpyDtoD(gpu_dst, gpu_src, size);
  if (res != hipSuccess) {
    return port::InternalError(port::Printf(
        "failed to synchronous memcpy from host to device: %s; GPU dst: %p; "
        "GPU src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), port::bit_cast<void *>(gpu_dst),
        port::bit_cast<void *>(gpu_src), size, size));
  }
  VLOG(2) << "successfully sync memcpy'd d2d of " << size << " bytes";
  return port::Status::OK();
}

/* static */ bool ROCMDriver::AsynchronousMemcpyD2H(ROCmContext* context,
                                                    void *host_dst,
                                                    hipDeviceptr_t gpu_src,
                                                    uint64 size,
                                                    hipStream_t stream) {
  ScopedActivateContext activation{context};
  hipError_t res = hipMemcpyDtoHAsync(host_dst, gpu_src, size, stream);
  if (res != hipSuccess) {
    LOG(ERROR) << port::Printf(
        "failed to enqueue async memcpy from device to host: %s; host dst: %p; "
        "GPU src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), host_dst, port::bit_cast<void *>(gpu_src), size, size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << port::bit_cast<void *>(gpu_src) << " to " << host_dst
          << " on stream " << stream;
  return true;
}

/* static */ bool ROCMDriver::AsynchronousMemcpyH2D(ROCmContext* context,
                                                    hipDeviceptr_t gpu_dst,
                                                    const void *host_src,
                                                    uint64 size,
                                                    hipStream_t stream) {
  ScopedActivateContext activation{context};
  hipError_t res = hipMemcpyHtoDAsync(gpu_dst, const_cast<void*>(host_src), size, stream);
  if (res != hipSuccess) {
    LOG(ERROR) << port::Printf(
        "failed to enqueue async memcpy from host to device: %s; GPU dst: %p; "
        "host src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), port::bit_cast<void *>(gpu_dst), host_src, size, size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " on stream " << stream;
  return true;
}

/* static */ bool ROCMDriver::AsynchronousMemcpyD2D(ROCmContext* context,
                                                    hipDeviceptr_t gpu_dst,
                                                    hipDeviceptr_t gpu_src,
                                                    uint64 size,
                                                    hipStream_t stream) {
  ScopedActivateContext activation{context};
  hipError_t result = hipMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream);
  if (result != hipSuccess) {
    LOG(ERROR) << port::Printf(
        "failed to enqueue async memcpy from device to device: %s"
        "; GPU dst: %p on %s %s"
        "; GPU src: %p on %s %s"
        "; can access? %s; size: %llu=0x%llx",
        ToString(result).c_str(), port::bit_cast<void *>(gpu_dst),
        ROCMPointerToMemorySpaceString(gpu_dst).c_str(),
        ROCMPointerToDeviceString(gpu_dst).c_str(), port::bit_cast<void *>(gpu_src),
        ROCMPointerToMemorySpaceString(gpu_src).c_str(),
        ROCMPointerToDeviceString(gpu_src).c_str(),
        ROCMPointersToCanAccessString(gpu_src, gpu_dst).c_str(), size, size);

    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes";
  return true;
}

/* static */ port::Status ROCMDriver::CreateEvent(ROCmContext* context,
                                                  hipEvent_t *result,
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
  hipError_t res = hipEventCreateWithFlags(result, hipflags);

  if (res == hipSuccess) {
    return port::Status::OK();
  } else if (res == hipErrorMemoryAllocation) {
    return port::Status{port::error::RESOURCE_EXHAUSTED,
                        "could not create ROCM event: out of device memory"};
  } else {
    return port::Status{
        port::error::FAILED_PRECONDITION,
        port::StrCat("could not create ROCM event: ", ToString(res))};
  }
}

/* static */ int ROCMDriver::GetDeviceCount() {
  int device_count = 0;
  hipError_t res = hipGetDeviceCount(&device_count);
  if (res != hipSuccess) {
    LOG(ERROR) << "could not retrieve ROCM device count: " << ToString(res);
    return 0;
  }

  if (FLAGS_gpuexec_rocm_device_0_only && device_count > 1) {
    device_count = 1;
  }
  return device_count;
}

/* static */ port::StatusOr<ROCmContext*> ROCMDriver::GetPointerContext(
    hipDeviceptr_t pointer) {
  ROCmContext* context = nullptr;
  hipError_t result = hipSuccess;
  // XXX FIXME
  if (result == hipSuccess) {
    CHECK(context != nullptr) << "success should entail non-null context";
    return context;
  }

  return port::Status{
      port::error::INTERNAL,
      port::StrCat("failed to query device pointer for context: ",
                   ToString(result))};
}

/* static */ port::StatusOr<MemorySpace> ROCMDriver::GetPointerMemorySpace(
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
        return port::Status{
            port::error::INTERNAL,
            port::StrCat("unknown memory space provided by ROCM API: ", value)};
    }
  }

  return port::Status{
      port::error::INTERNAL,
      port::StrCat("failed to query device pointer for memory space: ",
                   ToString(result))};
}

/* static */ port::Status ROCMDriver::GetPointerAddressRange(hipDeviceptr_t dptr,
                                                             hipDeviceptr_t *base,
                                                             size_t *size) {
  hipError_t result = hipMemGetAddressRange(base, size, dptr);
  if (result == hipSuccess) {
    return port::Status::OK();
  } else if (result == hipErrorNotFound) {
    // We differentiate between "this pointer is unknown" (return here) and
    // "there was an internal error while performing this operation" (return
    // below).
    return port::Status{
        port::error::NOT_FOUND,
        port::Printf("not a device pointer %p; %s",
                     reinterpret_cast<void *>(dptr), ToString(result).c_str())};
  }

  return port::Status{
      port::error::INTERNAL,
      port::Printf("failed to get pointer into for device pointer %p; %s",
                   reinterpret_cast<void *>(dptr), ToString(result).c_str())};
}

/* static */ port::StatusOr<hipDevice_t> ROCMDriver::GetPointerDevice(
    hipDeviceptr_t pointer) {
  auto result = GetPointerContext(pointer);
  if (!result.ok()) {
    return result.status();
  }

  return DeviceFromContext(result.ValueOrDie());
}

/* static */ port::Status ROCMDriver::GetAMDGPUISAVersion(int *version,
                                                          hipDevice_t device) {
  hipDeviceProp_t props;
  hipError_t result = hipGetDeviceProperties(&props, device);
  if (result == hipSuccess) {
    *version = props.gcnArch;
    return port::Status::OK();
  }
  *version = 0;
  return port::Status{
      port::error::INTERNAL,
      port::Printf("failed to determine AMDGPU ISA version for device %d", device)};
}

// Helper function that turns the integer output of hipDeviceGetAttribute to type
// T and wraps it in a StatusOr.
template <typename T>
static port::StatusOr<T> GetSimpleAttribute(hipDevice_t device,
                                            hipDeviceAttribute_t attribute) {
  int value = -1;
  hipError_t result = hipDeviceGetAttribute(&value, attribute, device);
  if (result != hipSuccess) {
    return port::Status{
        port::error::NOT_FOUND,
        port::StrCat("could not retrieve ROCM device attribute (", attribute,
                     "): ", ToString(result))};
  }
  T converted = value;
  return converted;
}

/* static */ port::StatusOr<int> ROCMDriver::GetMultiprocessorCount(
    hipDevice_t device) {
  return GetSimpleAttribute<int>(device,
                                 hipDeviceAttributeMultiprocessorCount);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetMaxSharedMemoryPerCore(
    hipDevice_t device) {
  return GetSimpleAttribute<int64>(
      device, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetMaxSharedMemoryPerBlock(
    hipDevice_t device) {
  return GetSimpleAttribute<int64>(
      device, hipDeviceAttributeMaxSharedMemoryPerBlock);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetMaxThreadsPerMultiprocessor(
    hipDevice_t device) {
  return GetSimpleAttribute<int64>(
      device, hipDeviceAttributeMaxThreadsPerMultiProcessor);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetMaxThreadsPerBlock(
    hipDevice_t device) {
  return GetSimpleAttribute<int64>(device,
                                   hipDeviceAttributeMaxThreadsPerBlock);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetMaxRegistersPerBlock(
    hipDevice_t device) {
  return GetSimpleAttribute<int64>(device,
                                   hipDeviceAttributeMaxRegistersPerBlock);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetThreadsPerWarp(
    hipDevice_t device) {
  return GetSimpleAttribute<int64>(device, hipDeviceAttributeWarpSize);
}

/* static */ bool ROCMDriver::GetGridLimits(int *x, int *y, int *z,
                                            hipDevice_t device) {
  int value;
  hipError_t res =
      hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimX, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query max grid dim x: " << ToString(res);
    return false;
  }
  *x = value;

  res =
      hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimY, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query max grid dim y: " << ToString(res);
    return false;
  }
  *y = value;

  res =
      hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimZ, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query max grid dim z: " << ToString(res);
    return false;
  }
  *z = value;
  return true;
}

/* static */ bool ROCMDriver::GetDriverVersion(int *driver_version) {
  hipError_t res = hipDriverGetVersion(driver_version);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query driver version: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool ROCMDriver::GetDeviceProperties(hipDeviceProp_t *device_properties,
                                                  int device_ordinal) {
  hipError_t res = hipGetDeviceProperties(device_properties, device_ordinal);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool ROCMDriver::IsEccEnabled(hipDevice_t device, bool *result) {
  int value = -1;
  hipError_t res = hipSuccess;
  // XXX FIXME
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query ECC status: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ bool ROCMDriver::GetDeviceMemoryInfo(ROCmContext* context,
                                                  int64 *free_out,
                                                  int64 *total_out) {
  ScopedActivateContext activation{context};
  size_t free = 0;
  size_t total = 0;
  hipError_t res = hipMemGetInfo(&free, &total);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device memory info: " << ToString(res);
    return false;
  }

  *free_out = free;
  *total_out = total;
  return true;
}

/* static */ bool ROCMDriver::GetDeviceTotalMemory(hipDevice_t device,
                                                   uint64 *result) {
  size_t value = -1;
  hipError_t res = hipDeviceTotalMem(&value, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query total available memory: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ string ROCMDriver::GetPCIBusID(hipDevice_t device) {
  string pci_bus_id;
  static const int kBufferSize = 64;
  port::InlinedVector<char, 4> chars(kBufferSize);
  chars[kBufferSize - 1] = '\0';
  hipError_t res = hipDeviceGetPCIBusId(chars.begin(), kBufferSize - 1, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << ToString(res);
    return pci_bus_id;
  }
  pci_bus_id = chars.begin();
  return pci_bus_id;
}

/* static */ bool ROCMDriver::CanEnablePeerAccess(ROCmContext* from,
                                                  ROCmContext* to) {
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
  hipError_t res = hipDeviceCanAccessPeer(
      &can_access_peer, from_device.ValueOrDie(), to_device.ValueOrDie());
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to detect peer access capability: " << ToString(res);
    return false;
  }

  return can_access_peer;
}

/* static */ port::Status ROCMDriver::EnablePeerAccess(ROCmContext* from,
                                                       ROCmContext* to) {
  if (from == to) {
    return port::Status::OK();  // A context can always access its own memory.
  }

  ScopedActivateContext activated{from};
  hipError_t result = hipCtxEnablePeerAccess(to->context(), 0 /* = flags */);
  if (result != hipSuccess &&
      result != hipErrorPeerAccessAlreadyEnabled) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf("failed to enable peer access from %p to %p: %s", from, to,
                     ToString(result).c_str())};
  }

  return port::Status::OK();
}

/* static */ port::StatusOr<int> ROCMDriver::GetMaxOccupiedBlocksPerCore(
    ROCmContext* context, hipFunction_t kernel, int threads_per_block,
    size_t dynamic_shared_memory_bytes) {
  ScopedActivateContext activation{context};

  int max_blocks = 0;
  hipError_t result = hipSuccess;
  // XXX FIXME
  if (result != hipSuccess) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf("failed to calculate occupancy of kernel %p: %s", kernel,
                     ToString(result).c_str())};
  }

  return max_blocks;
}

/* static */ hipCtx_t ROCMDriver::CurrentContextOrDie() {
  hipCtx_t current = nullptr;
  hipError_t result = hipCtxGetCurrent(&current);
  if (result != hipSuccess) {
    LOG(FATAL) << "failed to query current context: " << ToString(result);
  }
  return current;
}

}  // namespace rocm
}  // namespace stream_executor
