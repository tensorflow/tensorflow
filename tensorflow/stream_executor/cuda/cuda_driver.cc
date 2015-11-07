#include "tensorflow/stream_executor/cuda/cuda_driver.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <set>
#include "tensorflow/stream_executor/platform/port.h"

#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/dso_loader.h"
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

bool FLAGS_gpuexec_cuda_driver_inject_init_error = false;
bool FLAGS_gpuexec_cuda_sync_around_driver_calls = false;
bool FLAGS_gpuexec_cuda_device_0_only = false;

namespace perftools {
namespace gputools {
namespace cuda {

namespace dynload {

#define PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(__name)                              \
  struct DynLoadShim__##__name {                                             \
    static const char *kName;                                                \
    using FuncPointerT = std::add_pointer<decltype(::__name)>::type;         \
    static void *GetDsoHandle() {                                            \
      static auto status = internal::CachedDsoLoader::GetLibcudaDsoHandle(); \
      return status.ValueOrDie();                                            \
    }                                                                        \
    static FuncPointerT DynLoad() {                                          \
      static void *f = dlsym(GetDsoHandle(), kName);                         \
      CHECK(f != nullptr) << "could not find " << kName                      \
                          << "in libcuda DSO; dlerror: " << dlerror();       \
      return reinterpret_cast<FuncPointerT>(f);                              \
    }                                                                        \
    template <typename... Args>                                              \
    CUresult operator()(Args... args) {                                      \
      return DynLoad()(args...);                                             \
    }                                                                        \
  } __name;                                                                  \
  const char *DynLoadShim__##__name::kName = #__name;

PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuCtxCreate_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuCtxDestroy);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuCtxEnablePeerAccess);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuCtxGetCurrent);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuCtxGetDevice);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuCtxGetSharedMemConfig);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuCtxPopCurrent_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuCtxSetCurrent);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuCtxSetSharedMemConfig);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuCtxSynchronize);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuDeviceComputeCapability);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuDeviceCanAccessPeer);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuDeviceGet);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuDeviceGetAttribute);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuDeviceGetCount);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuDeviceGetName);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuDeviceGetPCIBusId);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuDeviceGetProperties);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuDeviceTotalMem);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuDriverGetVersion);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuEventCreate);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuEventDestroy_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuEventElapsedTime);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuEventQuery);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuEventRecord);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuFuncGetAttribute);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuFuncSetCacheConfig);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuGetErrorName);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuGetErrorString);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuInit);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuLaunchKernel);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemAlloc_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemcpyDtoD_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemcpyDtoH_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemcpyHtoD_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemcpyDtoDAsync_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemcpyDtoHAsync_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemcpyHtoDAsync_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemGetAddressRange_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemFree_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemFreeHost);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemGetInfo_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemHostAlloc);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemHostRegister_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemHostUnregister);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemsetD32_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemsetD32Async);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuMemsetD8_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuModuleGetFunction);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuModuleGetGlobal_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuModuleLoadDataEx);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuModuleLoadFatBinary);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuModuleUnload);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuOccupancyMaxActiveBlocksPerMultiprocessor);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuPointerGetAttribute);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuStreamAddCallback);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuStreamCreate);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuStreamDestroy_v2);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuStreamQuery);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuStreamSynchronize);
PERFTOOLS_GPUTOOLS_LIBCUDA_WRAP(cuStreamWaitEvent);

}  // namespace dynload

namespace {

// Manages the singleton set of contexts that we've created. This is used for
// checking that no CUDA-runtime-created contexts have been generated
// accidentally.  CUDA-runtime-created contexts are avoided, if triple angle
// brace launches are required, by using the scoped activations in
// cuda_activation.h.
class CreatedContexts {
 public:
  // Returns whether context is a member of the live set.
  static bool Has(CUcontext context) {
    shared_lock lock{mu_};
    return Live()->find(context) != Live()->end();
  }

  // Adds context to the live set.
  static void Add(CUcontext context) {
    CHECK(context != nullptr);
    mutex_lock lock{mu_};
    Live()->emplace(context);
  }

  // Removes context from the live set.
  static void Remove(CUcontext context) {
    CHECK(context != nullptr);
    mutex_lock lock{mu_};
    Live()->erase(context);
  }

 private:
  // Returns the live set singleton.
  static std::set<CUcontext> *Live() {
    static auto singleton = new std::set<CUcontext>;
    return singleton;
  }

  // Lock that guards access-to/mutation-of the live set.
  static mutex mu_;
};

/* static */ mutex CreatedContexts::mu_{LINKER_INITIALIZED};

// Formats CUresult to output prettified values into a log stream.
// Error summaries taken from:
// http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9
//
// TODO(leary) switch to cuGetErrorName when updated cuda.h is available.
string ToString(CUresult result) {
#define OSTREAM_CUDA_ERROR(__name) \
  case CUDA_ERROR_##__name:        \
    return "CUDA_ERROR_" #__name;

///////////////
// NOTE: here we specify return code values outside of the enum explicitly
// because our in-tree cuda.h is from the CUDA 5.5 SDK, but CUDA 6.0+ driver
// libraries are deployed in the fleet these error codes are backwards
// compatible, but if we see a "new" one, we want to be able to identify it in
// the logs.
//
// Once we get a cuda.h that has cuGetErrorName (TODO is above) we can
// eliminate this function and just rely on the driver to provide us these
// strings.
//
// NOTE: "Must reboot all context" below is shorthand for, "must
// destroy/recreate the offending context and any allocation which come from
// it if you are to continue using CUDA."
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
  switch (result) {
    OSTREAM_CUDA_ERROR(INVALID_VALUE)
    OSTREAM_CUDA_ERROR(OUT_OF_MEMORY)
    OSTREAM_CUDA_ERROR(NOT_INITIALIZED)
    OSTREAM_CUDA_ERROR(DEINITIALIZED)
    OSTREAM_CUDA_ERROR(NO_DEVICE)
    OSTREAM_CUDA_ERROR(INVALID_DEVICE)
    OSTREAM_CUDA_ERROR(INVALID_IMAGE)
    OSTREAM_CUDA_ERROR(INVALID_CONTEXT)
    OSTREAM_CUDA_ERROR(INVALID_HANDLE)
    OSTREAM_CUDA_ERROR(NOT_FOUND)
    OSTREAM_CUDA_ERROR(NOT_READY)
    OSTREAM_CUDA_ERROR(NO_BINARY_FOR_GPU)

    // Encountered an uncorrectable ECC error during execution.
    OSTREAM_CUDA_ERROR(ECC_UNCORRECTABLE)

    // Load/store on an invalid address. Must reboot all context.
    case 700:
      return "CUDA_ERROR_ILLEGAL_ADDRESS";
    // Passed too many / wrong arguments, too many threads for register count.
    case 701:
      return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    // Kernel took too long to execute.
    case 702:
      return "CUDA_ERROR_LAUNCH_TIMEOUT";
    // Kernel launch uses an incompatible texturing mode.
    case 703:
      return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
    // Trying to re-enable peer access that already has it enabled.
    case 704:
      return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    // Trying to disable peer access that has not yet been enabled.
    case 705:
      return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    // Primary context for the specified device has already been initialized.
    case 708:
      return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    // Context current to calling thread has been destroyed or is a primary
    // context that has not yet been initialized.
    case 709:
      return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
    // Device-side assert triggered during kernel execution. Must reboot all
    // context.
    case 710:
      return "CUDA_ERROR_ASSERT";
    // Hardware resources to enable peer access have been exhausted.
    case 711:
      return "CUDA_ERROR_TOO_MANY_PEERS";
    // Memory range has already been registered.
    case 712:
      return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    // Pointer does not correspond to any currently registered memory region.
    case 713:
      return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
    // Due to stack corruption or exceeding stack size limit. Must reboot all
    // context.
    case 714:
      return "CUDA_ERROR_HARDWARE_STACK_ERROR";
    case 715:
      return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
    // Load/store on an unaligned memory address. Must reboot all context.
    case 716:
      return "CUDA_ERROR_MISALIGNED_ADDRESS";
    // Device instruction with specific address space given address not
    // belonging to allowed address space. Must reboot all context.
    case 717:
      return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
    // Device program counter wrapped its address space. Must reboot all
    // context.
    case 718:
      return "CUDA_ERROR_INVALID_PC";
    // Exception on device while executing a kernel; e.g. deref invalid device
    // pointer, accessing OOB shared memory. Must reboot all context.
    case 719:
      return "CUDA_ERROR_LAUNCH_FAILED";

    OSTREAM_CUDA_ERROR(CONTEXT_ALREADY_IN_USE)
    OSTREAM_CUDA_ERROR(PEER_ACCESS_UNSUPPORTED)
    OSTREAM_CUDA_ERROR(NOT_PERMITTED)
    OSTREAM_CUDA_ERROR(NOT_SUPPORTED)
    OSTREAM_CUDA_ERROR(UNKNOWN)  // Unknown internal error to CUDA.
    default:
      return port::StrCat("CUresult(", static_cast<int>(result), ")");
  }
#pragma GCC diagnostic pop
}

// Returns the current context and checks that it is in the set of CUDA contexts
// created by StreamExecutor (to ensure that the CUDA runtime didn't create a
// context behind our backs).
CUcontext CurrentContext() {
  CUcontext current = nullptr;
  CUresult result = dynload::cuCtxGetCurrent(&current);
  if (result != CUDA_SUCCESS) {
    LOG(FATAL) << "failed to query current context: " << ToString(result);
  }
  if (current != nullptr && !CreatedContexts::Has(current)) {
    LOG(FATAL) << "current context was not created by the StreamExecutor "
                  "cuda_driver API: "
               << current
               << "; a CUDA runtime call "
                  "was likely performed without using a StreamExecutor context";
  }
  return current;
}

// "Pops" the current context, checks that it matches expected, and checks the
// postcondition that the current context is nullptr.
//
// This is not done when we're nested within a MultiOpActivation, as we want to
// persist the active context until the MultiOpActivation is popped.
void PopContextAndCheckNowNull(CUcontext expected) {
  CUcontext actual = CurrentContext();
  CHECK_EQ(expected, actual) << "would pop unexpected context";
  CUcontext popped;
  CHECK_EQ(CUDA_SUCCESS, dynload::cuCtxPopCurrent_v2(&popped));
  CHECK_EQ(expected, popped);
  CHECK(nullptr == CurrentContext());
  VLOG(3) << "popped context " << expected
          << " and current context is now null";
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


// Thread-local storage that indicates whether a CUDA context activation is
// being nested within an outer, MultiOpActivation. In that case, we should not
// pop the context to nullptr when we are done with the current activation.
SE_STATIC_THREAD_LOCAL_POD(bool, tls_in_multi_op_activation);

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

// Implementation note: the CUDA context is held, per-thread, in TLS. We avoid
// setting all the time because it's not clear what side effects might occur for
// a "set" operation, whereas a "get" operation we can reasonably assume is a
// TLS read.
//
// We cannot race here because CUcontext is associated with a particular thread
// and stored in TLS; and these interfaces should not be used from signal
// handlers.
ScopedActivateContext::ScopedActivateContext(CUcontext context,
                                             MultiOpActivation moa)
    : context_(CHECK_NOTNULL(context)),
      previously_in_multi_op_activation_(tls_in_multi_op_activation.get()) {
  if (static_cast<bool>(moa)) {
    tls_in_multi_op_activation.get() = true;
  }

  CUcontext current = prior_context_ = CurrentContext();
  if (current != context) {
    VLOG(3) << "ScopedActivateContext switching context from " << current
            << " to " << context;
    CHECK_EQ(CUDA_SUCCESS, dynload::cuCtxSetCurrent(context));
    if (FLAGS_gpuexec_cuda_sync_around_driver_calls) {
      auto res = dynload::cuCtxSynchronize();
      if (res != CUDA_SUCCESS) {
        LOG(FATAL) << "gpuexec_cuda_sync_around_driver_calls found "
                   << ToString(res)
                   << " immediately after establishing the device context "
                   << context << " :: " << port::CurrentStackTrace();
      }
    }
  }
}

ScopedActivateContext::~ScopedActivateContext() {
  if (tls_in_multi_op_activation.get()) {
    CHECK_EQ(context_, CurrentContext());
    if (FLAGS_gpuexec_cuda_sync_around_driver_calls) {
      auto res = dynload::cuCtxSynchronize();
      if (res != CUDA_SUCCESS) {
        LOG(FATAL) << "gpuexec_cuda_sync_around_driver_calls found "
                   << ToString(res)
                   << " immediately after de-establishing the device context "
                   << context_ << " :: " << port::CurrentStackTrace();
      }
    }
    CHECK_EQ(CUDA_SUCCESS, dynload::cuCtxSetCurrent(prior_context_));
  } else {
    PopContextAndCheckNowNull(context_);
  }
  tls_in_multi_op_activation.get() = previously_in_multi_op_activation_;
}

namespace {

// Returns a stringified device number associated with pointer, primarily for
// logging purposes. Returns "?" if the device could not be successfully
// queried.
string CUDAPointerToDeviceString(CUdeviceptr pointer) {
  auto value = CUDADriver::GetPointerDevice(pointer);
  if (value.ok()) {
    return port::StrCat(value.ValueOrDie());
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
  } else if (internal::CachedDsoLoader::GetLibcudaDsoHandle().ok()) {
    // We only call cuInit if we can dynload libcuda.
    
    res = dynload::cuInit(0 /* = flags */);
  }

  if (res == CUDA_SUCCESS) {
    return port::Status::OK();
  }

  LOG(ERROR) << "failed call to cuInit: " << ToString(res);
  Diagnostician::LogDiagnosticInformation();
  return port::Status{port::error::ABORTED,
                      port::StrCat("failed call to cuInit: ", ToString(res))};
}

}  // namespace

/* static */ port::Status CUDADriver::Init() {
  // Cached return value from calling InternalInit(), as cuInit need only be
  // called once, but CUDADriver::Init may be called many times.
  static port::Status init_retval;
  static bool set = false;
  static mutex init_mu(LINKER_INITIALIZED);

  mutex_lock lock(init_mu);
  if (!set) {
    init_retval = InternalInit();
    set = true;
  }

  return init_retval;
}

/* static */ port::Status CUDADriver::GetDevice(int device_ordinal,
                                                CUdevice *device) {
  CUresult res = dynload::cuDeviceGet(device, device_ordinal);
  if (res == CUDA_SUCCESS) {
    return port::Status::OK();
  }

  return port::Status{
      port::error::INTERNAL,
      port::StrCat("failed call to cuDeviceGet: ", ToString(res))};
}

/* static */ bool CUDADriver::GetDeviceName(CUdevice device,
                                            string *device_name) {
  static const size_t kCharLimit = 64;
  port::InlinedVector<char, 4> chars(kCharLimit);
  CUresult res =
      dynload::cuDeviceGetName(chars.begin(), kCharLimit - 1, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to get device name for " << device << ": "
               << ToString(res);
    return false;
  }
  chars[kCharLimit - 1] = '\0';
  *device_name = chars.begin();
  return true;
}

bool DeviceOptionsToContextFlags(DeviceOptions device_options, int *flags) {
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
    CUdevice device, DeviceOptions device_options, CUcontext *context) {
  CUcontext former_context = CurrentContext();
  if (former_context != nullptr) {
    LOG(WARNING) << "creating context when one is currently active; existing: "
                 << former_context;
  }

  int flags = 0;
  if (!DeviceOptionsToContextFlags(device_options, &flags)) {
    LOG(WARNING) << "could not convert all device options into context flags";
  }

  CUresult res;
  {
    // TODO(leary) Need to see if NVIDIA can expunge the leakiness in their
    // context creation: see http://b/13248943
    
    res = dynload::cuCtxCreate_v2(context, flags, device);
  }
  if (res == CUDA_SUCCESS) {
    CreatedContexts::Add(*context);
    PopContextAndCheckNowNull(*context);
    CHECK(*context != nullptr)
        << "success in this call must entail non-null result";
    VLOG(2) << "created context " << context << " for this thread";
    return port::Status::OK();
  }

  string message = "failed call to cuCtxCreate: " + ToString(res);
  if (res == CUDA_ERROR_OUT_OF_MEMORY) {
    uint64 total_memory;
    if (GetDeviceTotalMemory(device, &total_memory)) {
      port::StrAppend(&message, "; total memory reported: ", total_memory);
    } else {
      port::StrAppend(&message, "; could not query total memory");
    }
  }

  return port::Status{port::error::INTERNAL, message};
}

/* static */ void CUDADriver::DestroyContext(CUcontext context) {
  if (context == nullptr) {
    return;
  }

  CUresult res = dynload::cuCtxDestroy_v2(context);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to destroy CUDA context; leaking: " << ToString(res);
  }

  CreatedContexts::Remove(context);
}

/* static */ bool CUDADriver::FuncGetAttribute(CUfunction_attribute attribute,
                                               CUfunction func,
                                               int *attribute_value) {
  CUresult res = dynload::cuFuncGetAttribute(attribute_value, attribute, func);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query kernel attribute. kernel: " << func
               << ", attribute: " << attribute;
    return false;
  }
  return true;
}

/* static */ bool CUDADriver::FuncSetCacheConfig(CUfunction function,
                                                 CUfunc_cache cache_config) {
  CUresult res = dynload::cuFuncSetCacheConfig(function, cache_config);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to set CUDA kernel cache config. kernel: " << function
               << ", config: " << cache_config << ", result: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ port::StatusOr<CUsharedconfig>
CUDADriver::ContextGetSharedMemConfig(CUcontext context) {
  CUsharedconfig shared_mem_config;
  ScopedActivateContext activation{context};
  CUresult result = dynload::cuCtxGetSharedMemConfig(&shared_mem_config);
  if (result != CUDA_SUCCESS) {
    CUdevice device;
    dynload::cuCtxGetDevice(&device);
    LOG(ERROR) << "failed to get CUDA device shared memory config. "
               << "Context device ID: " << device
               << ", result: " << ToString(result);
    return port::Status{
        port::error::INTERNAL,
        port::StrCat("failed to get shared memory config: ", ToString(result))};
  }
  return shared_mem_config;
}

/* static */ port::Status CUDADriver::ContextSetSharedMemConfig(
    CUcontext context, CUsharedconfig shared_mem_config) {
  ScopedActivateContext activation{context};
  CUresult result = dynload::cuCtxSetSharedMemConfig(shared_mem_config);
  if (result != CUDA_SUCCESS) {
    CUdevice device;
    dynload::cuCtxGetDevice(&device);
    LOG(ERROR) << "failed to set CUDA device shared memory config. "
               << "Context device ID: " << device
               << ", config: " << shared_mem_config
               << ", result: " << ToString(result);
    return port::Status{
        port::error::INTERNAL,
        port::StrCat("failed to set shared memory config: ", ToString(result))};
  }
  return port::Status::OK();
}

/* static */ bool CUDADriver::LaunchKernel(
    CUcontext context, CUfunction function, unsigned int grid_dim_x,
    unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, CUstream stream, void **kernel_params,
    void **extra) {
  ScopedActivateContext activation{context};
  VLOG(2) << "launching kernel: " << function << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z;
  CUresult res = dynload::cuLaunchKernel(
      function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
      block_dim_z, shared_mem_bytes, stream, kernel_params, extra);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to launch CUDA kernel: " << function
               << "; result: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully launched kernel";
  return true;
}

/* static */ port::Status CUDADriver::LoadCubin(CUcontext context,
                                                const char *cubin_bytes,
                                                CUmodule *module) {
  ScopedActivateContext activation{context};
  CUresult result = dynload::cuModuleLoadFatBinary(module, cubin_bytes);
  if (result != CUDA_SUCCESS) {
    return port::Status{port::error::INTERNAL,
                        "failed to load in-memory CUBIN: " + ToString(result)};
  }

  return port::Status::OK();
}

/* static */ bool CUDADriver::LoadPtx(CUcontext context,
                                      const char *ptx_contents,
                                      CUmodule *module) {
  port::Notification notification;
  bool ret = true;
  GetDriverExecutor()->Schedule([context, ptx_contents, module, &ret,
                                 &notification]() {
    ScopedActivateContext activation{context};
    void *ptx_data = const_cast<char *>(ptx_contents);
    static const unsigned int kLogBufferBytesLimit = 1024;
    unsigned int error_log_buffer_bytes = kLogBufferBytesLimit;
    unsigned int info_log_buffer_bytes = kLogBufferBytesLimit;
    port::InlinedVector<char, 4> error_log_buffer(error_log_buffer_bytes);
    port::InlinedVector<char, 4> info_log_buffer(info_log_buffer_bytes);
    bool log_verbose = true;
    CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                              CU_JIT_ERROR_LOG_BUFFER,
                              CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                              CU_JIT_INFO_LOG_BUFFER, CU_JIT_LOG_VERBOSE};
    // Note that the driver API wants the contents of this values to be stored
    // in an array of void*s, so we coerce them accordingly.
    void *option_values[] = {
        port::bit_cast<void *>(uintptr_t(error_log_buffer_bytes)),
        port::bit_cast<void *>(error_log_buffer.data()),
        port::bit_cast<void *>(uintptr_t(info_log_buffer_bytes)),
        port::bit_cast<void *>(info_log_buffer.data()),
        port::bit_cast<void *>(uintptr_t(log_verbose))};
    CHECK(ARRAYSIZE(options) == ARRAYSIZE(option_values));

    CUresult res;
    {
      // TODO(leary) Need to see if NVIDIA can expunge the leakiness in their
      // module loading: see http://b/13248943
      
      res = dynload::cuModuleLoadDataEx(module, ptx_data, ARRAYSIZE(options),
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

/* static */ bool CUDADriver::SynchronousMemsetUint8(CUcontext context,
                                                     CUdeviceptr location,
                                                     uint8 value, size_t size) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuMemsetD8_v2(location, value, size);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to memset memory: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool CUDADriver::SynchronousMemsetUint32(CUcontext context,
                                                      CUdeviceptr location,
                                                      uint32 value,
                                                      size_t uint32_count) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuMemsetD32_v2(location, value, uint32_count);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to memset memory: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool CUDADriver::AsynchronousMemsetUint32(CUcontext context,
                                                       CUdeviceptr location,
                                                       uint32 value,
                                                       size_t uint32_count,
                                                       CUstream stream) {
  ScopedActivateContext activation{context};
  CUresult res =
      dynload::cuMemsetD32Async(location, value, uint32_count, stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to enqueue async memset operation: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully enqueued async memset operation";
  return true;
}

/* static */ bool CUDADriver::AddStreamCallback(CUcontext context,
                                                CUstream stream,
                                                StreamCallback callback,
                                                void *data) {
  // Note: flags param is required to be zero according to CUDA 6.0.
  CUresult res =
      dynload::cuStreamAddCallback(stream, callback, data, 0 /* = flags */);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "unable to add host callback: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool CUDADriver::GetModuleFunction(CUcontext context,
                                                CUmodule module,
                                                const char *kernel_name,
                                                CUfunction *function) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  CUresult res = dynload::cuModuleGetFunction(function, module, kernel_name);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to get PTX kernel \"" << kernel_name
               << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool CUDADriver::GetModuleSymbol(CUcontext context,
                                              CUmodule module,
                                              const char *symbol_name,
                                              CUdeviceptr *dptr,
                                              size_t *bytes) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  CUresult res =
      dynload::cuModuleGetGlobal_v2(dptr, bytes, module, symbol_name);
  if (res != CUDA_SUCCESS) {
    // symbol may not be found in the current module, but it may reside in
    // another module.
    VLOG(2) << "failed to get symbol \"" << symbol_name
            << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ void CUDADriver::UnloadModule(CUcontext context, CUmodule module) {
  ScopedActivateContext activated{context};
  CUresult res = dynload::cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << ToString(res);
  }
}

/* static */ port::StatusOr<CUdevice> CUDADriver::DeviceFromContext(
    CUcontext context) {
  ScopedActivateContext activated{context};
  CUdevice device = -1;
  CUresult result = dynload::cuCtxGetDevice(&device);
  if (result == CUDA_SUCCESS) {
    return device;
  }

  return port::Status{
      port::error::INTERNAL,
      port::StrCat("failed to get device for context: ", ToString(result))};
}

/* static */ bool CUDADriver::CreateStream(CUcontext context, CUstream *out) {
  // TODO(leary) can we switch this to CU_STREAM_NON_BLOCKING or will that mess
  // up synchronization with respect to memsets and any other things that have
  // to occur on the default stream?
  ScopedActivateContext activated{context};
  CUresult res = dynload::cuStreamCreate(out, 0);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not allocate CUDA stream for context " << context
               << ": " << ToString(res);
    return false;
  }

  VLOG(2) << "successfully created stream " << *out << " for context "
          << context << " on thread";
  return true;
}

/* static */ void CUDADriver::DestroyStream(CUcontext context,
                                            CUstream *stream) {
  if (*stream == nullptr) {
    return;
  }

  ScopedActivateContext activated{context};
  CUresult res = dynload::cuStreamDestroy_v2(*stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to destroy CUDA stream for context " << context
               << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << *stream << " for context "
            << context;
    *stream = nullptr;
  }
}

/* static */ void *CUDADriver::DeviceAllocate(CUcontext context, uint64 bytes) {
  ScopedActivateContext activated{context};
  CUdeviceptr result = 0;
  CUresult res = dynload::cuMemAlloc_v2(&result, bytes);
  if (res != CUDA_SUCCESS) {
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

/* static */ void CUDADriver::DeviceDeallocate(CUcontext context,
                                               void *location) {
  ScopedActivateContext activation{context};
  CUdeviceptr pointer = port::bit_cast<CUdeviceptr>(location);
  CUresult res = dynload::cuMemFree_v2(pointer);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated " << location << " for context " << context;
  }
}

/* static */ void *CUDADriver::HostAllocate(CUcontext context, uint64 bytes) {
  ScopedActivateContext activation{context};
  void *host_mem = nullptr;
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  CUresult res =
      dynload::cuMemHostAlloc(&host_mem, bytes, CU_MEMHOSTALLOC_PORTABLE);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes on host: " << ToString(res);
  }
  return host_mem;
}

/* static */ void CUDADriver::HostDeallocate(CUcontext context,
                                             void *location) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuMemFreeHost(location);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "error deallocating host memory at " << location << ": "
               << ToString(res);
  }
}

/* static */ bool CUDADriver::HostRegister(CUcontext context, void *location,
                                           uint64 bytes) {
  ScopedActivateContext activation{context};
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  CUresult res =
      dynload::cuMemHostRegister(location, bytes, CU_MEMHOSTREGISTER_PORTABLE);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool CUDADriver::HostUnregister(CUcontext context,
                                             void *location) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuMemHostUnregister(location);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ port::Status CUDADriver::DestroyEvent(CUcontext context,
                                                   CUevent *event) {
  if (*event == nullptr) {
    return port::Status{port::error::INVALID_ARGUMENT,
                        "input event cannot be null"};
  }

  ScopedActivateContext activated{context};
  CUresult res = dynload::cuEventDestroy_v2(*event);
  *event = nullptr;

  switch (res) {
    case CUDA_SUCCESS:
      return port::Status::OK();
    case CUDA_ERROR_DEINITIALIZED:
    case CUDA_ERROR_NOT_INITIALIZED:
      return port::Status{
          port::error::FAILED_PRECONDITION,
          port::Printf("error destroying CUDA event in context %p: %s", context,
                       ToString(res).c_str())};
    default:
      return port::Status{
          port::error::INTERNAL,
          port::Printf("error destroying CUDA event in context %p: %s", context,
                       ToString(res).c_str())};
  }
}

/* static */ port::Status CUDADriver::RecordEvent(CUcontext context,
                                                  CUevent event,
                                                  CUstream stream) {
  ScopedActivateContext activated{context};
  CUresult res = dynload::cuEventRecord(event, stream);
  switch (res) {
    case CUDA_SUCCESS:
      return port::Status::OK();
    case CUDA_ERROR_DEINITIALIZED:
    case CUDA_ERROR_NOT_INITIALIZED:
      return port::Status{
          port::error::FAILED_PRECONDITION,
          port::Printf("error recording CUDA event on stream %p: %s", stream,
                       ToString(res).c_str())};
    default:
      return port::Status{
          port::error::INVALID_ARGUMENT,
          port::Printf("error recording CUDA event on stream %p: %s", stream,
                       ToString(res).c_str())};
  }
}

/* static */ port::StatusOr<CUresult> CUDADriver::QueryEvent(CUcontext context,
                                                             CUevent event) {
  ScopedActivateContext activated{context};
  CUresult res = dynload::cuEventQuery(event);
  if (res != CUDA_SUCCESS && res != CUDA_ERROR_NOT_READY) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf("failed to query event: %s", ToString(res).c_str())};
  }

  return res;
}

/* static */ bool CUDADriver::GetEventElapsedTime(CUcontext context,
                                                  float *elapsed_milliseconds,
                                                  CUevent start, CUevent stop) {
  ScopedActivateContext activated{context};
  CUresult res = dynload::cuEventElapsedTime(elapsed_milliseconds, start, stop);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to get elapsed time between events: "
               << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool CUDADriver::WaitStreamOnEvent(CUcontext context,
                                                CUstream stream,
                                                CUevent event) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuStreamWaitEvent(stream, event, 0 /* = flags */);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not wait stream on event: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool CUDADriver::SynchronizeContext(CUcontext context) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuCtxSynchronize();
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not synchronize on CUDA context: " << ToString(res)
               << " :: " << port::CurrentStackTrace();
    return false;
  }

  return true;
}

/* static */ bool CUDADriver::SynchronizeStream(CUcontext context,
                                                CUstream stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  CUresult res = dynload::cuStreamSynchronize(stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not synchronize on CUDA stream: " << ToString(res)
               << " :: " << port::CurrentStackTrace();
    return false;
  }
  VLOG(2) << "successfully synchronized stream " << stream << " on context "
          << context;
  return true;
}

/* static */ bool CUDADriver::IsStreamIdle(CUcontext context, CUstream stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  CUresult res = dynload::cuStreamQuery(stream);
  if (res == CUDA_SUCCESS) {
    return true;
  }

  if (res != CUDA_ERROR_NOT_READY) {
    LOG(ERROR) << "stream in bad state on status query: " << ToString(res);
  }
  return false;
}

/* static */ bool CUDADriver::SynchronousMemcpyD2H(CUcontext context,
                                                   void *host_dst,
                                                   CUdeviceptr gpu_src,
                                                   uint64 size) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuMemcpyDtoH_v2(host_dst, gpu_src, size);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << port::Printf(
        "failed to synchronous memcpy from device to host: %s; "
        "host dst: %p; GPU src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), host_dst, port::bit_cast<void *>(gpu_src), size, size);
    return false;
  }
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return true;
}

/* static */ bool CUDADriver::SynchronousMemcpyH2D(CUcontext context,
                                                   CUdeviceptr gpu_dst,
                                                   const void *host_src,
                                                   uint64 size) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuMemcpyHtoD_v2(gpu_dst, host_src, size);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << port::Printf(
        "failed to synchronous memcpy from host to device: %s; GPU dst: %p;"
        " host src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), port::bit_cast<void *>(gpu_dst), host_src, size, size);
    return false;
  }
  VLOG(2) << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return true;
}

/* static */ bool CUDADriver::SynchronousMemcpyD2D(CUcontext context,
                                                   CUdeviceptr gpu_dst,
                                                   CUdeviceptr gpu_src,
                                                   uint64 size) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuMemcpyDtoD_v2(gpu_dst, gpu_src, size);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << port::Printf(
        "failed to synchronous memcpy from host to device: %s; GPU dst: %p; "
        "GPU src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), port::bit_cast<void *>(gpu_dst),
        port::bit_cast<void *>(gpu_src), size, size);
    return false;
  }
  VLOG(2) << "successfully sync memcpy'd d2d of " << size << " bytes";
  return true;
}

/* static */ bool CUDADriver::AsynchronousMemcpyD2H(CUcontext context,
                                                    void *host_dst,
                                                    CUdeviceptr gpu_src,
                                                    uint64 size,
                                                    CUstream stream) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuMemcpyDtoHAsync_v2(host_dst, gpu_src, size, stream);
  if (res != CUDA_SUCCESS) {
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

/* static */ bool CUDADriver::AsynchronousMemcpyH2D(CUcontext context,
                                                    CUdeviceptr gpu_dst,
                                                    const void *host_src,
                                                    uint64 size,
                                                    CUstream stream) {
  ScopedActivateContext activation{context};
  CUresult res = dynload::cuMemcpyHtoDAsync_v2(gpu_dst, host_src, size, stream);
  if (res != CUDA_SUCCESS) {
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

/* static */ bool CUDADriver::AsynchronousMemcpyD2D(CUcontext context,
                                                    CUdeviceptr gpu_dst,
                                                    CUdeviceptr gpu_src,
                                                    uint64 size,
                                                    CUstream stream) {
  ScopedActivateContext activation{context};
  CUresult result =
      dynload::cuMemcpyDtoDAsync_v2(gpu_dst, gpu_src, size, stream);
  if (result != CUDA_SUCCESS) {
    LOG(ERROR) << port::Printf(
        "failed to enqueue async memcpy from device to device: %s"
        "; GPU dst: %p on %s %s"
        "; GPU src: %p on %s %s"
        "; can access? %s; size: %llu=0x%llx",
        ToString(result).c_str(), port::bit_cast<void *>(gpu_dst),
        CUDAPointerToMemorySpaceString(gpu_dst).c_str(),
        CUDAPointerToDeviceString(gpu_dst).c_str(), port::bit_cast<void *>(gpu_src),
        CUDAPointerToMemorySpaceString(gpu_src).c_str(),
        CUDAPointerToDeviceString(gpu_src).c_str(),
        CUDAPointersToCanAccessString(gpu_src, gpu_dst).c_str(), size, size);

    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes";
  return true;
}

/* static */ port::Status CUDADriver::CreateEvent(CUcontext context,
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
  CUresult res = dynload::cuEventCreate(result, cuflags);

  if (res == CUDA_SUCCESS) {
    return port::Status::OK();
  } else if (res == CUDA_ERROR_OUT_OF_MEMORY) {
    return port::Status{port::error::RESOURCE_EXHAUSTED,
                        "could not create CUDA event: out of device memory"};
  } else {
    return port::Status{
        port::error::FAILED_PRECONDITION,
        port::StrCat("could not create CUDA event: ", ToString(res))};
  }
}

/* static */ int CUDADriver::GetDeviceCount() {
  int device_count = 0;
  CUresult res = dynload::cuDeviceGetCount(&device_count);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not retrieve CUDA device count: " << ToString(res);
    return 0;
  }

  if (FLAGS_gpuexec_cuda_device_0_only && device_count > 1) {
    device_count = 1;
  }
  return device_count;
}

/* static */ port::StatusOr<CUcontext> CUDADriver::GetPointerContext(
    CUdeviceptr pointer) {
  CUcontext context = nullptr;
  CUresult result = dynload::cuPointerGetAttribute(
      &context, CU_POINTER_ATTRIBUTE_CONTEXT, pointer);
  if (result == CUDA_SUCCESS) {
    CHECK(context != nullptr) << "success should entail non-null context";
    return context;
  }

  return port::Status{
      port::error::INTERNAL,
      port::StrCat("failed to query device pointer for context: ",
                   ToString(result))};
}

/* static */ port::StatusOr<MemorySpace> CUDADriver::GetPointerMemorySpace(
    CUdeviceptr pointer) {
  unsigned int value;
  CUresult result = dynload::cuPointerGetAttribute(
      &value, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer);
  if (result == CUDA_SUCCESS) {
    switch (value) {
      case CU_MEMORYTYPE_DEVICE:
        return MemorySpace::kDevice;
      case CU_MEMORYTYPE_HOST:
        return MemorySpace::kHost;
      default:
        return port::Status{
            port::error::INTERNAL,
            port::StrCat("unknown memory space provided by CUDA API: ", value)};
    }
  }

  return port::Status{
      port::error::INTERNAL,
      port::StrCat("failed to query device pointer for memory space: ",
                   ToString(result))};
}

/* static */ port::Status CUDADriver::GetPointerAddressRange(CUdeviceptr dptr,
                                                             CUdeviceptr *base,
                                                             size_t *size) {
  CUresult result = dynload::cuMemGetAddressRange(base, size, dptr);
  if (result == CUDA_SUCCESS) {
    return port::Status::OK();
  } else if (result == CUDA_ERROR_NOT_FOUND) {
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
  CUresult result =
      dynload::cuDeviceComputeCapability(cc_major, cc_minor, device);
  if (result == CUDA_SUCCESS) {
    return port::Status::OK();
  }

  return port::Status{
      port::error::INTERNAL,
      port::Printf("failed to get compute capability for device: %s; %d",
                   ToString(result).c_str(), device)};
}

// Helper function that turns the integer output of cuDeviceGetAttribute to type
// T and wraps it in a StatusOr.
template <typename T>
static port::StatusOr<T> GetSimpleAttribute(CUdevice device,
                                            CUdevice_attribute attribute) {
  int value = -1;
  CUresult result = dynload::cuDeviceGetAttribute(&value, attribute, device);
  if (result != CUDA_SUCCESS) {
    return port::Status{
        port::error::NOT_FOUND,
        port::StrCat("could not retrieve CUDA device attribute (", attribute,
                     "): ", ToString(result))};
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
  CUresult res = dynload::cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim x: " << ToString(res);
    return false;
  }
  *x = value;

  res = dynload::cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim y: " << ToString(res);
    return false;
  }
  *y = value;

  res = dynload::cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim z: " << ToString(res);
    return false;
  }
  *z = value;
  return true;
}

/* static */ bool CUDADriver::GetDriverVersion(int *driver_version) {
  CUresult res = dynload::cuDriverGetVersion(driver_version);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query driver version: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool CUDADriver::GetDeviceProperties(CUdevprop *device_properties,
                                                  int device_ordinal) {
  CUresult res =
      dynload::cuDeviceGetProperties(device_properties, device_ordinal);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool CUDADriver::IsEccEnabled(CUdevice device, bool *result) {
  int value = -1;
  CUresult res = dynload::cuDeviceGetAttribute(
      &value, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query ECC status: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ bool CUDADriver::GetDeviceMemoryInfo(CUcontext context,
                                                  int64 *free_out,
                                                  int64 *total_out) {
  ScopedActivateContext activation{context};
  size_t free = 0;
  size_t total = 0;
  CUresult res = dynload::cuMemGetInfo_v2(&free, &total);
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
  CUresult res = dynload::cuDeviceTotalMem_v2(&value, device);
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
  port::InlinedVector<char, 4> chars(kBufferSize);
  chars[kBufferSize - 1] = '\0';
  CUresult res =
      dynload::cuDeviceGetPCIBusId(chars.begin(), kBufferSize - 1, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << ToString(res);
    return pci_bus_id;
  }
  pci_bus_id = chars.begin();
  return pci_bus_id;
}

/* static */ bool CUDADriver::CanEnablePeerAccess(CUcontext from,
                                                  CUcontext to) {
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
  CUresult res = dynload::cuDeviceCanAccessPeer(
      &can_access_peer, from_device.ValueOrDie(), to_device.ValueOrDie());
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to detect peer access capability: " << ToString(res);
    return false;
  }

  return can_access_peer;
}

/* static */ port::Status CUDADriver::EnablePeerAccess(CUcontext from,
                                                       CUcontext to) {
  if (from == to) {
    return port::Status::OK();  // A context can always access its own memory.
  }

  ScopedActivateContext activated{from};
  CUresult result = dynload::cuCtxEnablePeerAccess(to, 0 /* = flags */);
  if (result != CUDA_SUCCESS &&
      result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf("failed to enable peer access from %p to %p: %s", from, to,
                     ToString(result).c_str())};
  }

  return port::Status::OK();
}

/* static */ port::StatusOr<int> CUDADriver::GetMaxOccupiedBlocksPerCore(
    CUcontext context, CUfunction kernel, int threads_per_block,
    size_t dynamic_shared_memory_bytes) {
  ScopedActivateContext activation{context};

  int max_blocks;
  CUresult result = dynload::cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks, kernel, threads_per_block, dynamic_shared_memory_bytes);
  if (result != CUDA_SUCCESS) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf("failed to calculate occupancy of kernel %p: %s", kernel,
                     ToString(result).c_str())};
  }

  return max_blocks;
}

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools
