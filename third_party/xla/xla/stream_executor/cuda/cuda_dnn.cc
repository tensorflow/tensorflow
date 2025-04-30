/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_dnn.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend/graph_properties.h"
#include "Eigen/Core"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_diagnostics.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cudnn_frontend_helpers.h"
#include "xla/stream_executor/data_type.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/tensor_float_32_utils.h"

// clang-format off
#include "third_party/gpus/cuda/include/library_types.h"
#include "third_party/gpus/cudnn/cudnn_version.h"

#if CUDNN_VERSION >= 90000
#include "third_party/gpus/cudnn/cudnn_adv.h"
#include "third_party/gpus/cudnn/cudnn_cnn.h"
#include "third_party/gpus/cudnn/cudnn_ops.h"
#else
#include "third_party/gpus/cudnn/cudnn_adv_infer.h"
#include "third_party/gpus/cudnn/cudnn_adv_train.h"
#include "third_party/gpus/cudnn/cudnn_cnn_infer.h"
#include "third_party/gpus/cudnn/cudnn_cnn_train.h"
#include "third_party/gpus/cudnn/cudnn_ops_infer.h"
#include "third_party/gpus/cudnn/cudnn_ops_train.h"
#endif

#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_utils.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_EngineConfig.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_Errata.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_ExecutionPlan.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_Filters.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_Heuristics.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_MatMulDesc.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_Operation.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_OperationGraph.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_PointWiseDesc.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_Tensor.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_VariantPack.h"
// clang-format on

#ifdef __clang__
#pragma clang diagnostic push

// Make sure that Eigen::half forward declaration in dnn.h matches the
// declaration in Eigen.
#pragma clang diagnostic warning "-Wmismatched-tags"
#endif

namespace stream_executor {
namespace gpu {

namespace {

static_assert(CUDNN_VERSION >= 8900, "cuDNN needs to be version 8.9 or higher");

// Exits the program if 'expr' doesn't return CUDNN_STATUS_SUCCESS.
#define CHECK_CUDNN_OK(expr) CHECK_EQ(expr, CUDNN_STATUS_SUCCESS)

// If 'expr' doesn't return CUDNN_STATUS_SUCCESS, returns from the current
// function with a non-successful absl::Status.
#define RETURN_IF_CUDNN_ERROR(expr)                                     \
  do {                                                                  \
    cudnnStatus_t _status = (expr);                                     \
    if (!ABSL_PREDICT_TRUE(_status == CUDNN_STATUS_SUCCESS)) {          \
      std::ostringstream oss;                                           \
      oss << CudnnStatusToString(_status) << "\nin " << __FILE__ << "(" \
          << __LINE__ << "): '" << #expr << "'";                        \
      return absl::UnknownError(oss.str());                             \
    }                                                                   \
  } while (false)

#define RETURN_MSG_IF_CUDNN_ERROR(expr)                                 \
  do {                                                                  \
    cudnnStatus_t _status = (expr).get_status();                        \
    if (!ABSL_PREDICT_TRUE(_status == CUDNN_STATUS_SUCCESS)) {          \
      std::ostringstream oss;                                           \
      oss << CudnnStatusToString(_status) << "\nin " << __FILE__ << "(" \
          << __LINE__ << "): '" << #expr << "' " << (expr).get_error(); \
      return absl::UnknownError(oss.str());                             \
    }                                                                   \
  } while (false)

#define RETURN_FALSE_IF_CUDNN_ERROR(expr)                                  \
  do {                                                                     \
    if (!ABSL_PREDICT_TRUE((expr).get_status() == CUDNN_STATUS_SUCCESS)) { \
      return false;                                                        \
    }                                                                      \
  } while (false)

// Converts (via narrowing) a type T value to a type U, and checks that the
// value has no value change due to the conversion.
template <typename WideT, typename NarrowT>
NarrowT CheckedNarrowing(const WideT& wide) {
  NarrowT narrow = wide;
  CHECK_EQ(narrow, wide)
      << "checked narrowing failed; values not equal post-conversion";
  return narrow;
}

std::string CudnnStatusToString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
      return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
      return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
    default:
      return absl::StrCat("<unknown cudnn status: ", static_cast<int>(status),
                          ">");
  }
}

// RAII wrapper for all calls to cuDNN with a cuDNN handle argument.
//
// See CudnnAccess::GetHandle() for details.
class CudnnHandle {
 public:
  // Takes ownership of the lock to access cuDNN using handle.
  CudnnHandle(StreamExecutor* executor, std::unique_ptr<absl::MutexLock> lock,
              cudnnHandle_t handle)
      : context_(executor->Activate()),
        lock_(std::move(lock)),
        handle_(handle) {}

  // Returns cuDNN handle. To be passed directly to cuDNN APIs, don't keep
  // a copy.
  cudnnHandle_t handle() const { return handle_; }

 private:
  std::unique_ptr<ActivateContext> context_;
  std::unique_ptr<absl::MutexLock> lock_;
  cudnnHandle_t handle_;  // Not owned.
};

// RAII wrapper for temporary cuDNN handles that are used for multithreaded
// compilation. Unlike with CudnnAccess these are not associated
// with GPU devices and are not locked.
class LocalCuDnnHandle {
 public:
  explicit LocalCuDnnHandle(cudnnHandle_t handle) : handle_(handle) {}
  ~LocalCuDnnHandle() { cudnnDestroy(handle_); }
  cudnnHandle_t handle() { return handle_; }

 private:
  cudnnHandle_t handle_;
};

// Major version is neither forward or backward compatible and therefore major
// versions needs to match between source and library.
//
// Minor version is backward-compatible and therefore minor version of library
// needs to be same or higher.
//
// Patch releases are always forward and backward compatible and therefore
// need not match.
bool IsSourceCompatibleWithCudnnLibrary(dnn::VersionInfo source_version,
                                        dnn::VersionInfo loaded_version) {
  return loaded_version.major_version() == source_version.major_version() &&
         loaded_version.minor_version() >= source_version.minor_version();
}

}  // namespace

// Wraps a cuDNN handle and provides access to it through CudnnHandle
// instances, which also locks a mutex, acquires the CUDA context, and sets
// the stream that cuDNN should use to enqueue any work.
//
// Note: CudnnSupport::cudnn_ should be the only instantiation of this class.
class CudnnAccess {
 public:
  // Takes ownership of the handle.
  explicit CudnnAccess(cudnnHandle_t handle) : handle_(handle) {}

  ~CudnnAccess() {
    absl::MutexLock lock(&mutex_);
    cudnnDestroy(handle_);
  }

  // Creates a CudnnHandle instance for stream.
  //
  // cuDNN API calls using the same handle instance need to be serialized
  // across threads. This is guaranteed by CudnnHandle instances locking the
  // mutex owned by this class.
  //
  // Most cuDNN APIs taking a handle perform work on a CUDA stream. The
  // CudnnHandle instance acquires the executor's CUDA context and sets cuDNN
  // to use the provided stream.
  //
  // The stream argument may be null, which translates to the legacy default
  // stream. See
  // https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html.
  // The legacy default stream synchronizes with all other streams and it is
  // therefore a bad idea (performance wise) to call any cuDNN APIs that
  // enqueue work in the stream.
  CudnnHandle GetHandle(StreamExecutor* executor, Stream* stream) {
    auto lock = std::make_unique<absl::MutexLock>(&mutex_);
    mutex_.AssertHeld();
    CUstream cu_stream = stream ? absl::bit_cast<CUstream>(
                                      stream->platform_specific_handle().stream)
                                : cudaStreamLegacy;
    if (!current_stream_ || cu_stream != *current_stream_) {
      current_stream_ = cu_stream;
      const auto status = cudnnSetStream(handle_, cu_stream);
      CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << "Failed to set cuDNN stream.";
    }
    return CudnnHandle(executor, std::move(lock), handle_);
  }

  absl::StatusOr<std::unique_ptr<LocalCuDnnHandle>> GetLocalHandle() {
    cudnnHandle_t handle = nullptr;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
      return absl::InternalError("Creation of local cudnn handle failed.");
    }
    return std::make_unique<LocalCuDnnHandle>(handle);
  }

  void NotifyStreamDestroyed(Stream* stream) {
    CUstream cu_stream =
        absl::bit_cast<CUstream>(stream->platform_specific_handle().stream);
    absl::MutexLock lock(&mutex_);
    if (current_stream_ && cu_stream == *current_stream_) {
      current_stream_.reset();
    }
  }

 private:
  // Guards current_stream_ and the enqueueing of cuDNN operations via the
  // handle_ below.
  absl::Mutex mutex_;

  // If set, indicates the stream currently active on handle_, to avoid the
  // overhead of re-setting the same stream unnecessarily.
  std::optional<CUstream> current_stream_ ABSL_GUARDED_BY(mutex_);

  // cuDNN library handle.
  cudnnHandle_t handle_ ABSL_GUARDED_BY(mutex_);  // Owned.
};

namespace {

// A helper function to return the internal compute type for
// RNNs in cudnn.
cudnnDataType_t GetRnnComputeType(dnn::DataType data_type);

cudnnConvolutionFwdAlgo_t ToConvForwardAlgo(dnn::AlgorithmDesc algorithm) {
  cudnnConvolutionFwdAlgo_t algo =
      cudnnConvolutionFwdAlgo_t(algorithm.algo_id());
  switch (algo) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
      return algo;
    default:
      LOG(FATAL) << "Unsupported Cudnn convolution forward algorithm: "
                 << algorithm.algo_id();
  }
}

cudnnConvolutionBwdDataAlgo_t ToConvBackwardDataAlgo(
    dnn::AlgorithmDesc algorithm) {
  cudnnConvolutionBwdDataAlgo_t algo =
      cudnnConvolutionBwdDataAlgo_t(algorithm.algo_id());
  switch (algo) {
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
      return algo;
    default:
      LOG(FATAL)
          << "Unsupported Cudnn convolution backward algorithm for data: "
          << algorithm.algo_id();
  }
}

cudnnConvolutionBwdFilterAlgo_t ToConvBackwardFilterAlgo(
    dnn::AlgorithmDesc algorithm) {
  cudnnConvolutionBwdFilterAlgo_t algo =
      cudnnConvolutionBwdFilterAlgo_t(algorithm.algo_id());
  switch (algo) {
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
    // Based on cudnn.h, the following is not implemented.
    // case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
      return algo;
    default:
      LOG(FATAL)
          << "Unsupported Cudnn convolution backward algorithm for filter: "
          << algorithm.algo_id();
  }
}

absl::StatusOr<int> GetCudnnProperty(libraryPropertyType type) {
  int value;
  RETURN_IF_CUDNN_ERROR(cudnnGetProperty(type, &value));
  return value;
}

cudnnRNNAlgo_t ToCudnnRNNAlgo(std::optional<dnn::AlgorithmDesc> algorithm) {
  if (!algorithm.has_value()) {
    return CUDNN_RNN_ALGO_STANDARD;
  }
  cudnnRNNAlgo_t algo = static_cast<cudnnRNNAlgo_t>(algorithm->algo_id());
  switch (algo) {
    case CUDNN_RNN_ALGO_STANDARD:
    case CUDNN_RNN_ALGO_PERSIST_STATIC:
    case CUDNN_RNN_ALGO_PERSIST_DYNAMIC:
      return algo;
    default:
      LOG(FATAL) << "Unsupported Cudnn RNN algorithm: " << algorithm->algo_id();
  }
}

absl::StatusOr<dnn::VersionInfo> GetLoadedCudnnVersion() {
  TF_ASSIGN_OR_RETURN(int major, GetCudnnProperty(MAJOR_VERSION));
  TF_ASSIGN_OR_RETURN(int minor, GetCudnnProperty(MINOR_VERSION));
  TF_ASSIGN_OR_RETURN(int patch_level, GetCudnnProperty(PATCH_LEVEL));
  return dnn::VersionInfo(major, minor, patch_level);
}

enum class PreloadCudnnType { ConvFwd, ConvBwdFilter, ConvBwdData, Rnn };

// Preload sub libs for cudnn 8.0.4+ to make sure that the loading time isn't
// measured in the autotuning.
void PreloadCudnnSubLibs(PreloadCudnnType type) {
  switch (type) {
    case PreloadCudnnType::ConvBwdFilter:
    case PreloadCudnnType::ConvBwdData: {
#if CUDNN_VERSION < 90000
      cudnnOpsTrainVersionCheck();
      cudnnCnnTrainVersionCheck();
#endif  // CUDNN_VERSION < 90000
      [[clang::fallthrough]];
    }
    case PreloadCudnnType::ConvFwd: {
#if CUDNN_VERSION >= 90000
      cudnnGraphVersionCheck();
      cudnnOpsVersionCheck();
#else
      cudnnOpsInferVersionCheck();
      cudnnCnnInferVersionCheck();
#endif  // CUDNN_VERSION >= 90000
      break;
    }
    case PreloadCudnnType::Rnn: {
#if CUDNN_VERSION >= 90000
      cudnnOpsVersionCheck();
      cudnnAdvVersionCheck();
#else
      cudnnOpsInferVersionCheck();
      cudnnAdvInferVersionCheck();
      cudnnOpsTrainVersionCheck();
      cudnnAdvTrainVersionCheck();
#endif  // CUDNN_VERSION >= 90000
      break;
    }
  }
}

void PreloadCudnnSubLibsHelper(dnn::ConvolutionKind kind) {
  switch (kind) {
    case dnn::ConvolutionKind::FORWARD:
    case dnn::ConvolutionKind::FORWARD_GRAPH: {
      PreloadCudnnSubLibs(PreloadCudnnType::ConvFwd);
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_DATA: {
      PreloadCudnnSubLibs(PreloadCudnnType::ConvBwdData);
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_FILTER: {
      PreloadCudnnSubLibs(PreloadCudnnType::ConvBwdFilter);
      break;
    }
    default: {
      LOG(WARNING) << "Unsupported dnn::ConvolutionKind: "
                   << static_cast<int>(kind) << " for cuDNN preload.";
      break;
    }
  }
}

}  // namespace

CudnnSupport::CudnnSupport(StreamExecutor* parent) : parent_(parent) {}

absl::Status CudnnSupport::Init() {
  std::unique_ptr<ActivateContext> context = parent_->Activate();

  // Peek at the last error to give more information in cases of errors.
  cudaError_t cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    // Printing the cuda_error value is useful when cudaGetErrorName doesn't
    // work.
    const std::string error =
        absl::StrCat("There was an error before creating cudnn handle (",
                     cuda_error, "): ", cudaGetErrorName(cuda_error), " : ",
                     cudaGetErrorString(cuda_error));
    LOG(ERROR) << error;
    return absl::InternalError(error);
  }

  cudnnHandle_t cudnn_handle = nullptr;
  const auto status = cudnnCreate(&cudnn_handle);
  if (status == CUDNN_STATUS_SUCCESS) {
    dnn::VersionInfo source_version(CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);

    TF_ASSIGN_OR_RETURN(dnn::VersionInfo loaded_version,
                        GetLoadedCudnnVersion());
    if (!IsSourceCompatibleWithCudnnLibrary(source_version, loaded_version)) {
      const std::string error = absl::StrCat(
          "Loaded runtime CuDNN library: ", loaded_version.ToString(),
          " but source was compiled with: ", source_version.ToString(),
          ".  CuDNN library needs to have matching major version and equal or "
          "higher minor version. If using a binary install, upgrade your CuDNN "
          "library.  If building from sources, make sure the library loaded at "
          "runtime is compatible with the version specified during compile "
          "configuration.");
      LOG(ERROR) << error;
      cudnnDestroy(cudnn_handle);
      return absl::InternalError(error);
    }

    cudnn_ = std::make_unique<CudnnAccess>(cudnn_handle);

    LOG(INFO) << "Loaded cuDNN version " << cudnnGetVersion();
    return absl::OkStatus();
  }

  CHECK_EQ(cudnn_handle, nullptr);
  LOG(ERROR) << "Could not create cudnn handle: "
             << CudnnStatusToString(status);
  int64_t free, total;
  parent_->DeviceMemoryUsage(&free, &total);
  LOG(ERROR) << "Memory usage: " << free << " bytes free, " << total
             << " bytes total.";

  if (status == CUDNN_STATUS_NOT_INITIALIZED) {
    auto result = cuda::Diagnostician::FindKernelDriverVersion();
    if (!result.ok()) {
      LOG(ERROR) << "Error retrieving driver version: "
                 << cuda::DriverVersionStatusToString(result);
    } else {
      const auto& version = result.value();
      LOG(ERROR) << "Possibly insufficient driver version: "
                 << cuda::DriverVersionToString(version);
    }
  }

  return absl::InternalError(
      absl::StrCat("cudnn library could not create a handle: ",
                   CudnnStatusToString(status)));
}

void CudnnSupport::NotifyStreamDestroyed(Stream* stream) /* override */ {
  cudnn_->NotifyStreamDestroyed(stream);
}

absl::StatusOr<stream_executor::dnn::VersionInfo> CudnnSupport::GetVersion() {
  return GetLoadedCudnnVersion();
}

namespace {

// Deleter functors for cuDNN types that need to be deleted.
struct TensorDescriptorDeleter {
  void operator()(cudnnTensorDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyTensorDescriptor(descriptor));
  }
};
struct RNNDataDescriptorDeleter {
  void operator()(cudnnRNNDataDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyRNNDataDescriptor(descriptor));
  }
};
struct FilterDescriptorDeleter {
  void operator()(cudnnFilterDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyFilterDescriptor(descriptor));
  }
};
struct ConvolutionDescriptorDeleter {
  void operator()(cudnnConvolutionDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyConvolutionDescriptor(descriptor));
  }
};
struct PoolingDescriptorDeleter {
  void operator()(cudnnPoolingDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyPoolingDescriptor(descriptor));
  }
};
struct LrnDescriptorDeleter {
  void operator()(cudnnLRNDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyLRNDescriptor(descriptor));
  }
};

struct ActivationDescriptorDeleter {
  void operator()(cudnnActivationDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyActivationDescriptor(descriptor));
  }
};
struct DropoutDescriptorDeleter {
  void operator()(cudnnDropoutDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyDropoutDescriptor(descriptor));
  }
};
struct RnnDescriptorDeleter {
  void operator()(cudnnRNNDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyRNNDescriptor(descriptor));
  }
};
struct CtcLossDescriptorDeleter {
  void operator()(cudnnCTCLossDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyCTCLossDescriptor(descriptor));
  }
};

// RAII wrappers for cuDNN types.
using TensorDescriptor =
    std::unique_ptr<cudnnTensorStruct, TensorDescriptorDeleter>;
using RNNDataDescriptor =
    std::unique_ptr<cudnnRNNDataStruct, RNNDataDescriptorDeleter>;
using FilterDescriptor =
    std::unique_ptr<cudnnFilterStruct, FilterDescriptorDeleter>;
using ConvolutionDescriptor =
    std::unique_ptr<cudnnConvolutionStruct, ConvolutionDescriptorDeleter>;
using PoolingDescriptor =
    std::unique_ptr<cudnnPoolingStruct, PoolingDescriptorDeleter>;
using LrnDescriptor = std::unique_ptr<cudnnLRNStruct, LrnDescriptorDeleter>;
using ActivationDescriptor =
    std::unique_ptr<cudnnActivationStruct, ActivationDescriptorDeleter>;
using DropoutDescriptor =
    std::unique_ptr<cudnnDropoutStruct, DropoutDescriptorDeleter>;
using RnnDescriptor = std::unique_ptr<cudnnRNNStruct, RnnDescriptorDeleter>;
struct DummyType {};
using PersistentRnnPlan = std::unique_ptr<DummyType>;
using CtcLossDescriptor =
    std::unique_ptr<cudnnCTCLossStruct, CtcLossDescriptorDeleter>;

// Factory methods for cuDNN types.
TensorDescriptor CreateTensorDescriptor() {
  cudnnTensorDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreateTensorDescriptor(&result));
  return TensorDescriptor(result);
}
RNNDataDescriptor CreateRNNDataDescriptor() {
  cudnnRNNDataDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreateRNNDataDescriptor(&result));
  return RNNDataDescriptor(result);
}
FilterDescriptor CreateFilterDescriptor() {
  cudnnFilterDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreateFilterDescriptor(&result));
  return FilterDescriptor(result);
}
ConvolutionDescriptor CreateConvolutionDescriptor() {
  cudnnConvolutionDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreateConvolutionDescriptor(&result));
  return ConvolutionDescriptor(result);
}
PoolingDescriptor CreatePoolingDescriptor() {
  cudnnPoolingDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreatePoolingDescriptor(&result));
  return PoolingDescriptor(result);
}
LrnDescriptor CreateLrnDescriptor() {
  cudnnLRNDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreateLRNDescriptor(&result));
  return LrnDescriptor(result);
}
ActivationDescriptor CreateActivationDescriptor() {
  cudnnActivationDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreateActivationDescriptor(&result));
  return ActivationDescriptor(result);
}
DropoutDescriptor CreateDropoutDescriptor() {
  cudnnDropoutDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreateDropoutDescriptor(&result));
  return DropoutDescriptor(result);
}
RnnDescriptor CreateRnnDescriptor() {
  cudnnRNNDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreateRNNDescriptor(&result));
  return RnnDescriptor(result);
}
CtcLossDescriptor CreateCtcLossDescriptor() {
  cudnnCTCLossDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreateCTCLossDescriptor(&result));
  return CtcLossDescriptor(result);
}

// Turns a BatchDescriptor structure into a cudnn tensor handle within a
// scope.
class CudnnTensorDescriptor {
 public:
  CudnnTensorDescriptor(const dnn::BatchDescriptor& batch_descriptor,
                        cudnnDataType_t elem_type)
      : handle_(CreateTensorDescriptor()) {
    switch (batch_descriptor.layout()) {
      case dnn::DataLayout::kBatchYXDepth:
      case dnn::DataLayout::kBatchDepthYX: {
        const int nd = batch_descriptor.ndims() + 2;
        // cuDNN requires the strides and dims to be ordered as BDYX.
        std::vector<int64_t> strides64 =
            batch_descriptor.full_strides(dnn::DataLayout::kBatchDepthYX);
        std::vector<int64_t> dims64 =
            batch_descriptor.full_dims(dnn::DataLayout::kBatchDepthYX);

        // cuDNN requires arrays of ints.
        std::vector<int> strides(nd);
        std::vector<int> dims(nd);
        std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                       &CheckedNarrowing<int64_t, int>);
        std::transform(dims64.cbegin(), dims64.cend(), dims.begin(),
                       &CheckedNarrowing<int64_t, int>);
        CHECK_CUDNN_OK(cudnnSetTensorNdDescriptor(handle_.get(), elem_type, nd,
                                                  dims.data(), strides.data()))
            << "batch_descriptor: " << batch_descriptor.ToString();
        break;
      }
      case dnn::DataLayout::kBatchDepthYX4:
      case dnn::DataLayout::kBatchDepthYX32: {
        auto expected_elem_ty =
            batch_descriptor.layout() == dnn::DataLayout::kBatchDepthYX4
                ? CUDNN_DATA_INT8x4
                : CUDNN_DATA_INT8x32;
        CHECK_EQ(elem_type, expected_elem_ty);
        CHECK_CUDNN_OK(cudnnSetTensor4dDescriptor(
            handle_.get(), CUDNN_TENSOR_NCHW_VECT_C, elem_type,
            batch_descriptor.count(), batch_descriptor.feature_map_count(),
            batch_descriptor.height(), batch_descriptor.width()))
            << "batch_descriptor: " << batch_descriptor.ToString();
        break;
      }
      default:
        LOG(FATAL) << "Unsupported tensor format "
                   << DataLayoutString(batch_descriptor.layout());
        break;
    }
  }

  cudnnTensorDescriptor_t handle() const { return handle_.get(); }

 private:
  TensorDescriptor handle_;
};

// Turns a FilterDescriptor structure into a cudnn filter handle within a
// scope.
class CudnnFilterDescriptor {
 public:
  CudnnFilterDescriptor(const dnn::FilterDescriptor& filter_descriptor,
                        cudnnDataType_t elem_type)
      : handle_(CreateFilterDescriptor()) {
    // TODO(b/23032134): Even if the filter layout is not supported,
    // cudnnSetFilter4DDescriptor_v4 will return CUDNN_STATUS_SUCCESS because
    // it does not take layout as an input. Maybe force cuDNN by giving wrong
    // inputs intentionally?
    cudnnTensorFormat_t format;
    switch (filter_descriptor.layout()) {
      case dnn::FilterLayout::kOutputInputYX:
        format = CUDNN_TENSOR_NCHW;
        break;
      case dnn::FilterLayout::kOutputYXInput:
        format = CUDNN_TENSOR_NHWC;
        break;
      case dnn::FilterLayout::kOutputInputYX4:
      case dnn::FilterLayout::kOutputInputYX32:
      case dnn::FilterLayout::kOutputInputYX32_CudnnReordered: {
        auto expected_elem_ty =
            filter_descriptor.layout() == dnn::FilterLayout::kOutputInputYX4
                ? CUDNN_DATA_INT8x4
                : CUDNN_DATA_INT8x32;
        CHECK_EQ(elem_type, expected_elem_ty);
        format = CUDNN_TENSOR_NCHW_VECT_C;
        break;
      }
      default:
        LOG(FATAL) << "Unsupported filter format "
                   << FilterLayoutString(filter_descriptor.layout());
        break;
    }

    std::vector<int> dims(2 + filter_descriptor.ndims());
    dims[0] = filter_descriptor.output_feature_map_count();
    dims[1] = filter_descriptor.input_feature_map_count();
    absl::Span<const int64_t> spatial_dims =
        filter_descriptor.input_filter_dims();
    std::copy(spatial_dims.begin(), spatial_dims.end(), dims.begin() + 2);

    CHECK_CUDNN_OK(cudnnSetFilterNdDescriptor(handle_.get(), elem_type, format,
                                              dims.size(), dims.data()));
  }

  cudnnFilterDescriptor_t handle() const { return handle_.get(); }

 private:
  FilterDescriptor handle_;  // Owned.
};

// The errata sheet (JSON format) for marking the cudnn engines that might be
// buggy. For example, we don't want the engine 999 of forward convolution:
// R"({ "version" : 1,
//      "rules"   : [
//        { "rule_id"             : "ConvFwd_eng999",
//          "operation"           : "ConvFwd",
//          "engine"              : 999,
//          "knob"                : [],
//          "cudnn_version_start" : 8000,
//          "cudnn_version_end"   : -1
//        }
// ]})"
// We skip a non-existing eng999 in the static filter as a placeholder.
// Additionally, users can specify an additional errata JSON file via
// CUDNN_ERRATA_JSON_FILE at runtime.
// We are also excluding two flavors of ConvFwd_eng42 due to b/234183340.
// Excluding ConvFwd_Add_Add_eng32 to avoid misaligned address on A100,
// see b/279920986.
const json* CudnnExecutionPlanEngineFilterStatic() {
  static absl::string_view filter_str = R"({
      "version" : 1,
        "rules"   : [
          { "rule_id"             : "ConvFwd_eng999",
            "operation"           : "ConvFwd",
            "engine"              : 999,
            "knob"                : [],
            "cudnn_version_start" : 8000,
            "cudnn_version_end"   : -1
          },
          { "rule_id"             : "ConvFwd_eng42_k2=2_k4=3_k5=0_k6=0_k7=0",
            "operation"           : "ConvFwd",
            "engine"              : 42,
            "knob"                :
            {
                                    "k2" : "2",
                                    "k4" : "3",
                                    "k5" : "0",
                                    "k6" : "0",
                                    "k7" : "0"
            },
            "cudnn_version_start" : 8000,
            "cudnn_version_end"   : -1
          },
          { "rule_id"             : "ConvFwd_eng42_k2=1_k4=3_k5=1_k6=0_k7=0",
            "operation"           : "ConvFwd",
            "engine"              : 42,
            "knob"                :
            {
                                    "k2" : "1",
                                    "k4" : "3",
                                    "k5" : "1",
                                    "k6" : "0",
                                    "k7" : "0"
            },
            "cudnn_version_start" : 8000,
            "cudnn_version_end"   : -1
          },
          { "rule_id"             : "ConvFwd_Add_Add_eng34_k24=11",
            "operation"           : "ConvFwd_Add_Add",
            "engine"              : 34,
            "cudnn_version_start" : 8700,
            "cudnn_version_end"   : 8900
          },
          { "rule_id"             : "ConvFwd_Add_Add_ReluFwd_eng15_k5=1_k6=0_k7=1_k10=1",
            "operation"           : "ConvFwd_Add_Add_ReluFwd",
            "engine"              : 15,
            "knob"                : ["k5=1", "k6=0", "k7=1", "k10=1"],
            "cudnn_version_start" : 8900,
            "cudnn_version_end"   : 8902,
            "comment"             : "b/281585171"
          },
          { "rule_id"             : "ConvFwd_Add_Add_eng15_k5=1_k6=0_k7=1_k10=1",
            "operation"           : "ConvFwd_Add_Add",
            "engine"              : 15,
            "knob"                : ["k5=1", "k6=0", "k7=1", "k10=1"],
            "cudnn_version_start" : 8900,
            "cudnn_version_end"   : 8902,
            "comment"             : "b/281887114"
          }
      ]})";
  static const json* const json_handle = new json(json::parse(filter_str));
  return json_handle;
}

const json* CudnnExecutionPlanEngineFilterRuntime() {
  static const json* json_handle = []() -> const json* {
    json j;
    if (cudnn_frontend::load_from_config(j, "")) {
      return new json(j);
    }
    return nullptr;
  }();
  return json_handle;
}

// A helper function to decide whether to use
// CUDNN_BATCHNORM_SPATIAL_PERSISTENT in batchnorm. This mode can be faster in
// some tasks because an optimized path may be selected for CUDNN_DATA_FLOAT
// and CUDNN_DATA_HALF data types, compute capability 6.0 or higher. The
// reason we set it to false by default is that this mode may use scaled
// atomic integer reduction that may cause a numerical overflow for certain
// input data range.
// TODO(yangzihao): Use autotune to choose between this mode and
// CUDNN_BATCHNORM_SPATIAL mode.
bool BatchnormSpatialPersistentEnabled() {
  static bool is_enabled = [] {
    bool is_enabled = false;
    TF_CHECK_OK(
        tsl::ReadBoolFromEnvVar("TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT",
                                /*default_val=*/false, &is_enabled));
    return is_enabled;
  }();
  return is_enabled;
}

// A helper function to decide whether to force the default conv algorithm.
bool ConvUseDefaultAlgorithm() {
  static bool use_default = [] {
    bool use_default = false;
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_USE_DEFAULT_CONV_ALGO",
                                        /*default_val=*/false, &use_default));
    return use_default;
  }();
  return use_default;
}

// Turns a ConvolutionDescriptor structure into a cudnn convolution handle
// within a scope.
class CudnnConvolutionDescriptor {
 public:
  CudnnConvolutionDescriptor(
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      cudnnDataType_t data_type)
      : handle_(CreateConvolutionDescriptor()) {
    absl::Span<const int64_t> strides64 = convolution_descriptor.strides();
    absl::Span<const int64_t> padding64 = convolution_descriptor.padding();
    absl::Span<const int64_t> dilations64 = convolution_descriptor.dilations();
    CHECK_NE(convolution_descriptor.pad_alignment(),
             dnn::PadAlignment::kTensorFlowPadding)
        << "TensorFlow padding alignment is not supported.";

    // cuDNN requires arrays of ints.
    std::vector<int> strides(convolution_descriptor.ndims());
    std::vector<int> padding(convolution_descriptor.ndims());
    std::vector<int> dilations(convolution_descriptor.ndims());
    std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                   &CheckedNarrowing<int64_t, int>);
    std::transform(padding64.cbegin(), padding64.cend(), padding.begin(),
                   &CheckedNarrowing<int64_t, int>);
    // TODO(yangzihao): Test with negative dilation to make sure that cudnn
    // doesn't crash.
    std::transform(dilations64.cbegin(), dilations64.cend(), dilations.begin(),
                   &CheckedNarrowing<int64_t, int>);

    CHECK_CUDNN_OK(cudnnSetConvolutionNdDescriptor(
        handle_.get(), convolution_descriptor.ndims(), padding.data(),
        strides.data(), dilations.data(),
        convolution_descriptor.convolution_not_crosscorr()
            ? CUDNN_CONVOLUTION
            : CUDNN_CROSS_CORRELATION,
        data_type));

    VLOG(2) << "Requesting grouped convolution: "
            << convolution_descriptor.group_count();
    CHECK_CUDNN_OK(cudnnSetConvolutionGroupCount(
        handle_.get(), convolution_descriptor.group_count()));
  }

  void set_use_tensor_op_math(bool use_tensor_op_math) {
    cudnnMathType_t math_type =
        (use_tensor_op_math ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH);
    CHECK_CUDNN_OK(cudnnSetConvolutionMathType(handle_.get(), math_type));
  }

  cudnnConvolutionDescriptor_t handle() const { return handle_.get(); }

 private:
  ConvolutionDescriptor handle_;  // Owned.
};

// A helper function to query if a CudnnConvolutionDescriptor has tensor_op_math
// set
static bool IsTensorMathOpSet(const CudnnConvolutionDescriptor& conv) {
  cudnnMathType_t math_type;
  CHECK_CUDNN_OK(cudnnGetConvolutionMathType(conv.handle(), &math_type));
  return math_type != CUDNN_FMA_MATH;
}

static bool TensorOpMathAvailable(
    CudaComputeCapability cuda_compute_capability) {
  return cuda_compute_capability.IsAtLeast(7);
}

static bool IsTensorMathEnabled(CudaComputeCapability cuda_compute_capability,
                                dnn::DataType input_type, bool allow_tf32) {
  if (!TensorOpMathAvailable(cuda_compute_capability)) {
    return false;
  }
  if (input_type == dnn::DataType::kFloat) {
    if (!allow_tf32 || !tsl::tensor_float_32_execution_enabled()) {
      return false;
    }
  }
  return true;
}

static bool IsTensorMathEnabled(Stream* stream, dnn::DataType input_type,
                                bool allow_tf32) {
  return IsTensorMathEnabled(stream->GetCudaComputeCapability(), input_type,
                             allow_tf32);
}

// Turns a PoolingDescriptor structure into a cudnn pooling descriptor handle
// within a scope.
class CudnnPoolingDescriptor {
 public:
  explicit CudnnPoolingDescriptor(
      const dnn::PoolingDescriptor& pooling_descriptor,
      const NumericOptions& numeric_options)
      : handle_(CreatePoolingDescriptor()) {
    absl::Span<const int64_t> strides64 = pooling_descriptor.strides();
    absl::Span<const int64_t> padding64 = pooling_descriptor.padding();
    absl::Span<const int64_t> shape64 = pooling_descriptor.window();

    const int nd = pooling_descriptor.ndims();
    std::vector<int> shape(nd);
    std::vector<int> padding(nd);
    std::vector<int> strides(nd);
    std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                   &CheckedNarrowing<int64_t, int>);
    std::transform(padding64.cbegin(), padding64.cend(), padding.begin(),
                   &CheckedNarrowing<int64_t, int>);
    std::transform(shape64.cbegin(), shape64.cend(), shape.begin(),
                   &CheckedNarrowing<int64_t, int>);
    bool propagate_nans = pooling_descriptor.propagate_nans();
    const auto cudnn_max_pooling_mode = numeric_options.require_determinism
                                            ? CUDNN_POOLING_MAX_DETERMINISTIC
                                            : CUDNN_POOLING_MAX;
    CHECK_CUDNN_OK(cudnnSetPoolingNdDescriptor(
        handle_.get(),
        (pooling_descriptor.mode() == dnn::PoolingMode::kMaximum
             ? cudnn_max_pooling_mode
             : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING),
        propagate_nans ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN, nd,
        shape.data(), padding.data(), strides.data()));
  }

  cudnnPoolingDescriptor_t handle() const { return handle_.get(); }

 private:
  PoolingDescriptor handle_;  // Owned.

  CudnnPoolingDescriptor(const CudnnPoolingDescriptor&) = delete;
  void operator=(const CudnnPoolingDescriptor&) = delete;
};

// Turns a NormalizeDescriptor structure into a cudnn LRN descriptor handle.
class CudnnNormalizeDescriptor {
 public:
  explicit CudnnNormalizeDescriptor(
      const dnn::NormalizeDescriptor& normalize_descriptor)
      : handle_(CreateLrnDescriptor()) {
    // The range specifies that the indices in the closed range
    // [i - range, i + range] should be included in the normalization for index
    // i. The lrnN value is the total number of elements in the range, so
    // lrnN = 2*range + 1.
    unsigned lrnN = 2 * normalize_descriptor.range() + 1;

    // Note that SE defines the normalization operation as
    //
    //  U_i = V_i / ((bias +  alpha      * (sum_j V_j^2)) ^ beta)
    //
    // but cuDNN defines it as
    //
    //  U_i = V_i / ((bias + (alpha / n) * (sum_j V_j^2)) ^ beta)
    //
    // i.e. there is a factor of n difference between the meaning of the alphas
    // in the two contexts. The cuDNN alpha is n times the SE alpha.
    double lrnAlpha = lrnN * normalize_descriptor.alpha();

    double lrnBeta = normalize_descriptor.beta();
    double lrnK = normalize_descriptor.bias();
    CHECK_CUDNN_OK(
        cudnnSetLRNDescriptor(handle_.get(), lrnN, lrnAlpha, lrnBeta, lrnK));
  }

  cudnnLRNDescriptor_t handle() const { return handle_.get(); }

 private:
  LrnDescriptor handle_;  // Owned.

  CudnnNormalizeDescriptor(const CudnnNormalizeDescriptor&) = delete;
  void operator=(const CudnnNormalizeDescriptor&) = delete;
};

// Turns a ActivationDescriptor structure into a cudnn activation
// descriptor handle within a scope.
class CudnnActivationDescriptor {
 public:
  CudnnActivationDescriptor(dnn::ActivationMode activation_mode,
                            cudnnNanPropagation_t nan_propagation,
                            double value_max)
      : handle_(CreateActivationDescriptor()) {
    double relu_ceiling = 0.0;
    cudnnActivationMode_t mode;
    switch (activation_mode) {
      case dnn::ActivationMode::kNone:
        mode = CUDNN_ACTIVATION_IDENTITY;
        break;
      case dnn::ActivationMode::kRelu6:
        relu_ceiling = 6.0;
        mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        break;
      case dnn::ActivationMode::kReluX:
        relu_ceiling = value_max;
        mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        break;
      case dnn::ActivationMode::kRelu:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case dnn::ActivationMode::kSigmoid:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case dnn::ActivationMode::kTanh:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      default:
        LOG(FATAL) << "unrecognized activation mode: "
                   << static_cast<int>(activation_mode);
    }

    CHECK_CUDNN_OK(cudnnSetActivationDescriptor(handle_.get(), mode,
                                                nan_propagation, relu_ceiling));
  }

  cudnnActivationDescriptor_t handle() const { return handle_.get(); }

 private:
  ActivationDescriptor handle_;  // Owned.
};

cudnn_frontend::DataType_t ToCudnnFrontendDataType(
    dnn::DataType data_type,
    dnn::DataLayout data_layout = dnn::DataLayout::kBatchDepthYX) {
  switch (data_type) {
    case dnn::DataType::kFloat:
      return cudnn_frontend::DataType_t::FLOAT;
    case dnn::DataType::kDouble:
      return cudnn_frontend::DataType_t::DOUBLE;
    case dnn::DataType::kHalf:
      return cudnn_frontend::DataType_t::HALF;
    case dnn::DataType::kInt8:
      switch (data_layout) {
        case dnn::DataLayout::kBatchDepthYX4:
          return cudnn_frontend::DataType_t::INT8x4;
        case dnn::DataLayout::kBatchDepthYX32:
          return cudnn_frontend::DataType_t::INT8x32;
        default:
          return cudnn_frontend::DataType_t::INT8;
      }
    case dnn::DataType::kInt32:
      return cudnn_frontend::DataType_t::INT32;
    case dnn::DataType::kInt64:
      return cudnn_frontend::DataType_t::INT64;
    case dnn::DataType::kBF16:
      return cudnn_frontend::DataType_t::BFLOAT16;
    case dnn::DataType::kF8E4M3FN:
      return cudnn_frontend::DataType_t::FP8_E4M3;
    case dnn::DataType::kF8E5M2:
      return cudnn_frontend::DataType_t::FP8_E5M2;
#if CUDNN_VERSION >= 90700
    case dnn::DataType::kF4E2M1FN:
      return cudnn_frontend::DataType_t::FP4_E2M1;
    case dnn::DataType::kF8E8M0FNU:
      return cudnn_frontend::DataType_t::FP8_E8M0;
#endif
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

cudnnDataType_t ToCudnnDataType(
    dnn::DataType data_type,
    dnn::DataLayout data_layout = dnn::DataLayout::kBatchDepthYX) {
  cudnnDataType_t type;
  CHECK_CUDNN_OK(cudnn_frontend::detail::convert_to_cudnn_type(
      ToCudnnFrontendDataType(data_type, data_layout), type));
  return type;
}

cudnn_frontend::DataType_t ToCudnnFrontendDataType(
    dnn::DataType data_type, dnn::FilterLayout filter_layout) {
  if (data_type == dnn::DataType::kInt8 &&
      filter_layout == dnn::FilterLayout::kOutputInputYX4) {
    return cudnn_frontend::DataType_t::INT8x4;
  }
  if (data_type == dnn::DataType::kInt8 &&
      (filter_layout == dnn::FilterLayout::kOutputInputYX32 ||
       filter_layout == dnn::FilterLayout::kOutputInputYX32_CudnnReordered)) {
    return cudnn_frontend::DataType_t::INT8x32;
  }
  return ToCudnnFrontendDataType(data_type);
}

cudnnDataType_t ToCudnnDataType(dnn::DataType data_type,
                                dnn::FilterLayout filter_layout) {
  cudnnDataType_t type;
  CHECK_CUDNN_OK(cudnn_frontend::detail::convert_to_cudnn_type(
      ToCudnnFrontendDataType(data_type, filter_layout), type));
  return type;
}

template <typename T>
cudnn_frontend::DataType_t GetCudnnFrontendDataType(
    dnn::DataLayout data_layout = dnn::DataLayout::kBatchDepthYX) {
  return ToCudnnFrontendDataType(dnn::ToDataType<T>::value, data_layout);
}

template <typename T>
cudnnDataType_t GetCudnnDataType(
    dnn::DataLayout data_layout = dnn::DataLayout::kBatchDepthYX) {
  return ToCudnnDataType(dnn::ToDataType<T>::value, data_layout);
}

template <typename T>
cudnn_frontend::DataType_t GetCudnnFrontendDataType(
    dnn::FilterLayout filter_layout) {
  return ToCudnnFrontendDataType(dnn::ToDataType<T>::value, filter_layout);
}

template <typename T>
cudnnDataType_t GetCudnnDataType(dnn::FilterLayout filter_layout) {
  return ToCudnnDataType(dnn::ToDataType<T>::value, filter_layout);
}

cudnnRNNInputMode_t ToCudnnRnnInputMode(dnn::RnnInputMode input_mode) {
  switch (input_mode) {
    case dnn::RnnInputMode::kRnnLinearSkip:
    case dnn::RnnInputMode::kRnnSkipInput:
      return static_cast<cudnnRNNInputMode_t>(input_mode);
    default:
      LOG(FATAL) << "Invalid RNN input mode: " << static_cast<int>(input_mode);
  }
}

cudnnDirectionMode_t ToCudnnRnnDirectionMode(
    dnn::RnnDirectionMode direction_mode) {
  switch (direction_mode) {
    case dnn::RnnDirectionMode::kRnnUnidirectional:
    case dnn::RnnDirectionMode::kRnnBidirectional:
      return static_cast<cudnnDirectionMode_t>(direction_mode);
    default:
      LOG(FATAL) << "Invalid RNN direction mode: "
                 << static_cast<int>(direction_mode);
  }
}

cudnnRNNMode_t ToCudnnRnnMode(dnn::RnnMode rnn_mode) {
  switch (rnn_mode) {
    case dnn::RnnMode::kRnnRelu:
    case dnn::RnnMode::kRnnTanh:
    case dnn::RnnMode::kRnnLstm:
    case dnn::RnnMode::kRnnGru:
      return static_cast<cudnnRNNMode_t>(rnn_mode);
    default:
      LOG(FATAL) << "Invalid RNN Mode: " << static_cast<int>(rnn_mode);
  }
}

int CudnnDataTypeToByteSize(cudnnDataType_t data_type) {
  switch (data_type) {
    case CUDNN_DATA_FLOAT:
      return sizeof(float);
    case CUDNN_DATA_DOUBLE:
      return sizeof(double);
    case CUDNN_DATA_HALF:
      return sizeof(Eigen::half);
    case CUDNN_DATA_BFLOAT16:
      return sizeof(Eigen::bfloat16);
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

class CudnnDropoutDescriptor {
  explicit CudnnDropoutDescriptor(DropoutDescriptor handle)
      : handle_(std::move(handle)) {}

 public:
  CudnnDropoutDescriptor(CudnnDropoutDescriptor&&) = default;

  static absl::StatusOr<CudnnDropoutDescriptor> Create(
      const CudnnHandle& cudnn, float dropout, uint64_t seed,
      ScratchAllocator* state_allocator) {
    DropoutDescriptor handle = CreateDropoutDescriptor();

    if (dropout == 0.0f) {
      // Return 'empty' dropout descriptor.
      return CudnnDropoutDescriptor(std::move(handle));
    }

    DeviceMemory<uint8_t> state_memory;
    if (state_allocator) {
      size_t state_sizes_in_bytes = 0;
      RETURN_IF_CUDNN_ERROR(
          cudnnDropoutGetStatesSize(cudnn.handle(), &state_sizes_in_bytes));
      TF_ASSIGN_OR_RETURN(state_memory,
                          state_allocator->AllocateBytes(state_sizes_in_bytes));
    }
    RETURN_IF_CUDNN_ERROR(cudnnSetDropoutDescriptor(
        handle.get(), cudnn.handle(), dropout, state_memory.opaque(),
        state_memory.size(), seed));

    return CudnnDropoutDescriptor(std::move(handle));
  }

  cudnnDropoutDescriptor_t handle() const { return handle_.get(); }

 private:
  DropoutDescriptor handle_;  // Owned.
  CudnnDropoutDescriptor(const CudnnDropoutDescriptor&) = delete;
  void operator=(const CudnnDropoutDescriptor&) = delete;
};

class CudnnRnnParamsDescriptor {
  typedef dnn::RnnDescriptor::ParamsRegions ParamsRegions;

  CudnnRnnParamsDescriptor(FilterDescriptor handle,
                           int64_t params_size_in_bytes, ParamsRegions weights,
                           ParamsRegions biases)
      : handle_(std::move(handle)),
        params_size_in_bytes_(params_size_in_bytes),
        weights_(std::move(weights)),
        biases_(std::move(biases)) {}

 public:
  CudnnRnnParamsDescriptor(CudnnRnnParamsDescriptor&&) = default;

  static absl::StatusOr<CudnnRnnParamsDescriptor> Create(
      const CudnnHandle& cudnn, int input_size, cudnnDataType_t data_type,
      cudnnRNNDescriptor_t rnn_desc, cudnnRNNMode_t rnn_mode,
      cudnnDirectionMode_t direction_mode, int num_layers);

  cudnnFilterDescriptor_t handle() const { return handle_.get(); }
  int64_t params_size_in_bytes() const { return params_size_in_bytes_; }
  ParamsRegions params_weights() const { return weights_; }
  ParamsRegions params_biases() const { return biases_; }

 private:
  FilterDescriptor handle_;
  int64_t params_size_in_bytes_;
  ParamsRegions weights_;
  ParamsRegions biases_;
  CudnnRnnParamsDescriptor(const CudnnRnnParamsDescriptor&) = delete;
  void operator=(const CudnnRnnParamsDescriptor&) = delete;
};

}  // namespace

class CudnnRnnDescriptor : public dnn::RnnDescriptor {
  CudnnRnnDescriptor(const CudnnHandle& cudnn, gpu::RnnDescriptor rnn_desc,
                     PersistentRnnPlan rnn_plan, int num_layers,
                     int hidden_size, int input_size, int cell_size,
                     int batch_size, cudnnRNNInputMode_t input_mode,
                     cudnnDirectionMode_t direction_mode,
                     cudnnRNNMode_t rnn_mode, cudnnDataType_t data_type,
                     cudnnDataType_t compute_type,
                     const dnn::AlgorithmConfig& algorithm_config,
                     CudnnDropoutDescriptor dropout_desc,
                     CudnnRnnParamsDescriptor params_desc)
      : rnn_desc_(std::move(rnn_desc)),
        rnn_plan_(std::move(rnn_plan)),
        num_layers_(num_layers),
        hidden_size_(hidden_size),
        input_size_(input_size),
        cell_size_(cell_size),
        batch_size_(batch_size),
        rnn_algo_(ToCudnnRNNAlgo(algorithm_config.algorithm())),
        input_mode_(input_mode),
        direction_mode_(direction_mode),
        rnn_mode_(rnn_mode),
        data_type_(data_type),
        compute_type_(compute_type),
        algorithm_config_(algorithm_config),
        dropout_desc_(std::move(dropout_desc)),
        params_desc_(std::move(params_desc)) {}

 public:
  CudnnRnnDescriptor(CudnnRnnDescriptor&& other) = default;

  static absl::StatusOr<CudnnRnnDescriptor> Create(
      const CudnnHandle& cudnn, int num_layers, int hidden_size, int input_size,
      int cell_size, int batch_size, cudnnRNNInputMode_t input_mode,
      cudnnDirectionMode_t direction_mode, cudnnRNNMode_t rnn_mode,
      cudnnDataType_t data_type, cudnnDataType_t compute_type,
      const dnn::AlgorithmConfig& algorithm_config,
      const NumericOptions& numeric_options, float dropout, uint64_t seed,
      ScratchAllocator* state_allocator, bool use_padded_io) {
    TF_ASSIGN_OR_RETURN(
        CudnnDropoutDescriptor dropout_desc,
        CudnnDropoutDescriptor::Create(cudnn, dropout, seed, state_allocator));

    gpu::RnnDescriptor rnn_desc = CreateRnnDescriptor();
    cudnnRNNAlgo_t rnn_algo = ToCudnnRNNAlgo(algorithm_config.algorithm());

    // TODO: allow the user to choose an algorithm.
    auto proj_size = hidden_size;
    hidden_size = std::max(hidden_size, cell_size);

    // Require explicit algorithm config to enable tensor cores. Some configs
    // return CUDNN_NOT_SUPPORTED when tensor ops are enabled (which is against
    // the idiom that enabling tensor ops is only a hint: see nvbugs/2172799).
    // We can only reasonably expect the user to handle the subsequent failure
    // in profile mode, which is run with algorithms returned from
    // GetRnnAlgorithms() (which are non-default and explicitly set whether to
    // use tensor ops). CuDNN 7.2.1 fixed this issue.
    // TODO(csigg): Minimal support cuDNN version is 7.3, clean up.
    bool allow_tensor_ops = data_type == CUDNN_DATA_HALF;
    if (data_type == CUDNN_DATA_FLOAT)
      allow_tensor_ops = numeric_options.allow_tf32 &&
                         tsl::tensor_float_32_execution_enabled();
    bool use_tensor_ops =
        algorithm_config.algorithm().has_value()
            ? algorithm_config.algorithm()->tensor_ops_enabled()
            : allow_tensor_ops;
    if (use_tensor_ops && !allow_tensor_ops) {
      return absl::InvalidArgumentError(
          "Algo requests disallowed tensor op evaluation.");
    }

    cudnnMathType_t math_type =
        use_tensor_ops ? CUDNN_TENSOR_OP_MATH : CUDNN_FMA_MATH;

    cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_DOUBLE_BIAS;
    uint32_t aux_flags = 0;
    if (use_padded_io) aux_flags |= CUDNN_RNN_PADDED_IO_ENABLED;
    RETURN_IF_CUDNN_ERROR(cudnnSetRNNDescriptor_v8(
        /*rnnDesc=*/rnn_desc.get(), /*algo=*/rnn_algo, /*cellMode=*/rnn_mode,
        /*biasMode=*/bias_mode, /*dirMode=*/direction_mode,
        /*inputMode=*/input_mode,
        /*dataType=*/data_type, /*mathPrec=*/compute_type,
        /*mathType=*/math_type,
        /*inputSize=*/input_size,
        /*hiddenSize=*/hidden_size, /*projSize=*/proj_size,
        /*numLayers=*/num_layers,
        /*dropoutDesc=*/dropout_desc.handle(),
        /*auxFlags=*/aux_flags));

    absl::StatusOr<PersistentRnnPlan> rnn_plan_wrapper;
    PersistentRnnPlan rnn_plan;
    if (rnn_algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
      CHECK_GE(batch_size, 0);
      RETURN_IF_CUDNN_ERROR(
          cudnnBuildRNNDynamic(cudnn.handle(), rnn_desc.get(), batch_size));
    }

    // Create the params handle.
    // TODO(kaixih@nvidia.com): Should be removed when cudnnRNNForward*** and
    // cudnnRNNForward***Ex are removed from the codebase, since the new API
    // doesn't need param descriptors any more.
    TF_ASSIGN_OR_RETURN(auto params_desc,
                        CudnnRnnParamsDescriptor::Create(
                            cudnn, input_size, data_type, rnn_desc.get(),
                            rnn_mode, direction_mode, num_layers));

    return CudnnRnnDescriptor(cudnn, std::move(rnn_desc), std::move(rnn_plan),
                              num_layers, hidden_size, input_size, cell_size,
                              batch_size, input_mode, direction_mode, rnn_mode,
                              data_type, compute_type, algorithm_config,
                              std::move(dropout_desc), std::move(params_desc));
  }

  cudnnRNNDescriptor_t handle() const { return rnn_desc_.get(); }
  int num_layers() const { return num_layers_; }
  int hidden_size() const { return hidden_size_; }
  int input_size() const { return input_size_; }
  int cell_size() const { return cell_size_; }
  int batch_size() const { return batch_size_; }
  cudnnRNNInputMode_t input_mode() const { return input_mode_; }
  cudnnDirectionMode_t direction_mode() const { return direction_mode_; }
  cudnnRNNMode_t rnn_mode() const { return rnn_mode_; }
  cudnnDataType_t data_type() const { return data_type_; }
  cudnnDataType_t compute_type() const { return compute_type_; }
  const dnn::AlgorithmConfig& algorithm_config() const {
    return algorithm_config_;
  }
  int64_t ParamsSizeInBytes() const override {
    return params_desc_.params_size_in_bytes();
  }
  cudnnFilterDescriptor_t params_handle() const {
    return params_desc_.handle();
  }
  ParamsRegions ParamsWeightRegions() const override {
    return params_desc_.params_weights();
  }
  ParamsRegions ParamsBiasRegions() const override {
    return params_desc_.params_biases();
  }

 private:
  gpu::RnnDescriptor rnn_desc_;
  PersistentRnnPlan rnn_plan_;
  int num_layers_;
  int hidden_size_;
  int input_size_;
  // cell_size_ is the size of cell state, which will be different from
  // hidden_size_ if the projection is used.
  int cell_size_;
  // batch_size_ is set to -1 when not using CUDNN_RNN_ALGO_PERSIST_DYNAMIC
  // algorithm.
  int batch_size_;
  cudnnRNNAlgo_t rnn_algo_;
  cudnnRNNInputMode_t input_mode_;
  cudnnDirectionMode_t direction_mode_;
  cudnnRNNMode_t rnn_mode_;
  cudnnDataType_t data_type_;
  cudnnDataType_t compute_type_;
  dnn::AlgorithmConfig algorithm_config_;
  CudnnDropoutDescriptor dropout_desc_;
  CudnnRnnParamsDescriptor params_desc_;
  CudnnRnnDescriptor(const CudnnRnnDescriptor&) = delete;
  void operator=(const CudnnRnnDescriptor&) = delete;
};

class CudnnCtcLossDescriptor {
 public:
  explicit CudnnCtcLossDescriptor(cudnnDataType_t data_type)
      : handle_(CreateCtcLossDescriptor()) {
    CHECK_CUDNN_OK(cudnnSetCTCLossDescriptorEx(
        /*ctcLossDesc=*/handle_.get(),
        /*compType=*/data_type,
        /*normMode=*/CUDNN_LOSS_NORMALIZATION_SOFTMAX,
        /*gradMode=*/CUDNN_NOT_PROPAGATE_NAN));
  }

  cudnnCTCLossDescriptor_t handle() const { return handle_.get(); }

 private:
  CtcLossDescriptor handle_;  // Owned

  CudnnCtcLossDescriptor(const CudnnCtcLossDescriptor&) = delete;
  void operator=(const CudnnCtcLossDescriptor&) = delete;
};

namespace {

// Check if the LSTM projection is used. If yes, an additional weight matrix
// (projection matrix) will be fetched to the 'weights'. Otherwise, nothing will
// be done.
absl::Status CheckAndFetchProjectionWeights(
    const CudnnHandle& cudnn, cudnnRNNDescriptor_t rnn_desc, const int layer,
    const TensorDescriptor& input_desc, const FilterDescriptor& filter_desc,
    int64_t params_size_in_bytes, const FilterDescriptor& region_desc_handle,
    dnn::RnnDescriptor::ParamsRegions* weights) {
  int hidden_size_v;
  int num_layers_v;
  cudnnDropoutDescriptor_t dropout_desc;
  cudnnRNNInputMode_t input_mode;
  cudnnDirectionMode_t direction;
  cudnnRNNMode_t mode;
  cudnnRNNAlgo_t algo;
  cudnnDataType_t data_type;
  int rec_proj_size_v;
  RETURN_IF_CUDNN_ERROR(cudnnGetRNNDescriptor_v8(
      /*rnnDesc=*/rnn_desc,
      /*algo=*/&algo,
      /*cellMode=*/&mode,
      /*biasMode=*/nullptr,
      /*dirMode=*/&direction,
      /*inputMode=*/&input_mode,
      /*dataType=*/nullptr,
      /*mathPrec=*/&data_type,
      /*mathType=*/nullptr,
      /*inputSize=*/nullptr,
      /*hiddenSize=*/&hidden_size_v,
      /*projSize=*/&rec_proj_size_v,
      /*numLayers=*/&num_layers_v,
      /*dropoutDesc=*/&dropout_desc,
      /*auxFlags=*/nullptr));
  if (rec_proj_size_v != hidden_size_v) {
    int region_id = 8;
    void* b_ptr = nullptr;
    void* m_ptr = nullptr;
    void* w_ptr = nullptr;
    TensorDescriptor m_region_desc_handle = CreateTensorDescriptor();
    TensorDescriptor b_region_desc_handle = CreateTensorDescriptor();
    RETURN_IF_CUDNN_ERROR(cudnnGetRNNWeightParams(
        /*handle=*/cudnn.handle(),
        /*rnnDesc=*/rnn_desc,
        /*pseudoLayer=*/layer,
        /*weightSpaceSize=*/params_size_in_bytes,
        /*weightSpace=*/w_ptr,
        /*linLayerID=*/region_id,
        /*mDesc=*/m_region_desc_handle.get(),
        /*mAddr=*/&m_ptr,
        /*bDesc=*/b_region_desc_handle.get(),
        /*bAddr=*/&b_ptr));
    int dims[] = {1, 1, 1};
    int strides[] = {1, 1, 1};
    cudnnDataType_t data_type;
    int n_dims;
    RETURN_IF_CUDNN_ERROR(cudnnGetTensorNdDescriptor(
        /*tensorDesc=*/m_region_desc_handle.get(),
        /*nbDimsRequested=*/sizeof(dims) / sizeof(dims[0]),
        /*dataType=*/&data_type,
        /*nbDims=*/&n_dims,
        /*dimA=*/dims,
        /*strideA*/ strides));
    int64_t size =
        dims[0] * dims[1] * dims[2] * CudnnDataTypeToByteSize(data_type);
    int64_t offset = static_cast<char*>(m_ptr) - static_cast<char*>(w_ptr);
    dnn::RnnDescriptor::ParamsRegion region = {static_cast<int64_t>(offset),
                                               size};
    weights->push_back(region);
  }
  return absl::OkStatus();
}

absl::StatusOr<CudnnRnnParamsDescriptor> CudnnRnnParamsDescriptor::Create(
    const CudnnHandle& cudnn, int input_size, cudnnDataType_t data_type,
    cudnnRNNDescriptor_t rnn_desc, cudnnRNNMode_t rnn_mode,
    cudnnDirectionMode_t direction_mode, int num_layers) {
  // Query the params size.
  TensorDescriptor input_desc = CreateTensorDescriptor();
  int tensor_dims[] = {1, input_size, 1};
  int strides[] = {tensor_dims[1] * tensor_dims[2], tensor_dims[2], 1};
  RETURN_IF_CUDNN_ERROR(cudnnSetTensorNdDescriptor(
      /*tensorDesc=*/input_desc.get(), /*dataType=*/data_type,
      /*nbDims=*/sizeof(tensor_dims) / sizeof(tensor_dims[0]),
      /*dimA=*/tensor_dims,
      /*strideA=*/strides));

  size_t params_size = 0;
  RETURN_IF_CUDNN_ERROR(cudnnGetRNNWeightSpaceSize(
      /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc,
      /*weightSpaceSize=*/&params_size));
  int64_t params_size_in_bytes = static_cast<int64_t>(params_size);

  FilterDescriptor filter_desc = CreateFilterDescriptor();
  int64_t filter_dim0 =
      params_size_in_bytes / CudnnDataTypeToByteSize(data_type);
  int filter_dims[] = {static_cast<int>(filter_dim0), 1, 1};
  RETURN_IF_CUDNN_ERROR(cudnnSetFilterNdDescriptor(
      /*filterDesc=*/filter_desc.get(), /*dataType=*/data_type,
      /*format=*/CUDNN_TENSOR_NCHW,
      /*nbDims=*/sizeof(filter_dims) / sizeof(filter_dims[0]),
      /*filterDimA=*/filter_dims));

  // Create the weights and biases into the params buffer
  int region_count_per_layer = [&] {
    switch (rnn_mode) {
      case CUDNN_RNN_RELU:
      case CUDNN_RNN_TANH:
        return 2;
      case CUDNN_LSTM:
        return 8;
      case CUDNN_GRU:
        return 6;
      default:
        LOG(FATAL) << "Invalid RNN Mode: " << static_cast<int>(rnn_mode);
        return 0;
    }
  }();

  FilterDescriptor region_desc_handle = CreateFilterDescriptor();
  const int layer_count =
      direction_mode == CUDNN_UNIDIRECTIONAL ? num_layers : 2 * num_layers;

  ParamsRegions weights;
  ParamsRegions biases;

  for (int layer = 0; layer < layer_count; layer++) {
    for (int region = 0; region < region_count_per_layer; region++) {
      void* m_ptr = nullptr;
      void* b_ptr = nullptr;
      void* w_ptr = nullptr;
      TensorDescriptor m_region_desc_handle = CreateTensorDescriptor();
      TensorDescriptor b_region_desc_handle = CreateTensorDescriptor();
      RETURN_IF_CUDNN_ERROR(cudnnGetRNNWeightParams(
          /*handle=*/cudnn.handle(),
          /*rnnDesc=*/rnn_desc,
          /*pseudoLayer=*/layer,
          /*weightsSize=*/params_size_in_bytes,
          /*weights=*/&w_ptr,
          /*linID=*/region,
          /*mDesc=*/m_region_desc_handle.get(),
          /*mAddr=*/&m_ptr,
          /*bDesc=*/b_region_desc_handle.get(),
          /*bAddr=*/&b_ptr));

      int dims[] = {1, 1, 1};
      int strides[] = {1, 1, 1};
      cudnnDataType_t data_type;
      int n_dims;
      auto get_size =
          [&](const TensorDescriptor& tensor_desc) -> absl::StatusOr<int64_t> {
        RETURN_IF_CUDNN_ERROR(cudnnGetTensorNdDescriptor(
            /*tensorDesc=*/m_region_desc_handle.get(),
            /*nbDimsRequested=*/sizeof(dims) / sizeof(dims[0]),
            /*dataType=*/&data_type,
            /*nbDims=*/&n_dims,
            /*dimA=*/dims,
            /*strideA*/ strides));
        int64_t size =
            dims[0] * dims[1] * dims[2] * CudnnDataTypeToByteSize(data_type);
        return size;
      };
      TF_ASSIGN_OR_RETURN(int64_t m_size, get_size(m_region_desc_handle));
      int64_t m_offset = static_cast<char*>(m_ptr) - static_cast<char*>(w_ptr);
      dnn::RnnDescriptor::ParamsRegion m_region = {m_offset, m_size};
      weights.push_back(m_region);

      TF_ASSIGN_OR_RETURN(int64_t b_size, get_size(b_region_desc_handle));
      int64_t b_offset = static_cast<char*>(b_ptr) - static_cast<char*>(w_ptr);
      dnn::RnnDescriptor::ParamsRegion b_region = {b_offset, b_size};
      biases.push_back(b_region);
    }
    TF_RETURN_IF_ERROR(CheckAndFetchProjectionWeights(
        cudnn, rnn_desc, layer, input_desc, filter_desc, params_size_in_bytes,
        region_desc_handle, &weights));
  }

  return CudnnRnnParamsDescriptor(std::move(filter_desc), params_size_in_bytes,
                                  weights, biases);
}

}  // namespace

class CudnnRnnSequenceTensorDescriptor
    : public dnn::RnnSequenceTensorDescriptor {
  CudnnRnnSequenceTensorDescriptor(StreamExecutor* parent, int max_seq_length,
                                   int batch_size, int data_size,
                                   RNNDataDescriptor data_handle,
                                   TensorDescriptor handle)
      : max_seq_length_(max_seq_length),
        batch_size_(batch_size),
        data_size_(data_size),
        handle_(std::move(handle)),
        rnn_data_handle_(std::move(data_handle)),
        handles_(max_seq_length, handle_.get()) {}

 public:
  CudnnRnnSequenceTensorDescriptor(CudnnRnnSequenceTensorDescriptor&&) =
      default;

  static absl::StatusOr<CudnnRnnSequenceTensorDescriptor> Create(
      StreamExecutor* parent, int max_seq_length, int batch_size, int data_size,
      cudnnDataType_t data_type) {
    if (max_seq_length <= 0) {
      return absl::InvalidArgumentError("max_seq_length <= 0");
    }
    int dims[] = {batch_size, data_size, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    TensorDescriptor tensor_desc = CreateTensorDescriptor();
    RETURN_IF_CUDNN_ERROR(cudnnSetTensorNdDescriptor(
        /*tensorDesc=*/tensor_desc.get(), /*dataType=*/data_type,
        /*nbDims=*/sizeof(dims) / sizeof(dims[0]), /*dimA=*/dims,
        /*strideA=*/strides));
    return CudnnRnnSequenceTensorDescriptor(parent, max_seq_length, batch_size,
                                            data_size, nullptr,
                                            std::move(tensor_desc));
  }

  static absl::StatusOr<CudnnRnnSequenceTensorDescriptor> Create(
      StreamExecutor* parent, int max_seq_length, int batch_size, int data_size,
      absl::Span<const int> seq_lengths, bool time_major,
      cudnnDataType_t data_type) {
    if (max_seq_length <= 0) {
      return absl::InvalidArgumentError("max_seq_length <= 0");
    }
    int dims[] = {batch_size, data_size, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    TensorDescriptor tensor_desc = CreateTensorDescriptor();
    RETURN_IF_CUDNN_ERROR(cudnnSetTensorNdDescriptor(
        /*tensorDesc=*/tensor_desc.get(), /*dataType=*/data_type,
        /*nbDims=*/sizeof(dims) / sizeof(dims[0]), /*dimA=*/dims,
        /*strideA=*/strides));
    const int* seq_lengths_array = seq_lengths.data();
    RNNDataDescriptor data_desc = CreateRNNDataDescriptor();
    float padding_fill = 0.0f;
    cudnnRNNDataLayout_t layout;
    if (time_major) {
      layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
    } else {
      layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
    }
    RETURN_IF_CUDNN_ERROR(cudnnSetRNNDataDescriptor(
        /*RNNDataDesc=*/data_desc.get(), /*dataType*/ data_type,
        /*layout=*/layout,
        /*maxSeqLength=*/max_seq_length,
        /*batchSize=*/batch_size, /*vectorSize=*/data_size,
        /*seqLengthArray=*/seq_lengths_array,
        /*paddingFill*/ (void*)&padding_fill));
    return CudnnRnnSequenceTensorDescriptor(parent, max_seq_length, batch_size,
                                            data_size, std::move(data_desc),
                                            std::move(tensor_desc));
  }

  const cudnnTensorDescriptor_t* handles() const { return handles_.data(); }
  cudnnRNNDataDescriptor_t data_handle() const {
    return rnn_data_handle_.get();
  }

  int max_seq_length() const { return max_seq_length_; }
  int batch_size() const { return batch_size_; }
  int data_size() const { return data_size_; }
  bool is_var_seq_lengths() const { return rnn_data_handle_ != nullptr; }

 private:
  int max_seq_length_;
  int batch_size_;
  int data_size_;
  TensorDescriptor handle_;
  RNNDataDescriptor rnn_data_handle_;
  std::vector<cudnnTensorDescriptor_t> handles_;  // Copies of handle_.
  CudnnRnnSequenceTensorDescriptor(const CudnnRnnSequenceTensorDescriptor&) =
      delete;
  void operator=(const CudnnRnnSequenceTensorDescriptor&) = delete;
};

class CudnnRnnStateTensorDescriptor : public dnn::RnnStateTensorDescriptor {
 public:
  CudnnRnnStateTensorDescriptor(StreamExecutor* parent, int num_layers,
                                int batch_size, int data_size,
                                cudnnDataType_t data_type)
      : handle_(CreateTensorDescriptor()),
        num_layers_(num_layers),
        batch_size_(batch_size),
        data_size_(data_size) {
    int dims[] = {num_layers, batch_size, data_size};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    CHECK_CUDNN_OK(cudnnSetTensorNdDescriptor(
        /*tensorDesc=*/handle_.get(), /*dataType=*/data_type,
        /*nbDims=*/sizeof(dims) / sizeof(dims[0]), /*dimA=*/dims,
        /*strideA=*/strides));
  }

  cudnnTensorDescriptor_t handle() const { return handle_.get(); }

  int num_layers() const { return num_layers_; }
  int batch_size() const { return batch_size_; }
  int data_size() const { return data_size_; }

 private:
  TensorDescriptor handle_;
  int num_layers_;
  int batch_size_;
  int data_size_;
  CudnnRnnStateTensorDescriptor(const CudnnRnnStateTensorDescriptor&) = delete;
  void operator=(const CudnnRnnStateTensorDescriptor&) = delete;
};

namespace {

struct RnnModelDims {
  int num_layers = 0;
  int batch_size = 0;
  int max_seq_length = 0;
  int hidden_size = 0;
  int input_size = 0;
  int cell_size = 0;
  int dir_count = 0;
};

template <class T>
absl::StatusOr<RnnModelDims> ExtractAndCheckRnnForward(
    const CudnnRnnDescriptor& rnn_desc,
    const CudnnRnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<T>& input_data,
    const CudnnRnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<T>& input_h_data,
    const CudnnRnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
    const CudnnRnnSequenceTensorDescriptor& output_desc,
    const DeviceMemory<T>& output_data,
    const CudnnRnnStateTensorDescriptor& output_h_desc,
    const DeviceMemory<T>& output_h_data,
    const CudnnRnnStateTensorDescriptor& output_c_desc,
    const DeviceMemory<T>& output_c_data) {
  // extract model parameters
  RnnModelDims model_dims;
  model_dims.num_layers = rnn_desc.num_layers();
  model_dims.batch_size = input_desc.batch_size();
  model_dims.max_seq_length = input_desc.max_seq_length();
  model_dims.hidden_size = rnn_desc.hidden_size();
  model_dims.input_size = input_desc.data_size();
  model_dims.cell_size = rnn_desc.cell_size();
  model_dims.dir_count =
      (rnn_desc.direction_mode() == CUDNN_BIDIRECTIONAL) ? 2 : 1;

  // check parameters
  if (!(input_h_desc.num_layers() ==
            model_dims.num_layers * model_dims.dir_count &&
        input_h_desc.batch_size() == model_dims.batch_size &&
        input_h_desc.data_size() == model_dims.hidden_size)) {
    return absl::InvalidArgumentError("Invalid input_h shape");
  }
  // The LSTM projection will be used if input_h_desc.data_size() <
  // input_c_desc.data_size()
  if (!(input_h_desc.num_layers() == input_c_desc.num_layers() &&
        input_h_desc.batch_size() == input_c_desc.batch_size() &&
        input_h_desc.data_size() <= input_c_desc.data_size())) {
    return absl::InvalidArgumentError("Invalid input_c shape");
  }
  if (!(output_desc.max_seq_length() == model_dims.max_seq_length &&
        output_desc.batch_size() == model_dims.batch_size &&
        output_desc.data_size() ==
            model_dims.hidden_size * model_dims.dir_count)) {
    return absl::InvalidArgumentError("Invalid output shape");
  }
  if (!(input_h_desc.num_layers() == output_h_desc.num_layers() &&
        input_h_desc.batch_size() == output_h_desc.batch_size() &&
        input_h_desc.data_size() == output_h_desc.data_size())) {
    return absl::InvalidArgumentError("Invalid output_h shape");
  }
  if (!(input_h_desc.num_layers() == output_c_desc.num_layers() &&
        input_h_desc.batch_size() == output_c_desc.batch_size() &&
        input_h_desc.data_size() <= output_c_desc.data_size())) {
    return absl::InvalidArgumentError("Invalid output_c shape");
  }

  return model_dims;
}

absl::Status CheckRNNParameterSize(
    const CudnnHandle& cudnn, const CudnnRnnDescriptor& rnn_desc,
    const CudnnRnnSequenceTensorDescriptor& input_desc) {
  size_t params_size_in_bytes = 0;
  RETURN_IF_CUDNN_ERROR(cudnnGetRNNWeightSpaceSize(
      /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
      /*sizeInBytes=*/&params_size_in_bytes));
  if (static_cast<int64_t>(params_size_in_bytes) !=
      rnn_desc.ParamsSizeInBytes()) {
    return absl::InvalidArgumentError("Mismatching RNN parameter size");
  }
  return absl::OkStatus();
}

absl::Status CreateRnnTempSpace(
    Stream* stream, const CudnnHandle& cudnn,
    const CudnnRnnDescriptor& rnn_desc, RnnModelDims model_dims,
    const CudnnRnnSequenceTensorDescriptor& input_desc,
    ScratchAllocator* workspace_allocator,
    ScratchAllocator* reserve_space_allocator, bool is_fwd_training,
    DeviceMemory<uint8_t>* workspace, DeviceMemory<uint8_t>* reserve_space) {
  size_t reserve_space_size_in_bytes = 0;
  size_t workspace_size_in_bytes = 0;
  if (input_desc.is_var_seq_lengths()) {
    auto rnn_fwd_mode =
        is_fwd_training ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;
    RETURN_IF_CUDNN_ERROR(cudnnGetRNNTempSpaceSizes(
        /*handle=*/cudnn.handle(),
        /*rnnDesc=*/rnn_desc.handle(),
        /*fMode=*/rnn_fwd_mode,
        /*xDesc=*/input_desc.data_handle(),
        /*workSpaceSize=*/&workspace_size_in_bytes,
        /*reserveSpaceSize=*/&reserve_space_size_in_bytes));
  } else {
#if CUDNN_VERSION >= 90000
    return tsl::errors::Internal(
        "Sequence lengths for RNN are required from CUDNN 9.0+");
#else
    RETURN_IF_CUDNN_ERROR(cudnnGetRNNWorkspaceSize(
        /*handle=*/cudnn.handle(),
        /*rnnDesc=*/rnn_desc.handle(),
        /*seqLength=*/input_desc.max_seq_length(),
        /*xDesc=*/input_desc.handles(),
        /*sizeInBytes=*/&workspace_size_in_bytes));
    if (is_fwd_training) {
      RETURN_IF_CUDNN_ERROR(cudnnGetRNNTrainingReserveSize(
          /*handle=*/cudnn.handle(),
          /*rnnDesc=*/rnn_desc.handle(),
          /*seqLength=*/model_dims.max_seq_length,
          /*xDesc=*/input_desc.handles(),
          /*sizeInBytes=*/&reserve_space_size_in_bytes));
    }
#endif  // CUDNN_VERSION >= 90000
  }

  if (workspace_size_in_bytes > 0) {
    TF_ASSIGN_OR_RETURN(*workspace, workspace_allocator->AllocateBytes(
                                        workspace_size_in_bytes));
  }
  if (reserve_space_allocator != nullptr && is_fwd_training &&
      reserve_space_size_in_bytes > 0) {
    TF_ASSIGN_OR_RETURN(*reserve_space, reserve_space_allocator->AllocateBytes(
                                            reserve_space_size_in_bytes));
  }
  return absl::OkStatus();
}

absl::StatusOr<DeviceMemory<uint8_t>> CreateBatchNormForwardWorkspace(
    Stream* stream, const CudnnHandle& cudnn, const cudnnBatchNormMode_t& mode,
    const cudnnBatchNormOps_t& bn_ops,
    const cudnnActivationDescriptor_t& activation_desc,
    const CudnnTensorDescriptor& x_descriptor,
    const CudnnTensorDescriptor& scale_offset_descriptor,
    ScratchAllocator* workspace_allocator) {
  // Query the workspace size.
  size_t workspace_size_in_bytes = 0;
  RETURN_IF_CUDNN_ERROR(
      cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
          /*handle=*/cudnn.handle(), /*mode=*/mode, /*bnOps=*/bn_ops,
          /*xDesc=*/x_descriptor.handle(), /*zDesc=*/x_descriptor.handle(),
          /*yDesc=*/x_descriptor.handle(),
          /*bnScaleBiasMeanVarDesc=*/scale_offset_descriptor.handle(),
          /*activationDesc=*/activation_desc,
          /*sizeInBytes=*/&workspace_size_in_bytes));
  // Allocate the workspace.
  if (workspace_size_in_bytes == 0) {
    return DeviceMemory<uint8_t>();
  }
  return workspace_allocator->AllocateBytes(workspace_size_in_bytes);
}

absl::StatusOr<DeviceMemory<uint8_t>> CreateBatchNormBackwardWorkspace(
    Stream* stream, const CudnnHandle& cudnn, const cudnnBatchNormMode_t& mode,
    const cudnnBatchNormOps_t& bn_ops,
    const cudnnActivationDescriptor_t& activation_desc,
    const CudnnTensorDescriptor& x_descriptor,
    const CudnnTensorDescriptor& scale_offset_descriptor,
    ScratchAllocator* workspace_allocator) {
  // Query the workspace size.
  size_t workspace_size_in_bytes = 0;
  RETURN_IF_CUDNN_ERROR(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
      /*handle=*/cudnn.handle(), /*mode=*/mode, /*bnOps=*/bn_ops,
      /*xDesc=*/x_descriptor.handle(),
      /*yDesc=*/x_descriptor.handle(),
      /*dyDesc=*/x_descriptor.handle(),
      /*dzDesc=*/x_descriptor.handle(),
      /*dxDesc=*/x_descriptor.handle(),
      /*dBnScaleBiasDesc=*/scale_offset_descriptor.handle(),
      /*activationDesc=*/activation_desc,
      /*sizeInBytes=*/&workspace_size_in_bytes));
  // Allocate the workspace.
  if (workspace_size_in_bytes == 0) {
    return DeviceMemory<uint8_t>();
  }
  return workspace_allocator->AllocateBytes(workspace_size_in_bytes);
}

}  // namespace

// Populates the profile result if not empty.
static absl::Status PopulateProfileFromTimer(
    EventBasedTimer* timer, const dnn::AlgorithmDesc& algorithm,
    dnn::ProfileResult* profile_result,
    std::optional<uint64_t> scratch_size = std::nullopt) {
  if (profile_result) {
    TF_ASSIGN_OR_RETURN(absl::Duration duration, timer->GetElapsedDuration());
    profile_result->set_algorithm(algorithm);
    profile_result->set_elapsed_time_in_ms(
        absl::ToDoubleMilliseconds(duration));
    if (scratch_size.has_value()) {
      profile_result->set_scratch_size(*scratch_size);
    }
  }
  return absl::OkStatus();
}

template <class T>
absl::Status CudnnSupport::DoRnnForwardImpl(
    Stream* stream, const CudnnRnnDescriptor& rnn_desc,
    const CudnnRnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<T>& input_data,
    const DeviceMemory<int>& seq_lengths_data,
    const CudnnRnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<T>& input_h_data,
    const CudnnRnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
    const CudnnRnnSequenceTensorDescriptor& output_desc,
    DeviceMemory<T>* output_data,
    const CudnnRnnStateTensorDescriptor& output_h_desc,
    DeviceMemory<T>* output_h_data,
    const CudnnRnnStateTensorDescriptor& output_c_desc,
    DeviceMemory<T>* output_c_data, bool is_training,
    ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
  TF_ASSIGN_OR_RETURN(
      RnnModelDims model_dims,
      ExtractAndCheckRnnForward(
          rnn_desc, input_desc, input_data, input_h_desc, input_h_data,
          input_c_desc, input_c_data, params, output_desc, *output_data,
          output_h_desc, *output_h_data, output_c_desc, *output_c_data));

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  TF_RETURN_IF_ERROR(CheckRNNParameterSize(cudnn, rnn_desc, input_desc));

  DeviceMemory<uint8_t> reserve_space;
  DeviceMemory<uint8_t> workspace;
  TF_RETURN_IF_ERROR(CreateRnnTempSpace(
      stream, cudnn, rnn_desc, model_dims, input_desc, workspace_allocator,
      reserve_space_allocator, is_training, &workspace, &reserve_space));

  std::unique_ptr<EventBasedTimer> timer;
  if (output_profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer,
                        stream->CreateEventBasedTimer(
                            output_profile_result->warmup_run_executed()));
  }

  if (input_desc.is_var_seq_lengths()) {
    auto rnn_fwd_mode =
        is_training ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;
    RETURN_IF_CUDNN_ERROR(cudnnRNNForward(
        /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
        /*fwdMode=*/rnn_fwd_mode,
        /*devSeqLengths=*/
        reinterpret_cast<const int*>(seq_lengths_data.opaque()),
        /*xDesc=*/input_desc.data_handle(), /*x=*/input_data.opaque(),
        /*yDesc=*/output_desc.data_handle(), /*y=*/output_data->opaque(),
        /*hxDesc=*/input_h_desc.handle(), /*hx=*/input_h_data.opaque(),
        /*hy=*/output_h_data->opaque(),
        /*cxDesc=*/input_c_desc.handle(), /*cx=*/input_c_data.opaque(),
        /*cy=*/output_c_data->opaque(),
        /*weightSpaceSize=*/rnn_desc.ParamsSizeInBytes(),
        /*weightSpace=*/params.opaque(),
        /*workSpaceSize=*/workspace.size(), /*workspace=*/workspace.opaque(),
        /*reserveSpaceSizeInBytes=*/reserve_space.size(),
        /*reserveSpace=*/reserve_space.opaque()));
  } else {
#if CUDNN_VERSION >= 90000
    return tsl::errors::Internal(
        "Sequence lengths for RNN are required from CUDNN 9.0+");
#else
    if (!is_training) {
      RETURN_IF_CUDNN_ERROR(cudnnRNNForwardInference(
          /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
          /*seqLength=*/model_dims.max_seq_length,
          /*xDesc=*/input_desc.handles(),
          /*x=*/input_data.opaque(), /*hxDesc=*/input_h_desc.handle(),
          /*hx=*/input_h_data.opaque(), /*cxDesc=*/input_c_desc.handle(),
          /*cx=*/input_c_data.opaque(), /*wDesc=*/rnn_desc.params_handle(),
          /*w=*/params.opaque(), /*yDesc=*/output_desc.handles(),
          /*y=*/output_data->opaque(), /*hyDesc=*/output_h_desc.handle(),
          /*hy=*/output_h_data->opaque(), /*cyDesc=*/output_c_desc.handle(),
          /*cy=*/output_c_data->opaque(), /*workspace=*/workspace.opaque(),
          /*workSpaceSizeInBytes=*/workspace.size()));
    } else {
      RETURN_IF_CUDNN_ERROR(cudnnRNNForwardTraining(
          /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
          /*seqLength=*/model_dims.max_seq_length,
          /*xDesc=*/input_desc.handles(),
          /*x=*/input_data.opaque(), /*hxDesc=*/input_h_desc.handle(),
          /*hx=*/input_h_data.opaque(), /*cxDesc=*/input_c_desc.handle(),
          /*cx=*/input_c_data.opaque(), /*wDesc=*/rnn_desc.params_handle(),
          /*w=*/params.opaque(), /*yDesc=*/output_desc.handles(),
          /*y=*/output_data->opaque(), /*hyDesc=*/output_h_desc.handle(),
          /*hy=*/output_h_data->opaque(), /*cyDesc=*/output_c_desc.handle(),
          /*cy=*/output_c_data->opaque(), /*workspace=*/workspace.opaque(),
          /*workSpaceSizeInBytes=*/workspace.size(),
          /*reserveSpace=*/reserve_space.opaque(),
          /*reserveSpaceSizeInBytes=*/reserve_space.size()));
    }
#endif  // CUDNN_VERSION >= 90000
  }

  if (timer != nullptr) {
    TF_RETURN_IF_ERROR(PopulateProfileFromTimer(
        timer.get(), *rnn_desc.algorithm_config().algorithm(),
        output_profile_result));
  }

  return absl::OkStatus();
}

template <class T>
absl::Status CudnnSupport::DoRnnBackwardImpl(
    Stream* stream, const CudnnRnnDescriptor& rnn_desc,
    const CudnnRnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<T>& input_data,
    const DeviceMemory<int>& seq_lengths_data,
    const CudnnRnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<T>& input_h_data,
    const CudnnRnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
    const CudnnRnnSequenceTensorDescriptor& output_desc,
    const DeviceMemory<T>& output_data,
    const CudnnRnnStateTensorDescriptor& output_h_desc,
    const DeviceMemory<T>& output_h_data,
    const CudnnRnnStateTensorDescriptor& output_c_desc,
    const DeviceMemory<T>& output_c_data,
    const DeviceMemory<T>& output_backprop_data,
    const DeviceMemory<T>& output_h_backprop_data,
    const DeviceMemory<T>& output_c_backprop_data,
    DeviceMemory<T>* input_backprop_data,
    DeviceMemory<T>* input_h_backprop_data,
    DeviceMemory<T>* input_c_backprop_data,
    DeviceMemory<T>* params_backprop_data,
    DeviceMemory<uint8_t>* reserve_space_data,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
  TF_ASSIGN_OR_RETURN(
      RnnModelDims model_dims,
      ExtractAndCheckRnnForward(rnn_desc, input_desc, input_data, input_h_desc,
                                input_h_data, input_c_desc, input_c_data,
                                params, output_desc, output_data, output_h_desc,
                                output_h_data, output_c_desc, output_c_data));

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  TF_RETURN_IF_ERROR(CheckRNNParameterSize(cudnn, rnn_desc, input_desc));

  DeviceMemory<uint8_t> workspace;
  TF_RETURN_IF_ERROR(CreateRnnTempSpace(stream, cudnn, rnn_desc, model_dims,
                                        input_desc, workspace_allocator,
                                        nullptr, true, &workspace, nullptr));

  std::unique_ptr<EventBasedTimer> timer;
  if (output_profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer,
                        stream->CreateEventBasedTimer(
                            output_profile_result->warmup_run_executed()));
  }

  if (input_desc.is_var_seq_lengths()) {
    RETURN_IF_CUDNN_ERROR(cudnnRNNBackwardData_v8(
        /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
        /*devSeqLengths=*/
        reinterpret_cast<const int*>(seq_lengths_data.opaque()),
        /*yDesc=*/output_desc.data_handle(), /*y=*/output_data.opaque(),
        /*dy=*/output_backprop_data.opaque(),
        /*xDesc=*/input_desc.data_handle(),
        /*dx=*/input_backprop_data->opaque(),
        /*hxDesc=*/input_h_desc.handle(), /*hx=*/input_h_data.opaque(),
        /*dhy=*/output_h_backprop_data.opaque(),
        /*dhx=*/input_h_backprop_data->opaque(),
        /*cxDesc=*/input_c_desc.handle(), /*cx=*/input_c_data.opaque(),
        /*dcy=*/output_c_backprop_data.opaque(),
        /*dcx=*/input_c_backprop_data->opaque(),
        /*weightSpaceSize=*/rnn_desc.ParamsSizeInBytes(),
        /*weightSpace=*/params.opaque(),
        /*workSpaceSize=*/workspace.size(), /*workSpace=*/workspace.opaque(),
        /*reserveSpaceSize=*/reserve_space_data->size(),
        /*reserveSpace=*/reserve_space_data->opaque()));

    if (params_backprop_data != nullptr) {
      // Clear the dw to zeros.
      TF_RETURN_IF_ERROR(
          stream->MemZero(params_backprop_data, params_backprop_data->size()));
      RETURN_IF_CUDNN_ERROR(cudnnRNNBackwardWeights_v8(
          /*handle=*/cudnn.handle(),
          /*rnnDesc=*/rnn_desc.handle(),
          /*addGrad=*/CUDNN_WGRAD_MODE_ADD,
          /*devSeqLengths=*/
          reinterpret_cast<const int*>(seq_lengths_data.opaque()),
          /*xDesc=*/input_desc.data_handle(),
          /*x=*/input_data.opaque(),
          /*hDesc=*/input_h_desc.handle(),
          /*hx=*/input_h_data.opaque(),
          /*yDesc=*/output_desc.data_handle(),
          /*y=*/output_data.opaque(),
          /*weightSpaceSize=*/rnn_desc.ParamsSizeInBytes(),
          /*dweightSpace=*/params_backprop_data->opaque(),
          /*workSpaceSize=*/workspace.size(),
          /*workSpace=*/workspace.opaque(),
          /*reserveSpaceSize=*/reserve_space_data->size(),
          /*reserveSpace=*/reserve_space_data->opaque()));
    }
  } else {
#if CUDNN_VERSION >= 90000
    return tsl::errors::Internal(
        "Sequence lengths for RNN are required from CUDNN 9.0+");
#else
    RETURN_IF_CUDNN_ERROR(cudnnRNNBackwardData(
        /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
        /*seqLength=*/model_dims.max_seq_length,
        /*yDesc=*/output_desc.handles(),
        /*y=*/output_data.opaque(), /*dyDesc=*/output_desc.handles(),
        /*dy=*/output_backprop_data.opaque(),
        /*dhyDesc=*/output_h_desc.handle(),
        /*dhy=*/output_h_backprop_data.opaque(),
        /*dcyDesc=*/output_c_desc.handle(),
        /*dcy=*/output_c_backprop_data.opaque(),
        /*wDesc=*/rnn_desc.params_handle(), /*w=*/params.opaque(),
        /*hxDesc=*/input_h_desc.handle(), /*hx=*/input_h_data.opaque(),
        /*cxDesc=*/input_c_desc.handle(), /*cx=*/input_c_data.opaque(),
        /*dxDesc=*/input_desc.handles(), /*dx=*/input_backprop_data->opaque(),
        /*dhxDesc=*/input_h_desc.handle(),
        /*dhx=*/input_h_backprop_data->opaque(),
        /*dcxDesc=*/input_c_desc.handle(),
        /*dcx=*/input_c_backprop_data->opaque(),
        /*workspace=*/workspace.opaque(),
        /*workSpaceSizeInBytes=*/workspace.size(),
        /*reserveSpace=*/reserve_space_data->opaque(),
        /*reserveSpaceSizeInBytes=*/reserve_space_data->size()));

    if (params_backprop_data != nullptr) {
      // Clear the dw to zeros.
      TF_RETURN_IF_ERROR(
          stream->MemZero(params_backprop_data, params_backprop_data->size()));
      // make the backward weight call
      RETURN_IF_CUDNN_ERROR(cudnnRNNBackwardWeights(
          /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
          /*seqLength=*/model_dims.max_seq_length,
          /*xDesc=*/input_desc.handles(),
          /*x=*/input_data.opaque(), /*hxDesc=*/input_h_desc.handle(),
          /*hx=*/input_h_data.opaque(), /*yDesc=*/output_desc.handles(),
          /*y=*/output_data.opaque(), /*workspace=*/workspace.opaque(),
          /*workSpaceSizeInBytes=*/workspace.size(),
          /*dwDesc=*/rnn_desc.params_handle(),
          /*dw=*/params_backprop_data->opaque(),
          /*reserveSpace=*/reserve_space_data->opaque(),
          /*reserveSpaceSizeInBytes=*/reserve_space_data->size()));
    }
#endif  // CUDNN_VERSION >= 90000
  }

  if (timer != nullptr) {
    TF_RETURN_IF_ERROR(PopulateProfileFromTimer(
        timer.get(), *rnn_desc.algorithm_config().algorithm(),
        output_profile_result));
  }

  return absl::OkStatus();
}

absl::Status CudnnSupport::DoCtcLossImpl(
    Stream* stream, const CudnnRnnStateTensorDescriptor& probs_desc,
    const DeviceMemoryBase probs_data, absl::Span<const int> labels_data,
    absl::Span<const int> labels_lengths_data,
    absl::Span<const int> input_lengths_data, DeviceMemoryBase costs_data,
    const CudnnRnnStateTensorDescriptor& grads_desc,
    DeviceMemoryBase grads_data, const CudnnCtcLossDescriptor& ctc_loss_desc,
    DeviceMemory<uint8_t> scratch_memory, int ctc_loss_algo_id) {
  auto cudnn = cudnn_->GetHandle(parent_, stream);

  int kNumTimestamps = probs_desc.num_layers();
  int kBatchSize = probs_desc.batch_size();
  int kNumLabels = probs_desc.data_size();
  int total_size = kNumLabels * kNumTimestamps * kBatchSize;
  (void)total_size;

  cudnnCTCLossAlgo_t ctc_loss_algo =
      static_cast<cudnnCTCLossAlgo_t>(ctc_loss_algo_id);
  RETURN_IF_CUDNN_ERROR(cudnnCTCLoss(
      /*handle=*/cudnn.handle(), /*probsDesc=*/probs_desc.handle(),
      /*probs=*/probs_data.opaque(), /*labels=*/labels_data.data(),
      /*labelLengths=*/labels_lengths_data.data(),
      /*inputLengths=*/input_lengths_data.data(),
      /*costs=*/costs_data.opaque(), /*gradientsDesc=*/grads_desc.handle(),
      /*gradients=*/grads_data.opaque(),
      /*algo=*/ctc_loss_algo,
      /*ctcLossDesc=*/ctc_loss_desc.handle(),
      /*workspace=*/scratch_memory.opaque(),
      /*workSpaceSizeInBytes=*/scratch_memory.size()));

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
CudnnSupport::CreateRnnDescriptor(
    int num_layers, int hidden_size, int input_size, int cell_size,
    int batch_size, dnn::RnnInputMode input_mode,
    dnn::RnnDirectionMode direction_mode, dnn::RnnMode rnn_mode,
    dnn::DataType data_type, const dnn::AlgorithmConfig& algorithm_config,
    const NumericOptions& numeric_options, float dropout, uint64_t seed,
    ScratchAllocator* state_allocator, bool use_padded_io) {
  // Setting up a cudnnRNNDescriptor requires a cuDNN handle, but because it's
  // not enqueueing anything into a stream, we pass in the null stream.
  auto cudnn = cudnn_->GetHandle(parent_, /*stream=*/nullptr);
  TF_ASSIGN_OR_RETURN(
      CudnnRnnDescriptor rnn_desc,
      CudnnRnnDescriptor::Create(
          cudnn, num_layers, hidden_size, input_size, cell_size, batch_size,
          ToCudnnRnnInputMode(input_mode),
          ToCudnnRnnDirectionMode(direction_mode), ToCudnnRnnMode(rnn_mode),
          ToCudnnDataType(data_type), GetRnnComputeType(data_type),
          algorithm_config, numeric_options, dropout, seed, state_allocator,
          use_padded_io));
  return std::unique_ptr<dnn::RnnDescriptor>(
      new CudnnRnnDescriptor(std::move(rnn_desc)));
}

absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
CudnnSupport::CreateRnnSequenceTensorDescriptor(int max_seq_length,
                                                int batch_size, int data_size,
                                                dnn::DataType data_type) {
  TF_ASSIGN_OR_RETURN(CudnnRnnSequenceTensorDescriptor descriptor,
                      CudnnRnnSequenceTensorDescriptor::Create(
                          parent_, max_seq_length, batch_size, data_size,
                          ToCudnnDataType(data_type)));
  return std::unique_ptr<dnn::RnnSequenceTensorDescriptor>(
      new CudnnRnnSequenceTensorDescriptor(std::move(descriptor)));
}

absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
CudnnSupport::CreateRnnSequenceTensorDescriptor(
    int max_seq_length, int batch_size, int data_size,
    const absl::Span<const int>& seq_lengths, bool time_major,
    dnn::DataType data_type) {
  TF_ASSIGN_OR_RETURN(CudnnRnnSequenceTensorDescriptor descriptor,
                      CudnnRnnSequenceTensorDescriptor::Create(
                          parent_, max_seq_length, batch_size, data_size,
                          seq_lengths, time_major, ToCudnnDataType(data_type)));
  return std::unique_ptr<dnn::RnnSequenceTensorDescriptor>(
      new CudnnRnnSequenceTensorDescriptor(std::move(descriptor)));
}

absl::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
CudnnSupport::CreateRnnStateTensorDescriptor(int num_layer, int batch_size,
                                             int data_size,
                                             dnn::DataType data_type) {
  return std::unique_ptr<dnn::RnnStateTensorDescriptor>(
      new CudnnRnnStateTensorDescriptor(parent_, num_layer, batch_size,
                                        data_size, ToCudnnDataType(data_type)));
}

bool CudnnSupport::DoRnnForward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<Eigen::half>& input_data,
    const DeviceMemory<int>& seq_lengths_data,
    const dnn::RnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<Eigen::half>& input_h_data,
    const dnn::RnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<Eigen::half>& input_c_data,
    const DeviceMemory<Eigen::half>& params,
    const dnn::RnnSequenceTensorDescriptor& output_desc,
    DeviceMemory<Eigen::half>* output_data,
    const dnn::RnnStateTensorDescriptor& output_h_desc,
    DeviceMemory<Eigen::half>* output_h_data,
    const dnn::RnnStateTensorDescriptor& output_c_desc,
    DeviceMemory<Eigen::half>* output_c_data, bool is_training,
    ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
  const CudnnRnnDescriptor& cudnn_rnn_desc =
      static_cast<const CudnnRnnDescriptor&>(rnn_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_input_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(input_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_c_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_output_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(output_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_c_desc);
  return IsStatusOk(
      DoRnnForwardImpl<Eigen::half>(
          stream, cudnn_rnn_desc, cudnn_input_desc, input_data,
          seq_lengths_data, cudnn_input_h_desc, input_h_data,
          cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
          output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
          output_c_data, is_training, reserve_space_allocator,
          workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoRnnForward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<float>& input_data,
    const DeviceMemory<int>& seq_lengths_data,
    const dnn::RnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<float>& input_h_data,
    const dnn::RnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<float>& input_c_data, const DeviceMemory<float>& params,
    const dnn::RnnSequenceTensorDescriptor& output_desc,
    DeviceMemory<float>* output_data,
    const dnn::RnnStateTensorDescriptor& output_h_desc,
    DeviceMemory<float>* output_h_data,
    const dnn::RnnStateTensorDescriptor& output_c_desc,
    DeviceMemory<float>* output_c_data, bool is_training,
    ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
  const CudnnRnnDescriptor& cudnn_rnn_desc =
      static_cast<const CudnnRnnDescriptor&>(rnn_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_input_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(input_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_c_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_output_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(output_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_c_desc);
  return IsStatusOk(
      DoRnnForwardImpl<float>(
          stream, cudnn_rnn_desc, cudnn_input_desc, input_data,
          seq_lengths_data, cudnn_input_h_desc, input_h_data,
          cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
          output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
          output_c_data, is_training, reserve_space_allocator,
          workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoRnnForward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<double>& input_data,
    const DeviceMemory<int>& seq_lengths_data,
    const dnn::RnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<double>& input_h_data,
    const dnn::RnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<double>& input_c_data,
    const DeviceMemory<double>& params,
    const dnn::RnnSequenceTensorDescriptor& output_desc,
    DeviceMemory<double>* output_data,
    const dnn::RnnStateTensorDescriptor& output_h_desc,
    DeviceMemory<double>* output_h_data,
    const dnn::RnnStateTensorDescriptor& output_c_desc,
    DeviceMemory<double>* output_c_data, bool is_training,
    ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
  const CudnnRnnDescriptor& cudnn_rnn_desc =
      static_cast<const CudnnRnnDescriptor&>(rnn_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_input_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(input_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_c_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_output_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(output_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_c_desc);
  return IsStatusOk(
      DoRnnForwardImpl<double>(
          stream, cudnn_rnn_desc, cudnn_input_desc, input_data,
          seq_lengths_data, cudnn_input_h_desc, input_h_data,
          cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
          output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
          output_c_data, is_training, reserve_space_allocator,
          workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoRnnBackward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<Eigen::half>& input_data,
    const DeviceMemory<int>& seq_lengths_data,
    const dnn::RnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<Eigen::half>& input_h_data,
    const dnn::RnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<Eigen::half>& input_c_data,
    const DeviceMemory<Eigen::half>& params,
    const dnn::RnnSequenceTensorDescriptor& output_desc,
    const DeviceMemory<Eigen::half>& output_data,
    const dnn::RnnStateTensorDescriptor& output_h_desc,
    const DeviceMemory<Eigen::half>& output_h_data,
    const dnn::RnnStateTensorDescriptor& output_c_desc,
    const DeviceMemory<Eigen::half>& output_c_data,
    const DeviceMemory<Eigen::half>& output_backprop_data,
    const DeviceMemory<Eigen::half>& output_h_backprop_data,
    const DeviceMemory<Eigen::half>& output_c_backprop_data,
    DeviceMemory<Eigen::half>* input_backprop_data,
    DeviceMemory<Eigen::half>* input_h_backprop_data,
    DeviceMemory<Eigen::half>* input_c_backprop_data,
    DeviceMemory<Eigen::half>* params_backprop_data,
    DeviceMemory<uint8_t>* reserve_space_data,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
  const CudnnRnnDescriptor& cudnn_rnn_desc =
      static_cast<const CudnnRnnDescriptor&>(rnn_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_input_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(input_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_c_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_output_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(output_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_c_desc);
  return IsStatusOk(
      DoRnnBackwardImpl<Eigen::half>(
          stream, cudnn_rnn_desc, cudnn_input_desc, input_data,
          seq_lengths_data, cudnn_input_h_desc, input_h_data,
          cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
          output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
          output_c_data, output_backprop_data, output_h_backprop_data,
          output_c_backprop_data, input_backprop_data, input_h_backprop_data,
          input_c_backprop_data, params_backprop_data, reserve_space_data,
          workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoRnnBackward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<float>& input_data,
    const DeviceMemory<int>& seq_lengths_data,
    const dnn::RnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<float>& input_h_data,
    const dnn::RnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<float>& input_c_data, const DeviceMemory<float>& params,
    const dnn::RnnSequenceTensorDescriptor& output_desc,
    const DeviceMemory<float>& output_data,
    const dnn::RnnStateTensorDescriptor& output_h_desc,
    const DeviceMemory<float>& output_h_data,
    const dnn::RnnStateTensorDescriptor& output_c_desc,
    const DeviceMemory<float>& output_c_data,
    const DeviceMemory<float>& output_backprop_data,
    const DeviceMemory<float>& output_h_backprop_data,
    const DeviceMemory<float>& output_c_backprop_data,
    DeviceMemory<float>* input_backprop_data,
    DeviceMemory<float>* input_h_backprop_data,
    DeviceMemory<float>* input_c_backprop_data,
    DeviceMemory<float>* params_backprop_data,
    DeviceMemory<uint8_t>* reserve_space_data,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
  const CudnnRnnDescriptor& cudnn_rnn_desc =
      static_cast<const CudnnRnnDescriptor&>(rnn_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_input_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(input_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_c_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_output_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(output_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_c_desc);
  return IsStatusOk(
      DoRnnBackwardImpl<float>(
          stream, cudnn_rnn_desc, cudnn_input_desc, input_data,
          seq_lengths_data, cudnn_input_h_desc, input_h_data,
          cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
          output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
          output_c_data, output_backprop_data, output_h_backprop_data,
          output_c_backprop_data, input_backprop_data, input_h_backprop_data,
          input_c_backprop_data, params_backprop_data, reserve_space_data,
          workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoRnnBackward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<double>& input_data,
    const DeviceMemory<int>& seq_lengths_data,
    const dnn::RnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<double>& input_h_data,
    const dnn::RnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<double>& input_c_data,
    const DeviceMemory<double>& params,
    const dnn::RnnSequenceTensorDescriptor& output_desc,
    const DeviceMemory<double>& output_data,
    const dnn::RnnStateTensorDescriptor& output_h_desc,
    const DeviceMemory<double>& output_h_data,
    const dnn::RnnStateTensorDescriptor& output_c_desc,
    const DeviceMemory<double>& output_c_data,
    const DeviceMemory<double>& output_backprop_data,
    const DeviceMemory<double>& output_h_backprop_data,
    const DeviceMemory<double>& output_c_backprop_data,
    DeviceMemory<double>* input_backprop_data,
    DeviceMemory<double>* input_h_backprop_data,
    DeviceMemory<double>* input_c_backprop_data,
    DeviceMemory<double>* params_backprop_data,
    DeviceMemory<uint8_t>* reserve_space_data,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
  const CudnnRnnDescriptor& cudnn_rnn_desc =
      static_cast<const CudnnRnnDescriptor&>(rnn_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_input_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(input_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_input_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_c_desc);
  const CudnnRnnSequenceTensorDescriptor& cudnn_output_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(output_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_h_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_output_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_c_desc);
  return IsStatusOk(
      DoRnnBackwardImpl<double>(
          stream, cudnn_rnn_desc, cudnn_input_desc, input_data,
          seq_lengths_data, cudnn_input_h_desc, input_h_data,
          cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
          output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
          output_c_data, output_backprop_data, output_h_backprop_data,
          output_c_backprop_data, input_backprop_data, input_h_backprop_data,
          input_c_backprop_data, params_backprop_data, reserve_space_data,
          workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

namespace {

// TODO(csigg): Merge a lot of duplicate code below for forward, backward data,
// and backward filter.

absl::StatusOr<cudnnConvolutionFwdAlgo_t> GetCudnnConvolutionForwardAlgo(
    const CudnnHandle& cudnn, const CudnnTensorDescriptor& input_nd,
    const CudnnFilterDescriptor& filter, const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd, bool specify_workspace_limit,
    size_t memory_limit_bytes) {
  const int num_requested_algos = 5;
  int num_returned_algos = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results[num_requested_algos];

  RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(
      cudnn.handle(), input_nd.handle(), filter.handle(), conv.handle(),
      output_nd.handle(), num_requested_algos, &num_returned_algos,
      perf_results));

  size_t mem_limit = specify_workspace_limit ? memory_limit_bytes : 0ULL;
  for (int r = 0; r < num_returned_algos; r++) {
    if (perf_results[r].status == CUDNN_STATUS_SUCCESS &&
        perf_results[r].algo != CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED &&
        perf_results[r].memory <= mem_limit) {
      return perf_results[r].algo;
    }
  }
  return absl::InternalError(
      "cudnnGetConvolutionForwardAlgorithm_v7 returned "
      "no suitable algorithms. This could be a cudnn bug.");
}

absl::StatusOr<cudnnConvolutionBwdDataAlgo_t>
GetCudnnConvolutionBackwardDataAlgo(const CudnnHandle& cudnn,
                                    const CudnnTensorDescriptor& input_nd,
                                    const CudnnFilterDescriptor& filter,
                                    const CudnnConvolutionDescriptor& conv,
                                    const CudnnTensorDescriptor& output_nd,
                                    bool specify_workspace_limit,
                                    size_t memory_limit_bytes) {
  const int num_requested_algos = 5;
  int num_returned_algos = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perf_results[num_requested_algos];

  RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionBackwardDataAlgorithm_v7(
      cudnn.handle(), filter.handle(), output_nd.handle(), conv.handle(),
      input_nd.handle(), num_requested_algos, &num_returned_algos,
      perf_results));

  size_t mem_limit = specify_workspace_limit ? memory_limit_bytes : 0ULL;
  for (int r = 0; r < num_returned_algos; r++) {
    if (perf_results[r].status == CUDNN_STATUS_SUCCESS &&
        perf_results[r].algo !=
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED &&
        perf_results[r].memory <= mem_limit) {
      return perf_results[r].algo;
    }
  }
  return absl::InternalError(
      "cudnnGetConvolutionBackwardDataAlgorithm_v7 returned "
      "no suitable algorithms. This could be a cudnn bug.");
}

absl::StatusOr<cudnnConvolutionBwdFilterAlgo_t>
GetCudnnConvolutionBackwardFilterAlgo(const CudnnHandle& cudnn,
                                      const CudnnTensorDescriptor& input_nd,
                                      const CudnnFilterDescriptor& filter,
                                      const CudnnConvolutionDescriptor& conv,
                                      const CudnnTensorDescriptor& output_nd,
                                      bool specify_workspace_limit,
                                      size_t memory_limit_bytes) {
  const int num_requested_algos = 5;
  int num_returned_algos = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t perf_results[num_requested_algos];

  RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
      cudnn.handle(), input_nd.handle(), output_nd.handle(), conv.handle(),
      filter.handle(), num_requested_algos, &num_returned_algos, perf_results));

  size_t mem_limit = specify_workspace_limit ? memory_limit_bytes : 0ULL;
  for (int r = 0; r < num_returned_algos; r++) {
    if (perf_results[r].status == CUDNN_STATUS_SUCCESS &&
        perf_results[r].algo !=
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED &&
        perf_results[r].memory <= mem_limit) {
      return perf_results[r].algo;
    }
  }
  return absl::InternalError(
      "cudnnGetConvolutionBackwardFilterAlgorithm_v7 returned "
      "no suitable algorithms. This could be a cudnn bug.");
}

absl::StatusOr<DeviceMemory<uint8_t>> AllocateCudnnConvolutionForwardWorkspace(
    Stream* stream, const CudnnHandle& cudnn,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd,
    const dnn::AlgorithmDesc& algorithm_desc,
    ScratchAllocator* scratch_allocator) {
  if (IsTensorMathOpSet(conv) != algorithm_desc.tensor_ops_enabled()) {
    return absl::InternalError(
        "Mismatch between cudnn conv and algorithm descriptors.");
  }

  // Query the size of the workspace and allocate it.
  size_t size_in_bytes;
  if (algorithm_desc.workspace_size()) {
    size_in_bytes = *algorithm_desc.workspace_size();
  } else {
    RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn.handle(),
        /*xDesc=*/input_nd.handle(),
        /*wDesc=*/filter.handle(), /*convDesc=*/conv.handle(),
        /*yDesc=*/output_nd.handle(),
        /*algo=*/ToConvForwardAlgo(algorithm_desc),
        /*sizeInBytes=*/&size_in_bytes));
  }

  int64_t size_in_bytes_int64_t = size_in_bytes;

  if (ABSL_PREDICT_FALSE(size_in_bytes_int64_t < 0)) {
    return absl::InternalError(
        "cudnnGetConvolutionForwardWorkspaceSize() returned "
        "negative sizeInBytes value. This could be a cudnn bug.");
  }

  if (size_in_bytes_int64_t == 0) {
    return DeviceMemory<uint8_t>();
  }

  if (ABSL_PREDICT_FALSE(!scratch_allocator)) {
    return absl::InvalidArgumentError("No scratch allocator provided");
  }

  return scratch_allocator->AllocateBytes(size_in_bytes);
}

absl::StatusOr<DeviceMemory<uint8_t>>
AllocateCudnnConvolutionBackwardDataWorkspace(
    Stream* stream, const CudnnHandle& cudnn,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd,
    const dnn::AlgorithmDesc& algorithm_desc,
    ScratchAllocator* scratch_allocator) {
  if (IsTensorMathOpSet(conv) != algorithm_desc.tensor_ops_enabled()) {
    return absl::InternalError(
        "Mismatch between cudnn conv and algorithm descriptors.");
  }

  // Query the size of the workspace and allocate it.
  size_t size_in_bytes;
  if (algorithm_desc.workspace_size()) {
    size_in_bytes = *algorithm_desc.workspace_size();
  } else {
    RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn.handle(),
        /*wDesc=*/filter.handle(),
        /*dyDesc=*/output_nd.handle(),
        /*convDesc=*/conv.handle(),
        /*dxDesc=*/input_nd.handle(),
        /*algo=*/ToConvBackwardDataAlgo(algorithm_desc),
        /*sizeInBytes=*/&size_in_bytes));
  }

  int64_t size_in_bytes_int64_t = size_in_bytes;

  if (ABSL_PREDICT_FALSE(size_in_bytes_int64_t < 0)) {
    return absl::InternalError(
        "cudnnGetConvolutionBackwardDataWorkspaceSize() returned "
        "negative sizeInBytes value. This could be a cudnn bug.");
  }

  if (size_in_bytes_int64_t == 0) {
    return DeviceMemory<uint8_t>();
  }

  if (ABSL_PREDICT_FALSE(!scratch_allocator)) {
    return absl::InvalidArgumentError("No scratch allocator provided");
  }

  return scratch_allocator->AllocateBytes(size_in_bytes);
}

absl::StatusOr<DeviceMemory<uint8_t>>
AllocateCudnnConvolutionBackwardFilterWorkspace(
    Stream* stream, const CudnnHandle& cudnn,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd,
    const dnn::AlgorithmDesc& algorithm_desc,
    ScratchAllocator* scratch_allocator) {
  if (IsTensorMathOpSet(conv) != algorithm_desc.tensor_ops_enabled()) {
    return absl::InternalError(
        "Mismatch between cudnn conv and algorithm descriptors.");
  }

  // Query the size of the workspace and allocate it.
  size_t size_in_bytes;
  if (algorithm_desc.workspace_size()) {
    size_in_bytes = *algorithm_desc.workspace_size();
  } else {
    RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn.handle(),
        /*xDesc=*/input_nd.handle(),
        /*dyDesc=*/output_nd.handle(),
        /*convDesc=*/conv.handle(),
        /*gradDesc=*/filter.handle(),
        /*algo=*/ToConvBackwardFilterAlgo(algorithm_desc),
        /*sizeInBytes=*/&size_in_bytes));
  }

  int64_t size_in_bytes_int64_t = size_in_bytes;

  if (ABSL_PREDICT_FALSE(size_in_bytes_int64_t < 0)) {
    return absl::InternalError(
        "cudnnGetConvolutionBackwardFilterWorkspaceSize() returned "
        "negative sizeInBytes value. This could be a cudnn bug.");
  }

  if (size_in_bytes_int64_t == 0) {
    return DeviceMemory<uint8_t>();
  }

  if (ABSL_PREDICT_FALSE(!scratch_allocator)) {
    return absl::InvalidArgumentError("No scratch allocator provided");
  }

  return scratch_allocator->AllocateBytes(size_in_bytes);
}

bool UseTensorOps(dnn::DataType input_type,
                  std::optional<dnn::AlgorithmDesc> desc) {
  if (desc.has_value()) {
    return desc->tensor_ops_enabled();
  } else {
    // It's unknown whether the user wants to use TensorFloat-32, which is used
    // with tensor ops when the inputs are FP32. For safety, assume the user
    // does not want TensorFloat-32 on FP32 inputs.
    return input_type != dnn::DataType::kFloat;
  }
}

cudnnDataType_t GetRnnComputeType(dnn::DataType data_type);
dnn::DataType GetConvAccumulatorType(dnn::DataType data_type);

absl::StatusOr<dnn::AlgorithmDesc> GetCudnnConvolutionForwardAlgorithm(
    Stream* stream, const CudnnHandle& cudnn,
    const dnn::AlgorithmConfig& algorithm_config,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    dnn::DataType element_type,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const CudnnTensorDescriptor& output_nd, ScratchAllocator* scratch_allocator,
    DeviceMemory<uint8_t>* scratch) {
  std::optional<dnn::AlgorithmDesc> algo_desc = algorithm_config.algorithm();

  CudnnConvolutionDescriptor conv(
      convolution_descriptor,
      ToCudnnDataType(GetConvAccumulatorType(element_type)));
  bool use_tensor_ops = UseTensorOps(element_type, algo_desc);
  conv.set_use_tensor_op_math(use_tensor_ops);

  if (!algo_desc.has_value()) {
    // Pick fastest algorithm within memory limit according to cuDNN's
    // heuristics.
    bool specify_workspace_limit = scratch_allocator != nullptr;
    auto memory_limit_bytes =
        specify_workspace_limit
            ? std::max(scratch_allocator->GetMemoryLimitInBytes(), int64_t{0})
            : int64_t{0};
    TF_ASSIGN_OR_RETURN(cudnnConvolutionFwdAlgo_t algo,
                        GetCudnnConvolutionForwardAlgo(
                            cudnn, input_nd, filter, conv, output_nd,
                            specify_workspace_limit, memory_limit_bytes));
    algo_desc = dnn::AlgorithmDesc(algo, use_tensor_ops);
  }

  const auto scratch_or = AllocateCudnnConvolutionForwardWorkspace(
      stream, cudnn, input_nd, filter, conv, output_nd, *algo_desc,
      scratch_allocator);

  if (scratch_or.ok()) {
    *scratch = scratch_or.value();
    return *algo_desc;
  }

  algo_desc = algorithm_config.algorithm_no_scratch();

  // Failed to allocate workspace for the first algorithm, fall back to the
  // no_scratch algorithm.
  if (!algo_desc.has_value()) {
    return absl::Status(
        scratch_or.status().code(),
        absl::StrCat("The primary convolution algorithm failed, ",
                     "while a secondary algorithm is not provided. ",
                     "Returned status: ", scratch_or.status().ToString()));
  }

  use_tensor_ops = UseTensorOps(element_type, algo_desc);
  conv.set_use_tensor_op_math(use_tensor_ops);
  TF_ASSIGN_OR_RETURN(*scratch, AllocateCudnnConvolutionForwardWorkspace(
                                    stream, cudnn, input_nd, filter, conv,
                                    output_nd, *algo_desc, scratch_allocator));
  return *algo_desc;
}

absl::StatusOr<dnn::AlgorithmDesc> GetCudnnConvolutionBackwardDataAlgorithm(
    Stream* stream, const CudnnHandle& cudnn,
    const dnn::AlgorithmConfig& algorithm_config,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    dnn::DataType element_type,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const CudnnTensorDescriptor& output_nd, ScratchAllocator* scratch_allocator,
    DeviceMemory<uint8_t>* scratch) {
  std::optional<dnn::AlgorithmDesc> algo_desc = algorithm_config.algorithm();
  CudnnConvolutionDescriptor conv(
      convolution_descriptor,
      ToCudnnDataType(GetConvAccumulatorType(element_type)));
  bool use_tensor_ops = UseTensorOps(element_type, algo_desc);
  conv.set_use_tensor_op_math(use_tensor_ops);

  if (!algo_desc.has_value()) {
    // Pick fastest algorithm within memory limit according to cuDNN's
    // heuristics.
    bool specify_workspace_limit = scratch_allocator != nullptr;
    auto memory_limit_bytes =
        specify_workspace_limit
            ? std::max(scratch_allocator->GetMemoryLimitInBytes(), int64_t{0})
            : int64_t{0};
    TF_ASSIGN_OR_RETURN(cudnnConvolutionBwdDataAlgo_t algo,
                        GetCudnnConvolutionBackwardDataAlgo(
                            cudnn, input_nd, filter, conv, output_nd,
                            specify_workspace_limit, memory_limit_bytes));
    algo_desc = dnn::AlgorithmDesc(algo, use_tensor_ops);
  }

  const auto scratch_or = AllocateCudnnConvolutionBackwardDataWorkspace(
      stream, cudnn, input_nd, filter, conv, output_nd, *algo_desc,
      scratch_allocator);

  if (scratch_or.ok()) {
    *scratch = scratch_or.value();
    return *algo_desc;
  }

  algo_desc = algorithm_config.algorithm_no_scratch();

  // Failed to allocate workspace for the first algorithm, fall back to the
  // no_scratch algorithm.
  if (!algo_desc.has_value()) {
    return absl::InvalidArgumentError(
        "The primary convolution algorithm failed memory allocation, "
        "while a secondary algorithm is not provided.");
  }

  use_tensor_ops = UseTensorOps(element_type, algo_desc);
  conv.set_use_tensor_op_math(use_tensor_ops);
  TF_ASSIGN_OR_RETURN(*scratch, AllocateCudnnConvolutionBackwardDataWorkspace(
                                    stream, cudnn, input_nd, filter, conv,
                                    output_nd, *algo_desc, scratch_allocator));
  return *algo_desc;
}

absl::StatusOr<dnn::AlgorithmDesc> GetCudnnConvolutionBackwardFilterAlgorithm(
    Stream* stream, const CudnnHandle& cudnn,
    const dnn::AlgorithmConfig& algorithm_config,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    dnn::DataType element_type,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const CudnnTensorDescriptor& output_nd, ScratchAllocator* scratch_allocator,
    DeviceMemory<uint8_t>* scratch) {
  std::optional<dnn::AlgorithmDesc> algo_desc = algorithm_config.algorithm();
  CudnnConvolutionDescriptor conv(
      convolution_descriptor,
      ToCudnnDataType(GetConvAccumulatorType(element_type)));
  bool use_tensor_ops = UseTensorOps(element_type, algo_desc);
  conv.set_use_tensor_op_math(use_tensor_ops);

  if (!algo_desc.has_value()) {
    // Pick fastest algorithm within memory limit according to cuDNN's
    // heuristics.
    bool specify_workspace_limit = scratch_allocator != nullptr;
    auto memory_limit_bytes =
        specify_workspace_limit
            ? std::max(scratch_allocator->GetMemoryLimitInBytes(), int64_t{0})
            : int64_t{0};
    TF_ASSIGN_OR_RETURN(cudnnConvolutionBwdFilterAlgo_t algo,
                        GetCudnnConvolutionBackwardFilterAlgo(
                            cudnn, input_nd, filter, conv, output_nd,
                            specify_workspace_limit, memory_limit_bytes));
    algo_desc = dnn::AlgorithmDesc(algo, use_tensor_ops);
  }

  absl::StatusOr<DeviceMemory<uint8_t>> scratch_or =
      AllocateCudnnConvolutionBackwardFilterWorkspace(
          stream, cudnn, input_nd, filter, conv, output_nd, *algo_desc,
          scratch_allocator);

  if (scratch_or.ok()) {
    *scratch = scratch_or.value();
    return *algo_desc;
  }

  algo_desc = algorithm_config.algorithm_no_scratch();

  // Failed to allocate workspace for the first algorithm, fall back to the
  // no_scratch algorithm.
  if (!algo_desc.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The primary convolution algorithm failed memory allocation, "
        "while a secondary algorithm is not provided. Actual error: ",
        scratch_or.status().ToString()));
  }

  use_tensor_ops = UseTensorOps(element_type, algo_desc);
  conv.set_use_tensor_op_math(use_tensor_ops);
  TF_ASSIGN_OR_RETURN(*scratch, AllocateCudnnConvolutionBackwardFilterWorkspace(
                                    stream, cudnn, input_nd, filter, conv,
                                    output_nd, *algo_desc, scratch_allocator));
  return *algo_desc;
}

// A helper class to set env-vars and choose options for cudnn-related
// algorithms.
template <typename EnvVar>
class CudnnEnvVar {
 public:
  static bool IsEnabled() {
    static bool is_enabled = IsEnabledImpl();
    return is_enabled;
  }

 private:
  static bool IsEnabledImpl() {
    const char* tf_env_var_val = getenv(EnvVar::kName);
    if (tf_env_var_val != nullptr) {
      absl::string_view tf_env_var_val_str(tf_env_var_val);
      if (tf_env_var_val_str == "0") {
        return false;
      }
      return true;
    }
    return EnvVar::kDefaultFlag;
  }
};

// A helper struct to decide whether to enable the FFT_TILING algorithms for
// forward convolution. It is disabled for cuDNN < 7 due to memory corruption
// caused by some shapes with this algorithm. Users can explicitly enable the
// algorithm through an env-var "TF_ENABLE_FFT_TILING_FORWARD=1".
struct FftTilingForward {
  static constexpr const char* kName = "TF_ENABLE_FFT_TILING_FORWARD";
  static constexpr bool kDefaultFlag = true;
};

// A helper struct to decide whether to enable the WINOGRAD_NONFUSED algorithms.
// By default it is turned on, users can explicitly disable them through an
// env-var "TF_ENABLE_WINOGRAD_NONFUSED=0".
// https://github.com/tensorflow/tensorflow/pull/4901
// For CUDNN v8.1, when this env-var is turned off, both the winograd and
// winograd-non-fused engines will be ruled out.
struct WinogradNonfused {
  static constexpr const char* kName = "TF_ENABLE_WINOGRAD_NONFUSED";
  static constexpr bool kDefaultFlag = true;
};

// A helper struct to decide whether to use FP32 as the internal compute type
// for convolution when the input data type is FP16. By default it is turned on,
// users can explicitly disable them (choose to use FP16 as the internal compute
// type) through an env-var "TF_FP16_CONV_USE_FP32_COMPUTE=0".
struct ConvDoFP32ComputationFP16Input {
  static constexpr const char* kName = "TF_FP16_CONV_USE_FP32_COMPUTE";
  // Using FP16 as the internal compute type for convolution when the input data
  // type is FP16 is only supported on architectures with true fp16 support
  // (compute capability 5.3 and 6.0). Setting this to false in an unsupported
  // architecture will cause internal errors.
  static constexpr bool kDefaultFlag = true;
};

// A helper struct to decide whether to use FP32 as the internal compute type
// for rnn when the input data type is FP16. At present it is turned off,
// users can explicitly control them through an env-var
// TF_FP16_RNN_USE_FP32_COMPUTE.
// After the TODO below is fixed, users should almost always use fp32 compute
// type for training. Using fp16 might suffer suboptimal accuracy due to loss
// in precision.
struct RnnDoFP32ComputationFP16Input {
  static constexpr const char* kName = "TF_FP16_RNN_USE_FP32_COMPUTE";
  static constexpr bool kDefaultFlag = true;
};

namespace {

bool GenericEngineFilter(cudnnBackendDescriptor_t engine_config,
                         bool disable_winograd, bool disable_nondeterminism,
                         bool disable_tensor_core) {
  bool ret = cudnn_frontend::hasNumericalNote<
      CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(engine_config);

  if (disable_winograd) {
    ret |= cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_WINOGRAD>(
        engine_config);
  }

  if (disable_nondeterminism) {
    ret |=
        cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(
            engine_config);
  }

  if (disable_tensor_core) {
    ret |= cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(
        engine_config);
  }

  return ret;
}

}  // namespace

cudnnDataType_t GetRnnComputeType(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kFloat:
      return CUDNN_DATA_FLOAT;
    case dnn::DataType::kDouble:
      return CUDNN_DATA_DOUBLE;
    case dnn::DataType::kHalf:
      if (CudnnEnvVar<RnnDoFP32ComputationFP16Input>::IsEnabled()) {
        return CUDNN_DATA_FLOAT;
      } else {
        return CUDNN_DATA_HALF;
      }
    default:
      LOG(FATAL) << "Invalid RNN data type: " << static_cast<int>(data_type);
  }
}

dnn::DataType GetConvActivationType(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kFloat:
    case dnn::DataType::kDouble:
      return data_type;
    // TODO(awpr): it's not clear whether float-precision activations on
    // half-precision convs are supported; double-check.
    case dnn::DataType::kHalf:
      return CudnnEnvVar<ConvDoFP32ComputationFP16Input>::IsEnabled()
                 ? dnn::DataType::kFloat
                 : dnn::DataType::kHalf;
    case dnn::DataType::kInt8:
    case dnn::DataType::kInt32:  // TODO(awpr): does int32 do blending in float?
      return dnn::DataType::kFloat;
    // TODO(awpr): as with kHalf, this is not clear.
    case dnn::DataType::kBF16:
      return CudnnEnvVar<ConvDoFP32ComputationFP16Input>::IsEnabled()
                 ? dnn::DataType::kFloat
                 : dnn::DataType::kBF16;
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

dnn::DataType GetConvAccumulatorType(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kFloat:
    case dnn::DataType::kDouble:
      return data_type;
    case dnn::DataType::kHalf:
      return CudnnEnvVar<ConvDoFP32ComputationFP16Input>::IsEnabled()
                 ? dnn::DataType::kFloat
                 : dnn::DataType::kHalf;
    case dnn::DataType::kInt8:
    case dnn::DataType::kInt32:
      return dnn::DataType::kInt32;
    case dnn::DataType::kBF16:
      return CudnnEnvVar<ConvDoFP32ComputationFP16Input>::IsEnabled()
                 ? dnn::DataType::kFloat
                 : dnn::DataType::kBF16;
    case dnn::DataType::kF8E4M3FN:
    case dnn::DataType::kF8E5M2:
      return dnn::DataType::kFloat;
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

namespace {
static bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
  (void)engine_config;
  return false;
}

cudnnBackendDescriptorType_t GetCudnnConvolutionType(
    dnn::ConvolutionKind kind) {
  cudnnBackendDescriptorType_t conv_mode;
  switch (kind) {
    case dnn::ConvolutionKind::FORWARD:
    case dnn::ConvolutionKind::FORWARD_GRAPH: {
      conv_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_DATA: {
      conv_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_FILTER: {
      conv_mode =
          CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
      break;
    }
    default:
      LOG(FATAL) << "Unexpected convolution kind " << static_cast<int>(kind);
      break;
  }
  return conv_mode;
}

// Cudnn only supports vectorization over the channel dimension (e.g., int8x4,
// or int8x32).
std::tuple<int, int> GetTensorVectorSizeAndDim(
    const dnn::BatchDescriptor& tensor, dnn::DataType element_type) {
  int vector_size = 1;
  int vector_dim = -1;
  if (element_type == dnn::DataType::kInt8) {
    if (tensor.layout() == dnn::DataLayout::kBatchDepthYX4) {
      vector_size = 4;
      vector_dim = 1;
    } else if (tensor.layout() == dnn::DataLayout::kBatchDepthYX32) {
      vector_size = 32;
      vector_dim = 1;
    }
  }
  return std::make_tuple(vector_size, vector_dim);
}

std::tuple<int, int> GetTensorVectorSizeAndDim(
    const dnn::FilterDescriptor& filter, dnn::DataType element_type) {
  int vector_size = 1;
  int vector_dim = -1;
  if (element_type == dnn::DataType::kInt8) {
    if (filter.layout() == dnn::FilterLayout::kOutputInputYX4) {
      vector_size = 4;
      vector_dim = 1;
    } else if (filter.layout() == dnn::FilterLayout::kOutputInputYX32 ||
               filter.layout() ==
                   dnn::FilterLayout::kOutputInputYX32_CudnnReordered) {
      vector_size = 32;
      vector_dim = 1;
    }
  }
  return std::make_tuple(vector_size, vector_dim);
}

absl::StatusOr<cudnn_frontend::Tensor> CreateCudnnTensor(
    absl::Span<const int64_t> dims, absl::Span<const int64_t> strides,
    int64_t uid, dnn::DataType dtype, int64_t vec_count, int64_t vec_dim,
    bool is_virtual = false,
    cudnnBackendTensorReordering_t cudnn_tensor_order_type =
        CUDNN_TENSOR_REORDERING_NONE,
    bool is_value = false) {
  auto tensor = cudnn_frontend::TensorBuilder()
                    .setDim(dims.size(), dims.data())
                    .setStride(strides.size(), strides.data())
                    .setId(uid)
                    .setAlignment(32)
                    .setDataType(ToCudnnDataType(dtype))
                    .setVectorCountAndDimension(vec_count, vec_dim)
                    .setVirtual(is_virtual)
                    .setReorderType(cudnn_tensor_order_type)
                    .setByValue(is_value)
                    .build();
  RETURN_MSG_IF_CUDNN_ERROR(tensor);
  return tensor;
}

absl::StatusOr<cudnn_frontend::Tensor> CreateCudnnTensor(
    const cudnn_frontend::Tensor& original, int64_t uid, dnn::DataType dtype,
    bool is_virtual = false) {
  auto tensor = cudnn_frontend::TensorBuilder()
                    .cloneFrom(original, uid)
                    .setAlignment(32)
                    .setDataType(ToCudnnDataType(dtype))
                    .setVirtual(is_virtual)
                    .build();
  RETURN_MSG_IF_CUDNN_ERROR(tensor);
  return tensor;
}

absl::StatusOr<cudnn_frontend::PointWiseDesc> CreatePwDesc(
    dnn::DataType dtype, cudnnPointwiseMode_t mode) {
  auto pw_desc_created = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(mode)
                             .setComputeType(ToCudnnDataType(dtype))
                             .build();
  RETURN_MSG_IF_CUDNN_ERROR(pw_desc_created);
  return pw_desc_created;
}

absl::StatusOr<cudnn_frontend::Operation> CreateUnaryPwOp(
    cudnn_frontend::Tensor const& xDesc, cudnn_frontend::Tensor const& yDesc,
    cudnn_frontend::PointWiseDesc const& pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(xDesc)
                           .setyDesc(yDesc)
                           .setpwDesc(pwDesc)
                           .build();
  RETURN_MSG_IF_CUDNN_ERROR(pw_op_created);
  return pw_op_created;
}

absl::StatusOr<cudnn_frontend::Operation> CreateBinaryPwOp(
    cudnn_frontend::Tensor const& xDesc, cudnn_frontend::Tensor const& bDesc,
    cudnn_frontend::Tensor const& yDesc,
    cudnn_frontend::PointWiseDesc const& pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(xDesc)
                           .setbDesc(bDesc)
                           .setyDesc(yDesc)
                           .setpwDesc(pwDesc)
                           .build();
  RETURN_MSG_IF_CUDNN_ERROR(pw_op_created);
  return pw_op_created;
}

absl::StatusOr<cudnn_frontend::Operation> CreateTernaryPwOp(
    cudnn_frontend::Tensor const& xDesc, cudnn_frontend::Tensor const& bDesc,
    cudnn_frontend::Tensor const& tDesc, cudnn_frontend::Tensor const& yDesc,
    cudnn_frontend::PointWiseDesc const& pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(xDesc)
                           .setbDesc(bDesc)
                           .settDesc(tDesc)
                           .setyDesc(yDesc)
                           .setpwDesc(pwDesc)
                           .build();
  RETURN_MSG_IF_CUDNN_ERROR(pw_op_created);
  return pw_op_created;
}

absl::StatusOr<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetCudnnOperationGraph(dnn::ConvolutionKind kind, dnn::DataType input_type,
                       dnn::DataType output_type,
                       const dnn::BatchDescriptor& input_descriptor,
                       const dnn::FilterDescriptor& filter_descriptor,
                       const dnn::BatchDescriptor& output_descriptor,
                       const dnn::ConvolutionDescriptor& convolution_descriptor,
                       CudnnHandle& cudnn) {
  PreloadCudnnSubLibsHelper(kind);

  cudnnBackendDescriptorType_t conv_mode = GetCudnnConvolutionType(kind);

  // x tensor.
  int vector_size, vector_dim;
  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(input_descriptor, input_type);
  std::vector<int64_t> input_dims = input_descriptor.vectorized_dims(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> input_strides = input_descriptor.vectorized_strides(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  TF_ASSIGN_OR_RETURN(auto tensor_x,
                      CreateCudnnTensor(input_dims, input_strides, 'x',
                                        input_type, vector_size, vector_dim));

  // y tensor.
  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(output_descriptor, output_type);
  std::vector<int64_t> output_dims = output_descriptor.vectorized_dims(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> output_strides = output_descriptor.vectorized_strides(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  TF_ASSIGN_OR_RETURN(auto tensor_y,
                      CreateCudnnTensor(output_dims, output_strides, 'y',
                                        output_type, vector_size, vector_dim));

  // w tensor.
  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(filter_descriptor, input_type);
  std::vector<int64_t> filter_dims = filter_descriptor.vectorized_dims(
      dnn::FilterLayout::kOutputInputYX, vector_size, vector_dim);
  std::vector<int64_t> filter_strides = filter_descriptor.vectorized_strides(
      dnn::FilterLayout::kOutputInputYX, vector_size, vector_dim);

  cudnnBackendTensorReordering_t tensor_ordering_type =
      filter_descriptor.layout() ==
              dnn::FilterLayout::kOutputInputYX32_CudnnReordered
          ? CUDNN_TENSOR_REORDERING_INT8x32
          : CUDNN_TENSOR_REORDERING_NONE;

  TF_ASSIGN_OR_RETURN(
      auto tensor_w,
      CreateCudnnTensor(filter_dims, filter_strides, 'w', input_type,
                        vector_size, vector_dim,
                        /*is_virtual=*/false, tensor_ordering_type));

  // conv_desc.
  auto mode = convolution_descriptor.convolution_not_crosscorr()
                  ? CUDNN_CONVOLUTION
                  : CUDNN_CROSS_CORRELATION;

  int conv_dim = convolution_descriptor.ndims();

  auto accumulator_type = ToCudnnDataType(GetConvAccumulatorType(input_type));
  CHECK_NE(convolution_descriptor.pad_alignment(),
           dnn::PadAlignment::kTensorFlowPadding)
      << "TensorFlow padding alignment is not supported.";

  auto conv_desc =
      cudnn_frontend::ConvDescBuilder()
          .setComputeType(accumulator_type)
          .setMathMode(mode)
          .setSpatialDimCount(conv_dim)
          .setSpatialStride(conv_dim, convolution_descriptor.strides().data())
          .setPrePadding(conv_dim, convolution_descriptor.padding().data())
          .setPostPadding(conv_dim, convolution_descriptor.padding().data())
          .setDilation(conv_dim, convolution_descriptor.dilations().data())
          .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  double alpha = 1.0;
  double beta = 0.0;

  // CUDNN Operation
  auto op = cudnn_frontend::OperationBuilder(conv_mode)
                .setxDesc(tensor_x)
                .setyDesc(tensor_y)
                .setwDesc(tensor_w)
                .setcDesc(conv_desc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
  RETURN_MSG_IF_CUDNN_ERROR(op);

  // CUDNN OperationGraph
  std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                     .setHandle(cudnn.handle())
                     .setOperationGraph(ops.size(), ops.data())
                     .build();
  RETURN_MSG_IF_CUDNN_ERROR(opGraph);

  VLOG(4) << "\nTensor_x: " << tensor_x.describe()
          << "\nTensor_y: " << tensor_y.describe()
          << "\nTensor_w: " << tensor_w.describe()
          << "\nConv: " << conv_desc.describe() << "\nOp: " << op.describe()
          << "\nOpGraph: " << opGraph.describe();

  return std::make_unique<cudnn_frontend::OperationGraph>(std::move(opGraph));
}

absl::StatusOr<dnn::DataType> PrimitiveTypeStringToDnnType(
    std::string data_type_string) {
  if (data_type_string == "f8e4m3fn") {
    return dnn::DataType::kF8E4M3FN;
  } else if (data_type_string == "f8e5m2") {
    return dnn::DataType::kF8E5M2;
  } else if (data_type_string == "bf16") {
    return dnn::DataType::kBF16;
  } else if (data_type_string == "f16") {
    return dnn::DataType::kHalf;
  } else if (data_type_string == "f32") {
    return dnn::DataType::kFloat;
  } else {
    return tsl::errors::Internal("Unsupported primitive type.");
  }
}

using OpMode = std::variant<cudnnConvolutionMode_t, cudnnPointwiseMode_t,
                            cudnnReduceTensorOp_t>;

enum class TensorKind { kNone, kScalar, kTensor };

absl::StatusOr<std::tuple<TensorKind, TensorKind, OpMode>>
OpNameStringToOperandKindAndMode(std::string opstring) {
#define KINDS_AND_MODE_FROM_OP_STRING(OPSTRING, BINARYOPERANDKIND,    \
                                      AUXOUTPUTKIND, PWMODE)          \
  if (opstring == OPSTRING) {                                         \
    return std::make_tuple(BINARYOPERANDKIND, AUXOUTPUTKIND, PWMODE); \
  }

  KINDS_AND_MODE_FROM_OP_STRING("add", TensorKind::kTensor, TensorKind::kTensor,
                                CUDNN_POINTWISE_ADD)
  KINDS_AND_MODE_FROM_OP_STRING("min", TensorKind::kTensor, TensorKind::kTensor,
                                CUDNN_POINTWISE_MIN)
  KINDS_AND_MODE_FROM_OP_STRING("max", TensorKind::kTensor, TensorKind::kTensor,
                                CUDNN_POINTWISE_MAX)
  KINDS_AND_MODE_FROM_OP_STRING("elu", TensorKind::kNone, TensorKind::kTensor,
                                CUDNN_POINTWISE_ELU_FWD)
  KINDS_AND_MODE_FROM_OP_STRING("relu", TensorKind::kNone, TensorKind::kTensor,
                                CUDNN_POINTWISE_RELU_FWD)
  KINDS_AND_MODE_FROM_OP_STRING("scale", TensorKind::kScalar,
                                TensorKind::kTensor, CUDNN_POINTWISE_MUL)
  KINDS_AND_MODE_FROM_OP_STRING("invscale", TensorKind::kScalar,
                                TensorKind::kTensor, CUDNN_POINTWISE_DIV)
  KINDS_AND_MODE_FROM_OP_STRING("amax", TensorKind::kNone, TensorKind::kScalar,
                                CUDNN_REDUCE_TENSOR_AMAX)
#undef KINDS_AND_MODE_FROM_OP_STRING

  return tsl::errors::Internal("Unknown op.");
}

// Struct describing the convolution, pointwise and reduction ops in the
// graph.
struct OpDescriptor {
  int uid;                        // The UID of the op.
  std::vector<int> operand_uids;  // The UIDs of the operands of the op that
                                  // are part of the graph.
  OpMode mode;                    // The mode describing the op.
  TensorKind operand_kind;        // The kind of a second operand.
  TensorKind result_kind;         // The kind of the output.
  dnn::DataType result_type;      // The type of the output.
  bool is_virtual;                // A virtual op has a user within the graph.
  int sequence_index;             // The index of the op in the sequence.
};

// Class describing the graph of ops to be fused into the cuDNN convolution
// Custom Call.
class OpGraph {
 public:
  OpGraph() = default;

  absl::Status AddOp(int uid, std::vector<int> operand_uids, OpMode mode,
                     TensorKind operand_kind, TensorKind result_kind,
                     dnn::DataType result_type) {
    ops_.emplace_back(OpDescriptor({uid, operand_uids, mode, operand_kind,
                                    result_kind, result_type, false, -1}));
    // If they exist, the operands are virtual.
    for (int operand_uid : operand_uids) {
      auto it = std::find_if(
          ops_.begin(), ops_.end(),
          [operand_uid](OpDescriptor op) { return op.uid == operand_uid; });
      if (it == ops_.end()) {
        return tsl::errors::Internal("Unknown ID.");
      }
      it->is_virtual = true;
    }
    return absl::OkStatus();
  }

  absl::StatusOr<OpDescriptor> FindOpDescriptor(int uid) const {
    auto it = std::find_if(ops_.begin(), ops_.end(),
                           [uid](OpDescriptor op) { return op.uid == uid; });
    if (it == ops_.end()) {
      return tsl::errors::Internal("Unknown ID.");
    }
    return *it;
  }

  absl::StatusOr<OpDescriptor> OpDescriptorAt(int index) const {
    if (index >= Size()) {
      return tsl::errors::Internal("Index exceeds bounds.");
    }
    return ops_[index];
  }

  absl::Status SetSequenceIndex(int uid, int index) {
    auto it = std::find_if(ops_.begin(), ops_.end(),
                           [uid](OpDescriptor op) { return op.uid == uid; });
    if (it == ops_.end()) {
      return tsl::errors::Internal("Unknown ID.");
    }
    it->sequence_index = index;
    return absl::OkStatus();
  }

  bool Empty() const { return ops_.empty(); }

  int Size() const { return ops_.size(); }

 private:
  std::vector<OpDescriptor> ops_;
};

// TODO(philipphack): Consider merging with GetCudnnOperationGraph and
// GetCudnnFusedOperationGraph.

// Returns a generic cuDNN OperationGraph for ForwardGraph convolutions with the
// fused ops listed in serialized_graph and the associated set of UIDs of
// non-virtual cuDNN tensors.
absl::StatusOr<std::pair<std::unique_ptr<cudnn_frontend::OperationGraph>,
                         std::vector<int64_t>>>
GetGenericCudnnOperationGraph(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    CudnnHandle& cudnn, std::string serialized_graph = "") {
  PreloadCudnnSubLibsHelper(kind);

  // The format of the serialized graph describing a sequence of ops fused
  // into the cuDNN convolution Custom Call is
  // "UID:[output_type]conv();UID:[output_type]op_name(operand
  // UIDs);UID:[output_type]op_name(operand UIDs);..." with the convolution
  // assumed to be the first op in the graph. Operand UIDs identifying ops
  // outside the serialized graph are elided.
  auto deserialize_cudnn_graph = [&]() -> absl::StatusOr<OpGraph> {
    OpGraph op_graph;
    std::string::size_type pos = 0;
    while (pos < serialized_graph.size()) {
      OpMode mode;
      dnn::DataType output_type;
      std::string::size_type m = serialized_graph.find('[', pos);
      std::string::size_type n = serialized_graph.find(']', pos);
      int uid = std::stoi(serialized_graph.substr(pos, m - pos - 1));
      std::string data_type_string = serialized_graph.substr(m + 1, n - m - 1);
      m = serialized_graph.find('(', pos);
      std::string op_string = serialized_graph.substr(n + 1, m - n - 1);
      std::vector<int> operands;
      std::string::size_type l = serialized_graph.find_first_of(",)", m + 1);
      while (l > m + 1) {
        operands.push_back(
            std::stoi(serialized_graph.substr(m + 1, l - m - 1)));
        if (serialized_graph[l] == ')') {
          break;
        }
        m = l;
        l = serialized_graph.find_first_of(",)", m + 1);
      }

      if (serialized_graph[l + 1] != ';') {
        return tsl::errors::Internal(
            "Unexpected character in graph serialization.");
      }
      pos = l + 2;

      TF_ASSIGN_OR_RETURN(output_type,
                          PrimitiveTypeStringToDnnType(data_type_string));
      TensorKind binary_operand_kind, output_kind;
      if (op_string == "conv") {
        if (!op_graph.Empty()) {
          return tsl::errors::Internal(
              "The graph must not contain more than one convolution op.");
        }
        if (!operands.empty()) {
          return tsl::errors::Internal(
              "Convolution op must not have operands in the graph.");
        }
        binary_operand_kind = TensorKind::kNone;
        output_kind = TensorKind::kTensor;
        mode = convolution_descriptor.convolution_not_crosscorr()
                   ? CUDNN_CONVOLUTION
                   : CUDNN_CROSS_CORRELATION;
      } else {
        if (op_graph.Empty()) {
          return tsl::errors::Internal(
              "The first op in the graph must be a convolution.");
        }
        if (operands.empty()) {
          return tsl::errors::Internal(
              "Non-convolution op must have one or more operands in the "
              "graph.");
        }
        TF_ASSIGN_OR_RETURN(std::tie(binary_operand_kind, output_kind, mode),
                            OpNameStringToOperandKindAndMode(op_string));
      }
      TF_RETURN_IF_ERROR(op_graph.AddOp(
          uid, operands, mode, binary_operand_kind, output_kind, output_type));
    }
    return op_graph;
  };

  TF_ASSIGN_OR_RETURN(OpGraph op_graph, deserialize_cudnn_graph());
  if (op_graph.Empty()) {
    return tsl::errors::Internal("No supported ops in convolution graph.");
  }

  std::vector<int64_t> virtual_uids, operand_uids, output_uids;
  std::vector<cudnn_frontend::Operation> ops;
  std::vector<cudnn_frontend::Tensor> result_tensors;

  auto next_uid = [&operand_uids, &output_uids, &virtual_uids](
                      bool is_operand, bool is_virtual) -> int64_t {
    DCHECK(!(is_operand && is_virtual));
    int64_t max_operand_uid =
        operand_uids.empty()
            ? 0
            : *std::max_element(operand_uids.begin(), operand_uids.end());
    int64_t max_output_uid =
        output_uids.empty()
            ? 0
            : *std::max_element(output_uids.begin(), output_uids.end());
    int64_t max_virtual_uid =
        virtual_uids.empty()
            ? 0
            : *std::max_element(virtual_uids.begin(), virtual_uids.end());
    int64_t next_uid =
        std::max({max_operand_uid, max_output_uid, max_virtual_uid}) + 1;

    if (is_operand) {
      operand_uids.push_back(next_uid);
    } else if (is_virtual) {
      virtual_uids.push_back(next_uid);
    } else {
      output_uids.push_back(next_uid);
    }
    return next_uid;
  };

  //  Input tensor.
  int vector_size, vector_dim;
  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(input_descriptor, input_type);
  std::vector<int64_t> input_dims = input_descriptor.vectorized_dims(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> input_strides = input_descriptor.vectorized_strides(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  TF_ASSIGN_OR_RETURN(
      auto tensor_x,
      CreateCudnnTensor(input_dims, input_strides,
                        next_uid(/*is_operand=*/true, /*is_virtual=*/false),
                        input_type, vector_size, vector_dim));

  // Filter tensor.
  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(filter_descriptor, input_type);
  std::vector<int64_t> filter_dims = filter_descriptor.vectorized_dims(
      dnn::FilterLayout::kOutputInputYX, vector_size, vector_dim);
  std::vector<int64_t> filter_strides = filter_descriptor.vectorized_strides(
      dnn::FilterLayout::kOutputInputYX, vector_size, vector_dim);

  cudnnBackendTensorReordering_t tensor_ordering_type =
      filter_descriptor.layout() ==
              dnn::FilterLayout::kOutputInputYX32_CudnnReordered
          ? CUDNN_TENSOR_REORDERING_INT8x32
          : CUDNN_TENSOR_REORDERING_NONE;
  TF_ASSIGN_OR_RETURN(
      auto tensor_w,
      CreateCudnnTensor(filter_dims, filter_strides,
                        next_uid(/*is_operand=*/true, /*is_virtual=*/false),
                        input_type, vector_size, vector_dim,
                        /*is_virtual=*/false, tensor_ordering_type));

  // Result tensor.
  TF_ASSIGN_OR_RETURN(OpDescriptor op_descriptor, op_graph.OpDescriptorAt(0));
  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(output_descriptor, op_descriptor.result_type);
  std::vector<int64_t> output_dims = output_descriptor.vectorized_dims(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> output_strides = output_descriptor.vectorized_strides(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  TF_ASSIGN_OR_RETURN(
      auto tensor_y,
      CreateCudnnTensor(output_dims, output_strides,
                        next_uid(/*is_operand=*/false,
                                 /*is_virtual=*/op_descriptor.is_virtual),
                        op_descriptor.result_type, vector_size, vector_dim,
                        /*is_virtual=*/op_descriptor.is_virtual));

  auto accumulator_type = ToCudnnDataType(GetConvAccumulatorType(input_type));
  CHECK_NE(convolution_descriptor.pad_alignment(),
           dnn::PadAlignment::kTensorFlowPadding)
      << "TensorFlow padding alignment is not supported.";

  int conv_dim = convolution_descriptor.ndims();
  auto conv_desc =
      cudnn_frontend::ConvDescBuilder()
          .setComputeType(accumulator_type)
          .setMathMode(std::get<cudnnConvolutionMode_t>(op_descriptor.mode))
          .setSpatialDimCount(conv_dim)
          .setSpatialStride(conv_dim, convolution_descriptor.strides().data())
          .setPrePadding(conv_dim, convolution_descriptor.padding().data())
          .setPostPadding(conv_dim, convolution_descriptor.padding().data())
          .setDilation(conv_dim, convolution_descriptor.dilations().data())
          .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  // CUDNN Operation
  double alpha = 1.0;
  double beta = 0.0;
  cudnnBackendDescriptorType_t conv_mode = GetCudnnConvolutionType(kind);
  cudnn_frontend::Operation op = cudnn_frontend::OperationBuilder(conv_mode)
                                     .setxDesc(tensor_x)
                                     .setyDesc(tensor_y)
                                     .setwDesc(tensor_w)
                                     .setcDesc(conv_desc)
                                     .setAlpha(alpha)
                                     .setBeta(beta)
                                     .build();
  RETURN_MSG_IF_CUDNN_ERROR(op);

  VLOG(4) << "\nTensor_x: " << tensor_x.describe()
          << "\nTensor_y: " << tensor_y.describe()
          << "\nTensor_w: " << tensor_w.describe()
          << "\nConv desc: " << conv_desc.describe()
          << "\nOp: " << op.describe();

  // Add the convolution to the cuDNN graph.
  ops.push_back(std::move(op));
  result_tensors.push_back(std::move(tensor_y));
  TF_RETURN_IF_ERROR(
      op_graph.SetSequenceIndex(op_descriptor.uid, ops.size() - 1));

  for (int op_index = 1; op_index < op_graph.Size(); ++op_index) {
    TF_ASSIGN_OR_RETURN(op_descriptor, op_graph.OpDescriptorAt(op_index));
    std::vector<OpDescriptor> preceding_ops;
    preceding_ops.reserve(op_descriptor.operand_uids.size());
    for (int operand_uid : op_descriptor.operand_uids) {
      preceding_ops.emplace_back(
          op_graph.FindOpDescriptor(operand_uid).value());
    }
    std::optional<cudnn_frontend::Tensor> external_operand;

    // Create a cuDNN tensor for the potential non-graph operand of
    // non-convolution binary ops (side input).
    if (op_descriptor.operand_kind == TensorKind::kScalar &&
        preceding_ops.size() == 1) {
      std::vector<int64_t> scale_dim(4, 1);
      TF_ASSIGN_OR_RETURN(
          external_operand,
          CreateCudnnTensor(scale_dim, scale_dim,
                            next_uid(/*is_operand=*/true, /*is_virtual=*/false),
                            preceding_ops[0].result_type, 1, -1));
      VLOG(4) << "\nPointwise operand: " << external_operand->describe();
    } else if (op_descriptor.operand_kind == TensorKind::kTensor &&
               preceding_ops.size() == 1) {
      TF_ASSIGN_OR_RETURN(
          external_operand,
          CreateCudnnTensor(tensor_y,
                            next_uid(/*is_operand=*/true, /*is_virtual=*/false),
                            preceding_ops[0].result_type,
                            /*is_virtual=*/false));
      VLOG(4) << "\nPointwise operand: " << external_operand->describe();
    }

    // Create the result tensor of the op.
    if (op_descriptor.result_kind == TensorKind::kScalar) {
      std::vector<int64_t> scale_dim(4, 1);
      TF_ASSIGN_OR_RETURN(cudnn_frontend::Tensor result,
                          CreateCudnnTensor(scale_dim, scale_dim,
                                            next_uid(/*is_operand=*/false,
                                                     /*is_virtual=*/false),
                                            op_descriptor.result_type, 1, -1));
      VLOG(4) << "\nScalar result: " << result.describe();
      result_tensors.push_back(std::move(result));
    } else if (op_descriptor.result_kind == TensorKind::kTensor) {
      TF_ASSIGN_OR_RETURN(
          cudnn_frontend::Tensor result,
          CreateCudnnTensor(tensor_y,
                            next_uid(/*is_operand=*/false,
                                     /*is_virtual=*/op_descriptor.is_virtual),
                            op_descriptor.result_type,
                            /*is_virtual=*/op_descriptor.is_virtual));
      VLOG(4) << "\nTensor result: " << result.describe();
      result_tensors.push_back(std::move(result));
    }

    if (std::holds_alternative<cudnnPointwiseMode_t>(op_descriptor.mode)) {
      // Create the descriptor for the pointwise op.
      cudnn_frontend::PointWiseDesc desc =
          cudnn_frontend::PointWiseDescBuilder()
              .setMode(std::get<cudnnPointwiseMode_t>(op_descriptor.mode))
              .setMathPrecision(CUDNN_DATA_FLOAT)
              .build();
      VLOG(4) << "\nPointwise op desc: " << desc.describe();
      // Add the op to the operation graph.
      if (external_operand.has_value()) {
        ops.emplace_back(
            cudnn_frontend::OperationBuilder(
                CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(result_tensors[preceding_ops[0].sequence_index])
                .setbDesc(external_operand.value())
                .setyDesc(result_tensors.back())
                .setpwDesc(desc)
                .build());
      } else if (preceding_ops.size() == 2) {
        ops.emplace_back(
            cudnn_frontend::OperationBuilder(
                CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(result_tensors[preceding_ops[0].sequence_index])
                .setbDesc(result_tensors[preceding_ops[1].sequence_index])
                .setyDesc(result_tensors.back())
                .setpwDesc(desc)
                .build());
      } else {
        ops.emplace_back(
            cudnn_frontend::OperationBuilder(
                CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(result_tensors[preceding_ops[0].sequence_index])
                .setyDesc(result_tensors.back())
                .setpwDesc(desc)
                .build());
      }
    } else if (std::holds_alternative<cudnnReduceTensorOp_t>(
                   op_descriptor.mode)) {
      // Create the descriptor for the reduction op.
      cudnn_frontend::ReductionDesc desc =
          cudnn_frontend::ReductionDescBuilder()
              .setMathPrecision(CUDNN_DATA_FLOAT)
              .setReductionOp(
                  std::get<cudnnReduceTensorOp_t>(op_descriptor.mode))
              .build();
      VLOG(4) << "\nReduction op desc: " << desc.describe();

      // Add the op to the operation graph.
      ops.emplace_back(
          cudnn_frontend::OperationBuilder(
              CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
              .setxDesc(result_tensors[preceding_ops[0].sequence_index])
              .setyDesc(result_tensors.back())
              .setreductionDesc(desc)
              .build());
    }
    TF_RETURN_IF_ERROR(
        op_graph.SetSequenceIndex(op_descriptor.uid, ops.size() - 1));
  }

  // Construct the cuDNN OperationGraph.
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                     .setHandle(cudnn.handle())
                     .setOperationGraph(ops)
                     .build();
  RETURN_MSG_IF_CUDNN_ERROR(opGraph);
  VLOG(4) << "\ncuDNN OperationGraph: " << opGraph.describe();

  // The non-virtual UIDS are the UIDs of the operands followed by the UIDs of
  // the outputs.
  std::vector<int64_t> non_virtual_uids = operand_uids;
  non_virtual_uids.insert(non_virtual_uids.end(), output_uids.begin(),
                          output_uids.end());

  return std::make_pair(
      std::make_unique<cudnn_frontend::OperationGraph>(std::move(opGraph)),
      non_virtual_uids);
}

bool SideInputNeeded(dnn::ActivationMode activation_mode, double conv_scale,
                     double side_input_scale) {
  // Cudnn uses precompiled kernels to perform the Conv-Add-BiasAdd-Act when the
  // activation is Relu or Identity and this requires the "side_input" for the
  // Add. For other activations, cudnn uses the runtime-compiled kernels.
  // However, for this case, we need to drop the Add node and use
  // Conv-BiasAdd-Act pattern to trigger the correct cudnn path.
  // TODO(kaixih@nvidia): We should remove this WAR when the cudnn fixes it.
  bool check_activation = activation_mode == dnn::ActivationMode::kNone ||
                          activation_mode == dnn::ActivationMode::kRelu;
  bool check_scale = conv_scale != 1.0 || side_input_scale != 0.0;
  return check_activation || check_scale;
}

absl::StatusOr<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetCudnnFusedOperationGraph(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType bias_type, dnn::DataType output_type, double alpha,
    double alpha2, double leakyrelu_alpha,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    dnn::BatchDescriptor bias_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::ActivationMode activation_mode, CudnnHandle& cudnn) {
  PreloadCudnnSubLibsHelper(kind);

  cudnnBackendDescriptorType_t conv_mode = GetCudnnConvolutionType(kind);
  dnn::DataType accumulator_type = GetConvAccumulatorType(input_type);
  dnn::DataType activation_type = GetConvActivationType(input_type);

  // CUDNN fused operation supports the pattern in the form of
  // Conv + Add + BiasAdd + Act. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // Conv   : input: tensor_x, tensor_w;    output: tensor_conv (virtual)
  // Add    : input: tensor_conv, tensor_z; output: tensor_add (virtual)
  // BiasAdd: input: tensor_add, tensor_b;  output: tensor_bias (virtual)
  // Act    : input: tensor_bias;           output: tensor_y
  int vector_size, vector_dim;
  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(input_descriptor, input_type);
  std::vector<int64_t> input_dims = input_descriptor.vectorized_dims(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> input_strides = input_descriptor.vectorized_strides(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  TF_ASSIGN_OR_RETURN(auto tensor_x,
                      CreateCudnnTensor(input_dims, input_strides, 'x',
                                        input_type, vector_size, vector_dim));

  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(output_descriptor, output_type);
  std::vector<int64_t> output_dims = output_descriptor.vectorized_dims(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> output_strides = output_descriptor.vectorized_strides(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  TF_ASSIGN_OR_RETURN(auto tensor_y,
                      CreateCudnnTensor(output_dims, output_strides, 'y',
                                        output_type, vector_size, vector_dim));

  TF_ASSIGN_OR_RETURN(auto tensor_z,
                      CreateCudnnTensor(output_dims, output_strides, 'z',
                                        output_type, vector_size, vector_dim));

  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(filter_descriptor, input_type);
  std::vector<int64_t> filter_dims = filter_descriptor.vectorized_dims(
      dnn::FilterLayout::kOutputInputYX, vector_size, vector_dim);
  std::vector<int64_t> filter_strides = filter_descriptor.vectorized_strides(
      dnn::FilterLayout::kOutputInputYX, vector_size, vector_dim);

  cudnnBackendTensorReordering_t tensor_ordering_type =
      filter_descriptor.layout() ==
              dnn::FilterLayout::kOutputInputYX32_CudnnReordered
          ? CUDNN_TENSOR_REORDERING_INT8x32
          : CUDNN_TENSOR_REORDERING_NONE;

  TF_ASSIGN_OR_RETURN(
      auto tensor_w,
      CreateCudnnTensor(filter_dims, filter_strides, 'w', input_type,
                        vector_size, vector_dim,
                        /*is_virtual=*/false,
                        tensor_ordering_type));  // cuDNN 8.3 fails here

  // For the purposes of the cudnn graph, say that the bias tensor has the same
  // layout as the output tensor.  It doesn't actually matter, because bias is a
  // 1D array.  But we need to get the correct vectorization, otherwise the
  // cudnn graph API rejects this tensor, even though vectorized float tensors
  // aren't even a thing in cuDNN.
  bias_descriptor.set_layout(output_descriptor.layout());

  // Even more unnecessarily subtle: since vectorized float tensors don't exist,
  // `GetVectorSizeAndDim` ignores vectorized layouts for floating-point types,
  // so we have to ask it for vector sizes as if the type were `input_type`, as
  // opposed to `bias_type`.  For non-int8 types, these are the same anyway, so
  // this only affects int8 convolutions.
  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(bias_descriptor, input_type);
  std::vector<int64_t> bias_dims = bias_descriptor.vectorized_dims(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> bias_strides = bias_descriptor.vectorized_strides(
      dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  // Experimental cuDNN versions handle int8x32 convolutions with a bias
  // differently from stable versions. Which versions exactly are affected is
  // unknown, but fortunately we can write code that works for both flavors.
  //
  // cuDNN version *8.3* expects the bias tensor to have the reordering set to
  // CUDNN_TENSOR_REORDERING_NONE, and fails with CUDNN_STATUS_BAD_PARAM if
  // it's set to something else.
  //
  // cuDNN version *8.7* expects the bias tensor to have the reordering set to
  // CUDNN_TENSOR_REORDERING_INT8x32 (which is weird, as the bias_type is
  // kFloat). If it's not, then cuDNN silently does the reordering under the
  // hood, which yields incorrect results as we already do the reordering
  // ourselves.
  auto maybe_tensor_b = CreateCudnnTensor(
      bias_dims, bias_strides, 'b', bias_type, vector_size, vector_dim,
      /*is_virtual=*/false, tensor_ordering_type);
  if (!maybe_tensor_b.ok()) {
    maybe_tensor_b = CreateCudnnTensor(bias_dims, bias_strides, 'b', bias_type,
                                       vector_size, vector_dim);
  }
  TF_ASSIGN_OR_RETURN(auto tensor_b, std::move(maybe_tensor_b));

  std::tie(vector_size, vector_dim) =
      GetTensorVectorSizeAndDim(output_descriptor, output_type);
  TF_ASSIGN_OR_RETURN(
      auto tensor_conv,
      CreateCudnnTensor(output_dims, output_strides, 'C', accumulator_type,
                        vector_size, vector_dim, /*is_virtual=*/true));

  TF_ASSIGN_OR_RETURN(
      auto tensor_add,
      CreateCudnnTensor(output_dims, output_strides, 'A', activation_type,
                        vector_size, vector_dim, /*is_virtual=*/true));

  TF_ASSIGN_OR_RETURN(
      auto tensor_bias,
      CreateCudnnTensor(output_dims, output_strides, 'B', activation_type,
                        vector_size, vector_dim, /*is_virtual=*/true));

  // conv_desc.
  auto mode = convolution_descriptor.convolution_not_crosscorr()
                  ? CUDNN_CONVOLUTION
                  : CUDNN_CROSS_CORRELATION;

  int conv_dim = convolution_descriptor.ndims();

  CHECK_NE(convolution_descriptor.pad_alignment(),
           dnn::PadAlignment::kTensorFlowPadding)
      << "TensorFlow padding alignment is not supported.";

  cudnnDataType_t cudnn_convolution_type = ToCudnnDataType(accumulator_type);
  cudnnDataType_t cudnn_activation_type = ToCudnnDataType(activation_type);
  auto conv_desc =
      cudnn_frontend::ConvDescBuilder()
          .setComputeType(cudnn_convolution_type)
          .setMathMode(mode)
          .setSpatialDimCount(conv_dim)
          .setSpatialStride(conv_dim, convolution_descriptor.strides().data())
          .setPrePadding(conv_dim, convolution_descriptor.padding().data())
          .setPostPadding(conv_dim, convolution_descriptor.padding().data())
          .setDilation(conv_dim, convolution_descriptor.dilations().data())
          .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_desc);

  // CUDNN Operation
  auto conv_op = cudnn_frontend::OperationBuilder(conv_mode)
                     .setxDesc(tensor_x)
                     .setyDesc(tensor_conv)
                     .setwDesc(tensor_w)
                     .setcDesc(conv_desc)
                     .setAlpha(1.0f)
                     .setBeta(0.0f)
                     .build();
  RETURN_MSG_IF_CUDNN_ERROR(conv_op);

  // CUDNN OperationGraph
  absl::InlinedVector<cudnn_frontend::Operation const*, 4> ops = {&conv_op};

  bool need_add_op = SideInputNeeded(activation_mode, alpha, alpha2);

  std::optional<cudnn_frontend::PointWiseDesc_v8> add_desc;
  std::optional<cudnn_frontend::Operation_v8> add_op;
  if (need_add_op) {
    add_desc.emplace(cudnn_frontend::PointWiseDescBuilder()
                         .setMode(CUDNN_POINTWISE_ADD)
                         .setMathPrecision(cudnn_activation_type)
                         .build());
    RETURN_MSG_IF_CUDNN_ERROR(*add_desc);
    add_op.emplace(cudnn_frontend::OperationBuilder(
                       CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                       .setxDesc(conv_op.getOutputTensor())
                       .setbDesc(tensor_z)
                       .setyDesc(tensor_add)
                       .setpwDesc(*add_desc)
                       .setAlpha(alpha)
                       .setAlpha2(alpha2)
                       .build());
    RETURN_MSG_IF_CUDNN_ERROR(*add_op);
    ops.push_back(&*add_op);
  }

  auto bias_add_desc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(cudnn_activation_type)
                           .build();

  // If the activation is the identity function, then the bias-add is the last
  // op, and it writes to the output, tensor_y.  Otherwise, it writes to the
  // "virtual tensor" (temp buffer) tensor_bias, to which we apply the
  // activation.
  auto& bias_out_desc =
      activation_mode == dnn::ActivationMode::kNone ? tensor_y : tensor_bias;
  auto& bias_in_desc = need_add_op ? tensor_add : tensor_conv;
  auto bias_add_op = cudnn_frontend::OperationBuilder(
                         CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setxDesc(bias_in_desc)
                         .setbDesc(tensor_b)
                         .setyDesc(bias_out_desc)
                         .setpwDesc(bias_add_desc)
                         .build();
  RETURN_MSG_IF_CUDNN_ERROR(bias_add_op);
  ops.push_back(&bias_add_op);

  std::optional<cudnn_frontend::PointWiseDesc_v8> act_desc;
  switch (activation_mode) {
    case dnn::ActivationMode::kNone:
      break;
    case dnn::ActivationMode::kRelu:
      act_desc.emplace(cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(cudnn_activation_type)
                           .build());
      RETURN_MSG_IF_CUDNN_ERROR(*act_desc);

      break;
    case dnn::ActivationMode::kRelu6:
      act_desc.emplace(cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setReluUpperClip(6.0)
                           .setMathPrecision(cudnn_activation_type)
                           .build());
      RETURN_MSG_IF_CUDNN_ERROR(*act_desc);
      break;
    case dnn::ActivationMode::kElu:
      act_desc.emplace(cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ELU_FWD)
                           .setMathPrecision(cudnn_activation_type)
                           .build());
      RETURN_MSG_IF_CUDNN_ERROR(*act_desc);
      break;
    case dnn::ActivationMode::kLeakyRelu:
      act_desc.emplace(cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setReluLowerClipSlope(leakyrelu_alpha)
                           .setMathPrecision(cudnn_activation_type)
                           .build());
      RETURN_MSG_IF_CUDNN_ERROR(*act_desc);
      break;
    default:
      return tsl::errors::Internal("Unimplemented activation mode ",
                                   dnn::ActivationModeString(activation_mode));
  }

  std::optional<cudnn_frontend::Operation_v8> act_op;
  if (activation_mode != dnn::ActivationMode::kNone) {
    act_op.emplace(cudnn_frontend::OperationBuilder(
                       CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                       .setxDesc(bias_add_op.getOutputTensor())
                       .setyDesc(tensor_y)
                       .setpwDesc(*act_desc)
                       .build());
    RETURN_MSG_IF_CUDNN_ERROR(*act_op);
    ops.push_back(&*act_op);
  }

  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(cudnn.handle())
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(op_graph);

  VLOG(4) << "\nTensor_x: " << tensor_x.describe()
          << "\nTensor_y: " << tensor_y.describe()
          << "\nTensor_z: " << tensor_z.describe()
          << "\nTensor_w: " << tensor_w.describe()
          << "\nTensor_b: " << tensor_b.describe()
          << "\nTensor_conv: " << tensor_conv.describe()
          << "\nTensor_add: " << tensor_add.describe()
          << "\nTensor_bias: " << tensor_bias.describe()
          << "\nConv: " << conv_desc.describe() << "\nAdd: "
          << (add_desc.has_value() ? add_desc->describe() : "(skipped)")
          << "\nBiasAdd: " << bias_add_desc.describe()  //
          << "\nAct: "
          << (act_desc.has_value() ? act_desc->describe() : "(identity)")
          << "\nConvOp: " << conv_op.describe() << "\nAddOp: "
          << (add_op.has_value() ? add_op->describe() : "(skipped)")
          << "\nBiasAddOp: " << bias_add_op.describe()  //
          << "\nActOp: "
          << (act_op.has_value() ? act_op->describe() : "(identity)")
          << "\nOpGraph: " << op_graph.describe();

  return std::make_unique<cudnn_frontend::OperationGraph>(std::move(op_graph));
}

absl::StatusOr<std::unique_ptr<cudnn_frontend::OperationGraph>>
GetCudnnFusedMatmulGraph(dnn::DataType input_type, dnn::DataType bias_type,
                         dnn::DataType output_type, bool trans_a, bool trans_b,
                         uint64_t m_u, uint64_t n_u, uint64_t k_u, int64_t lda,
                         int64_t ldb, int64_t ldc,
                         const dnn::ActivationMode activation_mode,
                         CudnnHandle& cudnn) {
  dnn::DataType accumulator_type = GetConvAccumulatorType(input_type);
  dnn::DataType activation_type = GetConvActivationType(input_type);
  cudnnDataType_t cudnn_activation_type = ToCudnnDataType(activation_type);

  // CUDNN fused operation supports the pattern in the form of
  // Conv + BiasAdd + Act. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // Conv   : input: tensor_a, tensor_b;      output: tensor_matmul (virtual)
  // BiasAdd: input: tensor_matmul, tensor_z; output: tensor_bias   (virtual)
  // Act    : input: tensor_bias;             output: tensor_c
  int64_t m = static_cast<int64_t>(m_u);
  int64_t n = static_cast<int64_t>(n_u);
  int64_t k = static_cast<int64_t>(k_u);
  int vector_size = 1, vector_dim = -1;
  std::vector<int64_t> a_dims = {1, m, k};
  int64_t stride1 = trans_a ? 1 : lda;
  int64_t stride2 = trans_a ? lda : 1;
  std::vector<int64_t> a_strides = {m * k, stride1, stride2};
  TF_ASSIGN_OR_RETURN(auto tensor_a,
                      CreateCudnnTensor(a_dims, a_strides, 'a', input_type,
                                        vector_size, vector_dim));

  std::vector<int64_t> b_dims = {1, k, n};
  stride1 = trans_b ? 1 : ldb;
  stride2 = trans_b ? ldb : 1;
  std::vector<int64_t> b_strides = {k * n, stride1, stride2};
  TF_ASSIGN_OR_RETURN(auto tensor_b,
                      CreateCudnnTensor(b_dims, b_strides, 'b', input_type,
                                        vector_size, vector_dim));

  std::vector<int64_t> c_dims = {1, m, n};
  std::vector<int64_t> c_strides = {m * n, ldc, 1};
  TF_ASSIGN_OR_RETURN(auto tensor_c,
                      CreateCudnnTensor(c_dims, c_strides, 'c', output_type,
                                        vector_size, vector_dim));

  std::vector<int64_t> z_dims = {1, 1, n};
  std::vector<int64_t> z_strides = {n, n, 1};
  TF_ASSIGN_OR_RETURN(auto tensor_z,
                      CreateCudnnTensor(z_dims, z_strides, 'z', bias_type,
                                        vector_size, vector_dim));

  TF_ASSIGN_OR_RETURN(
      auto tensor_matmul,
      CreateCudnnTensor(c_dims, c_strides, 'M', accumulator_type, vector_size,
                        vector_dim, /*is_virtual=*/true));

  TF_ASSIGN_OR_RETURN(
      auto tensor_bias,
      CreateCudnnTensor(c_dims, c_strides, 'B', activation_type, vector_size,
                        vector_dim, /*is_virtual=*/true));

  cudnnDataType_t cudnn_matmul_type = ToCudnnDataType(accumulator_type);
  auto matmul_desc = cudnn_frontend::MatMulDescBuilder()
                         .setMathPrecision(cudnn_matmul_type)
                         .build();
  RETURN_MSG_IF_CUDNN_ERROR(matmul_desc);
  auto matmul_op = cudnn_frontend::OperationBuilder(
                       CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                       .setmatmulDesc(matmul_desc)
                       .setaMatDesc(tensor_a)
                       .setbMatDesc(tensor_b)
                       .setcMatDesc(tensor_matmul)
                       .build();
  RETURN_MSG_IF_CUDNN_ERROR(matmul_op);

  auto bias_add_desc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(cudnn_activation_type)
                           .build();
  RETURN_MSG_IF_CUDNN_ERROR(bias_add_desc);
  auto bias_add_op = cudnn_frontend::OperationBuilder(
                         CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setxDesc(tensor_matmul)
                         .setbDesc(tensor_z)
                         .setyDesc(tensor_bias)
                         .setpwDesc(bias_add_desc)
                         .build();
  RETURN_MSG_IF_CUDNN_ERROR(bias_add_op);

  absl::InlinedVector<cudnn_frontend::Operation const*, 3> ops = {&matmul_op,
                                                                  &bias_add_op};

  cudnnPointwiseMode_t cudnn_activation_mode;
  switch (activation_mode) {
    case dnn::ActivationMode::kGeluExact:
      cudnn_activation_mode = CUDNN_POINTWISE_GELU_FWD;
      break;
    case dnn::ActivationMode::kTanh:
      cudnn_activation_mode = CUDNN_POINTWISE_TANH_FWD;
      break;
    case dnn::ActivationMode::kSigmoid:
      cudnn_activation_mode = CUDNN_POINTWISE_SIGMOID_FWD;
      break;
    default:
      return tsl::errors::Internal("Unimplemented activation mode ",
                                   dnn::ActivationModeString(activation_mode));
  }

  auto act_desc = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(cudnn_activation_mode)
                      .setMathPrecision(cudnn_activation_type)
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(act_desc);
  auto act_op = cudnn_frontend::OperationBuilder(
                    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                    .setxDesc(tensor_bias)
                    .setyDesc(tensor_c)
                    .setpwDesc(act_desc)
                    .build();
  RETURN_MSG_IF_CUDNN_ERROR(act_op);
  ops.push_back(&act_op);

  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(cudnn.handle())
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  RETURN_MSG_IF_CUDNN_ERROR(op_graph);

  VLOG(4) << "\nTensor_a: " << tensor_a.describe()
          << "\nTensor_b: " << tensor_b.describe()
          << "\nTensor_c: " << tensor_c.describe()
          << "\nTensor_z: " << tensor_z.describe()
          << "\nTensor_matmul: " << tensor_matmul.describe()
          << "\nTensor_bias: " << tensor_bias.describe()
          << "\nMatmul: " << matmul_desc.describe()
          << "\nBiasAdd: " << bias_add_desc.describe()  //
          << "\nActivation: " << act_desc.describe()
          << "\nMatmulOp: " << matmul_op.describe()
          << "\nBiasAddOp: " << bias_add_op.describe()  //
          << "\nActOp: " << act_op.describe()
          << "\nOpGraph: " << op_graph.describe();

  return std::make_unique<cudnn_frontend::OperationGraph>(std::move(op_graph));
}
}  // namespace

static absl::StatusOr<cudnn_frontend::ExecutionPlan> GetExecPlanFromHeuristics(
    cudnn_frontend::OperationGraph&& opGraph, const CudnnHandle& cudnn,
    bool include_fallback_heuristics = false) {
  cudnn_frontend::EngineConfigList engine_configs;
  if (!include_fallback_heuristics) {
    cudnn_frontend::get_heuristics_list<1>(
        {"heuristics_instant"}, opGraph, allowAllConfig, engine_configs, true);
  } else {
    cudnn_frontend::get_heuristics_list<2>(
        {"heuristics_instant", "heuristics_fallback"}, opGraph, allowAllConfig,
        engine_configs, true);
  }

  VLOG(4) << "Heuristic has " << engine_configs.size() << " configurations ";

  if (engine_configs.empty()) {
    return absl::InternalError(
        "No engine configurations found for this opGraph and heuristics.");
  }

  cudnnStatus_t status;
  for (auto engine_config : engine_configs) {
    cudnn_frontend::ExecutionPlan plan =
        cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(cudnn.handle())
            .setEngineConfig(engine_config, opGraph.getTag())
            .build();
    status = plan.get_status();
    if (status == CUDNN_STATUS_SUCCESS) {
      return plan;
    } else {
      VLOG(4) << "Failed to build cuDNN execution plan for opGraph "
              << opGraph.getTag()
              << ". absl::Status: " << CudnnStatusToString(status);
    }
  }

  LOG(FATAL) << "Failed to generate cuDNN execution plan for opGraph "
             << opGraph.getTag()
             << ". absl::Status of final plan: " << CudnnStatusToString(status);
}

static absl::StatusOr<cudnn_frontend::ExecutionPlan> RebuildExecutionPlan(
    const CudnnHandle& cudnn, const dnn::AlgorithmDesc& desc,
    const cudnn_frontend::OperationGraph& op_graph) {
  if (!desc.is_cudnn_frontend()) {
    return tsl::errors::Internal(
        "Got legacy cuDNN algorithm enum in RebuildExecutionPlan.");
  }

  // Errors encountered when building a cuDNN operation graph are surfaced in an
  // unprecedented and innovative way: they're written into a field of the
  // contained engine object, but then clobbered by the object's move
  // constructor which makes more cuDNN API calls and encounters further errors.
  // The only way to get the actual errors is to peek at them via the returned
  // rvalue reference before actually moving the object to finish its
  // initialization.
  cudnn_frontend::EngineBuilder engine_builder;
  engine_builder.setOperationGraph(op_graph).setGlobalEngineIdx(desc.algo_id());
  auto&& unmoved = engine_builder.build();
  RETURN_MSG_IF_CUDNN_ERROR(unmoved);
  cudnn_frontend::Engine engine = std::move(unmoved);
  RETURN_MSG_IF_CUDNN_ERROR(engine);

  // Miscellaneous compiler bugs and linker issues conspired to make it
  // impossible for AlgorithmDesc to just give us a map initially.  Get the
  // vector of tuning knobs and build the map locally.
  auto tuning_knobs_vec = desc.TuningKnobs();
  absl::flat_hash_map<int64_t, int64_t> tuning_knobs;
  tuning_knobs.reserve(tuning_knobs_vec.size());
  for (const auto& pair : tuning_knobs_vec) {
    tuning_knobs[pair.first] = pair.second;
  }

  for (auto& knob : engine.getSupportedKnobs()) {
    const auto it = tuning_knobs.find(static_cast<int64_t>(knob.getKnobType()));
    if (it != tuning_knobs.end()) {
      knob.setChoice(it->second);
    }
  }

  auto engine_config =
      cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
  RETURN_MSG_IF_CUDNN_ERROR(engine_config);

  auto plan = cudnn_frontend::ExecutionPlanBuilder()
                  .setHandle(cudnn.handle())
                  .setEngineConfig(engine_config)
                  .build();
  RETURN_MSG_IF_CUDNN_ERROR(plan);

  return {std::move(plan)};
}

}  // namespace

void FixDimsForRaggedOffset(std::vector<int64_t>& dims, int max_reg_per_batch) {
  dims[0] *= max_reg_per_batch;
}

absl::StatusOr<CudnnGraph> GetCudnnFlashAttentionOperationGraph(
    dnn::DnnSupport& dnn_support,
    const dnn::MatmulTensorDescriptor& q_descriptor,
    const dnn::MatmulTensorDescriptor& k_descriptor,
    const dnn::MatmulTensorDescriptor& v_descriptor,
    const dnn::TensorDescriptor& o_descriptor,
    const std::optional<dnn::TensorDescriptor> bias_descriptor,
    const std::optional<dnn::TensorDescriptor> stats_descriptor, double scale,
    const bool use_dropout, const std::optional<double> dropout_rate,
    const dnn::FMHAMaskKind mask_type, const int sliding_window_length,
    const int max_seg_per_batch) {
  using cudnn_frontend::graph::Tensor_attributes;

#if CUDNN_VERSION >= 90000
  VLOG(4) << "\n bmm1_lhs(q): " << q_descriptor.ToString()
          << "\n bmm1_rhs(k): " << k_descriptor.ToString()
          << "\n bmm2_rhs(v): " << v_descriptor.ToString()
          << "\n out(o): " << o_descriptor.ToString();
  if (bias_descriptor) {
    VLOG(4) << "\n bias(b): " << bias_descriptor->ToString();
  }
  if (stats_descriptor) {
    VLOG(4) << "\n activation(s): " << stats_descriptor->ToString();
  }

  cudnn_frontend::graph::Graph graph;
  dnn::DataType q_type = q_descriptor.type();
  dnn::DataType k_type = k_descriptor.type();
  dnn::DataType v_type = v_descriptor.type();
  dnn::DataType o_type = o_descriptor.type();
  if (!(q_type == k_type && k_type == v_type && v_type == o_type)) {
    return absl::InternalError("Input datatypes do not match");
  }
  cudnn_frontend::DataType_t ioDataType = ToCudnnFrontendDataType(q_type);

  graph.set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_io_data_type(ioDataType)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  auto next_uid = [uid = 0]() mutable -> int { return CuDnnTensorUID(uid++); };

  std::vector<int64_t> q_dims = q_descriptor.GetCudnnCompatibleDimensions(true);
  std::vector<int64_t> k_dims = k_descriptor.GetCudnnCompatibleDimensions(true);
  std::vector<int64_t> v_dims =
      v_descriptor.GetCudnnCompatibleDimensions(false);

  if (max_seg_per_batch > 1) {
    FixDimsForRaggedOffset(q_dims, max_seg_per_batch);
    FixDimsForRaggedOffset(k_dims, max_seg_per_batch);
    FixDimsForRaggedOffset(v_dims, max_seg_per_batch);
  }

  std::shared_ptr<Tensor_attributes> q_tensor =
      graph.tensor(Tensor_attributes()
                       .set_name("Q")
                       .set_dim(q_dims)
                       .set_stride(q_descriptor.GetCudnnCompatibleStrides(true))
                       .set_uid(next_uid()));

  std::shared_ptr<Tensor_attributes> k_tensor =
      graph.tensor(Tensor_attributes()
                       .set_name("K")
                       .set_dim(k_dims)
                       .set_stride(k_descriptor.GetCudnnCompatibleStrides(true))
                       .set_uid(next_uid()));
  std::shared_ptr<Tensor_attributes> v_tensor = graph.tensor(
      Tensor_attributes()
          .set_name("V")
          .set_dim(v_dims)
          .set_stride(v_descriptor.GetCudnnCompatibleStrides(false))
          .set_uid(next_uid()));

  // Setting sdpa, and is_inference
  bool is_causal = mask_type == dnn::FMHAMaskKind::CAUSAL ||
                   mask_type == dnn::FMHAMaskKind::PADDING_CAUSAL;
  cudnn_frontend::graph::SDPA_attributes sdpa_options;
  sdpa_options.set_name("flash_attention")
      .set_is_inference(stats_descriptor == std::nullopt)
      .set_causal_mask(is_causal)
      .set_attn_scale(scale);

  // Setting bias
  if (bias_descriptor.has_value()) {
    auto bias_tensor =
        graph.tensor(Tensor_attributes()
                         .set_name("bias")
                         .set_dim(bias_descriptor->dimensions())
                         .set_stride(bias_descriptor->GetLogicalStrides())
                         .set_uid(next_uid()));
    sdpa_options.set_bias(bias_tensor);
  }
  // Setting actual seqlen
  bool is_padding = mask_type == dnn::FMHAMaskKind::PADDING ||
                    mask_type == dnn::FMHAMaskKind::PADDING_CAUSAL;
  if (is_padding || max_seg_per_batch > 1) {
    // Get batch size
    auto b = q_dims[0];
    auto seq_q_tensor =
        graph.tensor(Tensor_attributes()
                         .set_name("seq_q")
                         .set_dim({b, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_uid(next_uid())
                         .set_data_type(cudnn_frontend::DataType_t::INT32));
    auto seq_kv_tensor =
        graph.tensor(Tensor_attributes()
                         .set_name("seq_kv")
                         .set_dim({b, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_uid(next_uid())
                         .set_data_type(cudnn_frontend::DataType_t::INT32));
    sdpa_options.set_padding_mask(true);
    sdpa_options.set_seq_len_q(seq_q_tensor);
    sdpa_options.set_seq_len_kv(seq_kv_tensor);
  }

  std::shared_ptr<Tensor_attributes> offset_q;
  if (max_seg_per_batch > 1) {
    // Get batch size
    auto b = q_dims[0];
    offset_q =
        graph.tensor(Tensor_attributes()
                         .set_name("offset_q")
                         .set_dim({b + 1, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_uid(next_uid())
                         .set_data_type(cudnn_frontend::DataType_t::INT32));
    auto offset_kv =
        graph.tensor(Tensor_attributes()
                         .set_name("offset_kv")
                         .set_dim({b + 1, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_uid(next_uid())
                         .set_data_type(cudnn_frontend::DataType_t::INT32));
    q_tensor->set_ragged_offset(offset_q);
    k_tensor->set_ragged_offset(offset_kv);
    v_tensor->set_ragged_offset(offset_kv);
  }

  // Setting seed and offset
  std::shared_ptr<Tensor_attributes> seed_tensor;
  std::shared_ptr<Tensor_attributes> offset_tensor;
  if (use_dropout) {
    // Skip setting UIDs: pass by value tensors go at the end.
    seed_tensor =
        graph.tensor(Tensor_attributes()
                         .set_name("seed")
                         .set_dim({1, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_data_type(cudnn_frontend::DataType_t::INT64)
                         .set_is_pass_by_value(true));
    offset_tensor =
        graph.tensor(Tensor_attributes()
                         .set_name("offset")
                         .set_dim({1, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_data_type(cudnn_frontend::DataType_t::INT64)
                         .set_is_pass_by_value(true));
    sdpa_options.set_dropout((float)dropout_rate.value(), seed_tensor,
                             offset_tensor);
  }

  if (sliding_window_length > 0) {
    sdpa_options.set_sliding_window_length(sliding_window_length);
  }
  // Add SDPA to the graph.
  auto [o_tensor, stats_tensor] =
      graph.sdpa(q_tensor, k_tensor, v_tensor, sdpa_options);

  auto o_dims = o_descriptor.dimensions();

  if (max_seg_per_batch > 1) {
    FixDimsForRaggedOffset(o_dims, max_seg_per_batch);
    o_tensor->set_ragged_offset(offset_q);
  }
  // Set output attributes.
  o_tensor->set_name("O")
      .set_output(true)
      .set_dim(o_dims)
      .set_stride(o_descriptor.GetLogicalStrides())
      .set_uid(next_uid());
  if (stats_descriptor.has_value()) {
    cudnn_frontend::DataType_t statsType =
        ToCudnnFrontendDataType(stats_descriptor->type());
    auto stat_dims = stats_descriptor->dimensions();
    auto stat_strides = stats_descriptor->GetLogicalStrides();
    stat_dims.push_back(1);
    stat_strides.push_back(1);
    stats_tensor->set_name("stats")
        .set_output(true)
        .set_data_type(statsType)
        .set_dim(stat_dims)
        .set_stride(stat_strides)
        .set_uid(next_uid());
  }
  if (seed_tensor != nullptr) {
    seed_tensor->set_uid(next_uid());
  }
  if (offset_tensor != nullptr) {
    offset_tensor->set_uid(next_uid());
  }
  CudnnGraph cudnnGraph(std::move(graph));
  TF_RETURN_IF_ERROR(cudnnGraph.Prepare(
      dnn_support, NumericOptions{/*require_determinism=*/false,
                                  /*allow_tf32=*/true}));
  TF_RETURN_IF_ERROR(cudnnGraph.Build(dnn_support, /*plan_id=*/std::nullopt));

  VLOG(4) << "\b flash attention operation graph: " << cudnnGraph.Graph();
  return cudnnGraph;
#else
  return absl::UnimplementedError(
      "Cudnn flash attention only supported with Cudnn >= 9.0.0");
#endif
}

absl::StatusOr<CudnnGraph> GetCudnnFlashAttentionF8OperationGraph(
    dnn::DnnSupport& dnn_support,
    const dnn::MatmulTensorDescriptor& q_descriptor,
    const dnn::MatmulTensorDescriptor& k_descriptor,
    const dnn::MatmulTensorDescriptor& v_descriptor,
    const dnn::TensorDescriptor& o_descriptor,
    const std::optional<dnn::TensorDescriptor>& stats_descriptor, double scale,
    const dnn::FMHAMaskKind mask_type) {
  using cudnn_frontend::graph::Tensor_attributes;

#if CUDNN_VERSION >= 90100
  VLOG(4) << "\n bmm1_lhs(q): " << q_descriptor.ToString()
          << "\n bmm1_rhs(k): " << k_descriptor.ToString()
          << "\n bmm2_rhs(v): " << v_descriptor.ToString()
          << "\n out(o): " << o_descriptor.ToString() << "\n scale: " << scale;
  if (stats_descriptor) {
    VLOG(4) << "\n activation(s): " << stats_descriptor->ToString();
  }

  cudnn_frontend::graph::Graph graph;
  dnn::DataType q_type = q_descriptor.type();
  dnn::DataType k_type = k_descriptor.type();
  dnn::DataType v_type = v_descriptor.type();
  dnn::DataType o_type = o_descriptor.type();
  if (!(q_type == k_type && k_type == v_type && v_type == o_type)) {
    return absl::InternalError("Input datatypes do not match");
  }
  cudnn_frontend::DataType_t ioDataType = ToCudnnFrontendDataType(q_type);

  graph.set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_io_data_type(ioDataType)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  auto next_uid = [uid = 0]() mutable -> int { return CuDnnTensorUID(uid++); };

  std::shared_ptr<Tensor_attributes> q_tensor =
      graph.tensor(Tensor_attributes()
                       .set_name("Q")
                       .set_dim(q_descriptor.GetCudnnCompatibleDimensions(true))
                       .set_stride(q_descriptor.GetCudnnCompatibleStrides(true))
                       .set_uid(next_uid()));

  std::shared_ptr<Tensor_attributes> k_tensor =
      graph.tensor(Tensor_attributes()
                       .set_name("K")
                       .set_dim(k_descriptor.GetCudnnCompatibleDimensions(true))
                       .set_stride(k_descriptor.GetCudnnCompatibleStrides(true))
                       .set_uid(next_uid()));
  std::shared_ptr<Tensor_attributes> v_tensor = graph.tensor(
      Tensor_attributes()
          .set_name("V")
          .set_dim(v_descriptor.GetCudnnCompatibleDimensions(false))
          .set_stride(v_descriptor.GetCudnnCompatibleStrides(false))
          .set_uid(next_uid()));

  auto descale_q =
      graph.tensor(Tensor_attributes()
                       .set_name("Descale_Q")
                       .set_dim({1, 1, 1, 1})
                       .set_stride({1, 1, 1, 1})
                       .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                       .set_uid(next_uid()));
  auto descale_k = graph.tensor_like(descale_q, "Descale_K");
  auto descale_v = graph.tensor_like(descale_q, "Descale_V");
  auto descale_s = graph.tensor_like(descale_q, "Descale_S");
  auto scale_s = graph.tensor_like(descale_q, "Scale_S");
  auto scale_o = graph.tensor_like(descale_q, "Scale_O");

  descale_k->set_uid(next_uid());
  descale_v->set_uid(next_uid());
  descale_s->set_uid(next_uid());
  scale_s->set_uid(next_uid());
  scale_o->set_uid(next_uid());

  // Setting sdpa, and is_inference
  bool is_causal = mask_type == dnn::FMHAMaskKind::CAUSAL;
  cudnn_frontend::graph::SDPA_fp8_attributes sdpa_fp8_options;
  sdpa_fp8_options.set_name("sdpa_fp8")
      .set_is_inference(stats_descriptor == std::nullopt)
      .set_causal_mask(is_causal)
      .set_attn_scale(scale);

  // Add SDPA to the graph.
  auto [o_tensor, stats_tensor, amax_s, amax_o] =
      graph.sdpa_fp8(q_tensor, k_tensor, v_tensor, descale_q, descale_k,
                     descale_v, descale_s, scale_s, scale_o, sdpa_fp8_options);

  // Set output attributes.
  o_tensor->set_name("O")
      .set_output(true)
      .set_dim(o_descriptor.dimensions())
      .set_stride(o_descriptor.GetLogicalStrides())
      .set_uid(next_uid());
  amax_s->set_output(true)
      .set_dim({1, 1, 1, 1})
      .set_stride({1, 1, 1, 1})
      .set_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_uid(next_uid());
  amax_o->set_output(true)
      .set_dim({1, 1, 1, 1})
      .set_stride({1, 1, 1, 1})
      .set_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_uid(next_uid());

  if (stats_descriptor.has_value()) {
    cudnn_frontend::DataType_t statsType =
        ToCudnnFrontendDataType(stats_descriptor->type());
    auto stat_dims = stats_descriptor->dimensions();
    auto stat_strides = stats_descriptor->GetLogicalStrides();
    stat_dims.push_back(1);
    stat_strides.push_back(1);
    stats_tensor->set_name("stats")
        .set_output(true)
        .set_data_type(statsType)
        .set_dim(stat_dims)
        .set_stride(stat_strides)
        .set_uid(next_uid());
  }
  CudnnGraph cudnnGraph(std::move(graph));
  TF_RETURN_IF_ERROR(cudnnGraph.Prepare(
      dnn_support, NumericOptions{/*require_determinism=*/false,
                                  /*allow_tf32=*/true}));
  TF_RETURN_IF_ERROR(cudnnGraph.Build(dnn_support, /*plan_id=*/std::nullopt));

  VLOG(4) << "\b workspace size:" << cudnnGraph.Graph().get_workspace_size();
  VLOG(4) << "\b flash attention operation graph: " << cudnnGraph.Graph();

  return cudnnGraph;
#else
  return absl::UnimplementedError(
      "Cudnn flash attention only supported with Cudnn >= 9.1.0");
#endif
}

absl::StatusOr<CudnnGraph> GetCudnnFlashAttentionBackwardF8OperationGraph(
    dnn::DnnSupport& dnn_support, const dnn::MatmulTensorDescriptor& q_desc,
    const dnn::MatmulTensorDescriptor& k_desc,
    const dnn::MatmulTensorDescriptor& p_desc,
    const dnn::MatmulTensorDescriptor& v_desc,
    const dnn::MatmulTensorDescriptor& do_desc,
    const dnn::TensorDescriptor& dq_desc, const dnn::TensorDescriptor& dk_desc,
    const dnn::TensorDescriptor& dv_desc, double scale,
    dnn::FMHAMaskKind mask_type) {
#if CUDNN_VERSION >= 90100
  VLOG(4) << "\n bmm1_grad_gemm1_rhs(q): " << q_desc.ToString()
          << "\n bmm1_grad_gemm2_rhs(k): " << k_desc.ToString()
          << "\n bmm2_grad_gemm1_lhs(p): " << p_desc.ToString()
          << "\n bmm2_grad_gemm2_rhs(v^t): " << v_desc.ToString()
          << "\n d_output(do): " << do_desc.ToString()
          << "\n d_bmm1_lhs(dq): " << dq_desc.ToString()
          << "\n d_bmm1_rhs(dk): " << dk_desc.ToString()
          << "\n d_bmm2_rhs(dv): " << dv_desc.ToString()
          << "\n scale: " << scale;

  using cudnn_frontend::graph::Tensor_attributes;
  cudnn_frontend::graph::Graph graph;
  if (!(q_desc.type() == k_desc.type() && v_desc.type() == do_desc.type() &&
        do_desc.type() == dq_desc.type() && dq_desc.type() == dk_desc.type() &&
        dk_desc.type() == dv_desc.type())) {
    return absl::InternalError("Input datatypes do not match.");
  }

  auto ioDataType = ToCudnnFrontendDataType(q_desc.type());
  graph.set_compute_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_io_data_type(ioDataType);

  auto next_uid = [uid = 0]() mutable -> int { return CuDnnTensorUID(uid++); };

  std::shared_ptr<Tensor_attributes> q =
      graph.tensor(Tensor_attributes()
                       .set_name("Q")
                       .set_dim(q_desc.GetCudnnCompatibleDimensions(false))
                       .set_stride(q_desc.GetCudnnCompatibleStrides(false))
                       .set_uid(next_uid())
                       .set_data_type(ioDataType));
  std::shared_ptr<Tensor_attributes> k =
      graph.tensor(Tensor_attributes()
                       .set_name("K")
                       .set_dim(k_desc.GetCudnnCompatibleDimensions(false))
                       .set_stride(k_desc.GetCudnnCompatibleStrides(false))
                       .set_uid(next_uid())
                       .set_data_type(ioDataType));
  std::shared_ptr<Tensor_attributes> v =
      graph.tensor(Tensor_attributes()
                       .set_name("V")
                       .set_dim(v_desc.GetCudnnCompatibleDimensions(true))
                       .set_stride(v_desc.GetCudnnCompatibleStrides(true))
                       .set_uid(next_uid())
                       .set_data_type(ioDataType));
  std::shared_ptr<Tensor_attributes> o =
      graph.tensor(Tensor_attributes()
                       .set_name("O")
                       .set_dim(do_desc.GetCudnnCompatibleDimensions(false))
                       .set_stride(do_desc.GetCudnnCompatibleStrides(false))
                       .set_uid(next_uid())
                       .set_data_type(ioDataType));
  std::shared_ptr<Tensor_attributes> dO =
      graph.tensor(Tensor_attributes()
                       .set_name("dO")
                       .set_dim(do_desc.GetCudnnCompatibleDimensions(false))
                       .set_stride(do_desc.GetCudnnCompatibleStrides(false))
                       .set_uid(next_uid())
                       .set_data_type(ioDataType));

  auto p_dims = p_desc.GetCudnnCompatibleDimensions(false);
  auto p_strides = p_desc.GetCudnnCompatibleStrides(false);
  std::vector<int64_t> p_reduction_dims(p_dims.begin(), p_dims.end() - 1);
  p_reduction_dims.push_back(1);

  // Divide every stride by the last dim value.
  std::vector<int64_t> p_reduction_strides;
  p_reduction_strides.reserve(p_strides.size());
  int64_t p_reduced_dim_len = p_dims.back();
  for (auto stride : p_strides) {
    p_reduction_strides.push_back(stride / p_reduced_dim_len);
  }
  p_reduction_strides[3] = 1;
  std::shared_ptr<Tensor_attributes> Stats =
      graph.tensor(Tensor_attributes()
                       .set_name("Stats")
                       .set_dim(p_reduction_dims)
                       .set_stride(p_reduction_strides)
                       .set_uid(next_uid())
                       .set_data_type(cudnn_frontend::DataType_t::FLOAT));

  auto descale_q =
      graph.tensor(Tensor_attributes()
                       .set_name("Descale_Q")
                       .set_dim({1, 1, 1, 1})
                       .set_stride({1, 1, 1, 1})
                       .set_uid(next_uid())
                       .set_data_type(cudnn_frontend::DataType_t::FLOAT));
  auto descale_k = graph.tensor_like(descale_q, "Descale_K");
  auto descale_v = graph.tensor_like(descale_q, "Descale_V");
  auto descale_s = graph.tensor_like(descale_q, "Descale_S");
  auto descale_o = graph.tensor_like(descale_q, "Descale_O");
  auto descale_dO = graph.tensor_like(descale_q, "Descale_dO");
  auto descale_dP = graph.tensor_like(descale_q, "Descale_dP");

  auto scale_s = graph.tensor_like(descale_q, "Scale_S");
  auto scale_dP = graph.tensor_like(descale_q, "Scale_dP");
  auto scale_dQ = graph.tensor_like(descale_q, "Scale_dQ");
  auto scale_dK = graph.tensor_like(descale_q, "Scale_dK");
  auto scale_dV = graph.tensor_like(descale_q, "Scale_dV");

  descale_k->set_uid(next_uid());
  descale_v->set_uid(next_uid());
  descale_s->set_uid(next_uid());
  descale_o->set_uid(next_uid());
  descale_dO->set_uid(next_uid());
  descale_dP->set_uid(next_uid());

  scale_s->set_uid(next_uid());
  scale_dP->set_uid(next_uid());
  scale_dQ->set_uid(next_uid());
  scale_dK->set_uid(next_uid());
  scale_dV->set_uid(next_uid());

  bool is_causal = mask_type == dnn::FMHAMaskKind::CAUSAL;
  auto sdpa_fp8_backwards_options =
      cudnn_frontend::graph::SDPA_fp8_backward_attributes()
          .set_name("sdpa_fp8_backward")
          .set_causal_mask(is_causal)
          .set_attn_scale(scale);

  auto [dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV, Amax_dP] =
      graph.sdpa_fp8_backward(q, k, v, o, dO, Stats, descale_q, descale_k,
                              descale_v, descale_o, descale_dO, descale_s,
                              descale_dP, scale_s, scale_dQ, scale_dK, scale_dV,
                              scale_dP, sdpa_fp8_backwards_options);

  dQ->set_output(true)
      .set_dim(dq_desc.dimensions())
      .set_stride(dq_desc.GetLogicalStrides())
      .set_name("dQ")
      .set_uid(next_uid())
      .set_data_type(ioDataType);
  dK->set_output(true)
      .set_dim(dk_desc.dimensions())
      .set_stride(dk_desc.GetLogicalStrides())
      .set_name("dK")
      .set_uid(next_uid())
      .set_data_type(ioDataType);
  dV->set_output(true)
      .set_dim(dv_desc.dimensions())
      .set_stride(dv_desc.GetLogicalStrides())
      .set_name("dV")
      .set_uid(next_uid())
      .set_data_type(ioDataType);
  Amax_dQ->set_output(true)
      .set_dim({1, 1, 1, 1})
      .set_stride({1, 1, 1, 1})
      .set_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_uid(next_uid());
  Amax_dK->set_output(true)
      .set_dim({1, 1, 1, 1})
      .set_stride({1, 1, 1, 1})
      .set_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_uid(next_uid());
  Amax_dV->set_output(true)
      .set_dim({1, 1, 1, 1})
      .set_stride({1, 1, 1, 1})
      .set_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_uid(next_uid());
  Amax_dP->set_output(true)
      .set_dim({1, 1, 1, 1})
      .set_stride({1, 1, 1, 1})
      .set_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_uid(next_uid());

  CudnnGraph cudnnGraph(std::move(graph));
  TF_RETURN_IF_ERROR(cudnnGraph.Prepare(
      dnn_support, NumericOptions{/*require_determinism=*/false,
                                  /*allow_tf32=*/true}));
  TF_RETURN_IF_ERROR(cudnnGraph.Build(dnn_support, /*plan_id=*/std::nullopt));

  VLOG(4) << "\b workspace size:" << cudnnGraph.Graph().get_workspace_size();
  VLOG(4) << "\b flash attention f8 operation backward graph: "
          << cudnnGraph.Graph();

  return cudnnGraph;
#else
  return absl::UnimplementedError(
      "Cudnn flash attention only supported with Cudnn >= 9.1.0");
#endif
}

absl::StatusOr<CudnnGraph> GetCudnnBlockScaledDotOperationGraph(
    dnn::DnnSupport& dnn_support, const dnn::TensorDescriptor& lhs_data,
    const dnn::TensorDescriptor& lhs_scale,
    const dnn::TensorDescriptor& rhs_data,
    const dnn::TensorDescriptor& rhs_scale, dnn::DataType result_type,
    int block_size) {
#if CUDNN_VERSION >= 90700
  using cudnn_frontend::graph::Block_scale_dequantize_attributes;
  using cudnn_frontend::graph::Matmul_attributes;
  using cudnn_frontend::graph::Tensor_attributes;

  VLOG(4) << "\n lhs_data: " << lhs_data.ToString()
          << "\n lhs_scale: " << lhs_scale.ToString()
          << "\n rhs_data: " << rhs_data.ToString()
          << "\n rhs_scale: " << rhs_scale.ToString()
          << "\n result_type: " << dnn::DataType_Name(result_type)
          << "\n block_size: " << block_size;

  cudnn_frontend::graph::Graph graph;
  auto compute_type = cudnn_frontend::DataType_t::FLOAT;
  graph.set_compute_data_type(compute_type);
  graph.set_intermediate_data_type(compute_type);

  auto next_uid = [uid = 0]() mutable -> int { return CuDnnTensorUID(uid++); };
  auto get_tensor_attr = [&](const dnn::TensorDescriptor& desc,
                             bool is_rhs) -> absl::StatusOr<Tensor_attributes> {
    TF_ASSIGN_OR_RETURN(std::vector<int64_t> dimensions,
                        desc.GetPhysicalDimensionsMajorToMinor());
    std::vector<int64_t> strides = desc.GetPhysicalStridesMajorToMinor();
    if (dimensions.size() == 2) {
      dimensions.insert(dimensions.begin(), 1);
      strides.insert(strides.begin(), dimensions[1] * dimensions[2]);
    }
    CHECK_EQ(dimensions.size(), 3);
    if (is_rhs) {
      std::swap(dimensions[1], dimensions[2]);
      std::swap(strides[1], strides[2]);
    }
    return Tensor_attributes()
        .set_uid(next_uid())
        .set_dim(dimensions)
        .set_stride(strides)
        .set_data_type(ToCudnnFrontendDataType(desc.type()));
  };
  TF_ASSIGN_OR_RETURN(auto a_data_attr, get_tensor_attr(lhs_data, false));
  TF_ASSIGN_OR_RETURN(auto b_data_attr, get_tensor_attr(rhs_data, true));
  TF_ASSIGN_OR_RETURN(auto a_scale_attr, get_tensor_attr(lhs_scale, false));
  TF_ASSIGN_OR_RETURN(auto b_scale_attr, get_tensor_attr(rhs_scale, true));

  a_scale_attr.set_reordering_type(
      cudnn_frontend::TensorReordering_t::F8_128x4);
  b_scale_attr.set_reordering_type(
      cudnn_frontend::TensorReordering_t::F8_128x4);

  auto a_data = graph.tensor(a_data_attr.set_name("a_data"));
  auto b_data = graph.tensor(b_data_attr.set_name("b_data"));
  auto a_scale = graph.tensor(a_scale_attr.set_name("a_scale"));
  auto b_scale = graph.tensor(b_scale_attr.set_name("b_scale"));

  auto dq_attr = Block_scale_dequantize_attributes().set_block_size(block_size);
  auto a_dq = graph.block_scale_dequantize(a_data, a_scale, dq_attr);
  auto b_dq = graph.block_scale_dequantize(b_data, b_scale, dq_attr);

  auto matmul_attr = Matmul_attributes().set_compute_data_type(compute_type);
  auto d_tensor = graph.matmul(a_dq, b_dq, matmul_attr);
  d_tensor->set_uid(next_uid());
  d_tensor->set_data_type(ToCudnnFrontendDataType(result_type));
  d_tensor->set_is_virtual(false);

  CudnnGraph cudnnGraph(std::move(graph));
  TF_RETURN_IF_ERROR(cudnnGraph.Prepare(
      dnn_support, NumericOptions{/*require_determinism=*/false,
                                  /*allow_tf32=*/true}));
  TF_RETURN_IF_ERROR(cudnnGraph.Build(dnn_support, /*plan_id=*/std::nullopt));

  VLOG(4) << "\b workspace size:" << cudnnGraph.Graph().get_workspace_size();
  VLOG(4) << "\b block scaled dot graph: " << cudnnGraph.Graph();

  return cudnnGraph;
#else
  return absl::UnimplementedError(
      "Cudnn block scaled dot only supported with Cudnn >= 9.7.0");
#endif
}

absl::StatusOr<CudnnGraph> GetCudnnFlashAttentionBackwardOperationGraph(
    dnn::DnnSupport& dnn_support, const dnn::MatmulTensorDescriptor& q_desc,
    const dnn::MatmulTensorDescriptor& k_desc,
    const dnn::MatmulTensorDescriptor& p_desc,
    const dnn::MatmulTensorDescriptor& v_desc,
    const dnn::MatmulTensorDescriptor& do_desc,
    const dnn::TensorDescriptor& dq_desc, const dnn::TensorDescriptor& dk_desc,
    const dnn::TensorDescriptor& dv_desc,
    const std::optional<dnn::TensorDescriptor> bias_descriptor,
    const std::optional<dnn::TensorDescriptor> dbias_descriptor,
    std::optional<double> dropout_rate, std::optional<int64_t> seed,
    double scale, bool use_dropout, bool use_bias, dnn::FMHAMaskKind mask_type,
    bool force_deterministic, const int sliding_window_length,
    const int max_seg_per_batch) {
#if CUDNN_VERSION >= 90000
  VLOG(4) << "\n bmm1_grad_gemm1_rhs(q): " << q_desc.ToString()
          << "\n bmm1_grad_gemm2_rhs(k): " << k_desc.ToString()
          << "\n bmm2_grad_gemm1_lhs(p): " << p_desc.ToString()
          << "\n bmm2_grad_gemm2_rhs(v^t): " << v_desc.ToString()
          << "\n d_output(do): " << do_desc.ToString()
          << "\n d_bmm1_lhs(dq): " << dq_desc.ToString()
          << "\n d_bmm1_rhs(dk): " << dk_desc.ToString()
          << "\n d_bmm2_rhs(dv): " << dv_desc.ToString();

  using cudnn_frontend::graph::Tensor_attributes;
  cudnn_frontend::graph::Graph graph;
  if (!(q_desc.type() == k_desc.type() && k_desc.type() == p_desc.type() &&
        p_desc.type() == v_desc.type() && v_desc.type() == do_desc.type() &&
        do_desc.type() == dq_desc.type() && dq_desc.type() == dk_desc.type() &&
        dk_desc.type() == dv_desc.type())) {
    return absl::InternalError("Input datatypes do not match.");
  }

  auto ioDataType = ToCudnnFrontendDataType(q_desc.type());
  graph.set_compute_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_io_data_type(ioDataType);

  // Get dims and strides
  std::vector<int64_t> q_dims = q_desc.GetCudnnCompatibleDimensions(false);
  std::vector<int64_t> k_dims = k_desc.GetCudnnCompatibleDimensions(false);
  std::vector<int64_t> v_dims = v_desc.GetCudnnCompatibleDimensions(true);
  std::vector<int64_t> p_dims = p_desc.GetCudnnCompatibleDimensions(false);
  std::vector<int64_t> p_strides = p_desc.GetCudnnCompatibleStrides(false);
  std::vector<int64_t> do_dims = do_desc.GetCudnnCompatibleDimensions(false);
  std::vector<int64_t> dq_dims = dq_desc.dimensions();
  std::vector<int64_t> dk_dims = dk_desc.dimensions();
  std::vector<int64_t> dv_dims = dv_desc.dimensions();
  std::vector<int64_t> stats_dims(p_dims.begin(), p_dims.end() - 1);
  stats_dims.push_back(1);
  // Divide every stride by the last dim value.
  std::vector<int64_t> stats_strides;
  stats_strides.reserve(p_strides.size());
  int64_t p_reduced_dim_len = p_dims.back();
  for (auto stride : p_strides) {
    stats_strides.push_back(stride / p_reduced_dim_len);
  }
  stats_strides[3] = 1;

  if (max_seg_per_batch > 1) {
    FixDimsForRaggedOffset(q_dims, max_seg_per_batch);
    FixDimsForRaggedOffset(k_dims, max_seg_per_batch);
    FixDimsForRaggedOffset(v_dims, max_seg_per_batch);
    FixDimsForRaggedOffset(p_dims, max_seg_per_batch);
    FixDimsForRaggedOffset(do_dims, max_seg_per_batch);
    FixDimsForRaggedOffset(dq_dims, max_seg_per_batch);
    FixDimsForRaggedOffset(dk_dims, max_seg_per_batch);
    FixDimsForRaggedOffset(dv_dims, max_seg_per_batch);
    FixDimsForRaggedOffset(stats_dims, max_seg_per_batch);
  }
  bool is_causal = mask_type == dnn::FMHAMaskKind::CAUSAL ||
                   mask_type == dnn::FMHAMaskKind::PADDING_CAUSAL;
  auto sdpa_backward_options =
      cudnn_frontend::graph::SDPA_backward_attributes()
          .set_name("flash_attention_backward")
          .set_causal_mask(is_causal)
          .set_attn_scale(scale)
          .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  auto next_uid = [uid = 0]() mutable -> int { return CuDnnTensorUID(uid++); };

  std::shared_ptr<Tensor_attributes> q =
      graph.tensor(Tensor_attributes()
                       .set_name("Q")
                       .set_dim(q_dims)
                       .set_stride(q_desc.GetCudnnCompatibleStrides(false))
                       .set_uid(next_uid())
                       .set_data_type(ioDataType));
  std::shared_ptr<Tensor_attributes> k =
      graph.tensor(Tensor_attributes()
                       .set_name("K")
                       .set_dim(k_dims)
                       .set_stride(k_desc.GetCudnnCompatibleStrides(false))
                       .set_uid(next_uid())
                       .set_data_type(ioDataType));
  std::shared_ptr<Tensor_attributes> v =
      graph.tensor(Tensor_attributes()
                       .set_name("V")
                       .set_dim(v_dims)
                       .set_stride(v_desc.GetCudnnCompatibleStrides(true))
                       .set_uid(next_uid())
                       .set_data_type(ioDataType));
  std::shared_ptr<Tensor_attributes> stats =
      graph.tensor(Tensor_attributes()
                       .set_name("stats")
                       .set_dim(stats_dims)
                       .set_stride(stats_strides)
                       .set_uid(next_uid())
                       .set_data_type(cudnn_frontend::DataType_t::FLOAT));
  std::shared_ptr<Tensor_attributes> dO =
      graph.tensor(Tensor_attributes()
                       .set_name("dO")
                       .set_dim(do_dims)
                       .set_stride(do_desc.GetCudnnCompatibleStrides(false))
                       .set_uid(next_uid())
                       .set_data_type(ioDataType));
  std::shared_ptr<Tensor_attributes> d_bias_tensor;
  if (use_bias) {
    DCHECK(bias_descriptor != std::nullopt);
    auto bias_dims = bias_descriptor->dimensions();
    auto bias_strides = bias_descriptor->GetLogicalStrides();
    auto b = bias_dims[0];
    auto n = bias_dims[1];
    auto q_n = q_dims[1];
    auto bias_tensor = graph.tensor(Tensor_attributes()
                                        .set_name("bias")
                                        .set_dim(bias_dims)
                                        .set_stride(bias_strides)
                                        .set_uid(next_uid()));
    sdpa_backward_options.set_bias(bias_tensor);

    // shapes [1, 1, s, s], [b, 1, s, s], [b, h, s, s] are not supported for
    // dbias calculation but they are supported for forward bias calculation
    // Set UID later: this is the last output tuple element.
    if (dbias_descriptor != std::nullopt) {
      DCHECK(b == 1 && n == q_n);
      d_bias_tensor =
          graph.tensor(Tensor_attributes()
                           .set_name("dBias")
                           .set_dim(dbias_descriptor->dimensions())
                           .set_stride(dbias_descriptor->GetLogicalStrides()));
      sdpa_backward_options.set_dbias(d_bias_tensor);
    }
  }
  std::shared_ptr<Tensor_attributes> o =
      graph.tensor(Tensor_attributes()
                       .set_name("O")
                       .set_dim(do_dims)
                       .set_stride(do_desc.GetCudnnCompatibleStrides(false))
                       .set_uid(next_uid())
                       .set_data_type(ioDataType));

  // Setting actual seqlen
  bool is_padding = mask_type == dnn::FMHAMaskKind::PADDING ||
                    mask_type == dnn::FMHAMaskKind::PADDING_CAUSAL;

  if (is_padding || max_seg_per_batch > 1) {
    // Get batch size
    auto b = q_dims[0];
    auto seq_q_tensor =
        graph.tensor(Tensor_attributes()
                         .set_name("seq_q")
                         .set_dim({b, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_uid(next_uid())
                         .set_data_type(cudnn_frontend::DataType_t::INT32));
    auto seq_kv_tensor =
        graph.tensor(Tensor_attributes()
                         .set_name("seq_kv")
                         .set_dim({b, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_uid(next_uid())
                         .set_data_type(cudnn_frontend::DataType_t::INT32));
    sdpa_backward_options.set_padding_mask(true);
    sdpa_backward_options.set_seq_len_q(seq_q_tensor);
    sdpa_backward_options.set_seq_len_kv(seq_kv_tensor);
  }

  std::shared_ptr<Tensor_attributes> offset_q, offset_kv;
  if (max_seg_per_batch > 1) {
    // Get batch size
    auto b = q_dims[0];
    offset_q =
        graph.tensor(Tensor_attributes()
                         .set_name("offset_q")
                         .set_dim({b + 1, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_uid(next_uid())
                         .set_data_type(cudnn_frontend::DataType_t::INT32));
    offset_kv =
        graph.tensor(Tensor_attributes()
                         .set_name("offset_k")
                         .set_dim({b + 1, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_uid(next_uid())
                         .set_data_type(cudnn_frontend::DataType_t::INT32));
    q->set_ragged_offset(offset_q);
    k->set_ragged_offset(offset_kv);
    v->set_ragged_offset(offset_kv);
    o->set_ragged_offset(offset_q);
    dO->set_ragged_offset(offset_q);
  }
  // Setting seed and offset
  std::shared_ptr<Tensor_attributes> seed_tensor;
  std::shared_ptr<Tensor_attributes> offset_tensor;
  if (use_dropout) {
    DCHECK(dropout_rate != std::nullopt);
    // Skip setting UIDs: pass by value tensors go at the end.
    seed_tensor =
        graph.tensor(Tensor_attributes()
                         .set_name("seed")
                         .set_dim({1, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_data_type(cudnn_frontend::DataType_t::INT64)
                         .set_is_pass_by_value(true));
    offset_tensor =
        graph.tensor(Tensor_attributes()
                         .set_name("offset")
                         .set_dim({1, 1, 1, 1})
                         .set_stride({1, 1, 1, 1})
                         .set_data_type(cudnn_frontend::DataType_t::INT64)
                         .set_is_pass_by_value(true));
    sdpa_backward_options.set_dropout((float)dropout_rate.value(), seed_tensor,
                                      offset_tensor);
  }

  if (force_deterministic) {
    sdpa_backward_options.set_deterministic_algorithm(true);
  }

  if (sliding_window_length > 0) {
    sdpa_backward_options.set_sliding_window_length(sliding_window_length);
  }

  auto [dQ, dK, dV] =
      graph.sdpa_backward(q, k, v, o, dO, stats, sdpa_backward_options);

  if (max_seg_per_batch > 1) {
    dQ->set_ragged_offset(offset_q);
    dK->set_ragged_offset(offset_kv);
    dV->set_ragged_offset(offset_kv);
  }
  dQ->set_output(true)
      .set_dim(dq_dims)
      .set_stride(dq_desc.GetLogicalStrides())
      .set_uid(next_uid())
      .set_name("dQ")
      .set_data_type(ioDataType);
  dK->set_output(true)
      .set_dim(dk_dims)
      .set_stride(dk_desc.GetLogicalStrides())
      .set_uid(next_uid())
      .set_name("dK")
      .set_data_type(ioDataType);
  dV->set_output(true)
      .set_dim(dv_dims)
      .set_stride(dv_desc.GetLogicalStrides())
      .set_uid(next_uid())
      .set_name("dV")
      .set_data_type(ioDataType);
  if (d_bias_tensor != nullptr) {
    d_bias_tensor->set_uid(next_uid());
  }
  if (seed_tensor != nullptr) {
    seed_tensor->set_uid(next_uid());
  }
  if (offset_tensor != nullptr) {
    offset_tensor->set_uid(next_uid());
  }

  CudnnGraph cudnnGraph(std::move(graph));
  TF_RETURN_IF_ERROR(
      cudnnGraph.Prepare(dnn_support, NumericOptions{force_deterministic,
                                                     /*allow_tf32=*/true}));
  TF_RETURN_IF_ERROR(cudnnGraph.Build(dnn_support, /*plan_id=*/std::nullopt));

  VLOG(4) << "\b flash attention operation backward graph: "
          << cudnnGraph.Graph();

  return cudnnGraph;
#else
  return absl::UnimplementedError(
      "Cudnn flash attention only supported with Cudnn >= 9.0.0");
#endif
}

absl::Status CudnnSupport::DoPrepareForConvolution(
    dnn::ConvolutionKind kind, dnn::DataType element_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::AlgorithmConfig& algorithm_config,
    ScratchAllocator* scratch_allocator, dnn::AlgorithmDesc* algorithm_desc,
    DeviceMemory<uint8_t>* scratch_memory) {
  CudnnTensorDescriptor input_nd(
      input_descriptor,
      ToCudnnDataType(element_type, input_descriptor.layout()));
  CudnnFilterDescriptor filter_nd(
      filter_descriptor,
      ToCudnnDataType(element_type, filter_descriptor.layout()));
  CudnnTensorDescriptor output_nd(
      output_descriptor,
      ToCudnnDataType(element_type, output_descriptor.layout()));

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  switch (kind) {
    case dnn::ConvolutionKind::FORWARD: {
      TF_ASSIGN_OR_RETURN(*algorithm_desc,
                          GetCudnnConvolutionForwardAlgorithm(
                              stream, cudnn, algorithm_config, input_nd,
                              filter_nd, element_type, convolution_descriptor,
                              output_nd, scratch_allocator, scratch_memory));
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_DATA: {
      TF_ASSIGN_OR_RETURN(*algorithm_desc,
                          GetCudnnConvolutionBackwardDataAlgorithm(
                              stream, cudnn, algorithm_config, input_nd,
                              filter_nd, element_type, convolution_descriptor,
                              output_nd, scratch_allocator, scratch_memory));
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_FILTER: {
      TF_ASSIGN_OR_RETURN(*algorithm_desc,
                          GetCudnnConvolutionBackwardFilterAlgorithm(
                              stream, cudnn, algorithm_config, input_nd,
                              filter_nd, element_type, convolution_descriptor,
                              output_nd, scratch_allocator, scratch_memory));
      break;
    }
    default:
      return tsl::errors::Internal("Unexpected convolution kind ",
                                   static_cast<int>(kind));
  }

  return absl::OkStatus();
}

class CudnnLegacyConvRunner : public dnn::ConvRunner {
 public:
  // Queries the workspace size and constructs a 'CudnnLegacyConvRunner'.
  static absl::StatusOr<CudnnLegacyConvRunner> Create(
      StreamExecutor* parent, Stream* stream, CudnnAccess* cudnn,
      const dnn::AlgorithmDesc& algo, dnn::DataType input_type,
      dnn::DataType output_type, dnn::ConvolutionKind kind,
      CudnnTensorDescriptor input_nd, CudnnTensorDescriptor output_nd,
      CudnnFilterDescriptor filter, CudnnConvolutionDescriptor conv) {
    size_t workspace_size;
    if (algo.workspace_size()) {
      workspace_size = *algo.workspace_size();
    } else {
      // For old AlgorithmProtos loaded from serialized autotune maps and for
      // AlgorithmDescs constructed by manually specifying an algorithm ID, we
      // need to compute the workspace size here.
      auto handle = cudnn->GetHandle(parent, stream);

      switch (kind) {
        case dnn::ConvolutionKind::FORWARD:
          RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
              handle.handle(),
              /*xDesc=*/input_nd.handle(),
              /*wDesc=*/filter.handle(), /*convDesc=*/conv.handle(),
              /*yDesc=*/output_nd.handle(),
              /*algo=*/ToConvForwardAlgo(algo),
              /*sizeInBytes=*/&workspace_size));
          break;
        case dnn::ConvolutionKind::BACKWARD_FILTER:
          RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
              handle.handle(),
              /*xDesc=*/input_nd.handle(),
              /*dyDesc=*/output_nd.handle(),
              /*convDesc=*/conv.handle(),
              /*gradDesc=*/filter.handle(),
              /*algo=*/ToConvBackwardFilterAlgo(algo),
              /*sizeInBytes=*/&workspace_size));
          break;
        case dnn::ConvolutionKind::BACKWARD_DATA:
          RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
              handle.handle(),
              /*wDesc=*/filter.handle(),
              /*dyDesc=*/output_nd.handle(),
              /*convDesc=*/conv.handle(),
              /*dxDesc=*/input_nd.handle(),
              /*algo=*/ToConvBackwardDataAlgo(algo),
              /*sizeInBytes=*/&workspace_size));
          break;
        default:
          return tsl::errors::Internal(
              "Invalid ConvolutionKind for CudnnLegacyConvRunner.");
      }
    }

    return {{parent, cudnn, algo.algo_id(), algo.tensor_ops_enabled(),
             workspace_size, input_type, output_type, kind, std::move(input_nd),
             std::move(output_nd), std::move(filter), std::move(conv)}};
  }

  std::string ToString() const override {
    return MakeAlgorithmDesc().ToString();
  }

  size_t GetWorkspaceSize() const override { return workspace_size_; }

  absl::StatusOr<dnn::AlgorithmDesc> ToAlgorithmDesc() const override {
    return MakeAlgorithmDesc();
  }

  absl::Status operator()(Stream* stream, dnn::ProfileResult* profile_result,
                          DeviceMemoryBase scratch_memory,
                          DeviceMemoryBase input_data,
                          DeviceMemoryBase filter_data,
                          DeviceMemoryBase output_data) const override {
    auto algo = MakeAlgorithmDesc();

    if (parent_ != stream->parent()) {
      return tsl::errors::Internal(
          "CudnnLegacyConvRunner cached across multiple StreamExecutors.");
    }

    auto cudnn = cudnn_->GetHandle(parent_, stream);
    // Alpha is the scaling factor for input.
    float falpha = 1.0;
    double dalpha = 1.0;
    void* alpha = input_type_ == dnn::DataType::kDouble
                      ? static_cast<void*>(&dalpha)
                      : static_cast<void*>(&falpha);
    // Beta is the scaling factor for output.
    float fbeta = 0.0;
    double dbeta = 0.0;
    void* beta = input_type_ == dnn::DataType::kDouble
                     ? static_cast<void*>(&dbeta)
                     : static_cast<void*>(&fbeta);

    std::unique_ptr<EventBasedTimer> timer;
    if (profile_result != nullptr) {
      TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                     profile_result->warmup_run_executed()));
    }

    const auto get_fwd_bugs = [&]() -> absl::Status {
      (void)output_type_;  // To stop clang-tidy saying it's unused.
      return absl::OkStatus();
    };

    auto get_bwd_data_bugs = [&]() -> absl::Status { return absl::OkStatus(); };

    const auto get_bwd_filter_bugs = [&]() -> absl::Status {
      return absl::OkStatus();
    };

    switch (kind_) {
      case dnn::ConvolutionKind::FORWARD: {
        TF_RETURN_IF_ERROR(get_fwd_bugs());
        RETURN_IF_CUDNN_ERROR(cudnnConvolutionForward(
            cudnn.handle(),
            /*alpha=*/alpha, /*srcDesc=*/input_nd_.handle(),
            /*srcData=*/input_data.opaque(), /*filterDesc=*/filter_.handle(),
            /*filterData=*/filter_data.opaque(), /*convDesc=*/conv_.handle(),
            /*algo=*/ToConvForwardAlgo(algo),
            /*workSpace=*/scratch_memory.opaque(),
            /*workSpaceSizeInBytes=*/scratch_memory.size(), /*beta=*/beta,
            /*yDesc=*/output_nd_.handle(), /*y=*/output_data.opaque()));
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_DATA: {
        TF_RETURN_IF_ERROR(get_bwd_data_bugs());
        RETURN_IF_CUDNN_ERROR(cudnnConvolutionBackwardData(
            cudnn.handle(),
            /*alpha=*/alpha,
            /*wDesc=*/filter_.handle(),
            /*w=*/filter_data.opaque(),
            /*dyDesc=*/output_nd_.handle(),
            /*dy=*/output_data.opaque(),
            /*convDesc=*/conv_.handle(),
            /*algo=*/ToConvBackwardDataAlgo(algo),
            /*workSpace=*/scratch_memory.opaque(),
            /*workSpaceSizeInBytes=*/scratch_memory.size(),
            /*beta=*/beta,
            /*dxDesc=*/input_nd_.handle(),
            /*dx=*/input_data.opaque()));
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_FILTER: {
        TF_RETURN_IF_ERROR(get_bwd_filter_bugs());
        RETURN_IF_CUDNN_ERROR(cudnnConvolutionBackwardFilter(
            cudnn.handle(),
            /*alpha=*/alpha,
            /*srcDesc=*/input_nd_.handle(),
            /*srcData=*/input_data.opaque(),
            /*diffDesc=*/output_nd_.handle(),
            /*diffData=*/output_data.opaque(),
            /*convDesc=*/conv_.handle(),
            /*algo=*/ToConvBackwardFilterAlgo(algo),
            /*workSpace=*/scratch_memory.opaque(),
            /*workSpaceSizeInBytes=*/scratch_memory.size(),
            /*beta=*/beta,
            /*gradDesc=*/filter_.handle(),
            /*dw=*/filter_data.opaque()));
        break;
      }
      default:
        return tsl::errors::Internal("Unexpected convolution kind ",
                                     static_cast<int>(kind_));
    }

    if (timer != nullptr) {
      TF_RETURN_IF_ERROR(PopulateProfileFromTimer(
          timer.get(), algo, profile_result, scratch_memory.size()));
    }

    return absl::OkStatus();
  }

 private:
  // Private to prevent passing in the wrong workspace_size.
  CudnnLegacyConvRunner(StreamExecutor* parent, CudnnAccess* cudnn,
                        int64_t algo_id, bool tensor_ops_enabled,
                        size_t workspace_size, dnn::DataType input_type,
                        dnn::DataType output_type, dnn::ConvolutionKind kind,
                        CudnnTensorDescriptor input_nd,
                        CudnnTensorDescriptor output_nd,
                        CudnnFilterDescriptor filter,
                        CudnnConvolutionDescriptor conv)
      : parent_(parent),
        cudnn_(cudnn),
        algo_id_(algo_id),
        tensor_ops_enabled_(tensor_ops_enabled),
        workspace_size_(workspace_size),
        kind_(kind),
        input_type_(input_type),
        output_type_(output_type),
        input_nd_(std::move(input_nd)),
        output_nd_(std::move(output_nd)),
        filter_(std::move(filter)),
        conv_(std::move(conv)) {}

  // Internal form of ToAlgorithmDesc without the absl::StatusOr.
  dnn::AlgorithmDesc MakeAlgorithmDesc() const {
    return {algo_id_, tensor_ops_enabled_, workspace_size_};
  }

  StreamExecutor* parent_;
  CudnnAccess* cudnn_;
  int64_t algo_id_;
  bool tensor_ops_enabled_;
  size_t workspace_size_;
  dnn::ConvolutionKind kind_;
  dnn::DataType input_type_;
  dnn::DataType output_type_;

  CudnnTensorDescriptor input_nd_;
  CudnnTensorDescriptor output_nd_;
  CudnnFilterDescriptor filter_;
  CudnnConvolutionDescriptor conv_;
};

absl::Status CudnnSupport::DoConvolve(
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::AlgorithmDesc algorithm_desc, DeviceMemory<uint8_t> scratch_memory,
    dnn::ProfileResult* profile_result) {
  cudnnDataType_t cudnn_type =
      ToCudnnDataType(element_type, input_descriptor.layout());
  CudnnTensorDescriptor input_nd(input_descriptor, cudnn_type);
  CudnnTensorDescriptor output_nd(
      output_descriptor,
      ToCudnnDataType(output_type, output_descriptor.layout()));
  CudnnFilterDescriptor filter_nd(
      filter_descriptor,
      ToCudnnDataType(element_type, filter_descriptor.layout()));

  auto accumulator_type = GetConvAccumulatorType(element_type);
  CudnnConvolutionDescriptor conv(convolution_descriptor,
                                  ToCudnnDataType(accumulator_type));
  bool use_tensor_ops = UseTensorOps(element_type, algorithm_desc);
  conv.set_use_tensor_op_math(use_tensor_ops);

  TF_ASSIGN_OR_RETURN(
      auto runner,
      CudnnLegacyConvRunner::Create(
          parent_, stream, cudnn_.get(), algorithm_desc, element_type,
          output_type, kind, std::move(input_nd), std::move(output_nd),
          std::move(filter_nd), std::move(conv)));
  return runner(stream, profile_result, scratch_memory, input_data, filter_data,
                output_data);
}

// Utility for dealing with CUDA's type-erased scaling parameters, where some
// sets of parameters expect a void* pointing at a float while others expect
// it to point at a double.
//
// This is rather ugly, but its purpose is to quarantine the corresponding
// ugliness that already exists in the CUDA API.
class ScalingParam {
 public:
  explicit ScalingParam(double value)
      : as_double_(value),
        as_float_(value),
        as_half_(value),
        as_bfloat16_(value),
        default_target_dtype_(dnn::DataType::kFloat) {}
  explicit ScalingParam(double value, dnn::DataType element_type)
      : as_double_(value),
        as_float_(value),
        as_half_(value),
        as_bfloat16_(value),
        default_target_dtype_(element_type) {}

  // Return a pointer to the appropriate representation type for the given
  // element type.
  //
  // See
  // https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters
  // for more info; the behavior for int8 result tensors is not described
  // there, but is maintained from the existing behavior (namely, using a
  // float scaling parameter).
  void* ToVoidPointer(dnn::DataType element_type) {
    if (element_type == dnn::DataType::kDouble) {
      return &as_double_;
    } else if (element_type == dnn::DataType::kHalf) {
      return &as_half_;
    } else if (element_type == dnn::DataType::kBF16) {
      return &as_bfloat16_;
    } else {
      return &as_float_;
    }
  }

  const void* ToVoidPointer() const {
    if (default_target_dtype_ == dnn::DataType::kDouble) {
      return &as_double_;
    } else if (default_target_dtype_ == dnn::DataType::kHalf) {
      return &as_half_;
    } else if (default_target_dtype_ == dnn::DataType::kBF16) {
      return &as_bfloat16_;
    } else {
      return &as_float_;
    }
  }

 private:
  double as_double_;
  float as_float_;
  Eigen::half as_half_;
  Eigen::bfloat16 as_bfloat16_;
  dnn::DataType default_target_dtype_;
};

struct BackendDescriptorDeleter {
  void operator()(cudnnBackendDescriptor_t desc) {
    cudnnBackendDestroyDescriptor(desc);
  }
};

using BackendDescriptor = std::unique_ptr<void, BackendDescriptorDeleter>;

absl::StatusOr<BackendDescriptor> CreateBackendDesc(
    cudnnBackendDescriptorType_t type) {
  void* result;
  RETURN_IF_CUDNN_ERROR(cudnnBackendCreateDescriptor(type, &result));
  return BackendDescriptor(result);
}

// Get the values of a CUDNN_TYPE_BACKEND_DESCRIPTOR attribute as a vector.
//
// This is fetching the entirety of a single sequence-valued attribute, as
// opposed to a sequence of multiple attributes.  The distinction is a bit
// meaningless, but this is the presentation the cuDNN docs use, so it may as
// well be consistent.
absl::StatusOr<std::vector<BackendDescriptor>> GetDescriptorAttribute(
    cudnnBackendDescriptor_t desc, cudnnBackendAttributeName_t name,
    cudnnBackendDescriptorType_t type) {
  int64_t n;
  RETURN_IF_CUDNN_ERROR(cudnnBackendGetAttribute(
      desc, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &n, nullptr));

  std::vector<BackendDescriptor> result(n);
  for (int i = 0; i < n; ++i) {
    TF_ASSIGN_OR_RETURN(result[i], CreateBackendDesc(type));
  }

  std::vector<cudnnBackendDescriptor_t> raw_ptrs;
  raw_ptrs.reserve(result.size());
  absl::c_transform(result, std::back_inserter(raw_ptrs),
                    [](const BackendDescriptor& ptr) { return ptr.get(); });

  // This API evidently does a deep copy of the descriptors into the pointers
  // in the output array, rather than writing pointers to the descriptors into
  // the output array.  So, this writes the memory behind each
  // BackendDescriptor in result, rather than writing the contents of
  // raw_ptrs.
  RETURN_IF_CUDNN_ERROR(cudnnBackendGetAttribute(
      desc, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, n, &n, raw_ptrs.data()));

  return result;
}

// Extract the engine ID and tuning knobs from the ExecutionPlan, and return
// them in the form of an AlgorithmDesc for use with RebuildExecutionPlan.
absl::StatusOr<dnn::AlgorithmDesc> ExecutionPlanToAlgorithmDesc(
    const cudnn_frontend::ExecutionPlan& plan, size_t workspace_size) {
  TF_ASSIGN_OR_RETURN(
      auto engine_cfgs,
      GetDescriptorAttribute(plan.get_raw_desc(),
                             CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                             CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
  if (engine_cfgs.size() != 1) {
    return tsl::errors::Internal(
        "CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG had more than one element.");
  }

  TF_ASSIGN_OR_RETURN(
      auto engines,
      GetDescriptorAttribute(engine_cfgs[0].get(), CUDNN_ATTR_ENGINECFG_ENGINE,
                             CUDNN_BACKEND_ENGINE_DESCRIPTOR));
  if (engines.size() != 1) {
    return tsl::errors::Internal(
        "CUDNN_ATTR_ENGINECFG_ENGINE had more than one element.");
  }

  int64_t n;
  int64_t engine_id;
  RETURN_IF_CUDNN_ERROR(
      cudnnBackendGetAttribute(engines[0].get(), CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                               CUDNN_TYPE_INT64, 1, &n, &engine_id));

  // Apparently for CUDNN_ATTR_ENGINECFG_KNOB_CHOICES only, trying to query
  // the number of elements in the attribute by using an output limit value of
  // 0 just returns 0; the only way to find out how many there are is to
  // pre-allocate space for every existing knob type (as an upper bound on the
  // number of knob choices a config can have), and then look back at how many
  // were filled.
  std::vector<BackendDescriptor> knobs(CUDNN_KNOB_TYPE_COUNTS);
  for (int i = 0; i < knobs.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        knobs[i], CreateBackendDesc(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR));
  }
  std::vector<cudnnBackendDescriptor_t> raw_knob_ptrs;
  raw_knob_ptrs.reserve(knobs.size());
  absl::c_transform(knobs, std::back_inserter(raw_knob_ptrs),
                    [](const BackendDescriptor& ptr) { return ptr.get(); });
  RETURN_IF_CUDNN_ERROR(cudnnBackendGetAttribute(
      engine_cfgs[0].get(), CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
      CUDNN_TYPE_BACKEND_DESCRIPTOR, raw_knob_ptrs.size(), &n,
      raw_knob_ptrs.data()));
  knobs.resize(n);

  absl::flat_hash_map<int64_t, int64_t> tuning_knobs;
  for (const auto& knob : knobs) {
    cudnnBackendKnobType_t knob_type;
    int64_t knob_value;

    RETURN_IF_CUDNN_ERROR(
        cudnnBackendGetAttribute(knob.get(), CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                                 CUDNN_TYPE_KNOB_TYPE, 1, &n, &knob_type));
    if (n != 1) {
      return tsl::errors::Internal(
          "Knob should have exactly one KNOB_TYPE; had ", n);
    }

    RETURN_IF_CUDNN_ERROR(
        cudnnBackendGetAttribute(knob.get(), CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                                 CUDNN_TYPE_INT64, 1, &n, &knob_value));
    if (n != 1) {
      return tsl::errors::Internal(
          "Knob should have exactly one KNOB_VALUE; had ", n);
    }

    auto emplaced = tuning_knobs.try_emplace(knob_type, knob_value).second;
    if (!emplaced) {
      return tsl::errors::Internal(absl::StrFormat(
          "cuDNN gave multiple knob values for the same knob type.\n"
          "  KNOB_TYPE: %d\n"
          "  new KNOB_VALUE: %d\n"
          "  old KNOB_VALUE: %d",
          knob_type, knob_value, tuning_knobs.at(knob_type)));
    }
  }

  std::vector<std::pair<int64_t, int64_t>> tuning_knobs_vec;
  tuning_knobs_vec.reserve(tuning_knobs.size());
  absl::c_copy(tuning_knobs, std::back_inserter(tuning_knobs_vec));

  return dnn::AlgorithmDesc(engine_id, tuning_knobs_vec, workspace_size);
}

template <typename Sig>
class CudnnExecutionPlanRunner;
// An OpRunner implemented by an ExecutionPlan.
//
// This is the class holding the implementation of ToString, GetWorkspaceSize,
// and operator() for use by the cudnn frontend op runners.
template <typename... Args>
class CudnnExecutionPlanRunner<void(Args...)>
    : public dnn::OpRunner<void(Args...)> {
 public:
  std::string ToString() const override { return plan_.getTag(); }

  size_t GetWorkspaceSize() const override { return workspace_size_; }

  absl::StatusOr<dnn::AlgorithmDesc> ToAlgorithmDesc() const override {
    return ExecutionPlanToAlgorithmDesc(plan_, workspace_size_);
  }

  absl::Status operator()(Stream* stream, dnn::ProfileResult* profile_result,
                          DeviceMemoryBase scratch_memory,
                          Args... inputs) const override {
    if (parent_ != stream->parent()) {
      return tsl::errors::Internal(
          "CudnnExecutionPlanRunner cached across multiple StreamExecutors.");
    }

    auto cudnn = cudnn_->GetHandle(parent_, stream);

    size_t workspace_size = plan_.getWorkspaceSize();

    RETURN_MSG_IF_CUDNN_ERROR(plan_);
    bool should_add_scalars =
        !scalar_input_uids_.empty() && !scalar_input_values_.empty();

    std::vector<int64_t> data_uids_vec = {data_uids_.cbegin(),
                                          data_uids_.cend()};
    std::vector<void*> data_ptrs_vec;

    // The operands of ForwardGraph convolutions and norm Custom Calls are
    // gathered dynamically. In these cases, Args... is
    // std::vector<DeviceMemoryBase>.
    if constexpr (sizeof...(Args) == 1 &&
                  std::is_same_v<std::tuple_element_t<0, std::tuple<Args...>>,
                                 std::vector<DeviceMemoryBase>>) {
      for (DeviceMemoryBase input : std::get<0>(std::tie(inputs...))) {
        data_ptrs_vec.push_back(input.opaque());
      }
    } else {
      data_ptrs_vec = {inputs.opaque()...};
      // We use need_side_input to determine if the side input 'z' from
      // {'x', 'w', 'z', 'b', 'y'} is needed for the conv-<add>-bias-act
      // patterns.
      if (sizeof...(Args) == 5 && !need_side_input_) {
        data_uids_vec.erase(data_uids_vec.begin() + 2);
        data_ptrs_vec.erase(data_ptrs_vec.begin() + 2);
      }
    }

    if (should_add_scalars) {
      data_uids_vec.insert(data_uids_vec.end(), scalar_input_uids_.begin(),
                           scalar_input_uids_.end());
      for (int64_t i = 0; i < scalar_input_values_.size(); i++) {
        data_ptrs_vec.push_back(
            const_cast<void*>(scalar_input_values_[i].ToVoidPointer()));
      }
    }

    auto variantPack =
        cudnn_frontend::VariantPackBuilder()
            .setWorkspacePointer(scratch_memory.opaque())
            .setDataPointers(data_ptrs_vec.size(), data_ptrs_vec.data())
            .setUids(data_uids_vec.size(), data_uids_vec.data())
            .build();
    RETURN_MSG_IF_CUDNN_ERROR(variantPack);
    VLOG(4) << "\nDo cudnn execution plan with plan tag: " << plan_.getTag()
            << "\nWorkspace size in bytes: " << workspace_size
            << "\nVariantPack: " << variantPack.describe();

    std::unique_ptr<EventBasedTimer> timer;
    if (profile_result != nullptr) {
      TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                     profile_result->warmup_run_executed()));
    }

    cudnnStatus_t status = cudnnBackendExecute(
        cudnn.handle(), plan_.get_raw_desc(), variantPack.get_raw_desc());
    RETURN_IF_CUDNN_ERROR(status);

    if (timer != nullptr) {
      TF_ASSIGN_OR_RETURN(auto desc, ToAlgorithmDesc());
      TF_RETURN_IF_ERROR(PopulateProfileFromTimer(
          timer.get(), desc, profile_result, scratch_memory.size()));

      VLOG(4) << "cudnn op with plan " << plan_.getTag()
              << ", workspace_size=" << workspace_size << " -> "
              << CudnnStatusToString(status) << " in "
              << profile_result->elapsed_time_in_ms() << "ms";
    }

    return absl::OkStatus();
  }

  static absl::StatusOr<CudnnExecutionPlanRunner> Create(
      StreamExecutor* parent, CudnnAccess* cudnn,
      cudnn_frontend::ExecutionPlan plan, absl::Span<const int64_t> uids,
      bool need_side_input) {
    auto workspace_size = static_cast<uint64_t>(plan.getWorkspaceSize());
    RETURN_MSG_IF_CUDNN_ERROR(plan);
    return {{parent,
             cudnn,
             std::move(plan),
             workspace_size,
             uids,
             need_side_input,
             {},
             {}}};
  }

  static absl::StatusOr<CudnnExecutionPlanRunner> Create(
      StreamExecutor* parent, CudnnAccess* cudnn,
      cudnn_frontend::ExecutionPlan plan, absl::Span<const int64_t> uids,
      bool need_side_input, std::vector<int64_t> scalar_input_uids,
      std::vector<ScalingParam> scalar_input_values) {
    auto workspace_size = static_cast<uint64_t>(plan.getWorkspaceSize());
    RETURN_MSG_IF_CUDNN_ERROR(plan);
    return {{parent, cudnn, std::move(plan), workspace_size, uids,
             need_side_input, scalar_input_uids, scalar_input_values}};
  }

 private:
  CudnnExecutionPlanRunner(StreamExecutor* parent, CudnnAccess* cudnn,
                           cudnn_frontend::ExecutionPlan plan,
                           size_t workspace_size,
                           absl::Span<const int64_t> uids, bool need_side_input,
                           std::vector<int64_t> scalar_input_uids,
                           std::vector<ScalingParam> scalar_input_values)
      : parent_(parent),
        cudnn_(cudnn),
        plan_(std::move(plan)),
        workspace_size_(workspace_size),
        data_uids_(uids.begin(), uids.end()),
        need_side_input_(need_side_input),
        scalar_input_uids_(scalar_input_uids),
        scalar_input_values_(scalar_input_values) {}
  StreamExecutor* parent_;
  CudnnAccess* cudnn_;
  cudnn_frontend::ExecutionPlan plan_;
  size_t workspace_size_;
  absl::InlinedVector<int64_t, sizeof...(Args)> data_uids_;
  bool need_side_input_;
  std::vector<int64_t> scalar_input_uids_;
  std::vector<ScalingParam> scalar_input_values_;
};

namespace {

template <typename Sig>
absl::Status CreateOpRunners(
    Stream* stream, CudnnHandle& cudnn, StreamExecutor* gpu_executor,
    CudnnAccess* cudnn_access,
    std::unique_ptr<cudnn_frontend::OperationGraph> op_graph,
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    absl::Span<const int64_t> input_uids, bool use_fallback,
    std::vector<std::unique_ptr<const dnn::OpRunner<Sig>>>* out_runners,
    bool need_side_input, const NumericOptions& numeric_options) {
  cudnn_frontend::EngineConfigList filtered_configs;
  const bool disable_winograd = !CudnnEnvVar<WinogradNonfused>::IsEnabled();
  const bool disable_tensor_core =
      !IsTensorMathEnabled(stream, input_type, numeric_options.allow_tf32);
  auto generic_filter_fn = [=](cudnnBackendDescriptor_t engine_config) -> bool {
    return GenericEngineFilter(engine_config, disable_winograd,
                               numeric_options.require_determinism,
                               disable_tensor_core);
  };
  VLOG(4) << "Filtering engine configs with disable_winograd="
          << disable_winograd
          << ", disable_nondeterminism=" << numeric_options.require_determinism
          << ", disable_tensor_core=" << disable_tensor_core;

  std::array<std::string, 1> heur_mode = {use_fallback ? "heuristics_fallback"
                                                       : "heuristics_mode_b"};
  std::vector<cudnnStatus_t> ret = cudnn_frontend::get_heuristics_list(
      heur_mode, *op_graph, generic_filter_fn, filtered_configs);
  for (auto status : ret) {
    RETURN_IF_CUDNN_ERROR(status);
  }
  // Also try heuristics_mode_a, because it may contain other fast algorithms
  // that are not included in heuristics_mode_b.
  // TODO(b/235475195): Remove once cuDNN 9.0.0 is available that improves
  // heuristics_mode_b.
  if (!use_fallback) {
    cudnn_frontend::EngineConfigList filtered_configs_a;
    ret = cudnn_frontend::get_heuristics_list(
        std::array<std::string, 1>{"heuristics_mode_a"}, *op_graph,
        generic_filter_fn, filtered_configs_a);
    for (auto status : ret) {
      RETURN_IF_CUDNN_ERROR(status);
    }
    filtered_configs.insert(filtered_configs.end(), filtered_configs_a.begin(),
                            filtered_configs_a.end());
  }

  auto fn = []() { return true; };
  auto maybe_json_handle_static = CudnnExecutionPlanEngineFilterStatic();
  auto maybe_json_handle_runtime = CudnnExecutionPlanEngineFilterRuntime();

  out_runners->clear();
  absl::flat_hash_set<dnn::AlgorithmDesc> algorithm_deduplication;
  for (int i = 0; i < filtered_configs.size(); i++) {
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(cudnn.handle())
                    .setEngineConfig(filtered_configs[i], op_graph->getTag())
                    .build();
    if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
      continue;
    }

    if (maybe_json_handle_static &&
        cudnn_frontend::check_errata(*maybe_json_handle_static, plan.getTag(),
                                     cudnn.handle(), fn)) {
      VLOG(4) << "Exclude engine (static): " << plan.getTag();
      continue;
    }
    if (maybe_json_handle_runtime &&
        cudnn_frontend::check_errata(*maybe_json_handle_runtime, plan.getTag(),
                                     cudnn.handle(), fn)) {
      VLOG(4) << "Exclude engine (runtime): " << plan.getTag();
      continue;
    }

    auto runner_or = CudnnExecutionPlanRunner<Sig>::Create(
        gpu_executor, cudnn_access, std::move(plan), input_uids,
        need_side_input);
    if (!runner_or.ok()) {
      // Note this can happen if cuDNN Frontend gives us partially-initialized
      // ExecutionPlans because its error handling is broken in non-exception
      // builds; those were meant to be filtered out earlier inside cuDNN
      // Frontend, but instead they get filtered out here.
      VLOG(4) << "Failed building runner from ExecutionPlan (i.e. failed "
                 "getting its workspace size): "
              << runner_or.status();
      continue;
    }
    // We currently collect a list of algorithms using heuristics_mode_a and
    // heuristics_mode_b, so we can potentially have duplicates. But we should
    // not actually autotune the same algorithm twice!
    if (!algorithm_deduplication.insert(runner_or->ToAlgorithmDesc().value())
             .second) {
      continue;
    }

    out_runners->push_back(std::make_unique<CudnnExecutionPlanRunner<Sig>>(
        std::move(runner_or).value()));

    // We will use the first working plan when determinism is required.
    if (numeric_options.require_determinism) {
      break;
    }
  }

  VLOG(4) << "\nReturned execution plans size: " << out_runners->size();

  return absl::OkStatus();
}

}  // namespace

absl::Status CudnnSupport::GetConvolveRunners(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor,
    DeviceMemoryBase /*input_data*/,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase /*filter_data*/,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase /*output_data*/,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    ScratchAllocator* /*scratch_allocator*/,
    const NumericOptions& numeric_options,
    std::vector<std::unique_ptr<const dnn::ConvRunner>>* out_exec_plans) {
  auto cudnn = cudnn_->GetHandle(parent_, stream);
  TF_ASSIGN_OR_RETURN(
      auto op_graph,
      GetCudnnOperationGraph(kind, input_type, output_type, input_descriptor,
                             filter_descriptor, output_descriptor,
                             convolution_descriptor, cudnn));

  return CreateOpRunners<dnn::ConvSignature>(
      stream, cudnn, parent_, cudnn_.get(), std::move(op_graph), kind,
      input_type, {'x', 'w', 'y'}, use_fallback, out_exec_plans,
      /*need_side_input=*/false, numeric_options);
}

absl::Status CudnnSupport::GetGraphConvolveRunners(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    const NumericOptions& numeric_options,
    std::vector<std::unique_ptr<const dnn::GraphConvRunner>>* out_exec_plans,
    std::string serialized_graph) {
  auto cudnn = cudnn_->GetHandle(parent_, stream);
  TF_ASSIGN_OR_RETURN(
      auto op_graph_and_uids,
      GetGenericCudnnOperationGraph(
          kind, input_type, input_descriptor, filter_descriptor,
          output_descriptor, convolution_descriptor, cudnn, serialized_graph));
  return CreateOpRunners<dnn::GraphConvSignature>(
      stream, cudnn, parent_, cudnn_.get(), std::move(op_graph_and_uids.first),
      kind, input_type, op_graph_and_uids.second, use_fallback, out_exec_plans,
      /*need_side_input=*/false, numeric_options);
}

absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>>
CudnnSupport::ConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor) {
  if (!algorithm_desc.is_cudnn_frontend()) {
    CudnnConvolutionDescriptor conv(
        convolution_descriptor,
        ToCudnnDataType(GetConvAccumulatorType(input_type)));
    conv.set_use_tensor_op_math(algorithm_desc.tensor_ops_enabled());

    if (filter_descriptor.layout() ==
        dnn::FilterLayout::kOutputInputYX32_CudnnReordered) {
      CHECK_CUDNN_OK(
          cudnnSetConvolutionReorderType(conv.handle(), CUDNN_NO_REORDER));
    }

    TF_ASSIGN_OR_RETURN(
        auto runner,
        CudnnLegacyConvRunner::Create(
            parent_, stream, cudnn_.get(), algorithm_desc, input_type,
            output_type, kind,
            /* input_nd = */
            CudnnTensorDescriptor(
                input_descriptor,
                ToCudnnDataType(input_type, input_descriptor.layout())),
            /* output_nd = */
            CudnnTensorDescriptor(
                output_descriptor,
                ToCudnnDataType(output_type, output_descriptor.layout())),
            /* filter = */
            CudnnFilterDescriptor(
                filter_descriptor,
                ToCudnnDataType(input_type, filter_descriptor.layout())),
            std::move(conv)));

    return {std::make_unique<CudnnLegacyConvRunner>(std::move(runner))};
  }

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  TF_ASSIGN_OR_RETURN(
      auto op_graph,
      GetCudnnOperationGraph(kind, input_type, output_type, input_descriptor,
                             filter_descriptor, output_descriptor,
                             convolution_descriptor, cudnn));

  TF_ASSIGN_OR_RETURN(auto execution_plan,
                      RebuildExecutionPlan(cudnn, algorithm_desc, *op_graph));

  TF_ASSIGN_OR_RETURN(
      auto runner,
      CudnnExecutionPlanRunner<dnn::ConvSignature>::Create(
          parent_, cudnn_.get(), std::move(execution_plan), {'x', 'w', 'y'},
          /*need_side_input=*/false));
  return {std::make_unique<CudnnExecutionPlanRunner<dnn::ConvSignature>>(
      std::move(runner))};
}

absl::StatusOr<std::unique_ptr<const dnn::GraphConvRunner>>
CudnnSupport::GraphConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    std::string serialized_graph) {
  if (!algorithm_desc.is_cudnn_frontend()) {
    return tsl::errors::Internal(
        "cuDNN graph execution requires the use of the cuDNN frontend.");
  }

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  TF_ASSIGN_OR_RETURN(
      auto op_graph_and_uids,
      GetGenericCudnnOperationGraph(
          kind, input_type, input_descriptor, filter_descriptor,
          output_descriptor, convolution_descriptor, cudnn, serialized_graph));

  TF_ASSIGN_OR_RETURN(
      auto execution_plan,
      RebuildExecutionPlan(cudnn, algorithm_desc, *op_graph_and_uids.first));

  TF_ASSIGN_OR_RETURN(auto runner,
                      CudnnExecutionPlanRunner<dnn::GraphConvSignature>::Create(
                          parent_, cudnn_.get(), std::move(execution_plan),
                          op_graph_and_uids.second,
                          /*need_side_input=*/false));
  return {std::make_unique<CudnnExecutionPlanRunner<dnn::GraphConvSignature>>(
      std::move(runner))};
}

class CudnnLegacyFusedConvRunner : public dnn::FusedConvRunner {
 public:
  // Queries the workspace size and constructs a 'CudnnLegacyFusedConvRunner'.
  static absl::StatusOr<CudnnLegacyFusedConvRunner> Create(
      StreamExecutor* parent, Stream* stream, CudnnAccess* cudnn,
      const dnn::AlgorithmDesc& algo, dnn::DataType input_type,
      double conv_scale, double side_input_scale,
      CudnnTensorDescriptor input_nd, CudnnTensorDescriptor output_nd,
      CudnnFilterDescriptor filter, CudnnTensorDescriptor bias_nd,
      CudnnConvolutionDescriptor conv,
      CudnnActivationDescriptor activation_desc) {
    size_t workspace_size;
    if (algo.workspace_size()) {
      workspace_size = *algo.workspace_size();
    } else {
      auto handle = cudnn->GetHandle(parent, stream);

      RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
          handle.handle(),
          /*xDesc=*/input_nd.handle(),
          /*wDesc=*/filter.handle(), /*convDesc=*/conv.handle(),
          /*yDesc=*/output_nd.handle(),
          /*algo=*/ToConvForwardAlgo(algo),
          /*sizeInBytes=*/&workspace_size));
    }

    return {{parent, cudnn, algo.algo_id(), algo.tensor_ops_enabled(),
             workspace_size, input_type, conv_scale, side_input_scale,
             std::move(input_nd), std::move(output_nd), std::move(filter),
             std::move(bias_nd), std::move(conv), std::move(activation_desc)}};
  }

  std::string ToString() const override {
    return MakeAlgorithmDesc().ToString();
  }

  uint64_t GetWorkspaceSize() const override { return workspace_size_; }

  absl::StatusOr<dnn::AlgorithmDesc> ToAlgorithmDesc() const override {
    return MakeAlgorithmDesc();
  }

  absl::Status operator()(Stream* stream, dnn::ProfileResult* profile_result,
                          DeviceMemoryBase scratch_memory,
                          DeviceMemoryBase input_data,
                          DeviceMemoryBase filter_data,
                          DeviceMemoryBase side_input_data,
                          DeviceMemoryBase bias_data,
                          DeviceMemoryBase output_data) const override {
    if (parent_ != stream->parent()) {
      return tsl::errors::Internal(
          "CudnnLegacyFusedConvRunner cached across multiple "
          "StreamExecutors.");
    }

    auto algo = MakeAlgorithmDesc();
    std::unique_ptr<EventBasedTimer> timer;

    if (profile_result != nullptr) {
      TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                     profile_result->warmup_run_executed()));
    }
    auto side_input_data_ptr = (side_input_scale_ == 0)
                                   ? output_data.opaque()
                                   : side_input_data.opaque();

    auto cudnn = cudnn_->GetHandle(parent_, stream);

    VLOG(2) << "\nconv_scale = " << conv_scale_
            << "\nconv_input_nd.handle() = " << input_nd_.handle()
            << "\nconv_input_data.opaque() = " << input_data.opaque()
            << "\nfilter.handle() = " << filter_.handle()
            << "\nfilter_data.opaque() = " << filter_data.opaque()
            << "\nconv.handle() = " << conv_.handle() << "\nalgo = " << algo_id_
            << ", tensor_ops_enabled=" << tensor_ops_enabled_
            << "\nscratch.opaque() = " << scratch_memory.opaque()
            << "\nscratch.size() = " << scratch_memory.size()
            << "\nside_input_scale = " << side_input_scale_
            << "\noutput_nd.handle() = " << output_nd_.handle()
            << "\nside_input_data_ptr = " << side_input_data_ptr
            << "\nbias_nd.handle() = " << bias_nd_.handle()
            << "\nbiases.opaque() = " << bias_data.opaque()
            << "\nactivation_desc.handle() = " << activation_desc_.handle()
            << "\noutput_nd.handle() = " << output_nd_.handle()
            << "\noutput_data.opaque() = " << output_data.opaque();

    if (IsTensorMathOpSet(conv_) != tensor_ops_enabled_) {
      return absl::FailedPreconditionError(
          "Tensor op math type in dnn::AlgorithmDesc does not "
          "match that of the CudnnConvolutionDescriptor");
    }

    // N.B. the scaling parameters alpha1 and alpha2 are pointers to
    // temporaries; this API doesn't persist the pointers beyond its own stack
    // frame.
    auto status = cudnnConvolutionBiasActivationForward(
        cudnn.handle(),
        /*alpha1=*/ScalingParam(conv_scale_).ToVoidPointer(input_type_),
        /*xDesc=*/input_nd_.handle(), /*x=*/input_data.opaque(),
        /*wDesc=*/filter_.handle(), /*w=*/filter_data.opaque(),
        /*convDesc=*/conv_.handle(), ToConvForwardAlgo(algo),
        /*workSpace=*/scratch_memory.opaque(),
        /*workSpaceSizeInBytes=*/scratch_memory.size(),
        /*alpha2=*/ScalingParam(side_input_scale_).ToVoidPointer(input_type_),
        /*zDesc=*/output_nd_.handle(), /*z=*/side_input_data_ptr,
        /*biasDesc=*/bias_nd_.handle(), /*bias=*/bias_data.opaque(),
        /*activationDesc=*/activation_desc_.handle(),
        /*yDesc=*/output_nd_.handle(), /*y=*/output_data.opaque());
    if (status != CUDNN_STATUS_SUCCESS || !profile_result) {
      VLOG(4) << "conv with algorithm " << ToConvForwardAlgo(algo)
              << ", workspace_size=" << scratch_memory.size() << " -> "
              << CudnnStatusToString(status);
    }
    RETURN_IF_CUDNN_ERROR(status);

    if (timer != nullptr) {
      TF_RETURN_IF_ERROR(PopulateProfileFromTimer(
          timer.get(), algo, profile_result, scratch_memory.size()));
      VLOG(4) << "conv with algorithm " << ToConvForwardAlgo(algo)
              << ", tensor_ops_enabled=" << tensor_ops_enabled_
              << ", workspace_size=" << scratch_memory.size() << " -> "
              << CudnnStatusToString(status) << " in "
              << profile_result->elapsed_time_in_ms() << "ms";
    }

    return absl::OkStatus();
  }

 private:
  // Private to prevent passing in the wrong workspace_size.
  CudnnLegacyFusedConvRunner(StreamExecutor* parent, CudnnAccess* cudnn,
                             int64_t algo_id, bool tensor_ops_enabled,
                             size_t workspace_size, dnn::DataType input_type,
                             double conv_scale, double side_input_scale,
                             CudnnTensorDescriptor input_nd,
                             CudnnTensorDescriptor output_nd,
                             CudnnFilterDescriptor filter,
                             CudnnTensorDescriptor bias_nd,
                             CudnnConvolutionDescriptor conv,
                             CudnnActivationDescriptor activation_desc)
      : parent_(parent),
        cudnn_(cudnn),
        algo_id_(algo_id),
        tensor_ops_enabled_(tensor_ops_enabled),
        workspace_size_(workspace_size),
        input_type_(input_type),
        conv_scale_(conv_scale),
        side_input_scale_(side_input_scale),
        input_nd_(std::move(input_nd)),
        output_nd_(std::move(output_nd)),
        filter_(std::move(filter)),
        bias_nd_(std::move(bias_nd)),
        conv_(std::move(conv)),
        activation_desc_(std::move(activation_desc)) {}

  // Internal form of ToAlgorithmDesc without the absl::StatusOr.
  dnn::AlgorithmDesc MakeAlgorithmDesc() const {
    return {algo_id_, tensor_ops_enabled_, workspace_size_};
  }

  StreamExecutor* parent_;
  CudnnAccess* cudnn_;
  int64_t algo_id_;
  bool tensor_ops_enabled_;
  size_t workspace_size_;
  dnn::DataType input_type_;
  double conv_scale_, side_input_scale_;

  CudnnTensorDescriptor input_nd_;
  CudnnTensorDescriptor output_nd_;
  CudnnFilterDescriptor filter_;
  CudnnTensorDescriptor bias_nd_;
  CudnnConvolutionDescriptor conv_;
  CudnnActivationDescriptor activation_desc_;
};

absl::StatusOr<std::unique_ptr<const dnn::FusedConvRunner>>
CudnnSupport::FusedConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType bias_type, dnn::DataType output_type, double conv_scale,
    double side_input_scale, double leakyrelu_alpha,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::ActivationMode activation_mode) {
  if (!algorithm_desc.is_cudnn_frontend()) {
    CudnnTensorDescriptor conv_input_nd(
        input_descriptor,
        ToCudnnDataType(input_type, input_descriptor.layout()));
    CudnnTensorDescriptor output_nd(
        output_descriptor,
        ToCudnnDataType(output_type, input_descriptor.layout()));
    CudnnFilterDescriptor filter(
        filter_descriptor,
        ToCudnnDataType(input_type, filter_descriptor.layout()));
    CudnnTensorDescriptor bias_nd(bias_descriptor, ToCudnnDataType(bias_type));

    CudnnConvolutionDescriptor conv(
        convolution_descriptor,
        ToCudnnDataType(GetConvAccumulatorType(input_type)));
    conv.set_use_tensor_op_math(algorithm_desc.tensor_ops_enabled());

    if (filter_descriptor.layout() ==
        dnn::FilterLayout::kOutputInputYX32_CudnnReordered) {
      CHECK_CUDNN_OK(
          cudnnSetConvolutionReorderType(conv.handle(), CUDNN_NO_REORDER));
    }

    // CUDNN v6 only supports CUDNN_NOT_PROPAGATE_NAN as the reluNanOpt for
    // activation descriptor. Note that this will change the nan propagation
    // behavior from separate conv, bias, and relu (which by default is
    // CUDNN_PROPAGATE_NAN).
    //
    // TODO(awpr): reevaluate this for newer cuDNN versions.
    CudnnActivationDescriptor activation_desc(activation_mode,
                                              CUDNN_NOT_PROPAGATE_NAN,
                                              output_descriptor.value_max());

    TF_ASSIGN_OR_RETURN(
        auto runner,
        CudnnLegacyFusedConvRunner::Create(
            parent_, stream, cudnn_.get(), algorithm_desc, input_type,
            conv_scale, side_input_scale, std::move(conv_input_nd),
            std::move(output_nd), std::move(filter), std::move(bias_nd),
            std::move(conv), std::move(activation_desc)));
    return {std::make_unique<CudnnLegacyFusedConvRunner>(std::move(runner))};
  }

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  TF_ASSIGN_OR_RETURN(auto op_graph,
                      GetCudnnFusedOperationGraph(
                          kind, input_type, bias_type, output_type, conv_scale,
                          side_input_scale, leakyrelu_alpha, input_descriptor,
                          filter_descriptor, bias_descriptor, output_descriptor,
                          convolution_descriptor, activation_mode, cudnn));

  TF_ASSIGN_OR_RETURN(auto execution_plan,
                      RebuildExecutionPlan(cudnn, algorithm_desc, *op_graph));

  bool need_side_input =
      SideInputNeeded(activation_mode, conv_scale, side_input_scale);
  TF_ASSIGN_OR_RETURN(auto runner,
                      CudnnExecutionPlanRunner<dnn::FusedConvSignature>::Create(
                          parent_, cudnn_.get(), std::move(execution_plan),
                          {'x', 'w', 'z', 'b', 'y'}, need_side_input));
  return {std::make_unique<CudnnExecutionPlanRunner<dnn::FusedConvSignature>>(
      std::move(runner))};
}

absl::Status CudnnSupport::GetFusedConvolveRunners(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType bias_type, dnn::DataType output_type, double conv_scale,
    double side_input_scale, double leakyrelu_alpha, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    const dnn::ActivationMode activation_mode,
    const NumericOptions& numeric_options,
    std::vector<std::unique_ptr<const dnn::FusedConvRunner>>* out_exec_plans) {
  // Fused convolutions with identity activations are broken in that they
  // implicitly do ReLU on some engines, and we can't reliably detect which
  // ones.

  if (input_type == dnn::DataType::kInt8 &&
      !stream->GetCudaComputeCapability().IsAtLeast(6, 1)) {
    return tsl::errors::Unimplemented(
        "cudnnConvolutionBiasActivationForward() for int8 is only supported "
        "on GPUs with compute capability 6.1 or later.");
  }

  if (activation_mode != dnn::ActivationMode::kRelu &&
      activation_mode != dnn::ActivationMode::kRelu6 &&
      activation_mode != dnn::ActivationMode::kElu &&
      activation_mode != dnn::ActivationMode::kLeakyRelu &&
      activation_mode != dnn::ActivationMode::kNone) {
    return absl::InvalidArgumentError(
        "CuDNN fusion only supports activations of "
        "{Relu, Relu6, Elu, <None>}.");
  }

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto op_graph_status = GetCudnnFusedOperationGraph(
      kind, input_type, bias_type, output_type, conv_scale, side_input_scale,
      leakyrelu_alpha, input_descriptor, filter_descriptor, bias_descriptor,
      output_descriptor, convolution_descriptor, activation_mode, cudnn);
  if (!op_graph_status.status().ok()) {
    return absl::InternalError(absl::StrCat(
        "Cudnn graph failed to build: ", op_graph_status.status().ToString()));
  }
  auto op_graph = std::move(op_graph_status).value();

  bool need_side_input =
      SideInputNeeded(activation_mode, conv_scale, side_input_scale);
  return CreateOpRunners<dnn::FusedConvSignature>(
      stream, cudnn, parent_, cudnn_.get(), std::move(op_graph), kind,
      input_type, {'x', 'w', 'z', 'b', 'y'}, use_fallback, out_exec_plans,
      need_side_input, numeric_options);
}

absl::Status CudnnSupport::GetFusedMatmulRunners(
    dnn::DataType input_type, dnn::DataType bias_type,
    dnn::DataType output_type, Stream* stream, bool trans_a, bool trans_b,
    uint64_t m, uint64_t n, uint64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    dnn::ActivationMode activation_mode, bool use_fallback,
    const NumericOptions& numeric_options,
    std::vector<std::unique_ptr<const dnn::FusedMatmulRunner>>*
        out_exec_plans) {
  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto op_graph_status = GetCudnnFusedMatmulGraph(
      input_type, bias_type, output_type, trans_a, trans_b, m, n, k, lda, ldb,
      ldc, activation_mode, cudnn);
  if (!op_graph_status.status().ok()) {
    return absl::InternalError(absl::StrCat(
        "Cudnn graph failed to build: ", op_graph_status.status().ToString()));
  }
  auto op_graph = std::move(op_graph_status).value();

  // The "need_side_input" will not actually affect the matmul execution. It
  // was proposed to work around a convolution issue with five inputs (see
  // SideInputNeeded()). Here, we set it true to make sure none of the inputs
  // get dropped in case the number of inputs get increased in the future.
  return CreateOpRunners<dnn::FusedMatmulSignature>(
      stream, cudnn, parent_, cudnn_.get(), std::move(op_graph),
      dnn::ConvolutionKind::INVALID, input_type, {'a', 'b', 'z', 'c'},
      use_fallback, out_exec_plans, /*need_side_input=*/true, numeric_options);
}

bool CudnnSupport::GetConvolveAlgorithms(
    CudaComputeCapability cuda_compute_capability, dnn::DataType input_type,
    const NumericOptions& numeric_options,
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  PreloadCudnnSubLibs(PreloadCudnnType::ConvFwd);

  bool tensor_op_math_available = IsTensorMathEnabled(
      cuda_compute_capability, input_type, numeric_options.allow_tf32);
  out_algorithms->clear();

  std::vector<dnn::AlgorithmDesc::Index> algo_types;
  if (ConvUseDefaultAlgorithm()) {
    // Force a fallback algorithm.
    algo_types = {CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM};
  } else {
    algo_types = {CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                  CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                  CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
                  CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD};
    if (CudnnEnvVar<FftTilingForward>::IsEnabled()) {
      algo_types.push_back(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING);
    }
    if (CudnnEnvVar<WinogradNonfused>::IsEnabled()) {
      algo_types.push_back(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
    }
  }

  // The algorithms are intentionally ordered for deterministic operation
  for (auto i : algo_types) {
    if (tensor_op_math_available) {
      out_algorithms->push_back({i, /*use_tensor_ops=*/true});
    }
    out_algorithms->push_back({i, /*use_tensor_ops=*/false});
  }

  return true;
}

absl::StatusOr<std::unique_ptr<const dnn::NormRunner>>
CudnnSupport::NormRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::NormKind kind, double epsilon,
    const dnn::TensorDescriptor& x_descriptor,
    const dnn::TensorDescriptor& scale_descriptor,
    const dnn::TensorDescriptor& y_or_dx_descriptor,
    std::optional<dnn::TensorDescriptor> bias_descriptor,
    std::optional<dnn::TensorDescriptor> dy_descriptor,
    std::optional<dnn::TensorDescriptor> expectation_descriptor,
    std::optional<dnn::TensorDescriptor> norm_factor_descriptor,
    std::optional<dnn::TensorDescriptor> dscale_descriptor,
    std::optional<dnn::TensorDescriptor> dbias_descriptor) {
#if (CUDNN_VERSION >= 8905)
  auto cudnn = cudnn_->GetHandle(parent_, stream);

  std::vector<int64_t> uids;
  auto next_uid = [&uids]() -> int64_t {
    if (uids.empty()) {
      uids.push_back(0);
    } else {
      uids.push_back(uids.back() + 1);
    }
    return uids.back();
  };

  auto create_cudnn_tensor = [next_uid](dnn::TensorDescriptor tensor_descriptor)
      -> absl::StatusOr<cudnn_frontend::Tensor> {
    return CreateCudnnTensor(tensor_descriptor.dimensions(),
                             tensor_descriptor.GetPhysicalStridesMajorToMinor(),
                             next_uid(), tensor_descriptor.type(), 1, -1);
  };

  TF_ASSIGN_OR_RETURN(auto x_tensor, create_cudnn_tensor(x_descriptor));
  TF_ASSIGN_OR_RETURN(auto scale_tensor, create_cudnn_tensor(scale_descriptor));
  TF_ASSIGN_OR_RETURN(auto y_or_dx_tensor,
                      create_cudnn_tensor(y_or_dx_descriptor));

  std::optional<cudnn_frontend::Tensor> bias_tensor, expectation_tensor,
      norm_factor_tensor, dy_tensor, dscale_tensor, dbias_tensor;
  if (kind == dnn::NormKind::LAYER_FWD_INFER ||
      kind == dnn::NormKind::LAYER_FWD_TRAIN) {
    TF_ASSIGN_OR_RETURN(bias_tensor,
                        create_cudnn_tensor(bias_descriptor.value()));
  }

  if (kind == dnn::LAYER_FWD_TRAIN) {
    TF_ASSIGN_OR_RETURN(expectation_tensor,
                        create_cudnn_tensor(expectation_descriptor.value()));
    TF_ASSIGN_OR_RETURN(norm_factor_tensor,
                        create_cudnn_tensor(norm_factor_descriptor.value()));
  }

  if (kind == dnn::LAYER_BWD) {
    TF_ASSIGN_OR_RETURN(dy_tensor, create_cudnn_tensor(dy_descriptor.value()));
    TF_ASSIGN_OR_RETURN(expectation_tensor,
                        create_cudnn_tensor(expectation_descriptor.value()));
    TF_ASSIGN_OR_RETURN(norm_factor_tensor,
                        create_cudnn_tensor(norm_factor_descriptor.value()));
    TF_ASSIGN_OR_RETURN(dscale_tensor,
                        create_cudnn_tensor(dscale_descriptor.value()));
    TF_ASSIGN_OR_RETURN(dbias_tensor,
                        create_cudnn_tensor(dbias_descriptor.value()));
  }

  std::vector<int64_t> scale_dim(4, 1), scalar_uids;
  TF_ASSIGN_OR_RETURN(
      auto epsilon_tensor,
      CreateCudnnTensor(scale_dim, scale_dim,
                        scalar_uids.emplace_back(uids.back() + 1),
                        dnn::DataType::kDouble, 1, -1, /*is_virtual=*/false,
                        CUDNN_TENSOR_REORDERING_NONE,
                        /*is_value=*/true));

  cudnnBackendNormMode_t normalizationMode = CUDNN_LAYER_NORM;

  std::optional<cudnn_frontend::Operation> norm_op;
  switch (kind) {
    case dnn::LAYER_FWD_INFER:
      norm_op = cudnn_frontend::OperationBuilder(
                    CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
                    .setNormalizationMode(normalizationMode)
                    .setNormFwdPhase(CUDNN_NORM_FWD_INFERENCE)
                    .setxDesc(x_tensor)
                    .setScaleAndBias(scale_tensor, bias_tensor.value())
                    .setEpsilonTensor(epsilon_tensor)
                    .setyDesc(y_or_dx_tensor)
                    .build();
      break;
    case dnn::LAYER_FWD_TRAIN:
      norm_op = cudnn_frontend::OperationBuilder(
                    CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
                    .setNormalizationMode(normalizationMode)
                    .setNormFwdPhase(CUDNN_NORM_FWD_TRAINING)
                    .setxDesc(x_tensor)
                    .setScaleAndBias(scale_tensor, bias_tensor.value())
                    .setEpsilonTensor(epsilon_tensor)
                    .setSavedMeanAndInvVar(expectation_tensor.value(),
                                           norm_factor_tensor.value())
                    .setyDesc(y_or_dx_tensor)
                    .build();
      break;
    case dnn::LAYER_BWD:
      norm_op =
          cudnn_frontend::OperationBuilder(
              CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
              .setNormalizationMode(normalizationMode)
              .setxDesc(x_tensor)
              .setScale(scale_tensor)
              .setSavedMeanAndInvVar(expectation_tensor.value(),
                                     norm_factor_tensor.value())
              .setDScaleAndDBias(dscale_tensor.value(), dbias_tensor.value())
              .setdyDesc(dy_tensor.value())
              .setdxDesc(y_or_dx_tensor)
              .build();
      break;
    default:
      break;
  }

  std::array<cudnn_frontend::Operation const*, 1> ops = {&norm_op.value()};
  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(cudnn.handle())
                      .setOperationGraph(ops.size(), ops.data())
                      .build();

  TF_ASSIGN_OR_RETURN(
      auto execution_plan,
      GetExecPlanFromHeuristics(std::move(op_graph), cudnn,
                                /*include_fallback_heuristics=*/true));
  std::vector<ScalingParam> scalar_input_values = {
      ScalingParam(epsilon, dnn::DataType::kDouble)};

  TF_ASSIGN_OR_RETURN(
      auto runner,
      CudnnExecutionPlanRunner<dnn::NormSignature>::Create(
          parent_, cudnn_.get(), std::move(execution_plan), uids,
          /*need_side_input=*/false, scalar_uids, scalar_input_values));
  return {std::make_unique<CudnnExecutionPlanRunner<dnn::NormSignature>>(
      std::move(runner))};

#else
  return absl::UnimplementedError(
      "Layer norm kernels require cuDNN 8.9.5 or higher.");
#endif  // CUDNN_VERSION >= 8905
}

bool CudnnSupport::GetRnnAlgorithms(
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  PreloadCudnnSubLibs(PreloadCudnnType::Rnn);

  std::vector<dnn::AlgorithmDesc::Index> algo_types = {
      // clang-format off
    CUDNN_RNN_ALGO_STANDARD,
    CUDNN_RNN_ALGO_PERSIST_STATIC,
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
      // clang-format on
  };

  out_algorithms->clear();
  for (auto i : algo_types) {
    out_algorithms->push_back({i, /*use_tensor_ops=*/false});
    out_algorithms->push_back({i, /*use_tensor_ops=*/true});
  }
  return true;
}

bool CudnnSupport::GetConvolveBackwardDataAlgorithms(
    CudaComputeCapability cuda_compute_capability, dnn::DataType input_type,
    const NumericOptions& numeric_options,
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  PreloadCudnnSubLibs(PreloadCudnnType::ConvBwdData);

  bool tensor_op_math_available = IsTensorMathEnabled(
      cuda_compute_capability, input_type, numeric_options.allow_tf32);
  out_algorithms->clear();

  std::vector<dnn::AlgorithmDesc::Index> algo_types = {
      // clang-format off
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
      // clang-format on
  };
  if (CudnnEnvVar<WinogradNonfused>::IsEnabled()) {
    algo_types.push_back(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);
  }
  if (numeric_options.require_determinism) {
    algo_types.push_back(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0);
  }

  // The algorithms are intentionally ordered for deterministic operation
  for (auto i : algo_types) {
    if (tensor_op_math_available) {
      out_algorithms->push_back({i, /*use_tensor_ops=*/true});
    }
    out_algorithms->push_back({i, /*use_tensor_ops=*/false});
  }

  return true;
}

bool CudnnSupport::GetConvolveBackwardFilterAlgorithms(
    CudaComputeCapability cuda_compute_capability, dnn::DataType input_type,
    const NumericOptions& numeric_options,
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  PreloadCudnnSubLibs(PreloadCudnnType::ConvBwdFilter);

  bool tensor_op_math_available = IsTensorMathEnabled(
      cuda_compute_capability, input_type, numeric_options.allow_tf32);
  out_algorithms->clear();

  std::vector<dnn::AlgorithmDesc::Index> algo_types = {
      // clang-format off
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
      // Based on cudnn.h, the following is not implemented.
      // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,

      // Produces incorrect results for some shapes. Disabled for now, see
      // NVIDIA bug 2072856. TODO(csigg): Only disable for subset of shapes.
      // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
      // clang-format on
  };
  if (CudnnEnvVar<WinogradNonfused>::IsEnabled()) {
    algo_types.push_back(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED);
  }
  if (!numeric_options.require_determinism) {
    algo_types.push_back(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0);
    algo_types.push_back(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3);
  }

  // The algorithms are intentionally ordered for deterministic operation
  for (auto i : algo_types) {
    if (tensor_op_math_available) {
      out_algorithms->push_back({i, /*use_tensor_ops=*/true});
    }
    out_algorithms->push_back({i, /*use_tensor_ops=*/false});
  }

  return true;
}

bool CudnnSupport::DoBatchNormalizationForward(
    Stream* stream, const DeviceMemory<float>& x,
    const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
    const DeviceMemory<float>& estimated_mean,
    const DeviceMemory<float>& estimated_variance,
    const DeviceMemory<float>& side_input, const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    const double exponential_average_factor,
    dnn::ActivationMode activation_mode, DeviceMemory<float>* y,
    DeviceMemory<float>* batch_mean, DeviceMemory<float>* batch_var,
    DeviceMemory<float>* saved_mean, DeviceMemory<float>* saved_inv_var,
    bool is_training, ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator) {
  return IsStatusOk(
      DoBatchNormalizationForwardImpl<float, float>(
          stream, dnn::DataType::kFloat, dnn::DataType::kFloat, x, scale,
          offset, estimated_mean, estimated_variance, side_input, x_desc,
          scale_offset_desc, epsilon, exponential_average_factor,
          activation_mode, y, batch_mean, batch_var, saved_mean, saved_inv_var,
          is_training, reserve_space_allocator, workspace_allocator),
      /*report_error=*/true);
}

bool CudnnSupport::DoBatchNormalizationForward(
    Stream* stream, const DeviceMemory<Eigen::half>& x,
    const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
    const DeviceMemory<float>& estimated_mean,
    const DeviceMemory<float>& estimated_variance,
    const DeviceMemory<Eigen::half>& side_input,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    const double exponential_average_factor,
    dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half>* y,
    DeviceMemory<float>* batch_mean, DeviceMemory<float>* batch_var,
    DeviceMemory<float>* saved_mean, DeviceMemory<float>* saved_inv_var,
    bool is_training, ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator) {
  return IsStatusOk(
      DoBatchNormalizationForwardImpl<Eigen::half, float>(
          stream, dnn::DataType::kHalf, dnn::DataType::kFloat, x, scale, offset,
          estimated_mean, estimated_variance, side_input, x_desc,
          scale_offset_desc, epsilon, exponential_average_factor,
          activation_mode, y, batch_mean, batch_var, saved_mean, saved_inv_var,
          is_training, reserve_space_allocator, workspace_allocator),
      /*report_error=*/true);
}

bool CudnnSupport::DoBatchNormalizationForward(
    Stream* stream, const DeviceMemory<Eigen::bfloat16>& x,
    const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
    const DeviceMemory<float>& estimated_mean,
    const DeviceMemory<float>& estimated_variance,
    const DeviceMemory<Eigen::bfloat16>& side_input,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    const double exponential_average_factor,
    dnn::ActivationMode activation_mode, DeviceMemory<Eigen::bfloat16>* y,
    DeviceMemory<float>* batch_mean, DeviceMemory<float>* batch_var,
    DeviceMemory<float>* saved_mean, DeviceMemory<float>* saved_inv_var,
    bool is_training, ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator) {
  return IsStatusOk(
      DoBatchNormalizationForwardImpl<Eigen::bfloat16, float>(
          stream, dnn::DataType::kBF16, dnn::DataType::kFloat, x, scale, offset,
          estimated_mean, estimated_variance, side_input, x_desc,
          scale_offset_desc, epsilon, exponential_average_factor,
          activation_mode, y, batch_mean, batch_var, saved_mean, saved_inv_var,
          is_training, reserve_space_allocator, workspace_allocator),
      /*report_error=*/true);
}

template <class T, class U>
absl::Status CudnnSupport::DoBatchNormalizationForwardImpl(
    Stream* stream, dnn::DataType input_data_type,
    dnn::DataType scale_data_type, const DeviceMemory<T>& x,
    const DeviceMemory<U>& scale, const DeviceMemory<U>& offset,
    const DeviceMemory<U>& estimated_mean,
    const DeviceMemory<U>& estimated_variance,
    const DeviceMemory<T>& side_input, const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    const double exponential_average_factor,
    dnn::ActivationMode activation_mode, DeviceMemory<T>* y,
    DeviceMemory<U>* batch_mean, DeviceMemory<U>* batch_var,
    DeviceMemory<U>* saved_mean, DeviceMemory<U>* saved_inv_var,
    bool is_training, ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator) {
  CudnnTensorDescriptor x_descriptor(x_desc, ToCudnnDataType(input_data_type));
  CudnnTensorDescriptor scale_offset_descriptor(
      scale_offset_desc, ToCudnnDataType(scale_data_type));
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
  if (BatchnormSpatialPersistentEnabled() && is_training) {
    mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  }
  float one = 1.0;
  float zero = 0.0;
  auto cudnn = cudnn_->GetHandle(parent_, stream);

  DeviceMemory<uint8_t> workspace;
  DeviceMemory<uint8_t> reserve_space;

  const auto get_bn_ops = [&]() -> cudnnBatchNormOps_t {
    if (side_input.is_null()) {
      return activation_mode == dnn::ActivationMode::kNone
                 ? CUDNN_BATCHNORM_OPS_BN
                 : CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    } else {
      return CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    }
  };
  const cudnnBatchNormOps_t bn_ops = get_bn_ops();

  // We use Nan propagation to be consistent with
  // CudnnSupport::DoActivate(...).
  CudnnActivationDescriptor activation_desc(
      activation_mode, CUDNN_PROPAGATE_NAN, x_desc.value_max());

  if (reserve_space_allocator != nullptr && workspace_allocator != nullptr) {
    TF_ASSIGN_OR_RETURN(
        workspace,
        CreateBatchNormForwardWorkspace(
            stream, cudnn, mode, bn_ops, activation_desc.handle(), x_descriptor,
            scale_offset_descriptor, workspace_allocator));
    if (is_training) {
      size_t reserve_space_size_in_bytes = 0;
      RETURN_IF_CUDNN_ERROR(
          cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
              /*handle=*/cudnn.handle(), /*mode=*/mode, /*bnOps=*/bn_ops,
              /*activationDesc=*/activation_desc.handle(),
              /*xDesc=*/x_descriptor.handle(),
              /*sizeInBytes=*/&reserve_space_size_in_bytes));
      TF_ASSIGN_OR_RETURN(reserve_space, reserve_space_allocator->AllocateBytes(
                                             reserve_space_size_in_bytes));
    }
  }

  auto check_no_side_input_or_activation = [&]() -> absl::Status {
    if (activation_mode != dnn::ActivationMode::kNone ||
        !side_input.is_null()) {
      return absl::InternalError(
          absl::StrCat("Side input and activation are not "
                       "supported by cuDNN version: ",
                       CUDNN_VERSION));
    } else {
      return absl::OkStatus();
    }
  };

  if (is_training) {
    CHECK_EQ(batch_mean->is_null(), batch_var->is_null())
        << "batch_mean and batch_var must both be null or both be non-null";

    void* batch_mean_opaque;
    void* batch_var_opaque;
    if (!batch_mean->is_null() && !batch_var->is_null()) {
      if (exponential_average_factor == 1.0) {
        TF_RETURN_IF_ERROR(stream->MemZero(batch_mean, batch_mean->size()));
        TF_RETURN_IF_ERROR(stream->MemZero(batch_var, batch_var->size()));
      }
      batch_mean_opaque = batch_mean->opaque();
      batch_var_opaque = batch_var->opaque();
    } else {
      batch_mean_opaque = nullptr;
      batch_var_opaque = nullptr;
    }

    bool called = false;
    if (reserve_space_allocator != nullptr && workspace_allocator != nullptr) {
      called = true;
      RETURN_IF_CUDNN_ERROR(cudnnBatchNormalizationForwardTrainingEx(
          /*handle=*/cudnn.handle(),
          /*mode=*/mode,
          /*bnOps=*/bn_ops,
          /*alpha=*/&one,
          /*beta=*/&zero,
          /*xDesc=*/x_descriptor.handle(),
          /*xData=*/x.opaque(),
          /*zDesc=*/x_descriptor.handle(),
          /*zData=*/side_input.opaque(),
          /*yDesc=*/x_descriptor.handle(),
          /*yData=*/y->opaque(),
          /*bnScaleBiasMeanVarDesc=*/scale_offset_descriptor.handle(),
          /*bnScale=*/scale.opaque(),
          /*bnBias=*/offset.opaque(),
          /*exponentialAverageFactor=*/exponential_average_factor,
          /*resultRunningMean=*/batch_mean_opaque,
          /*resultRunningVariance=*/batch_var_opaque,
          /*epsilon=*/epsilon,
          /*resultSaveMean=*/saved_mean->opaque(),
          /*resultSaveInvVariance=*/saved_inv_var->opaque(),
          /*activationDesc=*/activation_desc.handle(),
          /*workspace=*/workspace.opaque(),
          /*workSpaceSizeInBytes=*/workspace.size(),
          /*reserveSpace=*/reserve_space.opaque(),
          /*reserveSpaceSizeInBytes=*/reserve_space.size()));
    }
    if (!called) {
      TF_RETURN_IF_ERROR(check_no_side_input_or_activation());
      RETURN_IF_CUDNN_ERROR(cudnnBatchNormalizationForwardTraining(
          cudnn.handle(), mode, &one, &zero, x_descriptor.handle(), x.opaque(),
          x_descriptor.handle(), y->opaque(), scale_offset_descriptor.handle(),
          scale.opaque(), offset.opaque(), exponential_average_factor,
          batch_mean_opaque, batch_var_opaque, epsilon, saved_mean->opaque(),
          saved_inv_var->opaque()));
    }
  } else {
    const void* maybe_inv_var = estimated_variance.opaque();
    TF_RETURN_IF_ERROR(check_no_side_input_or_activation());
    RETURN_IF_CUDNN_ERROR(cudnnBatchNormalizationForwardInference(
        cudnn.handle(), mode, &one, &zero, x_descriptor.handle(), x.opaque(),
        x_descriptor.handle(), y->opaque(), scale_offset_descriptor.handle(),
        scale.opaque(), offset.opaque(), estimated_mean.opaque(), maybe_inv_var,
        epsilon));
  }
  return absl::OkStatus();
}

bool CudnnSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<float>& y_backprop,
    const DeviceMemory<float>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
    const DeviceMemory<float>& inv_var, const DeviceMemory<float>& y,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<float>* x_backprop,
    DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
    DeviceMemory<float>* side_input_backprop,
    DeviceMemory<uint8_t>* reserve_space_data,
    ScratchAllocator* workspace_allocator) {
  return IsStatusOk(
      DoBatchNormalizationBackwardImpl(
          stream, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, y_backprop, x, scale,
          offset, mean, inv_var, y, x_desc, scale_offset_desc, epsilon,
          activation_mode, x_backprop, scale_backprop, offset_backprop,
          side_input_backprop, reserve_space_data, workspace_allocator),
      /*report_error=*/true);
}

bool CudnnSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<Eigen::half>& y_backprop,
    const DeviceMemory<Eigen::half>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
    const DeviceMemory<float>& inv_var, const DeviceMemory<Eigen::half>& y,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half>* x_backprop,
    DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
    DeviceMemory<Eigen::half>* side_input_backprop,
    DeviceMemory<uint8_t>* reserve_space_data,
    ScratchAllocator* workspace_allocator) {
  return IsStatusOk(
      DoBatchNormalizationBackwardImpl(
          stream, CUDNN_DATA_HALF, CUDNN_DATA_FLOAT, y_backprop, x, scale,
          offset, mean, inv_var, y, x_desc, scale_offset_desc, epsilon,
          activation_mode, x_backprop, scale_backprop, offset_backprop,
          side_input_backprop, reserve_space_data, workspace_allocator),
      /*report_error=*/true);
}

bool CudnnSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<Eigen::bfloat16>& y_backprop,
    const DeviceMemory<Eigen::bfloat16>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
    const DeviceMemory<float>& inv_var, const DeviceMemory<Eigen::bfloat16>& y,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    dnn::ActivationMode activation_mode,
    DeviceMemory<Eigen::bfloat16>* x_backprop,
    DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
    DeviceMemory<Eigen::bfloat16>* side_input_backprop,
    DeviceMemory<uint8_t>* reserve_space_data,
    ScratchAllocator* workspace_allocator) {
  return IsStatusOk(
      DoBatchNormalizationBackwardImpl(
          stream, CUDNN_DATA_BFLOAT16, CUDNN_DATA_FLOAT, y_backprop, x, scale,
          offset, mean, inv_var, y, x_desc, scale_offset_desc, epsilon,
          activation_mode, x_backprop, scale_backprop, offset_backprop,
          side_input_backprop, reserve_space_data, workspace_allocator),
      /*report_error=*/true);
}

template <class T, class U>
absl::Status CudnnSupport::DoBatchNormalizationBackwardImpl(
    Stream* stream, int cudnn_input_type, int cudnn_scale_type,
    const DeviceMemory<T>& y_backprop, const DeviceMemory<T>& x,
    const DeviceMemory<U>& scale, const DeviceMemory<U>& offset,
    const DeviceMemory<U>& mean, const DeviceMemory<U>& inv_var,
    const DeviceMemory<T>& y, const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<T>* x_backprop,
    DeviceMemory<U>* scale_backprop, DeviceMemory<U>* offset_backprop,
    DeviceMemory<T>* side_input_backprop,
    DeviceMemory<uint8_t>* reserve_space_data,
    ScratchAllocator* workspace_allocator) {
  CudnnTensorDescriptor x_descriptor(
      x_desc, static_cast<cudnnDataType_t>(cudnn_input_type));
  CudnnTensorDescriptor scale_offset_descriptor(
      scale_offset_desc, static_cast<cudnnDataType_t>(cudnn_scale_type));
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
  if (BatchnormSpatialPersistentEnabled()) {
    mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  }
  float one = 1.0;
  float zero = 0.0;

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  bool called = false;
  if (reserve_space_data != nullptr && workspace_allocator != nullptr) {
    called = true;
    const cudnnBatchNormOps_t bn_ops = [&]() {
      if (side_input_backprop->is_null()) {
        return activation_mode == dnn::ActivationMode::kNone
                   ? CUDNN_BATCHNORM_OPS_BN
                   : CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
      } else {
        return CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
      }
    }();

    // We use Nan propagation to be consistent with
    // CudnnSupport::DoActivate(...).
    CudnnActivationDescriptor activation_desc(
        activation_mode, CUDNN_PROPAGATE_NAN, x_desc.value_max());

    TF_ASSIGN_OR_RETURN(
        DeviceMemory<uint8_t> workspace,
        CreateBatchNormBackwardWorkspace(
            stream, cudnn, mode, bn_ops, activation_desc.handle(), x_descriptor,
            scale_offset_descriptor, workspace_allocator));
    RETURN_IF_CUDNN_ERROR(cudnnBatchNormalizationBackwardEx(
        /*handle=*/cudnn.handle(),
        /*mode=*/mode,
        /*bnOps=*/bn_ops,
        /*alphaDataDiff=*/&one,
        /*betaDataDiff=*/&zero,
        /*alphaParamDiff=*/&one,
        /*betaParamDiff=*/&zero,
        /*xDesc=*/x_descriptor.handle(),
        /*xData=*/x.opaque(),
        /*yDesc=*/x_descriptor.handle(),
        /*yData=*/y.opaque(),
        /*dyDesc=*/x_descriptor.handle(),
        /*dyData=*/y_backprop.opaque(),
        /*dzDesc=*/x_descriptor.handle(),
        /*dzData=*/side_input_backprop->opaque(),
        /*dxDesc=*/x_descriptor.handle(),
        /*dxData=*/x_backprop->opaque(),
        /*dBnScaleBiasDesc=*/scale_offset_descriptor.handle(),
        /*bnScaleData=*/scale.opaque(),
        /*bnBiasData=*/offset.opaque(),
        /*dBnScaleData=*/scale_backprop->opaque(),
        /*dBnBiasData=*/offset_backprop->opaque(),
        /*epsilon=*/epsilon,
        /*savedMean=*/mean.opaque(),
        /*savedInvVariance=*/inv_var.opaque(),
        /*activationDesc=*/activation_desc.handle(),
        /*workspace=*/workspace.opaque(),
        /*workSpaceSizeInBytes=*/workspace.size(),
        /*reserveSpace=*/reserve_space_data->opaque(),
        /*reserveSpaceSizeInBytes=*/reserve_space_data->size()));
  }
  auto check_no_side_input_or_activation = [&]() -> absl::Status {
    if (activation_mode != dnn::ActivationMode::kNone ||
        !side_input_backprop->is_null()) {
      return tsl::errors::Internal(
          "Side input and activation are not supported by cuDNN version: ",
          CUDNN_VERSION);
    } else {
      return absl::OkStatus();
    }
  };

  if (!called && check_no_side_input_or_activation().ok()) {
    RETURN_IF_CUDNN_ERROR(cudnnBatchNormalizationBackward(
        cudnn.handle(), mode, &one, &zero, &one, &zero, x_descriptor.handle(),
        x.opaque(), x_descriptor.handle(), y_backprop.opaque(),
        x_descriptor.handle(), x_backprop->opaque(),
        scale_offset_descriptor.handle(), scale.opaque(),
        scale_backprop->opaque(), offset_backprop->opaque(), epsilon,
        mean.opaque(), inv_var.opaque()));
  }

  return absl::OkStatus();
}

absl::Status CudnnSupport::DoFusedConvolve(
    Stream* stream, dnn::DataType input_type, dnn::DataType side_input_type,
    dnn::DataType bias_type, dnn::DataType output_type,
    const dnn::BatchDescriptor& conv_input_descriptor,
    DeviceMemoryBase conv_input_data, double conv_scale,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    DeviceMemoryBase side_input_data, double side_input_scale,
    const dnn::BatchDescriptor& bias_descriptor, DeviceMemoryBase biases,
    dnn::ActivationMode activation_mode,
    const dnn::BatchDescriptor& output_descriptor, DeviceMemoryBase output_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  if (input_type == dnn::DataType::kInt8 &&
      !stream->GetCudaComputeCapability().IsAtLeast(6, 1)) {
    return tsl::errors::Unimplemented(
        "cudnnConvolutionBiasActivationForward() for int8 is only "
        "supported "
        "on GPUs with compute capability 6.1 or later.");
  }

  if (activation_mode != dnn::ActivationMode::kRelu &&
      activation_mode != dnn::ActivationMode::kNone) {
    return absl::InvalidArgumentError(
        "cudnnConvolutionBiasActivationForward() only supports "
        "Relu or None activation.");
  }

  CudnnTensorDescriptor conv_input_nd(
      conv_input_descriptor,
      ToCudnnDataType(input_type, conv_input_descriptor.layout()));
  CudnnTensorDescriptor output_nd(
      output_descriptor,
      ToCudnnDataType(output_type, conv_input_descriptor.layout()));
  CudnnFilterDescriptor filter(
      filter_descriptor,
      ToCudnnDataType(input_type, filter_descriptor.layout()));
  CudnnTensorDescriptor bias_nd(bias_descriptor, ToCudnnDataType(bias_type));

  DeviceMemory<uint8_t> scratch;
  dnn::AlgorithmDesc algo_desc;
  {
    auto cudnn = cudnn_->GetHandle(parent_, stream);
    TF_ASSIGN_OR_RETURN(
        algo_desc,
        GetCudnnConvolutionForwardAlgorithm(
            stream, cudnn, algorithm_config, conv_input_nd, filter, input_type,
            convolution_descriptor, output_nd, scratch_allocator, &scratch));
  }  // Explicitly release cuDNN handle.

  CudnnConvolutionDescriptor conv(
      convolution_descriptor,
      ToCudnnDataType(GetConvAccumulatorType(input_type)));
  conv.set_use_tensor_op_math(algo_desc.tensor_ops_enabled());

  // CUDNN v6 only supports CUDNN_NOT_PROPAGATE_NAN as the reluNanOpt for
  // activation descriptor. Note that this will change the nan propagation
  // behavior from separate conv, bias, and relu (which by default is
  // CUDNN_PROPAGATE_NAN).
  CudnnActivationDescriptor activation_desc(
      activation_mode, CUDNN_NOT_PROPAGATE_NAN, output_descriptor.value_max());

  TF_ASSIGN_OR_RETURN(
      auto runner,
      CudnnLegacyFusedConvRunner::Create(
          parent_, stream, cudnn_.get(), std::move(algo_desc), input_type,
          conv_scale, side_input_scale, std::move(conv_input_nd),
          std::move(output_nd), std::move(filter), std::move(bias_nd),
          std::move(conv), std::move(activation_desc)));

  return runner(stream, output_profile_result, scratch, conv_input_data,
                filter_data, side_input_data, biases, output_data);
}

absl::Status CudnnSupport::CudnnReorderConvolutionFilterAndBias(
    Stream* stream, const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<int8_t>& filter_input,
    DeviceMemory<int8_t>* filter_output,
    std::optional<const DeviceMemory<float>> bias_input,
    std::optional<DeviceMemory<float>> bias_output) {
  bool has_bias = bias_input.has_value();
  CHECK(!has_bias || bias_output.has_value());

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  CudnnFilterDescriptor filter_nd(filter_descriptor, CUDNN_DATA_INT8x32);

  cudnnStatus_t status = cudnnReorderFilterAndBias(
      /*handle=*/cudnn.handle(), /*filterDesc=*/filter_nd.handle(),
      /*reorderType=*/CUDNN_DEFAULT_REORDER,
      /*filterData=*/filter_input.opaque(),
      /*reorderedFilterData=*/filter_output->opaque(),
      /*reorderBias=*/has_bias ? 1 : 0,
      /*biasData=*/has_bias ? bias_input->opaque() : nullptr,
      /*reorderedBiasData=*/has_bias ? bias_output->opaque() : nullptr);
  RETURN_IF_CUDNN_ERROR(status);

  return absl::OkStatus();
}

absl::Status CudnnSupport::DoPrepareForCtcLoss(
    Stream* stream, dnn::DataType element_type,
    const dnn::RnnStateTensorDescriptor& probs_desc,
    const dnn::RnnStateTensorDescriptor& grads_desc,
    absl::Span<const int> labels_data,
    absl::Span<const int> labels_lengths_data,
    absl::Span<const int> input_lengths_data,
    const NumericOptions& numeric_options, ScratchAllocator* scratch_allocator,
    DeviceMemory<uint8_t>* scratch_memory, int* ctc_loss_algo_id) {
  auto cudnn = cudnn_->GetHandle(parent_, stream);
  // Query the workspace size.
  size_t workspace_size_in_bytes = 0;
  CudnnCtcLossDescriptor cudnn_ctc_loss_desc(ToCudnnDataType(element_type));
  const CudnnRnnStateTensorDescriptor& cudnn_probs_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(probs_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_grads_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(grads_desc);

  // Try running with `algo`, if successful then pick it. The
  // non-deterministic algorithm is first and thus preferentially picked
  // when determinism is not required.
  auto algo = numeric_options.require_determinism
                  ? CUDNN_CTC_LOSS_ALGO_DETERMINISTIC
                  : CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC;
  cudnnStatus_t status = cudnnGetCTCLossWorkspaceSize(
      /*handle=*/cudnn.handle(), /*probsDesc=*/cudnn_probs_desc.handle(),
      /*gradientsDesc=*/cudnn_grads_desc.handle(),
      /*labels=*/labels_data.data(),
      /*labelLengths=*/labels_lengths_data.data(),
      /*inputLengths=*/input_lengths_data.data(),
      /*algo=*/algo,
      /*ctcLossDesc=*/cudnn_ctc_loss_desc.handle(),
      /*sizeInBytes=*/&workspace_size_in_bytes);
  if (numeric_options.require_determinism) {
    RETURN_IF_CUDNN_ERROR(status);
  }

  if (status != CUDNN_STATUS_SUCCESS) {
    algo = CUDNN_CTC_LOSS_ALGO_DETERMINISTIC;
    RETURN_IF_CUDNN_ERROR(cudnnGetCTCLossWorkspaceSize(
        /*handle=*/cudnn.handle(), /*probsDesc=*/cudnn_probs_desc.handle(),
        /*gradientsDesc=*/cudnn_grads_desc.handle(),
        /*labels=*/labels_data.data(),
        /*labelLengths=*/labels_lengths_data.data(),
        /*inputLengths=*/input_lengths_data.data(),
        /*algo=*/algo,
        /*ctcLossDesc=*/cudnn_ctc_loss_desc.handle(),
        /*sizeInBytes=*/&workspace_size_in_bytes));
  }
  *ctc_loss_algo_id = algo;
  // Allocate the workspace.
  if (workspace_size_in_bytes == 0) {
    *scratch_memory = DeviceMemory<uint8_t>();
    return absl::OkStatus();
  }
  const auto scratch_or =
      scratch_allocator->AllocateBytes(workspace_size_in_bytes);
  if (scratch_or.ok()) {
    *scratch_memory = scratch_or.value();
    return absl::OkStatus();
  }
  return tsl::errors::Internal(
      "Failed to allocate scratch memory for the CuDNN CTC Loss");
}

absl::Status CudnnSupport::DoCtcLoss(
    Stream* stream, dnn::DataType element_type,
    const dnn::RnnStateTensorDescriptor& probs_desc,
    const DeviceMemoryBase probs_data, absl::Span<const int> labels_data,
    absl::Span<const int> labels_lengths_data,
    absl::Span<const int> input_lengths_data, DeviceMemoryBase costs_data,
    const dnn::RnnStateTensorDescriptor& grads_desc,
    DeviceMemoryBase grads_data, DeviceMemory<uint8_t> scratch_memory,
    int ctc_loss_algo_id) {
  // Current cuDNN CTC Loss only supports the float datatype
  if (element_type != dnn::DataType::kFloat) {
    return absl::InvalidArgumentError(
        "CudnnCtcLossDescriptor is supported only when the "
        "DataType is float");
  }
  CudnnCtcLossDescriptor cudnn_ctc_loss_desc(ToCudnnDataType(element_type));
  const CudnnRnnStateTensorDescriptor& cudnn_probs_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(probs_desc);
  const CudnnRnnStateTensorDescriptor& cudnn_grads_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(grads_desc);
  return DoCtcLossImpl(stream, cudnn_probs_desc, probs_data, labels_data,
                       labels_lengths_data, input_lengths_data, costs_data,
                       cudnn_grads_desc, grads_data, cudnn_ctc_loss_desc,
                       scratch_memory, ctc_loss_algo_id);
}

bool CudnnSupport::DoTransformTensor(Stream* stream,
                                     const dnn::BatchDescriptor& input_desc,
                                     dnn::DataType input_type,
                                     const DeviceMemoryBase& input_data,
                                     const dnn::BatchDescriptor& output_desc,
                                     dnn::DataType output_type, float scale,
                                     DeviceMemoryBase* output_data) {
  float beta = 0.0f;
  CudnnTensorDescriptor input_tensor_desc(
      input_desc, ToCudnnDataType(input_type, input_desc.layout()));
  CudnnTensorDescriptor output_tensor_desc(
      output_desc, ToCudnnDataType(output_type, output_desc.layout()));
  auto cudnn = cudnn_->GetHandle(parent_, stream);
  const auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnTransformTensor(
        cudnn.handle(), &scale, input_tensor_desc.handle(), input_data.opaque(),
        &beta, output_tensor_desc.handle(), output_data->opaque()));
    return absl::OkStatus();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

namespace {

// Cudnn legacy API only supports int32 indexing and can handle a maximum of
// 2^31-1 elements. For pooling operations, we split the big tensor along
// the batch axis into multiple small tensors when possible and then call
// cudnn API sequentially.
struct PoolingSplitsSpec {
  int64_t num_batches;
  int64_t input_offset_in_bytes;
  int64_t output_offset_in_bytes;
};

absl::StatusOr<std::vector<PoolingSplitsSpec>> GetTensorSplits(
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::BatchDescriptor& output_descriptor, dnn::DataType element_type) {
  std::vector<PoolingSplitsSpec> out;
  if (element_type == dnn::DataType::kInt8) {
    out.push_back({input_descriptor.count(), 0, 0});
    return out;
  }

  cudnnDataType_t cudnn_input_type =
      ToCudnnDataType(element_type, input_descriptor.layout());
  cudnnDataType_t cudnn_output_type =
      ToCudnnDataType(element_type, output_descriptor.layout());

  std::vector<int64_t> dims64 =
      input_descriptor.full_dims(dnn::DataLayout::kBatchDepthYX);

  int64_t num_batches = input_descriptor.count();
  int64_t elements_per_batch_input = input_descriptor.NodesAcrossFeatureMaps();
  int64_t elements_per_batch_output =
      output_descriptor.NodesAcrossFeatureMaps();

  int64_t max_batches_per_split =
      std::numeric_limits<int>::max() / elements_per_batch_input;

  if (max_batches_per_split == 0) {
    return absl::InternalError(absl::StrCat(
        "Tensor has too many elements for int32 indexing: batches=",
        num_batches, " elements_per_batch=", elements_per_batch_input, "."));
  }

  int64_t processed_batches = 0;
  while (processed_batches < num_batches) {
    int64_t num_batches_per_split =
        std::min(max_batches_per_split, num_batches - processed_batches);
    int64_t offset_input = processed_batches * elements_per_batch_input *
                           CudnnDataTypeToByteSize(cudnn_input_type);
    int64_t offset_output = processed_batches * elements_per_batch_output *
                            CudnnDataTypeToByteSize(cudnn_output_type);
    out.push_back({num_batches_per_split, offset_input, offset_output});
    processed_batches += num_batches_per_split;
  }
  return out;
}
}  // namespace

absl::Status CudnnSupport::DoPoolForward(
    dnn::DataType element_type, Stream* stream,
    const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
    const dnn::BatchDescriptor& output_dimensions, DeviceMemoryBase output_data,
    ScratchAllocator* workspace_allocator) {
  return DoPoolForward(element_type, stream, pooling_dimensions,
                       NumericOptions{}, input_dimensions, input_data,
                       output_dimensions, output_data, workspace_allocator);
}

absl::Status CudnnSupport::DoPoolForward(
    dnn::DataType element_type, Stream* stream,
    const dnn::PoolingDescriptor& pooling_dimensions,
    const NumericOptions& numeric_options,
    const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
    const dnn::BatchDescriptor& output_dimensions, DeviceMemoryBase output_data,
    ScratchAllocator* workspace_allocator) {
  // Alpha is the scaling factor for input.
  const float alpha_f = 1.0f;
  const double alpha_d = 1.0;
  const void* alpha = element_type == dnn::DataType::kDouble
                          ? static_cast<const void*>(&alpha_d)
                          : static_cast<const void*>(&alpha_f);
  // Beta is the scaling factor for output.
  const float beta_f = 0.0f;
  const double beta_d = 0.0;
  const void* beta = element_type == dnn::DataType::kDouble
                         ? static_cast<const void*>(&beta_d)
                         : static_cast<const void*>(&beta_f);

  cudnnDataType_t cudnn_input_type =
      ToCudnnDataType(element_type, input_dimensions.layout());
  cudnnDataType_t cudnn_output_type =
      ToCudnnDataType(element_type, output_dimensions.layout());
  CudnnPoolingDescriptor pooling_desc(pooling_dimensions, numeric_options);
  auto cudnn = cudnn_->GetHandle(parent_, stream);

  auto cudnn_launcher = [&](CudnnTensorDescriptor& src_desc,
                            CudnnTensorDescriptor& dest_desc,
                            const void* input_ptr, void* output_ptr) {
    RETURN_IF_CUDNN_ERROR(cudnnPoolingForward(
        cudnn.handle(), pooling_desc.handle(), alpha, src_desc.handle(),
        input_ptr, beta, dest_desc.handle(), output_ptr));
    return absl::OkStatus();
  };

  auto splits_or =
      GetTensorSplits(input_dimensions, output_dimensions, element_type);
  if (!splits_or.ok()) {
    return absl::InternalError("Cudnn pooling failed to split");
  }
  auto splits = std::move(splits_or.value());

  dnn::BatchDescriptor input_split = input_dimensions;
  dnn::BatchDescriptor output_split = output_dimensions;
  for (int i = 0; i < splits.size(); i++) {
    // It is safe to cap the batch dimension, since it is the leading
    // dimension and will have no effect on the computation of strides in
    // both kBatchYXDepth and kBatchDepthYX formats.
    input_split.set_count(splits[i].num_batches);
    output_split.set_count(splits[i].num_batches);
    CudnnTensorDescriptor src_desc(input_split, cudnn_input_type);
    CudnnTensorDescriptor dest_desc(output_split, cudnn_output_type);

    void* input_data_ptr = static_cast<char*>(input_data.opaque()) +
                           splits[i].input_offset_in_bytes;
    void* output_data_ptr = static_cast<char*>(output_data.opaque()) +
                            splits[i].output_offset_in_bytes;
    const auto status =
        cudnn_launcher(src_desc, dest_desc, input_data_ptr, output_data_ptr);
    if (!IsStatusOk(status, /*report_error=*/true)) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status CudnnSupport::DoPoolBackward(
    dnn::DataType element_type, Stream* stream,
    const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
    const dnn::BatchDescriptor& output_dimensions, DeviceMemoryBase output_data,
    DeviceMemoryBase input_diff_data, DeviceMemoryBase output_diff_data,
    ScratchAllocator* workspace_allocator) {
  return DoPoolBackward(element_type, stream, pooling_dimensions,
                        NumericOptions{}, input_dimensions, input_data,
                        output_dimensions, output_data, input_diff_data,
                        output_diff_data, workspace_allocator);
}

absl::Status CudnnSupport::DoPoolBackward(
    dnn::DataType element_type, Stream* stream,
    const dnn::PoolingDescriptor& pooling_dimensions,
    const NumericOptions& numeric_options,
    const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
    const dnn::BatchDescriptor& output_dimensions, DeviceMemoryBase output_data,
    DeviceMemoryBase input_diff_data, DeviceMemoryBase output_diff_data,
    ScratchAllocator* workspace_allocator) {
  // Alpha is the scaling factor for input.
  const float alpha_f = 1.0f;
  const double alpha_d = 1.0;
  const void* alpha = element_type == dnn::DataType::kDouble
                          ? static_cast<const void*>(&alpha_d)
                          : static_cast<const void*>(&alpha_f);
  // Beta is the scaling factor for output.
  const float beta_f = 0.0f;
  const double beta_d = 0.0;
  const void* beta = element_type == dnn::DataType::kDouble
                         ? static_cast<const void*>(&beta_d)
                         : static_cast<const void*>(&beta_f);

  cudnnDataType_t cudnn_input_type =
      ToCudnnDataType(element_type, input_dimensions.layout());
  cudnnDataType_t cudnn_output_type =
      ToCudnnDataType(element_type, output_dimensions.layout());
  CudnnPoolingDescriptor pooling_desc(pooling_dimensions, numeric_options);
  auto cudnn = cudnn_->GetHandle(parent_, stream);

  auto cudnn_launcher = [&](CudnnTensorDescriptor& src_desc,
                            CudnnTensorDescriptor& dest_desc,
                            const void* output_ptr, const void* input_diff_ptr,
                            const void* input_ptr, void* output_diff_ptr) {
    RETURN_IF_CUDNN_ERROR(cudnnPoolingBackward(
        cudnn.handle(), pooling_desc.handle(), alpha, dest_desc.handle(),
        output_ptr, dest_desc.handle(), input_diff_ptr, src_desc.handle(),
        input_ptr, beta, src_desc.handle(), output_diff_ptr));
    return absl::OkStatus();
  };

  auto splits_or =
      GetTensorSplits(input_dimensions, output_dimensions, element_type);
  if (!splits_or.ok()) {
    return absl::InternalError("Cudnn pooling failed to split");
  }
  auto splits = std::move(splits_or.value());

  dnn::BatchDescriptor input_split = input_dimensions;
  dnn::BatchDescriptor output_split = output_dimensions;
  for (int i = 0; i < splits.size(); i++) {
    // It is safe to cap the batch dimension, since it is the leading
    // dimension and will have no effect on the computation of strides in
    // both kBatchYXDepth and kBatchDepthYX formats.
    input_split.set_count(splits[i].num_batches);
    output_split.set_count(splits[i].num_batches);
    CudnnTensorDescriptor src_desc(input_split, cudnn_input_type);
    CudnnTensorDescriptor dest_desc(output_split, cudnn_output_type);

    void* output_data_ptr = static_cast<char*>(output_data.opaque()) +
                            splits[i].output_offset_in_bytes;
    void* input_diff_data_ptr = static_cast<char*>(input_diff_data.opaque()) +
                                splits[i].output_offset_in_bytes;
    void* input_data_ptr = static_cast<char*>(input_data.opaque()) +
                           splits[i].input_offset_in_bytes;
    void* output_diff_data_ptr = static_cast<char*>(output_diff_data.opaque()) +
                                 splits[i].input_offset_in_bytes;
    const auto status = cudnn_launcher(src_desc, dest_desc, output_data_ptr,
                                       input_diff_data_ptr, input_data_ptr,
                                       output_diff_data_ptr);
    if (!IsStatusOk(status, /*report_error=*/true)) {
      return status;
    }
  }
  return absl::OkStatus();
}

bool CudnnSupport::DoNormalizeWithDimensions(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const dnn::BatchDescriptor& dimensions,
    const DeviceMemory<float>& input_data, DeviceMemory<float>* output_data) {
  // Check for unsupported modes.
  if (normalize_descriptor.wrap_around()) {
    LOG(ERROR) << "CUDA LRN does not support cudnn-around mode";
    return false;
  }
  if (normalize_descriptor.segment_size()) {
    LOG(ERROR) << "CUDA LRN does not support segmentation";
    return false;
  }

  CudnnTensorDescriptor dims(dimensions, CUDNN_DATA_FLOAT);
  CudnnNormalizeDescriptor normalize(normalize_descriptor);

  // Alpha is the scaling factor for input.
  float alpha = 1.0f;
  // Beta is the scaling factor for output.
  float beta = 0.0f;

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  // Launch the normalization.
  const auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnLRNCrossChannelForward(
        cudnn.handle(), normalize.handle(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, dims.handle(), input_data.opaque(), &beta, dims.handle(),
        output_data->opaque()));
    return absl::OkStatus();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DoNormalizeBackwardWithDimensions(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const dnn::BatchDescriptor& dimensions, const DeviceMemory<float>& raw_data,
    const DeviceMemory<float>& normalized_data,
    const DeviceMemory<float>& normalized_variable_gradient,
    DeviceMemory<float>* raw_variable_gradient,
    ScratchAllocator* workspace_allocator) {
  // Check for unsupported modes.
  if (normalize_descriptor.wrap_around()) {
    LOG(ERROR) << "CUDA LRN does not support cudnn-around mode";
    return false;
  }
  if (normalize_descriptor.segment_size()) {
    LOG(ERROR) << "CUDA LRN does not support segmentation";
    return false;
  }

  CudnnTensorDescriptor dims(dimensions, CUDNN_DATA_FLOAT);
  CudnnNormalizeDescriptor normalize(normalize_descriptor);

  float alpha = 1.0f;
  float beta = 0.0f;

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  const auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnLRNCrossChannelBackward(
        cudnn.handle(), normalize.handle(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, dims.handle(), normalized_data.opaque(), dims.handle(),
        normalized_variable_gradient.opaque(), dims.handle(), raw_data.opaque(),
        &beta, dims.handle(), raw_variable_gradient->opaque()));
    return absl::OkStatus();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DeriveOutputBatchDescriptor(
    const dnn::BatchDescriptor& batch_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::BatchDescriptor* output_batch_descriptor) {
  CudnnTensorDescriptor input_nd(batch_descriptor, CUDNN_DATA_FLOAT);
  CudnnFilterDescriptor filter(filter_descriptor, CUDNN_DATA_FLOAT);
  CudnnConvolutionDescriptor conv(convolution_descriptor, CUDNN_DATA_FLOAT);

  int dn = batch_descriptor.ndims() + 2;
  std::vector<int> dims(dn);  // in BDYX
  const auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionNdForwardOutputDim(
        conv.handle(), input_nd.handle(), filter.handle(), dn, dims.data()));
    output_batch_descriptor->set_count(dims[0])
        .set_feature_map_count(dims[1])
        .set_layout(batch_descriptor.layout());

    for (int i = 0; i < batch_descriptor.ndims(); i++) {
      output_batch_descriptor->set_spatial_dim(static_cast<dnn::DimIndex>(i),
                                               dims.rbegin()[i]);
    }
    return absl::OkStatus();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

absl::StatusOr<std::unique_ptr<dnn::DnnGraph>> CudnnSupport::DeserializeGraph(
    Stream& stream, absl::string_view serialized_data) const {
  auto cudnn = cudnn_->GetHandle(stream.parent(), &stream);
  cudnn_frontend::graph::Graph graph;
  RETURN_IF_CUDNN_FRONTEND_ERROR(graph.deserialize(
      cudnn.handle(),
      std::vector<uint8_t>(serialized_data.data(),
                           serialized_data.data() + serialized_data.size())));
  return std::make_unique<CudnnGraph>(std::move(graph));
}

absl::Status CudnnGraph::Prepare(dnn::DnnSupport& dnn_support,
                                 const NumericOptions& numeric_options) {
  const CudnnSupport& cudnn_support = static_cast<CudnnSupport&>(dnn_support);
  TF_ASSIGN_OR_RETURN(auto cudnn, cudnn_support.cudnn_->GetLocalHandle());
  RETURN_IF_CUDNN_FRONTEND_ERROR(graph_.validate());
  RETURN_IF_CUDNN_FRONTEND_ERROR(graph_.build_operation_graph(cudnn->handle()));
  if (numeric_options.require_determinism) {
    graph_.deselect_numeric_notes(
        {cudnn_frontend::NumericalNote_t::NONDETERMINISTIC});
  }
  RETURN_IF_CUDNN_FRONTEND_ERROR(
      graph_.create_execution_plans({cudnn_frontend::HeurMode_t::A}));
  RETURN_CUDNN_FRONTEND_STATUS(graph_.check_support(cudnn->handle()));
}

absl::Status CudnnGraph::Build(dnn::DnnSupport& dnn_support,
                               const std::optional<int64_t> plan_id) {
  const CudnnSupport& cudnn_support = static_cast<CudnnSupport&>(dnn_support);
  TF_ASSIGN_OR_RETURN(auto cudnn, cudnn_support.cudnn_->GetLocalHandle());
  if (plan_id.has_value()) {
    RETURN_CUDNN_FRONTEND_STATUS(
        graph_.build_plan_at_index(cudnn->handle(), *plan_id));
  }
  RETURN_CUDNN_FRONTEND_STATUS(graph_.build_plans(cudnn->handle()));
}

CudnnGraph::VariantPack CudnnGraph::PackOperands(
    absl::Span<DeviceMemoryBase> operands, DeviceMemoryBase& workspace,
    std::optional<int64_t> local_device_ordinal) const {
  CudnnGraph::VariantPack tensor_to_ptr_map;
  absl::Span<DeviceMemoryBase> operands_without_workspace = operands;
  if (graph_.get_workspace_size() > 0) {
    workspace = operands.back();
    CHECK_EQ(graph_.get_workspace_size(), workspace.size());
  }
  if (graph_.get_workspace_size() > 0 || operands.back().size() == 0) {
    operands_without_workspace = operands.first(operands.size() - 1);
  }
  auto next_uid = [uid = 0]() mutable -> int { return CuDnnTensorUID(uid++); };
  for (DeviceMemoryBase operand : operands_without_workspace) {
    tensor_to_ptr_map[next_uid()] = operand.opaque();
  }

  if (dropout_rng_offset_increment_ > 0) {
    CHECK(local_device_ordinal.has_value());
    UpdateDropoutState(*local_device_ordinal);
    tensor_to_ptr_map[next_uid()] = (void*)&dropout_rng_seed_;
    tensor_to_ptr_map[next_uid()] =
        (void*)&current_dropout_rng_offset_[*local_device_ordinal];
  }

  return tensor_to_ptr_map;
}

absl::Status CudnnGraph::Execute(Stream& stream,
                                 absl::Span<DeviceMemoryBase> operands,
                                 int64_t local_device_ordinal) const {
  DeviceMemoryBase workspace;
  VariantPack tensor_to_ptr_map =
      PackOperands(operands, workspace, local_device_ordinal);

  const CudnnSupport& dnn_support =
      static_cast<CudnnSupport&>(*stream.parent()->AsDnn());
  CudnnHandle cudnn = dnn_support.cudnn_->GetHandle(stream.parent(), &stream);

  RETURN_CUDNN_FRONTEND_STATUS(
      graph_.execute(cudnn.handle(), tensor_to_ptr_map, workspace.opaque()));
}

absl::StatusOr<bool> CudnnGraph::SupportsExplicitCommandBufferConstruction()
    const {
  std::vector<cudnn_frontend::BehaviorNote_t> notes;
  RETURN_IF_CUDNN_FRONTEND_ERROR(graph_.get_behavior_notes(notes));
  bool result = absl::c_any_of(notes, [](cudnn_frontend::BehaviorNote_t n) {
    return n == cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API;
  });
  if (!result) {
    VLOG(5) << "Graph does not support CUDA graph native API:\n"
            << graph_.print();
  }
  return result;
}

absl::Status CudnnGraph::PopulateOrUpdateRawCommandBuffer(
    Stream& stream, absl::Span<DeviceMemoryBase> operands,
    RawCommandBufferHandle cuda_graph, bool do_update) {
  DeviceMemoryBase workspace;
  VariantPack tensor_to_ptr_map = PackOperands(operands, workspace);

  const CudnnSupport& dnn_support =
      static_cast<CudnnSupport&>(*stream.parent()->AsDnn());
  CudnnHandle cudnn = dnn_support.cudnn_->GetHandle(stream.parent(), &stream);

  if (do_update) {
    RETURN_CUDNN_FRONTEND_STATUS(
        graph_.update_cuda_graph(cudnn.handle(), tensor_to_ptr_map,
                                 workspace.opaque(), (cudaGraph_t)cuda_graph));
  } else {
    RETURN_CUDNN_FRONTEND_STATUS(graph_.populate_cuda_graph(
        cudnn.handle(), tensor_to_ptr_map, workspace.opaque(),
        (cudaGraph_t)cuda_graph));
  }
}

}  // namespace gpu

void initialize_cudnn() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::DnnFactory>(
          cuda::kCudaPlatformId, "cuDNN",
          [](StreamExecutor* parent) -> dnn::DnnSupport* {
            gpu::CudnnSupport* dnn = new gpu::CudnnSupport(parent);
            if (!dnn->Init().ok()) {
              // Note: Init() will log a more specific error.
              delete dnn;
              return nullptr;
            }
            return dnn;
          });

  if (!status.ok()) {
    LOG(INFO) << "Unable to register cuDNN factory: " << status.message();
  }
}

}  // namespace stream_executor

#ifdef __clang__
#pragma clang diagnostic pop
#endif

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_cudnn, {
  stream_executor::initialize_cudnn();
});
