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

#include "tensorflow/stream_executor/cuda/cuda_dnn.h"

#include <functional>
#include <memory>
#include <utility>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/cuda/cuda_timer.h"
#include "tensorflow/stream_executor/cuda/cudnn_version.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/mathutil.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringpiece.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
// clang-format off
#include "cuda/include/cudnn.h"
// clang-format on

namespace stream_executor {
namespace cuda {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuDnnPlugin);

namespace {

static_assert(CUDNN_VERSION >= 6000, "cuDNN needs to be version 6.0 or higher");

// Exits the program if 'expr' doesn't return CUDNN_STATUS_SUCCESS.
#define CHECK_CUDNN_OK(expr) CHECK_EQ(expr, CUDNN_STATUS_SUCCESS)

// If 'expr' doesn't return CUDNN_STATUS_SUCCESS, returns from the current
// function with a non-successful port::Status.
#define RETURN_IF_CUDNN_ERROR(expr)                                      \
  do {                                                                   \
    cudnnStatus_t _status = expr;                                        \
    if (!SE_PREDICT_TRUE(_status == CUDNN_STATUS_SUCCESS)) {             \
      std::ostringstream oss;                                            \
      oss << ToString(_status) << "\nin " << __FILE__ << "(" << __LINE__ \
          << "): '" << #expr << "'";                                     \
      return port::Status(port::error::UNKNOWN, oss.str().c_str());      \
    }                                                                    \
  } while (false)

// Returns whether status is 'ok', and potentially logs the error.
bool IsStatusOk(const port::Status& status, bool report_error) {
  if (status.ok()) {
    return true;
  }
  if (report_error) {
    LOG(ERROR) << status.error_message();
  }
  return false;
}

// Converts (via narrowing) a type T value to a type U, and checks that the
// value has no value change due to the conversion.
template <typename WideT, typename NarrowT>
NarrowT CheckedNarrowing(const WideT& wide) {
  NarrowT narrow = wide;
  CHECK_EQ(narrow, wide)
      << "checked narrowing failed; values not equal post-conversion";
  return narrow;
}

string ToString(cudnnStatus_t status) {
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
#if CUDNN_VERSION >= 7000
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
      return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
      return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
#endif
    default:
      return port::StrCat("<unknown cudnn status: ", static_cast<int>(status),
                          ">");
  }
}

template <typename T>
cudnnDataType_t GetCudnnDataType(
    dnn::DataLayout = dnn::DataLayout::kBatchDepthYX);

template <>
cudnnDataType_t GetCudnnDataType<double>(dnn::DataLayout) {
  return CUDNN_DATA_DOUBLE;
}

template <>
cudnnDataType_t GetCudnnDataType<float>(dnn::DataLayout) {
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t GetCudnnDataType<Eigen::half>(dnn::DataLayout) {
  return CUDNN_DATA_HALF;
}

template <>
cudnnDataType_t GetCudnnDataType<int8>(dnn::DataLayout layout) {
  switch (layout) {
    case dnn::DataLayout::kYXDepthBatch:
    case dnn::DataLayout::kYXBatchDepth:
    case dnn::DataLayout::kBatchYXDepth:
    case dnn::DataLayout::kBatchDepthYX:
      return CUDNN_DATA_INT8;
    case dnn::DataLayout::kBatchDepthYX4:
      return CUDNN_DATA_INT8x4;
  }
}

template <>
cudnnDataType_t GetCudnnDataType<int32>(dnn::DataLayout) {
  return CUDNN_DATA_INT32;
}

// RAII wrapper for all calls to cuDNN with a cuDNN handle argument.
//
// See CudnnAccess::GetHandle() for details.
class CudnnHandle {
 public:
  // Takes ownership of the executor context and the lock to access cuDNN
  // using handle.
  CudnnHandle(cuda::ScopedActivateExecutorContext context, mutex_lock lock,
              cudnnHandle_t handle)
      : context_(std::move(context)), lock_(std::move(lock)), handle_(handle) {}

  // Returns cuDNN handle. To be passed directly to cuDNN APIs, don't keep
  // a copy.
  cudnnHandle_t handle() const { return handle_; }

 private:
  cuda::ScopedActivateExecutorContext context_;
  mutex_lock lock_;
  cudnnHandle_t handle_;  // Not owned.
};

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
    mutex_lock lock(mutex_);
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
  CudnnHandle GetHandle(CUDAExecutor* executor, Stream* stream) {
    mutex_lock lock(mutex_);
    cuda::ScopedActivateExecutorContext context(executor);
    CUstream cu_stream = stream ? AsCUDAStreamValue(stream) : cudaStreamLegacy;
    auto status = cudnnSetStream(handle_, cu_stream);
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << "Failed to set cuDNN stream.";
    return CudnnHandle(std::move(context), std::move(lock), handle_);
  }

 private:
  // Guards the enqueueing of cuDNN operations via the handle_ below.
  mutex mutex_;

  // cuDNN library handle.
  cudnnHandle_t handle_ GUARDED_BY(mutex_);  // Owned.
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
      return algo;
    // Produces incorrect results for some shapes. Disabled for now, see
    // NVIDIA bug 2072856. TODO(csigg): Only disable for subset of shapes.
    // case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
    default:
      LOG(FATAL)
          << "Unsupported Cudnn convolution backward algorithm for filter: "
          << algorithm.algo_id();
  }
}

port::StatusOr<int> GetCudnnProperty(libraryPropertyType type) {
  int value;
  RETURN_IF_CUDNN_ERROR(cudnnGetProperty(type, &value));
  return value;
}

cudnnRNNAlgo_t ToCudnnRNNAlgo(const dnn::AlgorithmDesc& algorithm) {
  if (algorithm.is_default()) {
    return CUDNN_RNN_ALGO_STANDARD;
  } else {
    cudnnRNNAlgo_t algo = static_cast<cudnnRNNAlgo_t>(algorithm.algo_id());
    switch (algo) {
      case CUDNN_RNN_ALGO_STANDARD:
      case CUDNN_RNN_ALGO_PERSIST_STATIC:
      case CUDNN_RNN_ALGO_PERSIST_DYNAMIC:
        return algo;
      default:
        LOG(FATAL) << "Unsupported Cudnn RNN algorithm: "
                   << algorithm.algo_id();
    }
  }
}

port::Status GetLoadedCudnnVersion(CudnnVersion* version) {
  SE_ASSIGN_OR_RETURN(version->major_version, GetCudnnProperty(MAJOR_VERSION));
  SE_ASSIGN_OR_RETURN(version->minor_version, GetCudnnProperty(MINOR_VERSION));
  SE_ASSIGN_OR_RETURN(version->patch_level, GetCudnnProperty(PATCH_LEVEL));
  return port::Status::OK();
}

}  // namespace

CudnnSupport::CudnnSupport(CUDAExecutor* parent) : parent_(parent) {}

port::Status CudnnSupport::Init() {
  ScopedActivateExecutorContext context(parent_);
  cudnnHandle_t cudnn_handle = nullptr;
  auto status = cudnnCreate(&cudnn_handle);
  if (status == CUDNN_STATUS_SUCCESS) {
    CudnnVersion source_version(CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);

    CudnnVersion loaded_version;
    TF_RETURN_IF_ERROR(GetLoadedCudnnVersion(&loaded_version));
    if (!IsSourceCompatibleWithCudnnLibrary(source_version, loaded_version)) {
      const tensorflow::string error = port::StrCat(
          "Loaded runtime CuDNN library: ", loaded_version.ToString(),
          " but source was compiled with: ", source_version.ToString(),
          ".  CuDNN library major and minor version needs to match or have "
          "higher minor version in case of CuDNN 7.0 or later version. If "
          "using a binary install, upgrade your CuDNN library.  If building "
          "from sources, make sure the library loaded at runtime is "
          "compatible "
          "with the version specified during compile configuration.");
      LOG(ERROR) << error;
      cudnnDestroy(cudnn_handle);
      return port::Status(port::error::INTERNAL, error);
    }

    cudnn_.reset(new CudnnAccess(cudnn_handle));
    return port::Status::OK();
  }

  CHECK_EQ(cudnn_handle, nullptr);
  LOG(ERROR) << "Could not create cudnn handle: " << ToString(status);
  if (status == CUDNN_STATUS_NOT_INITIALIZED) {
    auto result = cuda::Diagnostician::FindKernelDriverVersion();
    if (!result.ok()) {
      LOG(ERROR) << "Error retrieving driver version: "
                 << DriverVersionStatusToString(result);
    } else {
      const auto& version = result.ValueOrDie();
      LOG(ERROR) << "Possibly insufficient driver version: "
                 << DriverVersionToString(version);
    }
  }

  return port::Status(port::error::INTERNAL,
                      port::StrCat("cudnn library could not create a handle: ",
                                   ToString(status)));
}

port::StatusOr<perftools::gputools::dnn::VersionInfo>
CudnnSupport::GetVersion() {
  CudnnVersion version;
  TF_RETURN_IF_ERROR(GetLoadedCudnnVersion(&version));
  return perftools::gputools::dnn::VersionInfo(
      version.major_version, version.minor_version, version.patch_level);
}

namespace {

// Deleter functors for cuDNN types that need to be deleted.
struct TensorDescriptorDeleter {
  void operator()(cudnnTensorDescriptor_t descriptor) const {
    CHECK_CUDNN_OK(cudnnDestroyTensorDescriptor(descriptor));
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
struct PersistentRnnPlanDeleter {
  void operator()(cudnnPersistentRNNPlan_t plan) const {
    CHECK_CUDNN_OK(cudnnDestroyPersistentRNNPlan(plan));
  }
};

// RAII wrappers for cuDNN types.
using TensorDescriptor =
    std::unique_ptr<cudnnTensorStruct, TensorDescriptorDeleter>;
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
using PersistentRnnPlan =
    std::unique_ptr<cudnnPersistentRNNPlan, PersistentRnnPlanDeleter>;

// Factory methods for cuDNN types.
TensorDescriptor CreateTensorDescriptor() {
  cudnnTensorDescriptor_t result;
  CHECK_CUDNN_OK(cudnnCreateTensorDescriptor(&result));
  return TensorDescriptor(result);
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
PersistentRnnPlan CreatePersistentRnnPlan(cudnnRNNDescriptor_t rnn_desc,
                                          int batch_size,
                                          cudnnDataType_t data_type) {
  cudnnPersistentRNNPlan_t result;
  CHECK_CUDNN_OK(
      cudnnCreatePersistentRNNPlan(rnn_desc, batch_size, data_type, &result));
  return PersistentRnnPlan(result);
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
        std::vector<int64> strides64 =
            batch_descriptor.full_strides(dnn::DataLayout::kBatchDepthYX);
        std::vector<int64> dims64 =
            batch_descriptor.full_dims(dnn::DataLayout::kBatchDepthYX);

        // cuDNN requires arrays of ints.
        std::vector<int> strides(nd);
        std::vector<int> dims(nd);
        std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                       &CheckedNarrowing<int64, int>);
        std::transform(dims64.cbegin(), dims64.cend(), dims.begin(),
                       &CheckedNarrowing<int64, int>);
        CHECK_CUDNN_OK(cudnnSetTensorNdDescriptor(handle_.get(), elem_type, nd,
                                                  dims.data(), strides.data()))
            << "batch_descriptor: " << batch_descriptor.ToString();
      } break;
      case dnn::DataLayout::kBatchDepthYX4: {
        CHECK_CUDNN_OK(cudnnSetTensor4dDescriptor(
            handle_.get(), CUDNN_TENSOR_NCHW_VECT_C, elem_type,
            batch_descriptor.count(), batch_descriptor.feature_map_count(),
            batch_descriptor.height(), batch_descriptor.width()))
            << "batch_descriptor: " << batch_descriptor.ToString();
      } break;
      default:
        LOG(FATAL) << "Unsupported tensor format "
                   << DataLayoutString(batch_descriptor.layout());
        break;
    }
  }

  cudnnTensorDescriptor_t handle() const { return handle_.get(); }

 private:
  TensorDescriptor handle_;

  SE_DISALLOW_COPY_AND_ASSIGN(CudnnTensorDescriptor);
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
        format = CUDNN_TENSOR_NCHW_VECT_C;
        break;
      default:
        LOG(FATAL) << "Unsupported filter format "
                   << FilterLayoutString(filter_descriptor.layout());
        break;
    }

    std::vector<int> dims(2 + filter_descriptor.ndims());
    dims[0] = filter_descriptor.output_feature_map_count();
    dims[1] = filter_descriptor.input_feature_map_count();
    const auto& spatial_dims = filter_descriptor.input_filter_dims();
    std::copy(spatial_dims.begin(), spatial_dims.end(), dims.begin() + 2);

    CHECK_CUDNN_OK(cudnnSetFilterNdDescriptor(handle_.get(), elem_type, format,
                                              dims.size(), dims.data()));
  }

  cudnnFilterDescriptor_t handle() const { return handle_.get(); }

 private:
  FilterDescriptor handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(CudnnFilterDescriptor);
};

// A helper function to decide whether to enable the TENSOR_OP_MATH math type
bool TensorOpMathEnabled() {
  static bool is_enabled = [] {
    bool is_disabled = false;
    TF_CHECK_OK(
        tensorflow::ReadBoolFromEnvVar("TF_DISABLE_CUDNN_TENSOR_OP_MATH",
                                       /*default_val=*/false, &is_disabled));
    return !is_disabled;
  }();
  return is_enabled;
}

// A helper function to decide whether to enable the TENSOR_OP_MATH math type
// for RNNs.
bool RnnTensorOpMathEnabled() {
  static bool is_enabled = [] {
    bool is_disabled = false;
    TF_CHECK_OK(
        tensorflow::ReadBoolFromEnvVar("TF_DISABLE_CUDNN_RNN_TENSOR_OP_MATH",
                                       /*default_val=*/false, &is_disabled));
    return !is_disabled;
  }();
  return is_enabled;
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
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
        "TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT",
        /*default_val=*/false, &is_enabled));
    return is_enabled;
  }();
  return is_enabled;
}

// Turns a ConvolutionDescriptor structure into a cudnn convolution handle
// within a scope.
class CudnnConvolutionDescriptor {
 public:
  CudnnConvolutionDescriptor(
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      cudnnDataType_t data_type)
      : handle_(CreateConvolutionDescriptor()) {
    const auto& strides64 = convolution_descriptor.strides();
    const auto& padding64 = convolution_descriptor.padding();
    const auto& dilations64 = convolution_descriptor.dilations();
    CHECK_NE(convolution_descriptor.pad_alignment(),
             dnn::PadAlignment::kTensorFlowPadding)
        << "TensorFlow padding alignment is not supported.";

    // cuDNN requires arrays of ints.
    std::vector<int> strides(convolution_descriptor.ndims());
    std::vector<int> padding(convolution_descriptor.ndims());
    std::vector<int> dilations(convolution_descriptor.ndims());
    std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                   &CheckedNarrowing<int64, int>);
    std::transform(padding64.cbegin(), padding64.cend(), padding.begin(),
                   &CheckedNarrowing<int64, int>);
    // TODO(yangzihao): Test with negative dilation to make sure that cudnn
    // doesn't crash.
    std::transform(dilations64.cbegin(), dilations64.cend(), dilations.begin(),
                   &CheckedNarrowing<int64, int>);

    CHECK_CUDNN_OK(cudnnSetConvolutionNdDescriptor(
        handle_.get(), convolution_descriptor.ndims(), padding.data(),
        strides.data(), dilations.data(),
        // NOTE(keveman): cuDNN supports convolution and cross correlation.
        // However, almost all the use cases do cross correlation, so just
        // hard coding it here.
        CUDNN_CROSS_CORRELATION, data_type));

    // NOTE(benbarsdell): This only applies if tensor op math is enabled
    //                      and algo selection is set to Default.
    this->set_use_tensor_op_math(true);

#if CUDNN_MAJOR >= 7
    VLOG(2) << "Requesting grouped convolution: "
            << convolution_descriptor.group_count();
    CHECK_CUDNN_OK(cudnnSetConvolutionGroupCount(
        handle_.get(), convolution_descriptor.group_count()));
#else
    CHECK_EQ(convolution_descriptor.group_count(), 1)
        << "Requested grouped convolution for cuDNN version < 7";
#endif
  }

  void set_use_tensor_op_math(bool use_tensor_op_math) const {
#if CUDNN_VERSION >= 7000
    cudnnMathType_t math_type =
        (use_tensor_op_math ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH);
    if (TensorOpMathEnabled()) {
      CHECK_CUDNN_OK(cudnnSetConvolutionMathType(handle_.get(), math_type));
    }
#endif
  }

  cudnnConvolutionDescriptor_t handle() const { return handle_.get(); }

 private:
  ConvolutionDescriptor handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(CudnnConvolutionDescriptor);
};

// Turns a PoolingDescriptor structure into a cudnn pooling descriptor handle
// within a scope.
class CudnnPoolingDescriptor {
 public:
  explicit CudnnPoolingDescriptor(
      const dnn::PoolingDescriptor& pooling_descriptor)
      : handle_(CreatePoolingDescriptor()) {
    const std::vector<int64> strides64 = pooling_descriptor.strides();
    const std::vector<int64> padding64 = pooling_descriptor.padding();
    const std::vector<int64> shape64 = pooling_descriptor.window();

    const int nd = pooling_descriptor.ndims();
    std::vector<int> shape(nd);
    std::vector<int> padding(nd);
    std::vector<int> strides(nd);
    std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                   &CheckedNarrowing<int64, int>);
    std::transform(padding64.cbegin(), padding64.cend(), padding.begin(),
                   &CheckedNarrowing<int64, int>);
    std::transform(shape64.cbegin(), shape64.cend(), shape.begin(),
                   &CheckedNarrowing<int64, int>);
    bool propagate_nans = pooling_descriptor.propagate_nans();
    CHECK_CUDNN_OK(cudnnSetPoolingNdDescriptor(
        handle_.get(),
        (pooling_descriptor.mode() == dnn::PoolingMode::kMaximum
             ? CUDNN_POOLING_MAX
             : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING),
        propagate_nans ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN, nd,
        shape.data(), padding.data(), strides.data()));
  }

  cudnnPoolingDescriptor_t handle() const { return handle_.get(); }

 private:
  PoolingDescriptor handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(CudnnPoolingDescriptor);
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

  SE_DISALLOW_COPY_AND_ASSIGN(CudnnNormalizeDescriptor);
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
#if CUDNN_VERSION >= 7100
      case dnn::ActivationMode::kNone:
        mode = CUDNN_ACTIVATION_IDENTITY;
        break;
#endif
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

  SE_DISALLOW_COPY_AND_ASSIGN(CudnnActivationDescriptor);
};

cudnnDataType_t ToCudnnDataType(
    dnn::DataType data_type,
    dnn::DataLayout data_layout = dnn::DataLayout::kBatchDepthYX) {
  switch (data_type) {
    case dnn::DataType::kFloat:
    case dnn::DataType::kDouble:
    case dnn::DataType::kHalf:
      return static_cast<cudnnDataType_t>(data_type);
    case dnn::DataType::kInt8:
      return data_layout == dnn::DataLayout::kBatchDepthYX4 ? CUDNN_DATA_INT8x4
                                                            : CUDNN_DATA_INT8;
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
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
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

class CudnnDropoutDescriptor {
  explicit CudnnDropoutDescriptor(DropoutDescriptor handle)
      : handle_(std::move(handle)) {}

 public:
  CudnnDropoutDescriptor(CudnnDropoutDescriptor&&) = default;

  static port::StatusOr<CudnnDropoutDescriptor> Create(
      const CudnnHandle& cudnn, float dropout, uint64 seed,
      ScratchAllocator* state_allocator) {
    DropoutDescriptor handle = CreateDropoutDescriptor();

    if (dropout == 0.0f) {
      // Return 'empty' dropout descriptor.
      return CudnnDropoutDescriptor(std::move(handle));
    }

    DeviceMemory<uint8> state_memory;
    if (state_allocator) {
      size_t state_sizes_in_bytes = 0;
      RETURN_IF_CUDNN_ERROR(
          cudnnDropoutGetStatesSize(cudnn.handle(), &state_sizes_in_bytes));
      SE_ASSIGN_OR_RETURN(state_memory, state_allocator->AllocateBytes(
                                            nullptr, state_sizes_in_bytes));
    }
    RETURN_IF_CUDNN_ERROR(cudnnSetDropoutDescriptor(
        handle.get(), cudnn.handle(), dropout, state_memory.opaque(),
        state_memory.size(), seed));

    return CudnnDropoutDescriptor(std::move(handle));
  }

  cudnnDropoutDescriptor_t handle() const { return handle_.get(); }

 private:
  DropoutDescriptor handle_;  // Owned.
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnDropoutDescriptor);
};

class CudnnRnnParamsDescriptor {
  typedef dnn::RnnDescriptor::ParamsRegions ParamsRegions;

  CudnnRnnParamsDescriptor(FilterDescriptor handle, int64 params_size_in_bytes,
                           ParamsRegions weights, ParamsRegions biases)
      : handle_(std::move(handle)),
        params_size_in_bytes_(params_size_in_bytes),
        weights_(std::move(weights)),
        biases_(std::move(biases)) {}

 public:
  CudnnRnnParamsDescriptor(CudnnRnnParamsDescriptor&&) = default;

  static port::StatusOr<CudnnRnnParamsDescriptor> Create(
      const CudnnHandle& cudnn, int input_size, cudnnDataType_t data_type,
      cudnnRNNDescriptor_t rnn_desc, cudnnRNNMode_t rnn_mode,
      cudnnDirectionMode_t direction_mode, int num_layers);

  cudnnFilterDescriptor_t handle() const { return handle_.get(); }
  int64 params_size_in_bytes() const { return params_size_in_bytes_; }
  ParamsRegions params_weights() const {
    return weights_;
  }
  ParamsRegions params_biases() const {
    return biases_;
  }

 private:
  FilterDescriptor handle_;
  int64 params_size_in_bytes_;
  ParamsRegions weights_;
  ParamsRegions biases_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnParamsDescriptor);
};

}  // namespace

class CudnnRnnDescriptor : public dnn::RnnDescriptor {
  CudnnRnnDescriptor(const CudnnHandle& cudnn, cuda::RnnDescriptor rnn_desc,
                     PersistentRnnPlan rnn_plan, int num_layers,
                     int hidden_size, int input_size, int batch_size,
                     cudnnRNNInputMode_t input_mode,
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

  static port::StatusOr<CudnnRnnDescriptor> Create(
      const CudnnHandle& cudnn, int num_layers, int hidden_size, int input_size,
      int batch_size, cudnnRNNInputMode_t input_mode,
      cudnnDirectionMode_t direction_mode, cudnnRNNMode_t rnn_mode,
      cudnnDataType_t data_type, cudnnDataType_t compute_type,
      const dnn::AlgorithmConfig& algorithm_config, float dropout, uint64 seed,
      ScratchAllocator* state_allocator) {
    SE_ASSIGN_OR_RETURN(
        CudnnDropoutDescriptor dropout_desc,
        CudnnDropoutDescriptor::Create(cudnn, dropout, seed, state_allocator));

    cuda::RnnDescriptor rnn_desc = CreateRnnDescriptor();
    cudnnRNNAlgo_t rnn_algo = ToCudnnRNNAlgo(algorithm_config.algorithm());

    // TODO: allow the user to choose an algorithm.
    RETURN_IF_CUDNN_ERROR(cudnnSetRNNDescriptor_v6(
        cudnn.handle(), /*rnnDesc=*/rnn_desc.get(), /*hiddenSize=*/hidden_size,
        /*numLayers=*/num_layers, /*dropoutDesc=*/dropout_desc.handle(),
        /*inputMode=*/input_mode, /*direction=*/direction_mode,
        /*mode=*/rnn_mode, /*algo=*/rnn_algo,
        /*dataType=*/compute_type));

    PersistentRnnPlan rnn_plan;
    if (rnn_algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
      CHECK_GE(batch_size, 0);
      rnn_plan = CreatePersistentRnnPlan(rnn_desc.get(), batch_size, data_type);
      RETURN_IF_CUDNN_ERROR(
          cudnnSetPersistentRNNPlan(rnn_desc.get(), rnn_plan.get()));
    }

    // Create the params handle.
    SE_ASSIGN_OR_RETURN(auto params_desc,
                        CudnnRnnParamsDescriptor::Create(
                            cudnn, input_size, data_type, rnn_desc.get(),
                            rnn_mode, direction_mode, num_layers));

#if CUDNN_VERSION >= 7000
    // Require explicit algorithm config to enable tensor cores. Some configs
    // return CUDNN_NOT_SUPPORTED when tensor ops are enabled (which is against
    // the idiom that enabling tensor ops is only a hint: see nvbugs/2172799).
    // We can only reasonably expect the user to handle the subsequent failure
    // in profile mode, which is run with algorithms returned from
    // GetRnnAlgorithms() (which are non-default and explicitly set whether to
    // use tensor ops).
    if (RnnTensorOpMathEnabled() &&
        !algorithm_config.algorithm().is_default()) {
      cudnnMathType_t math_type =
          algorithm_config.algorithm().tensor_ops_enabled()
              ? CUDNN_TENSOR_OP_MATH
              : CUDNN_DEFAULT_MATH;
      CHECK_CUDNN_OK(cudnnSetRNNMatrixMathType(rnn_desc.get(), math_type));
    }
#endif

    return CudnnRnnDescriptor(cudnn, std::move(rnn_desc), std::move(rnn_plan),
                              num_layers, hidden_size, input_size, batch_size,
                              input_mode, direction_mode, rnn_mode, data_type,
                              compute_type, algorithm_config,
                              std::move(dropout_desc), std::move(params_desc));
  }

  cudnnRNNDescriptor_t handle() const { return rnn_desc_.get(); }
  int num_layers() const { return num_layers_; }
  int hidden_size() const { return hidden_size_; }
  int input_size() const { return input_size_; }
  int batch_size() const { return batch_size_; }
  cudnnRNNInputMode_t input_mode() const { return input_mode_; }
  cudnnDirectionMode_t direction_mode() const { return direction_mode_; }
  cudnnRNNMode_t rnn_mode() const { return rnn_mode_; }
  cudnnDataType_t data_type() const { return data_type_; }
  cudnnDataType_t compute_type() const { return compute_type_; }
  const dnn::AlgorithmConfig& algorithm_config() const {
    return algorithm_config_;
  }
  int64 ParamsSizeInBytes() const override {
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
  cuda::RnnDescriptor rnn_desc_;
  PersistentRnnPlan rnn_plan_;
  int num_layers_;
  int hidden_size_;
  int input_size_;
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
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnDescriptor);
};

namespace {

port::StatusOr<CudnnRnnParamsDescriptor> CudnnRnnParamsDescriptor::Create(
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
  RETURN_IF_CUDNN_ERROR(cudnnGetRNNParamsSize(
      /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc,
      /*xDesc=*/input_desc.get(), /*sizeInBytes=*/&params_size,
      /*dataType=*/data_type));
  int64 params_size_in_bytes = static_cast<int64>(params_size);

  FilterDescriptor filter_desc = CreateFilterDescriptor();
  int filter_dims[] = {static_cast<int>(params_size_in_bytes), 1, 1};
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
      for (int type = 0; type < 2; type++) {
        void* offset = nullptr;
        RETURN_IF_CUDNN_ERROR((type == 0 ? cudnnGetRNNLinLayerMatrixParams
                                         : cudnnGetRNNLinLayerBiasParams)(
            /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc,
            /*layer=*/layer, /*xDesc=*/input_desc.get(),
            /*wDesc=*/filter_desc.get(),
            /*w=*/nullptr, /*linLayerID=*/region,
            /*linLayerMatDesc=*/region_desc_handle.get(),
            /*linLayerMat or linLayerBias=*/&offset));
        int dims[] = {1, 1, 1};
        cudnnDataType_t data_type;
        cudnnTensorFormat_t tensor_format;
        int n_dims;
        RETURN_IF_CUDNN_ERROR(cudnnGetFilterNdDescriptor(
            /*filterDesc=*/region_desc_handle.get(),
            /*nbDimsRequested=*/sizeof(dims) / sizeof(dims[0]),
            /*dataType=*/&data_type, /*format=*/&tensor_format,
            /*nbDims=*/&n_dims, /*filterDimA=*/dims));
        int64 size =
            dims[0] * dims[1] * dims[2] * CudnnDataTypeToByteSize(data_type);
        dnn::RnnDescriptor::ParamsRegion region = {
            reinterpret_cast<int64>(offset), size};
        (type == 0 ? weights : biases).push_back(region);
      }
    }
  }

  return CudnnRnnParamsDescriptor(std::move(filter_desc), params_size_in_bytes,
                                  weights, biases);
}

}  // namespace

class CudnnRnnSequenceTensorDescriptor
    : public dnn::RnnSequenceTensorDescriptor {
  CudnnRnnSequenceTensorDescriptor(CUDAExecutor* parent, int seq_length,
                                   int batch_size, int data_size,
                                   cudnnDataType_t data_type,
                                   TensorDescriptor handle)
      : parent_(parent),
        seq_length_(seq_length),
        batch_size_(batch_size),
        data_size_(data_size),
        data_type_(data_type),
        handle_(std::move(handle)),
        handles_(seq_length, handle_.get()) {}

 public:
  CudnnRnnSequenceTensorDescriptor(CudnnRnnSequenceTensorDescriptor&&) =
      default;

  static port::StatusOr<CudnnRnnSequenceTensorDescriptor> Create(
      CUDAExecutor* parent, int seq_length, int batch_size, int data_size,
      cudnnDataType_t data_type) {
    CHECK_GT(seq_length, 0);
    int dims[] = {batch_size, data_size, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    TensorDescriptor tensor_desc = CreateTensorDescriptor();
    RETURN_IF_CUDNN_ERROR(cudnnSetTensorNdDescriptor(
        /*tensorDesc=*/tensor_desc.get(), /*dataType=*/data_type,
        /*nbDims=*/sizeof(dims) / sizeof(dims[0]), /*dimA=*/dims,
        /*strideA=*/strides));
    return CudnnRnnSequenceTensorDescriptor(parent, seq_length, batch_size,
                                            data_size, data_type,
                                            std::move(tensor_desc));
  }

  const cudnnTensorDescriptor_t* handles() const {
    return handles_.data();
  }

  int seq_length() const { return seq_length_; }
  int batch_size() const { return batch_size_; }
  int data_size() const { return data_size_; }

 private:
  CUDAExecutor* parent_;
  int seq_length_;
  int batch_size_;
  int data_size_;
  cudnnDataType_t data_type_;
  TensorDescriptor handle_;
  std::vector<cudnnTensorDescriptor_t> handles_;  // Copies of handle_.
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnSequenceTensorDescriptor);
};

class CudnnRnnStateTensorDescriptor : public dnn::RnnStateTensorDescriptor {
 public:
  CudnnRnnStateTensorDescriptor(CUDAExecutor* parent, int num_layers,
                                int batch_size, int data_size,
                                cudnnDataType_t data_type)
      : parent_(parent),
        handle_(CreateTensorDescriptor()),
        num_layers_(num_layers),
        batch_size_(batch_size),
        data_size_(data_size),
        data_type_(data_type) {
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
  CUDAExecutor* parent_;
  TensorDescriptor handle_;
  int num_layers_;
  int batch_size_;
  int data_size_;
  cudnnDataType_t data_type_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnStateTensorDescriptor);
};

namespace {

struct RnnModelDims {
  int num_layers = 0;
  int batch_size = 0;
  int seq_length = 0;
  int hidden_size = 0;
  int input_size = 0;
  int dir_count = 0;
};

template <class T>
port::StatusOr<RnnModelDims> ExtractAndCheckRnnForward(
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
  model_dims.seq_length = input_desc.seq_length();
  model_dims.hidden_size = rnn_desc.hidden_size();
  model_dims.input_size = input_desc.data_size();
  model_dims.dir_count =
      (rnn_desc.direction_mode() == CUDNN_BIDIRECTIONAL) ? 2 : 1;

  // check parameters
  if (!(input_h_desc.num_layers() ==
            model_dims.num_layers * model_dims.dir_count &&
        input_h_desc.batch_size() == model_dims.batch_size &&
        input_h_desc.data_size() == model_dims.hidden_size)) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid input_h shape");
  }
  if (!(input_h_desc.num_layers() == input_c_desc.num_layers() &&
        input_h_desc.batch_size() == input_c_desc.batch_size() &&
        input_h_desc.data_size() == input_c_desc.data_size())) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid input_c shape");
  }
  if (!(output_desc.seq_length() == model_dims.seq_length &&
        output_desc.batch_size() == model_dims.batch_size &&
        output_desc.data_size() ==
            model_dims.hidden_size * model_dims.dir_count)) {
    return port::Status(port::error::INVALID_ARGUMENT, "Invalid output shape");
  }
  if (!(input_h_desc.num_layers() == output_h_desc.num_layers() &&
        input_h_desc.batch_size() == output_h_desc.batch_size() &&
        input_h_desc.data_size() == output_h_desc.data_size())) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "Invalid output_h shape");
  }
  if (!(input_h_desc.num_layers() == output_c_desc.num_layers() &&
        input_h_desc.batch_size() == output_c_desc.batch_size() &&
        input_h_desc.data_size() == output_c_desc.data_size())) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "Invalid output_c shape");
  }

  return model_dims;
}

port::Status CheckRNNParameterSize(
    const CudnnHandle& cudnn, const CudnnRnnDescriptor& rnn_desc,
    const CudnnRnnSequenceTensorDescriptor& input_desc) {
  size_t params_size_in_bytes = 0;
  RETURN_IF_CUDNN_ERROR(cudnnGetRNNParamsSize(
      /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
      /*xDesc=*/input_desc.handles()[0], /*sizeInBytes=*/&params_size_in_bytes,
      /*dataType=*/rnn_desc.data_type()));
  if (static_cast<int64>(params_size_in_bytes) !=
      rnn_desc.ParamsSizeInBytes()) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "Mismatching RNN parameter size");
  }
  return port::Status::OK();
}

port::StatusOr<DeviceMemory<uint8>> CreateRnnWorkspace(
    Stream* stream, const CudnnHandle& cudnn,
    const CudnnRnnDescriptor& rnn_desc,
    const CudnnRnnSequenceTensorDescriptor& input_desc,
    ScratchAllocator* workspace_allocator) {
  // Query the workspace size.
  size_t workspace_size_in_bytes = 0;
  RETURN_IF_CUDNN_ERROR(cudnnGetRNNWorkspaceSize(
      /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
      /*seqLength=*/input_desc.seq_length(), /*xDesc=*/input_desc.handles(),
      /*sizeInBytes=*/&workspace_size_in_bytes));
  // Allocate the workspace.
  if (workspace_size_in_bytes == 0) {
    return DeviceMemory<uint8>();
  }
  return workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
}

}  // namespace

template <class T>
port::Status CudnnSupport::DoRnnForwardImpl(
    Stream* stream, const CudnnRnnDescriptor& rnn_desc,
    const CudnnRnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<T>& input_data,
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
  SE_ASSIGN_OR_RETURN(
      RnnModelDims model_dims,
      ExtractAndCheckRnnForward(
          rnn_desc, input_desc, input_data, input_h_desc, input_h_data,
          input_c_desc, input_c_data, params, output_desc, *output_data,
          output_h_desc, *output_h_data, output_c_desc, *output_c_data));

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  SE_RETURN_IF_ERROR(CheckRNNParameterSize(cudnn, rnn_desc, input_desc));
  SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> workspace,
                      CreateRnnWorkspace(stream, cudnn, rnn_desc, input_desc,
                                         workspace_allocator))

  // query the reserve space size
  // allocate the reserve space
  DeviceMemory<uint8> reserve_space;
  if (is_training) {
    size_t reserve_space_size_in_bytes = 0;
    RETURN_IF_CUDNN_ERROR(cudnnGetRNNTrainingReserveSize(
        /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
        /*seqLength=*/model_dims.seq_length, /*xDesc=*/input_desc.handles(),
        /*sizeInBytes=*/&reserve_space_size_in_bytes));

    if (reserve_space_size_in_bytes > 0) {
      SE_ASSIGN_OR_RETURN(reserve_space,
                          reserve_space_allocator->AllocateBytes(
                              stream, reserve_space_size_in_bytes));
    }
  }

  std::unique_ptr<CUDATimer, TimerDeleter> timer;
  const bool is_profiling = output_profile_result != nullptr;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Init() || !timer->Start(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to start timer");
    }
  }

  if (!is_training) {
    RETURN_IF_CUDNN_ERROR(cudnnRNNForwardInference(
        /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
        /*seqLength=*/model_dims.seq_length, /*xDesc=*/input_desc.handles(),
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
        /*seqLength=*/model_dims.seq_length, /*xDesc=*/input_desc.handles(),
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

  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to stop timer");
    }
    auto algo_desc = rnn_desc.algorithm_config().algorithm();
    output_profile_result->set_algorithm(algo_desc);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }

  return port::Status::OK();
}

template <class T>
port::Status CudnnSupport::DoRnnBackwardImpl(
    Stream* stream, const CudnnRnnDescriptor& rnn_desc,
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
    const DeviceMemory<T>& output_c_data,
    const DeviceMemory<T>& output_backprop_data,
    const DeviceMemory<T>& output_h_backprop_data,
    const DeviceMemory<T>& output_c_backprop_data,
    DeviceMemory<T>* input_backprop_data,
    DeviceMemory<T>* input_h_backprop_data,
    DeviceMemory<T>* input_c_backprop_data,
    DeviceMemory<T>* params_backprop_data,
    DeviceMemory<uint8>* reserve_space_data,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
  SE_ASSIGN_OR_RETURN(
      RnnModelDims model_dims,
      ExtractAndCheckRnnForward(rnn_desc, input_desc, input_data, input_h_desc,
                                input_h_data, input_c_desc, input_c_data,
                                params, output_desc, output_data, output_h_desc,
                                output_h_data, output_c_desc, output_c_data));

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  SE_RETURN_IF_ERROR(CheckRNNParameterSize(cudnn, rnn_desc, input_desc));
  SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> workspace,
                      CreateRnnWorkspace(stream, cudnn, rnn_desc, input_desc,
                                         workspace_allocator));

  std::unique_ptr<CUDATimer, TimerDeleter> timer;
  const bool is_profiling = output_profile_result != nullptr;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Init() || !timer->Start(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to start timer");
    }
  }

  RETURN_IF_CUDNN_ERROR(cudnnRNNBackwardData(
      /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
      /*seqLength=*/model_dims.seq_length, /*yDesc=*/output_desc.handles(),
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
    stream->ThenMemZero(params_backprop_data, params_backprop_data->size());
    // make the backward weight call
    RETURN_IF_CUDNN_ERROR(cudnnRNNBackwardWeights(
        /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
        /*seqLength=*/model_dims.seq_length, /*xDesc=*/input_desc.handles(),
        /*x=*/input_data.opaque(), /*hxDesc=*/input_h_desc.handle(),
        /*hx=*/input_h_data.opaque(), /*yDesc=*/output_desc.handles(),
        /*y=*/output_data.opaque(), /*workspace=*/workspace.opaque(),
        /*workSpaceSizeInBytes=*/workspace.size(),
        /*dwDesc=*/rnn_desc.params_handle(),
        /*dw=*/params_backprop_data->opaque(),
        /*reserveSpace=*/reserve_space_data->opaque(),
        /*reserveSpaceSizeInBytes=*/reserve_space_data->size()));
  }

  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to stop timer");
    }
    auto algo_desc = rnn_desc.algorithm_config().algorithm();
    output_profile_result->set_algorithm(algo_desc);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }

  return port::Status::OK();
}

port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
CudnnSupport::createRnnDescriptor(
    int num_layers, int hidden_size, int input_size, int batch_size,
    dnn::RnnInputMode input_mode, dnn::RnnDirectionMode direction_mode,
    dnn::RnnMode rnn_mode, dnn::DataType data_type,
    const dnn::AlgorithmConfig& algorithm_config, float dropout, uint64 seed,
    ScratchAllocator* state_allocator) {
  // Setting up a cudnnRNNDescriptor requires a cuDNN handle, but because it's
  // not enqueueing anything into a stream, we pass in the null stream.
  auto cudnn = cudnn_->GetHandle(parent_, /*stream=*/nullptr);
  SE_ASSIGN_OR_RETURN(
      CudnnRnnDescriptor rnn_desc,
      CudnnRnnDescriptor::Create(
          cudnn, num_layers, hidden_size, input_size, batch_size,
          ToCudnnRnnInputMode(input_mode),
          ToCudnnRnnDirectionMode(direction_mode), ToCudnnRnnMode(rnn_mode),
          ToCudnnDataType(data_type), GetRnnComputeType(data_type),
          algorithm_config, dropout, seed, state_allocator));
  return std::unique_ptr<dnn::RnnDescriptor>(
      new CudnnRnnDescriptor(std::move(rnn_desc)));
}

port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
CudnnSupport::createRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                                int data_size,
                                                dnn::DataType data_type) {
  SE_ASSIGN_OR_RETURN(CudnnRnnSequenceTensorDescriptor descriptor,
                      CudnnRnnSequenceTensorDescriptor::Create(
                          parent_, seq_length, batch_size, data_size,
                          ToCudnnDataType(data_type)));
  return std::unique_ptr<dnn::RnnSequenceTensorDescriptor>(
      new CudnnRnnSequenceTensorDescriptor(std::move(descriptor)));
}

port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
CudnnSupport::createRnnStateTensorDescriptor(int num_layer, int batch_size,
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
          cudnn_input_h_desc, input_h_data, cudnn_input_c_desc, input_c_data,
          params, cudnn_output_desc, output_data, cudnn_output_h_desc,
          output_h_data, cudnn_output_c_desc, output_c_data, is_training,
          reserve_space_allocator, workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoRnnForward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<float>& input_data,
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
          cudnn_input_h_desc, input_h_data, cudnn_input_c_desc, input_c_data,
          params, cudnn_output_desc, output_data, cudnn_output_h_desc,
          output_h_data, cudnn_output_c_desc, output_c_data, is_training,
          reserve_space_allocator, workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoRnnForward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<double>& input_data,
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
          cudnn_input_h_desc, input_h_data, cudnn_input_c_desc, input_c_data,
          params, cudnn_output_desc, output_data, cudnn_output_h_desc,
          output_h_data, cudnn_output_c_desc, output_c_data, is_training,
          reserve_space_allocator, workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoRnnBackward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<Eigen::half>& input_data,
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
    DeviceMemory<uint8>* reserve_space_data,
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
          cudnn_input_h_desc, input_h_data, cudnn_input_c_desc, input_c_data,
          params, cudnn_output_desc, output_data, cudnn_output_h_desc,
          output_h_data, cudnn_output_c_desc, output_c_data,
          output_backprop_data, output_h_backprop_data, output_c_backprop_data,
          input_backprop_data, input_h_backprop_data, input_c_backprop_data,
          params_backprop_data, reserve_space_data, workspace_allocator,
          output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoRnnBackward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<float>& input_data,
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
    DeviceMemory<uint8>* reserve_space_data,
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
          cudnn_input_h_desc, input_h_data, cudnn_input_c_desc, input_c_data,
          params, cudnn_output_desc, output_data, cudnn_output_h_desc,
          output_h_data, cudnn_output_c_desc, output_c_data,
          output_backprop_data, output_h_backprop_data, output_c_backprop_data,
          input_backprop_data, input_h_backprop_data, input_c_backprop_data,
          params_backprop_data, reserve_space_data, workspace_allocator,
          output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoRnnBackward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<double>& input_data,
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
    DeviceMemory<uint8>* reserve_space_data,
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
          cudnn_input_h_desc, input_h_data, cudnn_input_c_desc, input_c_data,
          params, cudnn_output_desc, output_data, cudnn_output_h_desc,
          output_h_data, cudnn_output_c_desc, output_c_data,
          output_backprop_data, output_h_backprop_data, output_c_backprop_data,
          input_backprop_data, input_h_backprop_data, input_c_backprop_data,
          params_backprop_data, reserve_space_data, workspace_allocator,
          output_profile_result),
      /*report_error=*/!output_profile_result);
}

namespace {

// TODO(csigg): Merge a lot of duplicate code below for forward, backward data,
// and backward filter.

port::StatusOr<cudnnConvolutionFwdAlgo_t> GetCudnnConvolutionForwardAlgo(
    const CudnnHandle& cudnn, const CudnnTensorDescriptor& input_nd,
    const CudnnFilterDescriptor& filter, const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd, bool specify_workspace_limit,
    size_t memory_limit_bytes) {
  cudnnConvolutionFwdPreference_t preference =
      specify_workspace_limit ? CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
                              : CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
  cudnnConvolutionFwdAlgo_t algo_to_use;
  RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(
      cudnn.handle(), input_nd.handle(), filter.handle(), conv.handle(),
      output_nd.handle(), preference, memory_limit_bytes, &algo_to_use));
  return algo_to_use;
}

port::StatusOr<cudnnConvolutionBwdDataAlgo_t>
GetCudnnConvolutionBackwardDataAlgo(const CudnnHandle& cudnn,
                                    const CudnnTensorDescriptor& input_nd,
                                    const CudnnFilterDescriptor& filter,
                                    const CudnnConvolutionDescriptor& conv,
                                    const CudnnTensorDescriptor& output_nd,
                                    bool specify_workspace_limit,
                                    size_t memory_limit_bytes) {
  cudnnConvolutionBwdDataPreference_t preference =
      specify_workspace_limit
          ? CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
          : CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
  cudnnConvolutionBwdDataAlgo_t algo_to_use;
  RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn.handle(), filter.handle(), output_nd.handle(), conv.handle(),
      input_nd.handle(), preference, memory_limit_bytes, &algo_to_use));
  return algo_to_use;
}

port::StatusOr<cudnnConvolutionBwdFilterAlgo_t>
GetCudnnConvolutionBackwardFilterAlgo(const CudnnHandle& cudnn,
                                      const CudnnTensorDescriptor& input_nd,
                                      const CudnnFilterDescriptor& filter,
                                      const CudnnConvolutionDescriptor& conv,
                                      const CudnnTensorDescriptor& output_nd,
                                      bool specify_workspace_limit,
                                      size_t memory_limit_bytes) {
  cudnnConvolutionBwdFilterPreference_t preference =
      specify_workspace_limit
          ? CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
          : CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
  cudnnConvolutionBwdFilterAlgo_t algo_to_use;
  RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn.handle(), input_nd.handle(), output_nd.handle(), conv.handle(),
      filter.handle(), preference, memory_limit_bytes, &algo_to_use));
  return algo_to_use;
}

port::StatusOr<DeviceMemory<uint8>> AllocateCudnnConvolutionForwardWorkspace(
    Stream* stream, const CudnnHandle& cudnn,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd, dnn::AlgorithmDesc* algorithm_desc,
    ScratchAllocator* scratch_allocator) {
  // TODO(csigg): This has side effects on the convolution descriptor. It is
  // functionally correct because the convolution is run with the algorithm of
  // the last call to this function, but should be fixed anyway.
  conv.set_use_tensor_op_math(algorithm_desc->tensor_ops_enabled());

  // Query the size of the workspace and allocate it.
  size_t size_in_bytes;
  RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn.handle(),
      /*xDesc=*/input_nd.handle(),
      /*wDesc=*/filter.handle(), /*convDesc=*/conv.handle(),
      /*yDesc=*/output_nd.handle(), /*algo=*/ToConvForwardAlgo(*algorithm_desc),
      /*sizeInBytes=*/&size_in_bytes));

  if (TF_PREDICT_FALSE(!algorithm_desc)) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "No AlgorithmDesc provided");
  }
  algorithm_desc->set_scratch_size(size_in_bytes);
  int64 size_in_bytes_int64 = size_in_bytes;

  if (TF_PREDICT_FALSE(size_in_bytes_int64 < 0)) {
    return port::Status(
        port::error::INTERNAL,
        "cudnnGetConvolutionForwardWorkspaceSize() returned "
        "negative sizeInBytes value. This could be a cudnn bug.");
  }

  if (size_in_bytes_int64 == 0) {
    return DeviceMemory<uint8>();
  }

  if (TF_PREDICT_FALSE(!scratch_allocator)) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "No scratch allocator provided");
  }

  return scratch_allocator->AllocateBytes(stream, size_in_bytes);
}

port::StatusOr<DeviceMemory<uint8>>
AllocateCudnnConvolutionBackwardDataWorkspace(
    Stream* stream, const CudnnHandle& cudnn,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd, dnn::AlgorithmDesc* algorithm_desc,
    ScratchAllocator* scratch_allocator) {
  // TODO(csigg): This has side effects on the convolution descriptor. It is
  // functionally correct because the convolution is run with the algorithm of
  // the last call to this function, but should be fixed anyway.
  conv.set_use_tensor_op_math(algorithm_desc->tensor_ops_enabled());

  // Query the size of the workspace and allocate it.
  size_t size_in_bytes;
  RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn.handle(),
      /*wDesc=*/filter.handle(),
      /*dyDesc=*/output_nd.handle(),
      /*convDesc=*/conv.handle(),
      /*dxDesc=*/input_nd.handle(),
      /*algo=*/ToConvBackwardDataAlgo(*algorithm_desc),
      /*sizeInBytes=*/&size_in_bytes));

  if (TF_PREDICT_FALSE(!algorithm_desc)) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "No AlgorithmDesc provided");
  }
  algorithm_desc->set_scratch_size(size_in_bytes);
  int64 size_in_bytes_int64 = size_in_bytes;

  if (TF_PREDICT_FALSE(size_in_bytes_int64 < 0)) {
    return port::Status(
        port::error::INTERNAL,
        "cudnnGetConvolutionBackwardDataWorkspaceSize() returned "
        "negative sizeInBytes value. This could be a cudnn bug.");
  }

  if (size_in_bytes_int64 == 0) {
    return DeviceMemory<uint8>();
  }

  if (TF_PREDICT_FALSE(!scratch_allocator)) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "No scratch allocator provided");
  }

  return scratch_allocator->AllocateBytes(stream, size_in_bytes);
}

port::StatusOr<DeviceMemory<uint8>>
AllocateCudnnConvolutionBackwardFilterWorkspace(
    Stream* stream, const CudnnHandle& cudnn,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd, dnn::AlgorithmDesc* algorithm_desc,
    ScratchAllocator* scratch_allocator) {
  // TODO(csigg): This has side effects on the convolution descriptor. It is
  // functionally correct because the convolution is run with the algorithm of
  // the last call to this function, but should be fixed anyway.
  conv.set_use_tensor_op_math(algorithm_desc->tensor_ops_enabled());

  // Query the size of the workspace and allocate it.
  size_t size_in_bytes;
  RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn.handle(),
      /*xDesc=*/input_nd.handle(),
      /*dyDesc=*/output_nd.handle(),
      /*convDesc=*/conv.handle(),
      /*gradDesc=*/filter.handle(),
      /*algo=*/ToConvBackwardFilterAlgo(*algorithm_desc),
      /*sizeInBytes=*/&size_in_bytes));

  if (TF_PREDICT_FALSE(!algorithm_desc)) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "No AlgorithmDesc provided");
  }
  algorithm_desc->set_scratch_size(size_in_bytes);
  int64 size_in_bytes_int64 = size_in_bytes;

  if (TF_PREDICT_FALSE(size_in_bytes_int64 < 0)) {
    return port::Status(
        port::error::INTERNAL,
        "cudnnGetConvolutionBackwardFilterWorkspaceSize() returned "
        "negative sizeInBytes value. This could be a cudnn bug.");
  }

  if (size_in_bytes_int64 == 0) {
    return DeviceMemory<uint8>();
  }

  if (TF_PREDICT_FALSE(!scratch_allocator)) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "No scratch allocator provided");
  }

  return scratch_allocator->AllocateBytes(stream, size_in_bytes);
}

port::StatusOr<dnn::AlgorithmDesc> GetCudnnConvolutionForwardAlgorithm(
    Stream* stream, const CudnnHandle& cudnn,
    const dnn::AlgorithmConfig& algorithm_config,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd, ScratchAllocator* scratch_allocator,
    DeviceMemory<uint8>* scratch) {
  dnn::AlgorithmDesc algo_desc = algorithm_config.algorithm();
  if (algorithm_config.algorithm().is_default()) {
    // Pick fastest algorithm within memory limit according to cuDNN's
    // heuristics.
    bool specify_workspace_limit = scratch_allocator != nullptr;
    auto memory_limit_bytes =
        specify_workspace_limit
            ? std::max(scratch_allocator->GetMemoryLimitInBytes(stream), 0ll)
            : 0ll;
    SE_ASSIGN_OR_RETURN(cudnnConvolutionFwdAlgo_t algo,
                        GetCudnnConvolutionForwardAlgo(
                            cudnn, input_nd, filter, conv, output_nd,
                            specify_workspace_limit, memory_limit_bytes));
    algo_desc = dnn::AlgorithmDesc(
        algo, algorithm_config.algorithm().tensor_ops_enabled());
  }

  auto scratch_or = AllocateCudnnConvolutionForwardWorkspace(
      stream, cudnn, input_nd, filter, conv, output_nd, &algo_desc,
      scratch_allocator);

  if (scratch_or.ok()) {
    *scratch = scratch_or.ValueOrDie();
    return algo_desc;
  }

  // Failed to allocate workspace for the first algorithm, fall back to the
  // no_scratch algorithm.
  if (algorithm_config.algorithm_no_scratch().is_default()) {
    return port::Status(
        port::error::INVALID_ARGUMENT,
        "The primary convolution algorithm failed memory allocation, "
        "while a secondary algorithm is not provided.");
  }

  algo_desc = algorithm_config.algorithm_no_scratch();
  SE_ASSIGN_OR_RETURN(*scratch, AllocateCudnnConvolutionForwardWorkspace(
                                    stream, cudnn, input_nd, filter, conv,
                                    output_nd, &algo_desc, scratch_allocator));
  return algo_desc;
}

port::StatusOr<dnn::AlgorithmDesc> GetCudnnConvolutionBackwardDataAlgorithm(
    Stream* stream, const CudnnHandle& cudnn,
    const dnn::AlgorithmConfig& algorithm_config,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd, ScratchAllocator* scratch_allocator,
    DeviceMemory<uint8>* scratch) {
  dnn::AlgorithmDesc algo_desc = algorithm_config.algorithm();
  if (algorithm_config.algorithm().is_default()) {
    // Pick fastest algorithm within memory limit according to cuDNN's
    // heuristics.
    bool specify_workspace_limit = scratch_allocator != nullptr;
    auto memory_limit_bytes =
        specify_workspace_limit
            ? std::max(scratch_allocator->GetMemoryLimitInBytes(stream), 0ll)
            : 0ll;
    SE_ASSIGN_OR_RETURN(cudnnConvolutionBwdDataAlgo_t algo,
                        GetCudnnConvolutionBackwardDataAlgo(
                            cudnn, input_nd, filter, conv, output_nd,
                            specify_workspace_limit, memory_limit_bytes));
    algo_desc = dnn::AlgorithmDesc(
        algo, algorithm_config.algorithm().tensor_ops_enabled());
  }

  auto scratch_or = AllocateCudnnConvolutionBackwardDataWorkspace(
      stream, cudnn, input_nd, filter, conv, output_nd, &algo_desc,
      scratch_allocator);

  if (scratch_or.ok()) {
    *scratch = scratch_or.ValueOrDie();
    return algo_desc;
  }

  // Failed to allocate workspace for the first algorithm, fall back to the
  // no_scratch algorithm.
  if (algorithm_config.algorithm_no_scratch().is_default()) {
    return port::Status(
        port::error::INVALID_ARGUMENT,
        "The primary convolution algorithm failed memory allocation, "
        "while a secondary algorithm is not provided.");
  }

  algo_desc = algorithm_config.algorithm_no_scratch();
  SE_ASSIGN_OR_RETURN(*scratch, AllocateCudnnConvolutionBackwardDataWorkspace(
                                    stream, cudnn, input_nd, filter, conv,
                                    output_nd, &algo_desc, scratch_allocator));
  return algo_desc;
}

port::StatusOr<dnn::AlgorithmDesc> GetCudnnConvolutionBackwardFilterAlgorithm(
    Stream* stream, const CudnnHandle& cudnn,
    const dnn::AlgorithmConfig& algorithm_config,
    const CudnnTensorDescriptor& input_nd, const CudnnFilterDescriptor& filter,
    const CudnnConvolutionDescriptor& conv,
    const CudnnTensorDescriptor& output_nd, ScratchAllocator* scratch_allocator,
    DeviceMemory<uint8>* scratch) {
  dnn::AlgorithmDesc algo_desc = algorithm_config.algorithm();
  if (algorithm_config.algorithm().is_default()) {
    // Pick fastest algorithm within memory limit according to cuDNN's
    // heuristics.
    bool specify_workspace_limit = scratch_allocator != nullptr;
    auto memory_limit_bytes =
        specify_workspace_limit
            ? std::max(scratch_allocator->GetMemoryLimitInBytes(stream), 0ll)
            : 0ll;
    SE_ASSIGN_OR_RETURN(cudnnConvolutionBwdFilterAlgo_t algo,
                        GetCudnnConvolutionBackwardFilterAlgo(
                            cudnn, input_nd, filter, conv, output_nd,
                            specify_workspace_limit, memory_limit_bytes));
    algo_desc = dnn::AlgorithmDesc(
        algo, algorithm_config.algorithm().tensor_ops_enabled());
  }

  auto scratch_or = AllocateCudnnConvolutionBackwardFilterWorkspace(
      stream, cudnn, input_nd, filter, conv, output_nd, &algo_desc,
      scratch_allocator);

  if (scratch_or.ok()) {
    *scratch = scratch_or.ValueOrDie();
    return algo_desc;
  }

  // Failed to allocate workspace for the first algorithm, fall back to the
  // no_scratch algorithm.
  if (algorithm_config.algorithm_no_scratch().is_default()) {
    return port::Status(
        port::error::INVALID_ARGUMENT,
        "The primary convolution algorithm failed memory allocation, "
        "while a secondary algorithm is not provided.");
  }

  algo_desc = algorithm_config.algorithm_no_scratch();
  SE_ASSIGN_OR_RETURN(*scratch, AllocateCudnnConvolutionBackwardFilterWorkspace(
                                    stream, cudnn, input_nd, filter, conv,
                                    output_nd, &algo_desc, scratch_allocator));
  return algo_desc;
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
      port::StringPiece tf_env_var_val_str(tf_env_var_val);
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
  static constexpr bool kDefaultFlag = CUDNN_VERSION >= 7000;
};

// A helper struct to decide whether to enable the WINOGRAD_NONFUSED algorithms.
// By default it is turned on, users can explicitly disable them through an
// env-var "TF_ENABLE_WINOGRAD_NONFUSED=0".
// https://github.com/tensorflow/tensorflow/pull/4901
struct WinogradNonfused {
  static constexpr const char* kName = "TF_ENABLE_WINOGRAD_NONFUSED";
  // NVIDIA has fixed winograd nonfused bug for cudnn v>=7. For older versions,
  // we have a workaround.
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

// A group of helper functions to return the internal compute type for
// convolutions in cudnn.
template <typename T>
cudnnDataType_t GetConvComputeType() {
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t GetConvComputeType<Eigen::half>() {
  if (CudnnEnvVar<ConvDoFP32ComputationFP16Input>::IsEnabled()) {
    return CUDNN_DATA_FLOAT;
  } else {
    return CUDNN_DATA_HALF;
  }
}

template <>
cudnnDataType_t GetConvComputeType<double>() {
  return CUDNN_DATA_DOUBLE;
}

// A helper struct to decide whether to use FP32 as the internal compute type
// for rnn when the input data type is FP16. At present it is turned off,
// users can explicitly control them through an env-var
// TF_FP16_RNN_USE_FP32_COMPUTE.
// After the TODO below is fixed, users should almost always use fp32 compute
// type for training. Using fp16 might suffer suboptimal accuracy due to loss
// in precision.
struct RnnDoFP32ComputationFP16Input {
  static constexpr const char* kName = "TF_FP16_RNN_USE_FP32_COMPUTE";
  // TODO(jamesqin): b/78182362 flip to true when cudnn 7.1.4 fixes the bug.
  // Before cudnn 7.1.4 RNN are always done in fp32, no matter what math
  // precision is set.
  // Set it temporary to false s.t. no error is raised when using fp16 inputs,
  // fp32 math precision.
  static constexpr bool kDefaultFlag = false;
};

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

// Determines whether we can safely perform a winograd non-fused convolution for
// the given input and output shapes.  This works around b/68264959, an integer
// overflow in cuDNNv5 and cuDNNv6.
#if CUDNN_VERSION >= 7000
bool ShouldIncludeWinogradNonfusedAlgo(const dnn::BatchDescriptor&,
                                       const dnn::BatchDescriptor&) {
  return true;
}
#else
bool ShouldIncludeWinogradNonfusedAlgo(
    const dnn::BatchDescriptor& input_desc,
    const dnn::BatchDescriptor& output_desc) {
  int64 batch = input_desc.count();
  int64 in_depths = input_desc.feature_map_count();
  int64 in_rows = input_desc.height();
  int64 in_cols = input_desc.ndims() == 1 ? 1 : input_desc.width();
  int64 out_depths = output_desc.feature_map_count();

  int64 total_size = port::MathUtil::CeilOfRatio(batch, int64{16}) *
                     std::max(in_depths, out_depths) * in_cols * in_rows *
                     sizeof(float);

  const int64 threshold = 1L << 31;
  return total_size < threshold;
}
#endif

}  // namespace

template <class T>
port::Status CudnnSupport::DoConvolveImpl(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<T>& input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<T>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::BatchDescriptor& output_descriptor, DeviceMemory<T>* output_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  cudnnDataType_t cudnn_type = GetCudnnDataType<T>();
  CudnnTensorDescriptor input_nd(input_descriptor, cudnn_type);
  CudnnTensorDescriptor output_nd(output_descriptor, cudnn_type);
  CudnnFilterDescriptor filter(filter_descriptor, cudnn_type);
  CudnnConvolutionDescriptor conv(convolution_descriptor,
                                  GetConvComputeType<T>());

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  // Alpha is the scaling factor for input.
  float falpha = 1.0;
  double dalpha = 1.0;
  void* alpha = cudnn_type == CUDNN_DATA_DOUBLE ? static_cast<void*>(&dalpha)
                                                : static_cast<void*>(&falpha);
  // Beta is the scaling factor for output.
  float fbeta = 0.0;
  double dbeta = 0.0;
  void* beta = cudnn_type == CUDNN_DATA_DOUBLE ? static_cast<void*>(&dbeta)
                                               : static_cast<void*>(&fbeta);

  const bool is_profiling = output_profile_result != nullptr;

  DeviceMemory<uint8> scratch;
  SE_ASSIGN_OR_RETURN(dnn::AlgorithmDesc algo_desc,
                      GetCudnnConvolutionForwardAlgorithm(
                          stream, cudnn, algorithm_config, input_nd, filter,
                          conv, output_nd, scratch_allocator, &scratch));

  std::unique_ptr<CUDATimer, TimerDeleter> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));  // NOLINT
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Init() || !timer->Start(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to start timer");
    }
  }

  // Report an error if we might be hitting a cuDNN bug that accesses illegal
  // memory. See nvbugs/2138754, b/80018418.
  SE_RETURN_IF_ERROR([&] {
    if (algo_desc.algo_id() != CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) {
      return port::Status::OK();
    }
    if (input_descriptor.ndims() < 3) {
      return port::Status::OK();
    }
    // Checks that a*b is within the valid range (as provided by NVIDIA).
    auto check_sizes = [](size_t a, size_t b) {
      if ((a * b * 4608 - 1) >> 31 == 0) {
        return port::Status::OK();
      }
      return port::Status(
          port::error::FAILED_PRECONDITION,
          "This configuration potentially accesses illegal memory.");
    };
    SE_RETURN_IF_ERROR(check_sizes(input_descriptor.feature_map_count(),
                                   output_descriptor.feature_map_count()));
    SE_RETURN_IF_ERROR(check_sizes(input_descriptor.count(),
                                   input_descriptor.feature_map_count()));
    SE_RETURN_IF_ERROR(check_sizes(input_descriptor.count(),
                                   output_descriptor.feature_map_count()));
    return port::Status::OK();
  }());

  if (algo_desc.algo_id() == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED &&
      !ShouldIncludeWinogradNonfusedAlgo(input_descriptor, output_descriptor)) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "This configuration has potential integer overflow in "
                        "cuDNNv5 and cuDNNv6. See b/68264959.");
  }

  RETURN_IF_CUDNN_ERROR(cudnnConvolutionForward(
      cudnn.handle(),
      /*alpha=*/alpha, /*srcDesc=*/input_nd.handle(),
      /*srcData=*/input_data.opaque(), /*filterDesc=*/filter.handle(),
      /*filterData=*/filter_data.opaque(), /*convDesc=*/conv.handle(),
      /*algo=*/ToConvForwardAlgo(algo_desc), /*workSpace=*/scratch.opaque(),
      /*workSpaceSizeInBytes=*/scratch.size(), /*beta=*/beta,
      /*yDesc=*/output_nd.handle(), /*y=*/output_data->opaque()));

  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to stop timer");
    }
    output_profile_result->set_algorithm(algo_desc);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }

  return port::Status::OK();
}

template <typename AccumulatorType, typename ElementType, typename BiasType,
          typename ScaleType>
port::Status CudnnSupport::DoFusedConvolveImpl(
    Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
    const DeviceMemory<ElementType>& conv_input_data,
    ScaleType conv_input_scale, const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<ElementType>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const DeviceMemory<ElementType>& side_input_data,
    ScaleType side_input_scale, const dnn::BatchDescriptor& bias_descriptor,
    const DeviceMemory<BiasType>& biases, dnn::ActivationMode activation_mode,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<ElementType>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  if (activation_mode != dnn::ActivationMode::kRelu &&
      activation_mode != dnn::ActivationMode::kNone) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "cudnnConvolutionBiasActivationForward() only supports "
                        "Relu or None activation.");
  }

  CudnnTensorDescriptor conv_input_nd(
      conv_input_descriptor,
      GetCudnnDataType<ElementType>(conv_input_descriptor.layout()));
  CudnnTensorDescriptor output_nd(
      output_descriptor,
      GetCudnnDataType<ElementType>(conv_input_descriptor.layout()));
  CudnnFilterDescriptor filter(
      filter_descriptor,
      GetCudnnDataType<ElementType>(conv_input_descriptor.layout()));
  CudnnTensorDescriptor bias_nd(bias_descriptor, GetCudnnDataType<BiasType>());
  CudnnConvolutionDescriptor conv(convolution_descriptor,
                                  GetCudnnDataType<AccumulatorType>());

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  const bool is_profiling = output_profile_result != nullptr;

  DeviceMemory<uint8> scratch;
  SE_ASSIGN_OR_RETURN(
      dnn::AlgorithmDesc algo_desc,
      GetCudnnConvolutionForwardAlgorithm(
          stream, cudnn, algorithm_config, conv_input_nd, filter, conv,
          output_nd, scratch_allocator, &scratch));

  std::unique_ptr<CUDATimer, TimerDeleter> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));  // NOLINT
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Init() || !timer->Start(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to start timer");
    }
  }
  // CUDNN v6 only supports CUDNN_NOT_PROPAGATE_NAN as the reluNanOpt for
  // activation descriptor. Note that this will change the nan propagation
  // behavior from separate conv, bias, and relu (which by default is
  // CUDNN_PROPAGATE_NAN.
  CudnnActivationDescriptor activation_desc(
      activation_mode, CUDNN_NOT_PROPAGATE_NAN, output_descriptor.value_max());
  auto side_input_data_ptr = (side_input_scale == 0) ? output_data->opaque()
                                                     : side_input_data.opaque();

  VLOG(2) << "\nconv_input_scale = " << conv_input_scale
          << "\nconv_input_nd.handle() = " << conv_input_nd.handle()
          << "\nconv_input_data.opaque() = " << conv_input_data.opaque()
          << "\nfilter.handle() = " << filter.handle()
          << "\nfilter_data.opaque() = " << filter_data.opaque()
          << "\nconv.handle() = " << conv.handle()
          << "\nalgo = " << algo_desc.algo_id()
          << "\nscratch.opaque() = " << scratch.opaque()
          << "\nscratch.size() = " << scratch.size()
          << "\nside_input_scale = " << side_input_scale
          << "\noutput_nd.handle() = " << output_nd.handle()
          << "\nside_input_data_ptr = " << side_input_data_ptr
          << "\nbias_nd.handle() = " << bias_nd.handle()
          << "\nbiases.opaque() = " << biases.opaque()
          << "\nactivation_desc.handle() = " << activation_desc.handle()
          << "\noutput_nd.handle() = " << output_nd.handle()
          << "\noutput_data->opaque() = " << output_data->opaque();

  if (algo_desc.algo_id() == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED &&
      !ShouldIncludeWinogradNonfusedAlgo(conv_input_descriptor,
                                         output_descriptor)) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "This configuration has potential integer overflow in "
                        "cuDNNv5 and cuDNNv6. See around b/68264959.");
  }

  RETURN_IF_CUDNN_ERROR(cudnnConvolutionBiasActivationForward(
      cudnn.handle(),
      /*alpha1=*/&conv_input_scale,
      /*srcDesc=*/conv_input_nd.handle(), /*srcData=*/conv_input_data.opaque(),
      /*filterDesc=*/filter.handle(), /*filterData=*/filter_data.opaque(),
      /*convDesc=*/conv.handle(), ToConvForwardAlgo(algo_desc),
      /*workSpace=*/scratch.opaque(),
      /*workSpaceSizeInBytes=*/scratch.size(), /*alpha2=*/&side_input_scale,
      /*zDesc=*/output_nd.handle(), /*z=*/side_input_data_ptr,
      /*biasDesc=*/bias_nd.handle(), /*bias=*/biases.opaque(),
      /*activationDesc=*/activation_desc.handle(),
      /*yDesc=*/output_nd.handle(), /*y=*/output_data->opaque()));

  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to stop timer");
    }
    output_profile_result->set_algorithm(algo_desc);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }

  return port::Status::OK();
}

bool CudnnSupport::GetConvolveAlgorithms(
    bool with_winograd_nonfused, int cc_major, int cc_minor,
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  std::vector<dnn::AlgorithmDesc::Index> algo_types = {
    // clang-format off
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    // clang-format on
  };
  if (CudnnEnvVar<FftTilingForward>::IsEnabled()) {
    algo_types.push_back(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING);
  }
  if (CudnnEnvVar<WinogradNonfused>::IsEnabled() && with_winograd_nonfused) {
    algo_types.push_back(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
  }

  out_algorithms->clear();
  for (auto i : algo_types) {
    out_algorithms->push_back({i, /*use_tensor_ops=*/false});
    if (cc_major >= 7 && CUDNN_VERSION >= 7000 && TensorOpMathEnabled()) {
      out_algorithms->push_back({i, /*use_tensor_ops=*/true});
    }
  }
  return true;
}

bool CudnnSupport::GetRnnAlgorithms(
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
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
#if CUDNN_VERSION >= 7100
    if (RnnTensorOpMathEnabled()) {
      out_algorithms->push_back({i, /*use_tensor_ops=*/true});
    }
#endif
  }
  return true;
}

bool CudnnSupport::GetConvolveBackwardDataAlgorithms(
    bool with_winograd_nonfused, int cc_major, int cc_minor,
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  std::vector<dnn::AlgorithmDesc::Index> algo_types = {
      // clang-format off
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
      // clang-format on
  };
  if (CudnnEnvVar<WinogradNonfused>::IsEnabled() && with_winograd_nonfused) {
    algo_types.push_back(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);
  }

  out_algorithms->clear();
  for (auto i : algo_types) {
    out_algorithms->push_back({i, /*use_tensor_ops=*/false});
    if (cc_major >= 7 && CUDNN_VERSION >= 7000 && TensorOpMathEnabled()) {
      out_algorithms->push_back({i, /*use_tensor_ops=*/true});
    }
  }
  return true;
}

bool CudnnSupport::GetConvolveBackwardFilterAlgorithms(
    bool with_winograd_nonfused, int cc_major, int cc_minor,
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  std::vector<dnn::AlgorithmDesc::Index> algo_types = {
      // clang-format off
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
      // Based on cudnn.h, the following is not implemented.
      // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,

      // Produces incorrect results for some shapes. Disabled for now, see
      // NVIDIA bug 2072856. TODO(csigg): Only disable for subset of shapes.
      // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
      // clang-format on
  };
  if (CudnnEnvVar<WinogradNonfused>::IsEnabled() && with_winograd_nonfused) {
    algo_types.push_back(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED);
  }

  out_algorithms->clear();
  for (auto i : algo_types) {
    out_algorithms->push_back({i, /*use_tensor_ops=*/false});
    if (cc_major >= 7 && CUDNN_VERSION >= 7000 && TensorOpMathEnabled()) {
      out_algorithms->push_back({i, /*use_tensor_ops=*/true});
    }
  }
  return true;
}

bool CudnnSupport::DoBatchNormalizationForward(
    Stream* stream, const DeviceMemory<float>& x,
    const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
    const DeviceMemory<float>& estimated_mean,
    const DeviceMemory<float>& estimated_variance,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<float>* y, DeviceMemory<float>* batch_mean,
    DeviceMemory<float>* batch_var, DeviceMemory<float>* saved_mean,
    DeviceMemory<float>* saved_inv_var, bool is_training,
    std::function<const DeviceMemory<float>&()> var_to_inv_var,
    std::function<void()> inv_var_to_var) {
  return IsStatusOk(
      DoBatchNormalizationForwardImpl<float, float>(
          stream, dnn::DataType::kFloat, dnn::DataType::kFloat, x, scale,
          offset, estimated_mean, estimated_variance, x_desc, scale_offset_desc,
          epsilon, y, batch_mean, batch_var, saved_mean, saved_inv_var,
          is_training, std::move(var_to_inv_var), std::move(inv_var_to_var)),
      /*report_error=*/true);
}

bool CudnnSupport::DoBatchNormalizationForward(
    Stream* stream, const DeviceMemory<Eigen::half>& x,
    const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
    const DeviceMemory<float>& estimated_mean,
    const DeviceMemory<float>& estimated_variance,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<Eigen::half>* y, DeviceMemory<float>* batch_mean,
    DeviceMemory<float>* batch_var, DeviceMemory<float>* saved_mean,
    DeviceMemory<float>* saved_inv_var, bool is_training,
    std::function<const DeviceMemory<float>&()> var_to_inv_var,
    std::function<void()> inv_var_to_var) {
  return IsStatusOk(
      DoBatchNormalizationForwardImpl<Eigen::half, float>(
          stream, dnn::DataType::kHalf, dnn::DataType::kFloat, x, scale, offset,
          estimated_mean, estimated_variance, x_desc, scale_offset_desc,
          epsilon, y, batch_mean, batch_var, saved_mean, saved_inv_var,
          is_training, std::move(var_to_inv_var), std::move(inv_var_to_var)),
      /*report_error=*/true);
}

template <class T, class U>
port::Status CudnnSupport::DoBatchNormalizationForwardImpl(
    Stream* stream, dnn::DataType input_data_type,
    dnn::DataType scale_data_type, const DeviceMemory<T>& x,
    const DeviceMemory<U>& scale, const DeviceMemory<U>& offset,
    const DeviceMemory<U>& estimated_mean,
    const DeviceMemory<U>& estimated_variance,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<T>* y, DeviceMemory<U>* batch_mean, DeviceMemory<U>* batch_var,
    DeviceMemory<U>* saved_mean, DeviceMemory<U>* saved_inv_var,
    bool is_training, std::function<const DeviceMemory<U>&()> var_to_inv_var,
    std::function<void()> inv_var_to_var) {
  CudnnTensorDescriptor x_descriptor(x_desc, ToCudnnDataType(input_data_type));
  CudnnTensorDescriptor scale_offset_descriptor(
      scale_offset_desc, ToCudnnDataType(scale_data_type));
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7000
  if (BatchnormSpatialPersistentEnabled() && is_training) {
    mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  }
#endif
  float one = 1.0;
  float zero = 0.0;
  auto cudnn = cudnn_->GetHandle(parent_, stream);

  if (is_training) {
    CHECK_EQ(batch_mean->is_null(), batch_var->is_null())
        << "batch_mean and batch_var must both be null or both be non-null";

    void* batch_mean_opaque;
    void* batch_var_opaque;
    if (!batch_mean->is_null() && !batch_var->is_null()) {
      stream->ThenMemZero(batch_mean, batch_mean->size());
      stream->ThenMemZero(batch_var, batch_var->size());
      batch_mean_opaque = batch_mean->opaque();
      batch_var_opaque = batch_var->opaque();
    } else {
      batch_mean_opaque = nullptr;
      batch_var_opaque = nullptr;
    }

    RETURN_IF_CUDNN_ERROR(cudnnBatchNormalizationForwardTraining(
        cudnn.handle(), mode, &one, &zero, x_descriptor.handle(), x.opaque(),
        x_descriptor.handle(), y->opaque(), scale_offset_descriptor.handle(),
        scale.opaque(), offset.opaque(), 1.0, batch_mean_opaque,
        batch_var_opaque, epsilon, saved_mean->opaque(),
        saved_inv_var->opaque()));
  } else {
    const void* maybe_inv_var = estimated_variance.opaque();
    RETURN_IF_CUDNN_ERROR(cudnnBatchNormalizationForwardInference(
        cudnn.handle(), mode, &one, &zero, x_descriptor.handle(), x.opaque(),
        x_descriptor.handle(), y->opaque(), scale_offset_descriptor.handle(),
        scale.opaque(), offset.opaque(), estimated_mean.opaque(), maybe_inv_var,
        epsilon));
  }
  return port::Status::OK();
}

bool CudnnSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<float>& y_backprop,
    const DeviceMemory<float>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& mean, const DeviceMemory<float>& inv_var,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<float>* x_backprop, DeviceMemory<float>* scale_backprop,
    DeviceMemory<float>* offset_backprop) {
  return IsStatusOk(DoBatchNormalizationBackwardImpl(
                        stream, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, y_backprop,
                        x, scale, mean, inv_var, x_desc, scale_offset_desc,
                        epsilon, x_backprop, scale_backprop, offset_backprop),
                    /*report_error=*/true);
}

bool CudnnSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<Eigen::half>& y_backprop,
    const DeviceMemory<Eigen::half>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& mean, const DeviceMemory<float>& inv_var,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<Eigen::half>* x_backprop, DeviceMemory<float>* scale_backprop,
    DeviceMemory<float>* offset_backprop) {
  return IsStatusOk(DoBatchNormalizationBackwardImpl(
                        stream, CUDNN_DATA_HALF, CUDNN_DATA_FLOAT, y_backprop,
                        x, scale, mean, inv_var, x_desc, scale_offset_desc,
                        epsilon, x_backprop, scale_backprop, offset_backprop),
                    /*report_error=*/true);
}

template <class T, class U>
port::Status CudnnSupport::DoBatchNormalizationBackwardImpl(
    Stream* stream, int cudnn_input_type, int cudnn_scale_type,
    const DeviceMemory<T>& y_backprop, const DeviceMemory<T>& x,
    const DeviceMemory<U>& scale, const DeviceMemory<U>& mean,
    const DeviceMemory<U>& inv_var, const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<T>* x_backprop, DeviceMemory<U>* scale_backprop,
    DeviceMemory<U>* offset_backprop) {
  CudnnTensorDescriptor x_descriptor(
      x_desc, static_cast<cudnnDataType_t>(cudnn_input_type));
  CudnnTensorDescriptor scale_offset_descriptor(
      scale_offset_desc, static_cast<cudnnDataType_t>(cudnn_scale_type));
  cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7000
  if (BatchnormSpatialPersistentEnabled()) {
    mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  }
#endif
  float one = 1.0;
  float zero = 0.0;

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  RETURN_IF_CUDNN_ERROR(cudnnBatchNormalizationBackward(
      cudnn.handle(), mode, &one, &zero, &one, &zero, x_descriptor.handle(),
      x.opaque(), x_descriptor.handle(), y_backprop.opaque(),
      x_descriptor.handle(), x_backprop->opaque(),
      scale_offset_descriptor.handle(), scale.opaque(),
      scale_backprop->opaque(), offset_backprop->opaque(), epsilon,
      mean.opaque(), inv_var.opaque()));
  return port::Status::OK();
}

bool CudnnSupport::DoConvolve(
    Stream* stream, const dnn::BatchDescriptor& batch_descriptor,
    const DeviceMemory<float>& input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<float>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<float>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoConvolveImpl<float>(
          stream, batch_descriptor, input_data, filter_descriptor, filter_data,
          convolution_descriptor, output_descriptor, output_data,
          scratch_allocator, algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoConvolve(
    Stream* stream, const dnn::BatchDescriptor& batch_descriptor,
    const DeviceMemory<double>& input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<double>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<double>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoConvolveImpl<double>(
          stream, batch_descriptor, input_data, filter_descriptor, filter_data,
          convolution_descriptor, output_descriptor, output_data,
          scratch_allocator, algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoConvolve(
    Stream* stream, const dnn::BatchDescriptor& batch_descriptor,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<Eigen::half>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<Eigen::half>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoConvolveImpl<Eigen::half>(
          stream, batch_descriptor, input_data, filter_descriptor, filter_data,
          convolution_descriptor, output_descriptor, output_data,
          scratch_allocator, algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoFusedConvolve(
    Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
    const DeviceMemory<double>& conv_input_data, double conv_input_scale,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<double>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const DeviceMemory<double>& side_input_data, double side_input_scale,
    const dnn::BatchDescriptor& bias_descriptor,
    const DeviceMemory<double>& biases, dnn::ActivationMode activation_mode,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<double>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoFusedConvolveImpl<double>(
          stream, conv_input_descriptor, conv_input_data, conv_input_scale,
          filter_descriptor, filter_data, convolution_descriptor,
          side_input_data, side_input_scale, bias_descriptor, biases,
          activation_mode, output_descriptor, output_data, scratch_allocator,
          algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoFusedConvolve(
    Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
    const DeviceMemory<float>& conv_input_data, float conv_input_scale,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<float>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const DeviceMemory<float>& side_input_data, float side_input_scale,
    const dnn::BatchDescriptor& bias_descriptor,
    const DeviceMemory<float>& biases, dnn::ActivationMode activation_mode,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<float>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoFusedConvolveImpl<float>(
          stream, conv_input_descriptor, conv_input_data, conv_input_scale,
          filter_descriptor, filter_data, convolution_descriptor,
          side_input_data, side_input_scale, bias_descriptor, biases,
          activation_mode, output_descriptor, output_data, scratch_allocator,
          algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoFusedConvolve(
    Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
    const DeviceMemory<Eigen::half>& conv_input_data, float conv_input_scale,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<Eigen::half>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const DeviceMemory<Eigen::half>& side_input_data, float side_input_scale,
    const dnn::BatchDescriptor& bias_descriptor,
    const DeviceMemory<Eigen::half>& biases,
    dnn::ActivationMode activation_mode,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<Eigen::half>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoFusedConvolveImpl<float>(
          stream, conv_input_descriptor, conv_input_data, conv_input_scale,
          filter_descriptor, filter_data, convolution_descriptor,
          side_input_data, side_input_scale, bias_descriptor, biases,
          activation_mode, output_descriptor, output_data, scratch_allocator,
          algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoFusedConvolve(
    Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
    const DeviceMemory<int8>& conv_input_data, float conv_input_scale,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<int8>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const DeviceMemory<int8>& side_input_data, float side_input_scale,
    const dnn::BatchDescriptor& bias_descriptor,
    const DeviceMemory<float>& biases, dnn::ActivationMode activation_mode,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<int8>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  int cc_major, cc_minor;
  stream->parent()->GetDeviceDescription().cuda_compute_capability(&cc_major,
                                                                   &cc_minor);
  if (cc_major < 6 || (cc_major == 6 && cc_minor < 1)) {
    LOG(WARNING) << "cudnnConvolutionBiasActivationForward() for int8 is only "
                    "supported on GPUs with compute capability 6.1 or later.";
    return false;
  }
  return IsStatusOk(
      DoFusedConvolveImpl<int32>(
          stream, conv_input_descriptor, conv_input_data, conv_input_scale,
          filter_descriptor, filter_data, convolution_descriptor,
          side_input_data, side_input_scale, bias_descriptor, biases,
          activation_mode, output_descriptor, output_data, scratch_allocator,
          algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
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
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnTransformTensor(
        cudnn.handle(), &scale, input_tensor_desc.handle(), input_data.opaque(),
        &beta, output_tensor_desc.handle(), output_data->opaque()));
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

template <class T>
port::Status CudnnSupport::DoConvolveBackwardDataImpl(
    Stream* stream, const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<T>& filter_data,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<T> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::BatchDescriptor& input_descriptor,
    DeviceMemory<T>* backward_input_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  cudnnDataType_t cudnn_type = GetCudnnDataType<T>();
  // Alpha is the scaling factor for input.
  float falpha = 1.0;
  double dalpha = 1.0;
  void* alpha = cudnn_type == CUDNN_DATA_DOUBLE ? static_cast<void*>(&dalpha)
                                                : static_cast<void*>(&falpha);
  // Beta is the scaling factor for output.
  float fbeta = 0.0;
  double dbeta = 0.0;
  void* beta = cudnn_type == CUDNN_DATA_DOUBLE ? static_cast<void*>(&dbeta)
                                               : static_cast<void*>(&fbeta);

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  CudnnTensorDescriptor out_back_nd(output_descriptor, cudnn_type);
  CudnnTensorDescriptor in_back_nd(input_descriptor, cudnn_type);
  CudnnFilterDescriptor filter(filter_descriptor, cudnn_type);
  CudnnConvolutionDescriptor conv(convolution_descriptor,
                                  GetConvComputeType<T>());

  const bool is_profiling = output_profile_result != nullptr;

  DeviceMemory<uint8> scratch;
  SE_ASSIGN_OR_RETURN(dnn::AlgorithmDesc algo_desc,
                      GetCudnnConvolutionBackwardDataAlgorithm(
                          stream, cudnn, algorithm_config, in_back_nd, filter,
                          conv, out_back_nd, scratch_allocator, &scratch));

  std::unique_ptr<CUDATimer, TimerDeleter> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));  // NOLINT
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Init() || !timer->Start(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to start timer");
    }
  }

  if (algo_desc.algo_id() == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED &&
      !ShouldIncludeWinogradNonfusedAlgo(input_descriptor, output_descriptor)) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "This configuration has potential integer overflow in "
                        "cuDNNv5 and cuDNNv6. See b/68264959.");
  }

  // Cudnn 7.1.4 has a bug if the workspace of the following convolution is not
  // zero-initialized, nvbugs/2254619.
  if (CUDNN_VERSION >= 7000 &&
      algorithm_config.algorithm().algo_id() ==
          CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 &&
      cudnn_type == CUDNN_DATA_HALF &&
      algorithm_config.algorithm().tensor_ops_enabled() &&
      input_descriptor.layout() == dnn::DataLayout::kBatchYXDepth &&
      filter_descriptor.layout() == dnn::FilterLayout::kOutputInputYX &&
      output_descriptor.layout() == dnn::DataLayout::kBatchDepthYX &&
      (convolution_descriptor.vertical_filter_stride() > 1 ||
       convolution_descriptor.horizontal_filter_stride() > 1)) {
    stream->ThenMemZero(&scratch, scratch.size());
  }

  RETURN_IF_CUDNN_ERROR(
      cudnnConvolutionBackwardData(cudnn.handle(),
                                   /*alpha=*/alpha,
                                   /*wDesc=*/filter.handle(),
                                   /*w=*/filter_data.opaque(),
                                   /*dyDesc=*/out_back_nd.handle(),
                                   /*dy=*/backward_output_data.opaque(),
                                   /*convDesc=*/conv.handle(),
                                   /*algo=*/ToConvBackwardDataAlgo(algo_desc),
                                   /*workSpace=*/scratch.opaque(),
                                   /*workSpaceSizeInBytes=*/scratch.size(),
                                   /*beta=*/beta,
                                   /*dxDesc=*/in_back_nd.handle(),
                                   /*dx=*/backward_input_data->opaque()));
  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to stop timer");
    }
    output_profile_result->set_algorithm(algo_desc);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }

  return port::Status::OK();
}

bool CudnnSupport::DoConvolveBackwardData(
    Stream* stream, const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<double>& filter_data,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<double> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::BatchDescriptor& input_descriptor,
    DeviceMemory<double>* backward_input_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoConvolveBackwardDataImpl(stream, filter_descriptor, filter_data,
                                 output_descriptor, backward_output_data,
                                 convolution_descriptor, input_descriptor,
                                 backward_input_data, scratch_allocator,
                                 algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoConvolveBackwardData(
    Stream* stream, const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<float>& filter_data,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<float> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::BatchDescriptor& input_descriptor,
    DeviceMemory<float>* backward_input_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoConvolveBackwardDataImpl(stream, filter_descriptor, filter_data,
                                 output_descriptor, backward_output_data,
                                 convolution_descriptor, input_descriptor,
                                 backward_input_data, scratch_allocator,
                                 algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoConvolveBackwardData(
    Stream* stream, const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<Eigen::half>& filter_data,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<Eigen::half> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::BatchDescriptor& input_descriptor,
    DeviceMemory<Eigen::half>* backward_input_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoConvolveBackwardDataImpl(stream, filter_descriptor, filter_data,
                                 output_descriptor, backward_output_data,
                                 convolution_descriptor, input_descriptor,
                                 backward_input_data, scratch_allocator,
                                 algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

template <class T>
port::Status CudnnSupport::DoConvolveBackwardFilterImpl(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<T>& input_data,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<T> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemory<T>* backward_filter_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  cudnnDataType_t cudnn_type = GetCudnnDataType<T>();
  // Alpha is the scaling factor for input.
  float falpha = 1.0;
  double dalpha = 1.0;
  void* alpha = cudnn_type == CUDNN_DATA_DOUBLE ? static_cast<void*>(&dalpha)
                                                : static_cast<void*>(&falpha);
  // Beta is the scaling factor for output.
  float fbeta = 0.0;
  double dbeta = 0.0;
  void* beta = cudnn_type == CUDNN_DATA_DOUBLE ? static_cast<void*>(&dbeta)
                                               : static_cast<void*>(&fbeta);

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  CudnnTensorDescriptor out_back_nd(output_descriptor, cudnn_type);
  CudnnTensorDescriptor input_nd(input_descriptor, cudnn_type);
  CudnnFilterDescriptor filter(filter_descriptor, cudnn_type);
  CudnnConvolutionDescriptor conv(convolution_descriptor,
                                  GetConvComputeType<T>());

  const bool is_profiling = output_profile_result != nullptr;

  DeviceMemory<uint8> scratch;
  SE_ASSIGN_OR_RETURN(dnn::AlgorithmDesc algo_desc,
                      GetCudnnConvolutionBackwardFilterAlgorithm(
                          stream, cudnn, algorithm_config, input_nd, filter,
                          conv, out_back_nd, scratch_allocator, &scratch));

  std::unique_ptr<CUDATimer, TimerDeleter> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));  // NOLINT
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Init() || !timer->Start(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to start timer");
    }
  }

  // Report an error if we might be hitting a cuDNN bug that produces incorrect
  // results. See nvbugs/2072856
  SE_RETURN_IF_ERROR([&] {
    if (algo_desc.algo_id() != CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING) {
      return port::Status::OK();
    }
    if (output_descriptor.height() > 1 && output_descriptor.width() > 1) {
      return port::Status::OK();
    }
    int convolution_size = output_descriptor.height() > 1
                               ? filter_descriptor.input_filter_height()
                               : filter_descriptor.input_filter_width();
    if (convolution_size <= 32) {
      return port::Status::OK();
    }
    cudnnConvolutionMode_t convolution_mode;
    cudnnDataType_t compute_type;
    RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionNdDescriptor(
        conv.handle(), 0, nullptr, nullptr, nullptr, nullptr, &convolution_mode,
        &compute_type));
    if (convolution_mode != CUDNN_CONVOLUTION) {
      return port::Status::OK();
    }
    return port::Status(
        port::error::FAILED_PRECONDITION,
        "This configuration potentially produces incorrect results.");
  }());

  if (algo_desc.algo_id() == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED &&
      !ShouldIncludeWinogradNonfusedAlgo(input_descriptor, output_descriptor)) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "This configuration has potential integer overflow in "
                        "cuDNNv5 and cuDNNv6. See b/68264959.");
  }

  // Zero out the result buffer for strided conv backward filter for NHWC
  // layouts. cuDNN 7.1.4 and 7.2 has non-determinisic bug if the buffer is not
  // zeroed.
  //
  // This wrong result caused by the bug is very flaky. It needs to be run for
  // up to 20 times to produce a mismatch.
  //
  // TODO(timshen): add a nvbugs link.
  if (CUDNN_VERSION >= 7100 &&
      algorithm_config.algorithm().algo_id() ==
          CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 &&
      cudnn_type == CUDNN_DATA_HALF &&
      input_descriptor.layout() == dnn::DataLayout::kBatchYXDepth &&
      filter_descriptor.layout() == dnn::FilterLayout::kOutputYXInput &&
      output_descriptor.layout() == dnn::DataLayout::kBatchYXDepth &&
      (convolution_descriptor.vertical_filter_stride() > 1 ||
       convolution_descriptor.horizontal_filter_stride() > 1)) {
    stream->ThenMemZero(backward_filter_data, backward_filter_data->size());
  }

  RETURN_IF_CUDNN_ERROR(cudnnConvolutionBackwardFilter(
      cudnn.handle(),
      /*alpha=*/alpha,
      /*srcDesc=*/input_nd.handle(),
      /*srcData=*/input_data.opaque(),
      /*diffDesc=*/out_back_nd.handle(),
      /*diffData=*/backward_output_data.opaque(),
      /*convDesc=*/conv.handle(),
      /*algo=*/ToConvBackwardFilterAlgo(algo_desc),
      /*workSpace=*/scratch.opaque(),
      /*workSpaceSizeInBytes=*/scratch.size(),
      /*beta=*/beta,
      /*gradDesc=*/filter.handle(),
      /*dw=*/backward_filter_data->opaque()));
  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      return port::Status(port::error::INTERNAL, "Failed to stop timer");
    }
    output_profile_result->set_algorithm(algo_desc);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }

  return port::Status::OK();
}

bool CudnnSupport::DoConvolveBackwardFilter(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<double>& input_data,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<double> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemory<double>* backward_filter_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoConvolveBackwardFilterImpl(stream, input_descriptor, input_data,
                                   output_descriptor, backward_output_data,
                                   convolution_descriptor, filter_descriptor,
                                   backward_filter_data, scratch_allocator,
                                   algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoConvolveBackwardFilter(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<float> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemory<float>* backward_filter_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoConvolveBackwardFilterImpl(stream, input_descriptor, input_data,
                                   output_descriptor, backward_output_data,
                                   convolution_descriptor, filter_descriptor,
                                   backward_filter_data, scratch_allocator,
                                   algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool CudnnSupport::DoConvolveBackwardFilter(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<Eigen::half> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemory<Eigen::half>* backward_filter_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return IsStatusOk(
      DoConvolveBackwardFilterImpl(stream, input_descriptor, input_data,
                                   output_descriptor, backward_output_data,
                                   convolution_descriptor, filter_descriptor,
                                   backward_filter_data, scratch_allocator,
                                   algorithm_config, output_profile_result),
      /*report_error=*/!output_profile_result);
}

template <class T>
port::Status CudnnSupport::DoConvolveBackwardBiasImpl(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<T>& input_data,
    const dnn::BatchDescriptor& bias_descriptor,
    DeviceMemory<T>* backward_bias_data) {
  cudnnDataType_t cudnn_type = GetCudnnDataType<T>();
  CudnnTensorDescriptor input_nd(input_descriptor, cudnn_type);
  CudnnTensorDescriptor bias_nd(bias_descriptor, cudnn_type);

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  RETURN_IF_CUDNN_ERROR(cudnnConvolutionBackwardBias(
      cudnn.handle(), &alpha, input_nd.handle(), input_data.opaque(), &beta,
      bias_nd.handle(), backward_bias_data->opaque()));
  return port::Status::OK();
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<double>& input_data,
    const dnn::BatchDescriptor& bias_descriptor,
    DeviceMemory<double>* backward_bias_data) {
  return IsStatusOk(
      DoConvolveBackwardBiasImpl(stream, input_descriptor, input_data,
                                 bias_descriptor, backward_bias_data),
      /*report_error=*/true);
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& bias_descriptor,
    DeviceMemory<float>* backward_bias_data) {
  return IsStatusOk(
      DoConvolveBackwardBiasImpl(stream, input_descriptor, input_data,
                                 bias_descriptor, backward_bias_data),
      /*report_error=*/true);
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& bias_descriptor,
    DeviceMemory<Eigen::half>* backward_bias_data) {
  return IsStatusOk(
      DoConvolveBackwardBiasImpl(stream, input_descriptor, input_data,
                                 bias_descriptor, backward_bias_data),
      /*report_error=*/true);
}

bool CudnnSupport::DoMatMul(Stream* stream,
                            const DeviceMemory<float>& input_data,
                            const DeviceMemory<float>& weights,
                            const dnn::BatchDescriptor& input_dimensions,
                            const dnn::BatchDescriptor& output_dimensions,
                            DeviceMemory<float>* output_data) {
  if (input_dimensions.count() != output_dimensions.count()) {
    LOG(ERROR) << "MatMul input and output dimensions are not compatible.";
    return false;
  }

  // We do not permute the input or output, instead we just
  // reinterpret the layout. We are working with row-major matrices
  // and the rows of the input and output correspond to batch, so
  // batch has to be outermost in both the input and output.
  //
  // By adding transposes to the BLAS gemm call we could perhaps make
  // the kYXDepthBatch layout work as well, but there has been no need
  // for that so far.
  if (input_dimensions.layout() != dnn::DataLayout::kBatchYXDepth &&
      input_dimensions.layout() != dnn::DataLayout::kBatchDepthYX) {
    LOG(ERROR) << "Unsupported MatMul input layout.";
    return false;
  }
  if (output_dimensions.layout() != dnn::DataLayout::kBatchYXDepth &&
      output_dimensions.layout() != dnn::DataLayout::kBatchDepthYX) {
    LOG(ERROR) << "Unsupported MatMul output layout.";
    return false;
  }

  if (output_dimensions.width() == 1 && output_dimensions.height() == 1) {
    // This is a fast path that also supports the kBatchYXDepth layout.

    // The matrices here are in row-major format while BLAS expects
    // column-major, i.e. our matrices are transposed as far as BLAS
    // is concerned. So we need to compute output^T =
    // input^T*weights^T. There is no parameter for transposing the
    // output in BLAS gemm, but instead we can transpose both sides of
    // the equality to see that this is equivalent to
    // output=weights*input. So we only need to swap the order of
    // weights and input in the matrix product to correct for the
    // row-major versus column-major difference.
    const float alpha = 1.0f;  // Take the matrix product without scaling it.
    const float beta = 0.0f;   // Ignore the original values in output_data.
    const int64 m = output_dimensions.NodesAcrossFeatureMaps();
    const int64 n = input_dimensions.count();
    const int64 k = input_dimensions.NodesAcrossFeatureMaps();
    stream->ThenBlasGemm(blas::Transpose::kNoTranspose,
                         blas::Transpose::kNoTranspose, m, n, k, alpha, weights,
                         m, input_data, k, beta, output_data, m);
  } else {
    // This is a slower and more complex path that supports output
    // width() * height() > 1, though it only supports the
    // kBatchYXDepth layout. Does support kBatchDepthYX if output
    // feature_map_count() == 1, as then there is no difference
    // between the two layouts.
    //
    // The operation here is the same as above, except that we have to
    // do the matrix multiplication for each (y,x) output coordinate
    // separately. We then interpret weights as containing K = width()
    // * height() different matrices, which we all multiply onto the
    // matrix from input_data, yielding K matrix products. We then
    // combine these together into one matrix by concatenating all the
    // first rows of these matrices, then all the seconds rows and so
    // on. We can do this with a batched matrix multiplication, where
    // the result is written to a different submatrix of the output
    // for each matrix multiplication.
    //
    // The reason that we only support the kBatchYXDepth output layout
    // is that we have to do something in the depth for each (y,x)
    // coordinate. The kBatchYXDepth layout has the depth information
    // for each point (y,x) in contiguous memory while the
    // kBatchDepthYX layout does not.
    //
    // TODO(broune): Consider a special case for when output depth ==
    // 1, as then possibly this could all be done as one matrix
    // multiplication instead of a batched one, which should be
    // faster. Another possibility would be to add a weights layout
    // parameter and then support kBatchDepthYX for a different
    // weights layout.
    if (output_dimensions.layout() != dnn::DataLayout::kBatchYXDepth &&
        !(output_dimensions.layout() == dnn::DataLayout::kBatchDepthYX &&
          output_dimensions.feature_map_count() == 1)) {
      LOG(ERROR) << "Unsupported MatMul output layout.";
      return false;
    }

    const float alpha = 1.0f;  // Take the matrix product without scaling it.
    const float beta = 0.0f;   // Ignore the original values in output_data.
    const uint64 m = output_dimensions.feature_map_count();
    const uint64 n = input_dimensions.count();
    const uint64 k = input_dimensions.NodesAcrossFeatureMaps();
    const int lda = m;
    const int ldb = k;
    const int ldc = output_dimensions.NodesAcrossFeatureMaps();
    const int batch_count = output_dimensions.NodesPerFeatureMap();

    std::vector<DeviceMemory<float>> a(batch_count);
    std::vector<DeviceMemory<float>> b(batch_count);
    std::vector<DeviceMemory<float>> c(batch_count);
    for (int i = 0; i < batch_count; ++i) {
      const int weights_offset = i * input_dimensions.NodesAcrossFeatureMaps() *
                                 output_dimensions.feature_map_count();
      a[i] = DeviceMemory<float>::MakeFromByteSize(
          const_cast<float*>(reinterpret_cast<const float*>(weights.opaque())) +
              weights_offset,
          weights.ElementCount() - weights_offset);

      b[i] = input_data;

      const int output_offset = i * output_dimensions.feature_map_count();
      c[i] = DeviceMemory<float>::MakeFromByteSize(
          const_cast<float*>(
              reinterpret_cast<const float*>(output_data->opaque())) +
              output_offset,
          output_data->ElementCount() - output_offset);
    }
    const auto toPtrs = [](std::vector<DeviceMemory<float>>& v) {
      std::vector<DeviceMemory<float>*> ptrs;
      ptrs.reserve(v.size());
      for (auto& mem : v) {
        ptrs.push_back(&mem);
      }
      return ptrs;
    };

    stream->ThenBlasGemmBatched(blas::Transpose::kNoTranspose,
                                blas::Transpose::kNoTranspose, m, n, k, alpha,
                                toPtrs(a), lda, toPtrs(b), ldb, beta, toPtrs(c),
                                ldc, batch_count);
  }

  return stream->ok();
}

bool CudnnSupport::DoBiasAdd(Stream* stream,
                             const DeviceMemory<float>& input_data,
                             const DeviceMemory<float>& biases,
                             const dnn::BatchDescriptor& dimensions,
                             DeviceMemory<float>* output_data) {
  CudnnTensorDescriptor input_descriptor(dimensions, CUDNN_DATA_FLOAT);

  dnn::BatchDescriptor bias_dimensions;
  bias_dimensions.set_count(1)
      .set_feature_map_count(dimensions.feature_map_count())
      .set_height(1)
      .set_width(1)
      .set_layout(dnn::DataLayout::kBatchYXDepth);
  CudnnTensorDescriptor bias_descriptor(bias_dimensions, CUDNN_DATA_FLOAT);

  // cudnnAddTensor after R3 is in-place, so we need to copy input_data to
  // output_data before doing the addition, unless the input and
  // output are at the same address.
  if (input_data.opaque() != output_data->opaque()) {
    stream->ThenMemcpy(output_data, input_data,
                       dimensions.ElementCount() * sizeof(float));
    if (!stream->ok()) {
      LOG(ERROR)
          << "stream " << stream
          << " could not enqueue a tensor copy as part of bias addition.";
      return false;
    }
  }

  const float alpha = 1.0f;
  const float beta = 1.0f;

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnAddTensor(
        cudnn.handle(), &alpha, bias_descriptor.handle(), biases.opaque(),
        &beta, input_descriptor.handle(), output_data->opaque()));
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DoActivate(Stream* stream,
                              dnn::ActivationMode activation_mode,
                              const dnn::BatchDescriptor& dimensions,
                              const DeviceMemory<float>& input_data,
                              DeviceMemory<float>* output_data,
                              uint64 options) {
  CudnnActivationDescriptor activation_desc(
      activation_mode, CUDNN_PROPAGATE_NAN, dimensions.value_max());

  CudnnTensorDescriptor input_nd(dimensions, CUDNN_DATA_FLOAT);
  // Alpha is the input scaling factor.
  float alpha = 1.0;
  // Beta is the output scaling factor.
  float beta = 0.0;

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnActivationForward(
        cudnn.handle(), activation_desc.handle(), &alpha, input_nd.handle(),
        input_data.opaque(), &beta, input_nd.handle(), output_data->opaque()));
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<double>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<double>* output_data, ScratchAllocator* workspace_allocator) {
  // Alpha is the scaling factor for input.
  double alpha = 1.0;
  // Beta is the scaling factor for output.
  double beta = 0.0;

  CudnnTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_DOUBLE);
  CudnnTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_DOUBLE);
  CudnnPoolingDescriptor pooling_desc(pooling_dimensions);

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnPoolingForward(
        cudnn.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
        input_data.opaque(), &beta, dest_desc.handle(), output_data->opaque()));
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<float>* output_data, ScratchAllocator* workspace_allocator) {
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  CudnnTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_FLOAT);
  CudnnTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_FLOAT);
  CudnnPoolingDescriptor pooling_desc(pooling_dimensions);

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnPoolingForward(
        cudnn.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
        input_data.opaque(), &beta, dest_desc.handle(), output_data->opaque()));
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<Eigen::half>* output_data,
    ScratchAllocator* workspace_allocator) {
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  CudnnTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_HALF);
  CudnnTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_HALF);
  CudnnPoolingDescriptor pooling_desc(pooling_dimensions);
  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnPoolingForward(
        cudnn.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
        input_data.opaque(), &beta, dest_desc.handle(), output_data->opaque()));
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<double>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<double>& output_data,
    const DeviceMemory<double>& input_diff_data,
    DeviceMemory<double>* output_diff_data,
    ScratchAllocator* workspace_allocator) {
  // Alpha is the scaling factor for input.
  double alpha = 1.0;
  // Beta is the scaling factor for output.
  double beta = 0.0;

  CudnnTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_DOUBLE);
  CudnnTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_DOUBLE);
  CudnnPoolingDescriptor pooling_desc(pooling_dimensions);

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnPoolingBackward(
        cudnn.handle(), pooling_desc.handle(), &alpha, dest_desc.handle(),
        output_data.opaque(), dest_desc.handle(), input_diff_data.opaque(),
        src_desc.handle(), input_data.opaque(), &beta, src_desc.handle(),
        output_diff_data->opaque()));
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<float>& output_data,
    const DeviceMemory<float>& input_diff_data,
    DeviceMemory<float>* output_diff_data,
    ScratchAllocator* workspace_allocator) {
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  CudnnTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_FLOAT);
  CudnnTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_FLOAT);
  CudnnPoolingDescriptor pooling_desc(pooling_dimensions);

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnPoolingBackward(
        cudnn.handle(), pooling_desc.handle(), &alpha, dest_desc.handle(),
        output_data.opaque(), dest_desc.handle(), input_diff_data.opaque(),
        src_desc.handle(), input_data.opaque(), &beta, src_desc.handle(),
        output_diff_data->opaque()));
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<Eigen::half>& output_data,
    const DeviceMemory<Eigen::half>& input_diff_data,
    DeviceMemory<Eigen::half>* output_diff_data,
    ScratchAllocator* workspace_allocator) {
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  CudnnTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_HALF);
  CudnnTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_HALF);
  CudnnPoolingDescriptor pooling_desc(pooling_dimensions);

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnPoolingBackward(
        cudnn.handle(), pooling_desc.handle(), &alpha, dest_desc.handle(),
        output_data.opaque(), dest_desc.handle(), input_diff_data.opaque(),
        src_desc.handle(), input_data.opaque(), &beta, src_desc.handle(),
        output_diff_data->opaque()));
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DoNormalize(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const DeviceMemory<float>& input_data, DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
  return false;
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
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnLRNCrossChannelForward(
        cudnn.handle(), normalize.handle(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, dims.handle(), input_data.opaque(), &beta, dims.handle(),
        output_data->opaque()));
    return port::Status::OK();
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
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnLRNCrossChannelBackward(
        cudnn.handle(), normalize.handle(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, dims.handle(), normalized_data.opaque(), dims.handle(),
        normalized_variable_gradient.opaque(), dims.handle(), raw_data.opaque(),
        &beta, dims.handle(), raw_variable_gradient->opaque()));
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

bool CudnnSupport::DoDepthConcatenate(
    Stream* stream, port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float>*> input_data,
    DeviceMemory<float>* output_data) {
  CHECK_EQ(input_dimensions.size(), input_data.size());

  for (const auto& dimensions : input_dimensions) {
    if (dimensions.layout() != dnn::DataLayout::kBatchDepthYX) {
      LOG(ERROR) << "CudnnSupport::DoDepthConcatenate currently only "
                    "supports the kBatchDepthYX layout.";
      return false;
    }
  }

  if (input_dimensions.empty()) {
    return true;  // Nothing to do.
  }

  dnn::BatchDescriptor output_dimensions =
      dnn::BatchDescriptor::DepthConcatenateOutputDescriptor(input_dimensions);

  const int64 area = output_dimensions.width() * output_dimensions.height();
  const auto index = [area](int64 batch, int64 depth, int64 yx,
                            int64 max_depth) {
    return (batch * max_depth + depth) * area + yx;
  };

  std::vector<float> output_host(output_dimensions.ElementCount());
  std::vector<float> tmp;
  int64 depth_sum = 0;
  for (size_t i = 0; i < input_data.size(); ++i) {
    const auto& dimensions = input_dimensions[i];
    tmp.resize(dimensions.ElementCount());
    stream->ThenMemcpyD2H<float>(*input_data[i], absl::MakeSpan(tmp));
    port::Status block_status = stream->BlockHostUntilDone();
    if (!block_status.ok()) {
      LOG(ERROR) << "BlockHostUntilDone failed: " << block_status;
      return false;
    }

    for (int64 batch = 0; batch < output_dimensions.count(); ++batch) {
      for (int64 yx = 0; yx < area; ++yx) {
        for (int64 depth = 0; depth < dimensions.feature_map_count(); ++depth) {
          LOG(INFO) << output_dimensions.ElementCount() << ' ' << batch << ' '
                    << yx << ' ' << depth;
          output_host[index(batch, depth + depth_sum, yx,
                            output_dimensions.feature_map_count())] =
              tmp[index(batch, depth, yx, dimensions.feature_map_count())];
        }
      }
    }
    depth_sum += dimensions.feature_map_count();
  }
  stream->ThenMemcpyH2D<float>(output_host, output_data);
  return true;
}

bool CudnnSupport::DoElementwiseOperate(
    Stream* stream, dnn::ElementwiseOperation operation,
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float>*> input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
  return false;
}

bool CudnnSupport::DoXYPad(Stream* stream,
                           const dnn::BatchDescriptor& dimensions,
                           const DeviceMemory<float>& input_data,
                           int64 left_pad, int64 right_pad, int64 top_pad,
                           int64 bottom_pad, DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
  return false;
}

bool CudnnSupport::DoXYSlice(Stream* stream,
                             const dnn::BatchDescriptor& dimensions,
                             const DeviceMemory<float>& input_data,
                             int64 left_trim, int64 right_trim, int64 top_trim,
                             int64 bottom_trim,
                             DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
  return false;
}

bool CudnnSupport::DoMemcpyD2HQuantized(
    Stream* stream, const DeviceMemory<float>& gpu_unquantized_src,
    dnn::QuantizedActivationMode mode, void* host_dst, int64 size) {
  LOG(ERROR) << "quantized memcpy not supported by cuDNN";
  return false;
}

bool CudnnSupport::DoMemcpyH2DQuantized(
    Stream* stream, const void* host_src, int64 size,
    dnn::QuantizedActivationMode mode,
    DeviceMemory<float>* gpu_unquantized_dst) {
  LOG(ERROR) << "quantized memcpy not supported by cuDNN";
  return false;
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
  auto status = [&] {
    RETURN_IF_CUDNN_ERROR(cudnnGetConvolutionNdForwardOutputDim(
        conv.handle(), input_nd.handle(), filter.handle(), dn, dims.data()));
    output_batch_descriptor->set_count(dims[0])
        .set_feature_map_count(dims[1])
        .set_layout(batch_descriptor.layout());

    for (int i = 0; i < batch_descriptor.ndims(); i++) {
      output_batch_descriptor->set_spatial_dim(static_cast<dnn::DimIndex>(i),
                                               dims.rbegin()[i]);
    }
    return port::Status::OK();
  }();
  return IsStatusOk(status, /*report_error=*/true);
}

}  // namespace cuda

void initialize_cudnn() {
  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::DnnFactory>(
          cuda::kCudaPlatformId, cuda::kCuDnnPlugin, "cuDNN",
          [](internal::StreamExecutorInterface* parent) -> dnn::DnnSupport* {
            cuda::CUDAExecutor* cuda_executor =
                dynamic_cast<cuda::CUDAExecutor*>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR) << "Attempting to initialize an instance of the cuDNN "
                         << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            cuda::CudnnSupport* dnn = new cuda::CudnnSupport(cuda_executor);
            if (!dnn->Init().ok()) {
              // Note: Init() will log a more specific error.
              delete dnn;
              return nullptr;
            }
            return dnn;
          });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuDNN factory: "
               << status.error_message();
  }

  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kDnn, cuda::kCuDnnPlugin);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_cudnn,
                            { stream_executor::initialize_cudnn(); });
