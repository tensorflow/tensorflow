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

// TODO(csigg): remove dnn namespace qualifier from the RNN code below.
using ::stream_executor::dnn::BatchDescriptor;
using ::stream_executor::dnn::ConvolutionDescriptor;
using ::stream_executor::dnn::FilterDescriptor;
using ::stream_executor::dnn::NormalizeDescriptor;
using ::stream_executor::dnn::PoolingDescriptor;

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
    default:
      return port::StrCat("<unknown cudnn status: ", static_cast<int>(status),
                          ">");
  }
}

#if CUDNN_VERSION >= 6000
string ToString(libraryPropertyType type) {
  switch (type) {
    case MAJOR_VERSION:
      return "MAJOR_VERSION";
    case MINOR_VERSION:
      return "MINOR_VERSION";
    case PATCH_LEVEL:
      return "PATCH_LEVEL";
    default:
      return port::StrCat(
          "<unknown libraryPropertyType: ", static_cast<int>(type), ">");
  }
}
#endif

template <typename T>
cudnnDataType_t GetCudnnDataType();

template <>
cudnnDataType_t GetCudnnDataType<double>() {
  return CUDNN_DATA_DOUBLE;
}

template <>
cudnnDataType_t GetCudnnDataType<float>() {
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t GetCudnnDataType<Eigen::half>() {
  return CUDNN_DATA_HALF;
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

// Wraps a cuDNN handle and provides access to it through CudnnHandle instances,
// which also locks a mutex, acquires the CUDA context, and sets the stream
// that cuDNN should use to enqueue any work.
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
  // cuDNN API calls using the same handle instance need to be serialized across
  // threads. This is guaranteed by CudnnHandle instances locking the mutex
  // owned by this class.
  //
  // Most cuDNN APIs taking a handle perform work on a CUDA stream. The
  // CudnnHandle instance acquires the executor's CUDA context and sets cuDNN to
  // use the provided stream.
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
    using my_mutex_lock = mutex_lock;
    return CudnnHandle(std::move(context), std::move(lock), handle_);
  }

 private:
  // Guards the enqueueing of cuDNN operations via the handle_ below.
  mutex mutex_;

  // cuDNN library handle.
  cudnnHandle_t handle_ GUARDED_BY(mutex_);  // Owned.
};

namespace {

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
#if CUDNN_VERSION >= 5000
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
#endif
#if CUDNN_VERSION >= 5100
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
#endif
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
#if CUDNN_VERSION >= 5000
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
#endif
#if CUDNN_VERSION >= 5100
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
#endif
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
#if CUDNN_VERSION >= 5100
    // Based on cudnn.h, the following is not implemented.
    // case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
#endif
      return algo;
    default:
      LOG(FATAL)
          << "Unsupported Cudnn convolution backward algorithm for filter: "
          << algorithm.algo_id();
  }
}

#if CUDNN_VERSION >= 6000
port::Status GetCudnnProperty(libraryPropertyType type, int* value) {
  cudnnStatus_t status = cudnnGetProperty(type, value);
  if (status != CUDNN_STATUS_SUCCESS) {
    const string error =
        port::StrCat("cudnnGetProperty failed for type: ", ToString(type),
                     " with status: ", ToString(status));
    LOG(ERROR) << error;
    return port::Status(port::error::INTERNAL, error);
  }
  return port::Status::OK();
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
#endif

port::Status GetLoadedCudnnVersion(CudnnVersion* version) {
#if CUDNN_VERSION >= 6000
  TF_RETURN_IF_ERROR(GetCudnnProperty(MAJOR_VERSION, &version->major_version));
  TF_RETURN_IF_ERROR(GetCudnnProperty(MINOR_VERSION, &version->minor_version));
  TF_RETURN_IF_ERROR(GetCudnnProperty(PATCH_LEVEL, &version->patch_level));
#else
  size_t loaded_version = ::cudnnGetVersion();
  version->major_version = loaded_version / 1000;
  version->minor_version = (loaded_version / 100) % 10;
  version->patch_level = loaded_version % 100;
#endif
  return port::Status::OK();
}

}  // namespace

CudnnSupport::CudnnSupport(CUDAExecutor* parent) : parent_(parent) {}

port::Status CudnnSupport::Init() {
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
          "from sources, make sure the library loaded at runtime is compatible "
          "with the version specified during compile configuration.");
      LOG(ERROR) << error;
      return port::Status(port::error::INTERNAL, error);
    }

    cudnn_.reset(new CudnnAccess(cudnn_handle));
    return port::Status::OK();
  }

  LOG(ERROR) << "could not create cudnn handle: " << ToString(status);
  if (status == CUDNN_STATUS_NOT_INITIALIZED) {
    auto result = cuda::Diagnostician::FindKernelDriverVersion();
    if (!result.ok()) {
      LOG(ERROR) << "error retrieving driver version: "
                 << DriverVersionStatusToString(result);
    } else {
      const auto& version = result.ValueOrDie();
      LOG(ERROR) << "possibly insufficient driver version: "
                 << DriverVersionToString(version);
      // OS X kernel driver does not report version accurately
#if !defined(__APPLE__)
      if (std::get<0>(version) < 340) {
        LOG(ERROR)
            << "cudnn library is only supported on 340.XX+ driver versions";
      }
#endif
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

// Turns a BatchDescriptor structure into a cudnn tensor handle within a scope.
class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor(const BatchDescriptor& batch_descriptor,
                         cudnnDataType_t elem_type)
      : handle_(nullptr) {
    cudnnStatus_t status = cudnnCreateTensorDescriptor(&handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not create cudnn tensor descriptor: "
                 << ToString(status);
    }

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
        status = cudnnSetTensorNdDescriptor(handle_, elem_type, nd, dims.data(),
                                            strides.data());

        if (status != CUDNN_STATUS_SUCCESS) {
          LOG(FATAL) << "could not convert BatchDescriptor "
                     << batch_descriptor.ToString()
                     << " to cudnn tensor descriptor: " << ToString(status);
        }
      } break;
#if CUDNN_VERSION >= 6000
      case dnn::DataLayout::kBatchDepthYX4: {
        status = cudnnSetTensor4dDescriptor(
            handle_, CUDNN_TENSOR_NCHW_VECT_C, elem_type,
            batch_descriptor.count(), batch_descriptor.feature_map_count(),
            batch_descriptor.height(), batch_descriptor.width());
        if (status != CUDNN_STATUS_SUCCESS) {
          LOG(FATAL) << "could not convert BatchDescriptor "
                     << batch_descriptor.ToString()
                     << " to cudnn tensor descriptor: " << ToString(status);
        }
      } break;
#endif
      default:
        LOG(FATAL) << "Unsupported tensor format "
                   << DataLayoutString(batch_descriptor.layout());
        break;
    }
  }

  ~ScopedTensorDescriptor() {
    cudnnStatus_t status = cudnnDestroyTensorDescriptor(handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(ERROR) << "could not destroy cudnn tensor descriptor: "
                 << ToString(status);
    }
  }

  cudnnTensorDescriptor_t handle() const { return handle_; }

 private:
  cudnnTensorDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

// Turns a FilterDescriptor structure into a cudnn filter handle within a scope.
class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor(const FilterDescriptor& filter_descriptor,
                         cudnnDataType_t elem_type)
      : handle_(nullptr) {
    cudnnStatus_t status = cudnnCreateFilterDescriptor(&handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not create cudnn filter descriptor: "
                 << ToString(status);
    }

#if CUDNN_VERSION >= 5000
    // TODO(b/23032134): Even if the filter layout is not supported,
    // cudnnSetFilter4DDescriptor_v4 will return CUDNN_STATUS_SUCCESS because it
    // does not take layout as an input. Maybe force cuDNN by giving wrong
    // inputs intentionally?
    cudnnTensorFormat_t format;
    switch (filter_descriptor.layout()) {
      case dnn::FilterLayout::kOutputInputYX:
        format = CUDNN_TENSOR_NCHW;
        break;
#if CUDNN_VERSION >= 6000
      case dnn::FilterLayout::kOutputInputYX4:
        format = CUDNN_TENSOR_NCHW_VECT_C;
        break;
#endif
      default:
        LOG(FATAL) << "Unsupported filter format "
                   << FilterLayoutString(filter_descriptor.layout());
        break;
    }
#endif

    std::vector<int> dims(2 + filter_descriptor.ndims());
    dims[0] = filter_descriptor.output_feature_map_count();
    dims[1] = filter_descriptor.input_feature_map_count();
    const auto& spatial_dims = filter_descriptor.input_filter_dims();
    std::copy(spatial_dims.begin(), spatial_dims.end(), dims.begin() + 2);

    status = cudnnSetFilterNdDescriptor(handle_, elem_type,
#if CUDNN_VERSION >= 5000
                                        format,
#endif
                                        dims.size(), dims.data());
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not set cudnn filter descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedFilterDescriptor() {
    cudnnStatus_t status = cudnnDestroyFilterDescriptor(handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(ERROR) << "could not destroy cudnn filter descriptor: "
                 << ToString(status);
    }
  }

  cudnnFilterDescriptor_t handle() const { return handle_; }

 private:
  cudnnFilterDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

// A helper function to decide whether to enable the TENSOR_OP_MATH math type
static bool TensorOpMathEnabled() {
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
static bool RnnTensorOpMathEnabled() {
  static bool is_enabled = [] {
    bool is_disabled = false;
    TF_CHECK_OK(
        tensorflow::ReadBoolFromEnvVar("TF_DISABLE_CUDNN_RNN_TENSOR_OP_MATH",
                                       /*default_val=*/false, &is_disabled));
    return !is_disabled;
  }();
  return is_enabled;
}

// A helper function to decide whether to use CUDNN_BATCHNORM_SPATIAL_PERSISTENT
// in batchnorm. This mode can be faster in some tasks because an optimized path
// may be selected for CUDNN_DATA_FLOAT and CUDNN_DATA_HALF data types, compute
// capability 6.0 or higher. The reason we set it to false by default is that
// this mode may use scaled atomic integer reduction that may cause a numerical
// overflow for certain input data range.
// TODO(yangzihao): Use autotune to choose between this mode and
// CUDNN_BATCHNORM_SPATIAL mode.
static bool BatchnormSpatialPersistentEnabled() {
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
class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor(
      const ConvolutionDescriptor& convolution_descriptor,
      cudnnDataType_t data_type)
      : handle_(nullptr) {
    cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not create cudnn convolution descriptor: "
                 << ToString(status);
    }
    const auto& strides64 = convolution_descriptor.strides();
    const auto& padding64 = convolution_descriptor.padding();
    const auto& dilations64 = convolution_descriptor.dilations();
    if (convolution_descriptor.pad_alignment() ==
        dnn::PadAlignment::kTensorFlowPadding) {
      LOG(ERROR) << "TensorFlow padding alignment is not supported.";
    }

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

    status = cudnnSetConvolutionNdDescriptor(
        handle_, convolution_descriptor.ndims(), padding.data(), strides.data(),
        dilations.data(),
        // NOTE(keveman): cuDNN supports convolution and cross correlation.
        // However, almost all the use cases do cross correlation, so just
        // hard coding it here.
        CUDNN_CROSS_CORRELATION, data_type);

    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not set cudnn convolution descriptor: "
                 << ToString(status);
    }
    // NOTE(benbarsdell): This only applies if tensor op math is enabled
    //                      and algo selection is set to Default.
    this->set_use_tensor_op_math(true);

#if CUDNN_MAJOR >= 7
    VLOG(2) << "Requesting grouped convolution: "
            << convolution_descriptor.group_count();
    status = cudnnSetConvolutionGroupCount(
        handle_, convolution_descriptor.group_count());
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not set cudnn convolution group count: "
                 << ToString(status);
    }
#else
    CHECK_EQ(convolution_descriptor.group_count(), 1)
        << "Requested grouped convolution for cuDNN version < 7";
#endif
  }

  void set_use_tensor_op_math(bool use_tensor_op_math) {
#if CUDNN_VERSION >= 7000
    cudnnMathType_t math_type =
        (use_tensor_op_math ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH);
    if (TensorOpMathEnabled()) {
      cudnnStatus_t status = cudnnSetConvolutionMathType(handle_, math_type);
      if (status != CUDNN_STATUS_SUCCESS) {
        LOG(FATAL) << "could not set cudnn convolution math type: "
                   << ToString(status);
      }
    }
#endif
  }

  ~ScopedConvolutionDescriptor() {
    cudnnStatus_t status = cudnnDestroyConvolutionDescriptor(handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(ERROR) << "could not destroy cudnn convolution descriptor: "
                 << ToString(status);
    }
  }

  cudnnConvolutionDescriptor_t handle() const { return handle_; }

 private:
  cudnnConvolutionDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedConvolutionDescriptor);
};

// Turns a PoolingDescriptor structure into a cudnn pooling descriptor handle
// within a scope.
class ScopedPoolingDescriptor {
 public:
  explicit ScopedPoolingDescriptor(const PoolingDescriptor& pooling_descriptor)
      : handle_(nullptr) {
    cudnnStatus_t status = cudnnCreatePoolingDescriptor(&handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not create cudnn pooling descriptor: "
                 << ToString(status);
    }
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
    status = cudnnSetPoolingNdDescriptor(
        handle_,
        (pooling_descriptor.mode() == dnn::PoolingMode::kMaximum
             ? CUDNN_POOLING_MAX
             : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING),
#if CUDNN_VERSION >= 5000
        propagate_nans ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN,
#endif
        nd, shape.data(), padding.data(), strides.data());
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not set cudnn pooling descriptor: "
                 << ToString(status);
    }
  }
  ~ScopedPoolingDescriptor() {
    cudnnStatus_t status = cudnnDestroyPoolingDescriptor(handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(ERROR) << "could not destroy cudnn pooling descriptor: "
                 << ToString(status);
    }
  }

  cudnnPoolingDescriptor_t handle() const { return handle_; }

 private:
  cudnnPoolingDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

// Turns a NormalizeDescriptor structure into a cudnn LRN descriptor handle.
class ScopedNormalizeDescriptor {
 public:
  explicit ScopedNormalizeDescriptor(
      const NormalizeDescriptor& normalize_descriptor)
      : handle_(nullptr) {
    cudnnStatus_t status = cudnnCreateLRNDescriptor(&handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not create cudnn LRN descriptor: "
                 << ToString(status);
    }

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
    status = cudnnSetLRNDescriptor(handle_, lrnN, lrnAlpha, lrnBeta, lrnK);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not set cudnn LRN descriptor: " << ToString(status);
    }
  }

  ~ScopedNormalizeDescriptor() {
    cudnnStatus_t status = cudnnDestroyLRNDescriptor(handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(ERROR) << "could not destroy cudnn LRN descriptor: "
                 << ToString(status);
    }
  }

  cudnnLRNDescriptor_t handle() const { return handle_; }

 private:
  cudnnLRNDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedNormalizeDescriptor);
};

#if CUDNN_VERSION >= 5000
// Turns a ActivationDescriptor structure into a cudnn activation
// descriptor handle within a scope.
class ScopedActivationDescriptor {
 public:
  ScopedActivationDescriptor(dnn::ActivationMode activation_mode,
                             cudnnNanPropagation_t nan_propagation,
                             double value_max)
      : handle_(nullptr) {
    cudnnStatus_t status = cudnnCreateActivationDescriptor(&handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not create cudnn activation descriptor: "
                 << ToString(status);
    }

    double relu_ceiling = 0.0;
    cudnnActivationMode_t mode;
    switch (activation_mode) {
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

    status = cudnnSetActivationDescriptor(handle_, mode, nan_propagation,
                                          relu_ceiling);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(FATAL) << "could not set cudnn activation descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedActivationDescriptor() {
    cudnnStatus_t status = cudnnDestroyActivationDescriptor(handle_);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(ERROR) << "could not destroy cudnn activation descriptor: "
                 << ToString(status);
    }
  }

  cudnnActivationDescriptor_t handle() const { return handle_; }

 private:
  cudnnActivationDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedActivationDescriptor);
};
#endif

cudnnDataType_t ToCudnnDataType(
    dnn::DataType data_type,
    dnn::DataLayout data_layout = dnn::DataLayout::kBatchDepthYX) {
  switch (data_type) {
    case dnn::DataType::kFloat:
    case dnn::DataType::kDouble:
    case dnn::DataType::kHalf:
      return static_cast<cudnnDataType_t>(data_type);
#if CUDNN_VERSION >= 6000
    case dnn::DataType::kInt8:
      return data_layout == dnn::DataLayout::kBatchDepthYX4 ? CUDNN_DATA_INT8x4
                                                            : CUDNN_DATA_INT8;
#endif
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

#if CUDNN_VERSION >= 5000

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

#endif  // CUDNN_VERSION

template <typename Base>
class MixinBase : public Base {};
template <>
class MixinBase<void> {};

#if CUDNN_VERSION >= 5000

#define CUDNN_RETURN_IF_FAIL(STATUS, ...)                                \
  if (!SE_PREDICT_TRUE((STATUS) == CUDNN_STATUS_SUCCESS)) {              \
    string error_msg = port::StrCat(ToString(STATUS), " ", __VA_ARGS__); \
    SetFailure(port::Status(port::error::UNKNOWN, error_msg));           \
    LOG(ERROR) << error_msg;                                             \
    return;                                                              \
  }

// TODO(csigg): Remove inheritance for code reuse.
template <typename Base>
class CudnnDescriptorCommon : public MixinBase<Base> {
 public:
  bool ok() const { return status_.ok(); }
  port::Status Status() const { return status_; }

 protected:
  void SetFailure(const port::Status& status) { status_.Update(status); }
  port::Status status_;
};

class CudnnDropoutDescriptor : public CudnnDescriptorCommon<void> {
 public:
  CudnnDropoutDescriptor(const CudnnHandle& cudnn, float dropout, uint64 seed,
                         ScratchAllocator* state_allocator)
      : handle_(nullptr) {
    cudnnStatus_t status;
    status = cudnnCreateDropoutDescriptor(&handle_);
    CUDNN_RETURN_IF_FAIL(status, "Failed to create dropout descriptor");

    if (dropout == 0.f) {
      return;
    }

    DeviceMemory<uint8> state_memory;
    if (state_allocator) {
      size_t state_sizes_in_bytes = 0;
      status = cudnnDropoutGetStatesSize(cudnn.handle(), &state_sizes_in_bytes);
      CUDNN_RETURN_IF_FAIL(status, "Failed to query dropout state sizes");

      auto allocated =
          state_allocator->AllocateBytes(nullptr, state_sizes_in_bytes);
      if (!allocated.ok() ||
          (state_memory = allocated.ValueOrDie()) == nullptr) {
        string error_msg =
            port::StrCat("Failed to allocate Cudnn dropout state memory of ",
                         state_sizes_in_bytes, " bytes.");
        status_ = port::Status(port::error::UNKNOWN, error_msg);
        LOG(ERROR) << error_msg;
        return;
      }
    }
    status = cudnnSetDropoutDescriptor(handle_, cudnn.handle(), dropout,
                                       state_memory.opaque(),
                                       state_memory.size(), seed);
    CUDNN_RETURN_IF_FAIL(
        status, port::StrCat(
                    "Failed to set dropout descriptor with state memory size: ",
                    state_memory.size(), " bytes."));
  }

  ~CudnnDropoutDescriptor() {
    cudnnStatus_t status = cudnnDestroyDropoutDescriptor(handle_);
    // TODO(csigg): This is a no-op (error is not reported). Same below.
    CUDNN_RETURN_IF_FAIL(status, "Failed to destroy Cudnn dropout handle: ");
  }

  cudnnDropoutDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return handle_;
  }

 private:
  cudnnDropoutDescriptor_t handle_;  // Owned.
  float dropout_;
  uint64 seed_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnDropoutDescriptor);
};

class CudnnRnnParamsDescriptor : public CudnnDescriptorCommon<void> {
 public:
  typedef dnn::RnnDescriptor::ParamsRegion ParamsRegion;
  typedef dnn::RnnDescriptor::ParamsRegions ParamsRegions;
  CudnnRnnParamsDescriptor(const CudnnHandle& cudnn,
                           const CudnnRnnDescriptor& rnn_desc);
  ~CudnnRnnParamsDescriptor() {
    cudnnStatus_t status = cudnnDestroyFilterDescriptor(handle_);
    CUDNN_RETURN_IF_FAIL(status, "Failed to destroy RNN filter descriptor");
  }
  cudnnFilterDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return handle_;
  }
  int64 params_size_in_bytes() const { return params_size_in_bytes_; }
  ParamsRegions params_weights() const {
    if (!ok()) return ParamsRegions();
    return weights_;
  }
  ParamsRegions params_biases() const {
    if (!ok()) return ParamsRegions();
    return biases_;
  }

 private:
  int GetRegionCountPerLayer() const;
  cudnnFilterDescriptor_t handle_;
  const CudnnRnnDescriptor* rnn_desc_;
  int64 params_size_in_bytes_;
  ParamsRegions weights_;
  ParamsRegions biases_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnParamsDescriptor);
};

}  // namespace

class CudnnRnnDescriptor : public CudnnDescriptorCommon<dnn::RnnDescriptor> {
 public:
  CudnnRnnDescriptor(const CudnnHandle& cudnn, int num_layers, int hidden_size,
                     int input_size, int batch_size,
                     cudnnRNNInputMode_t input_mode,
                     cudnnDirectionMode_t direction_mode,
                     cudnnRNNMode_t rnn_mode, cudnnDataType_t data_type,
                     cudnnDataType_t compute_type,
                     const dnn::AlgorithmConfig& algorithm_config,
                     float dropout, uint64 seed,
                     ScratchAllocator* state_allocator)
      : rnn_desc_(nullptr),
        num_layers_(num_layers),
        hidden_size_(hidden_size),
        input_size_(input_size),
        batch_size_(batch_size),
#if CUDNN_VERSION >= 6000
        rnn_plan_(nullptr),
#endif
        input_mode_(input_mode),
        direction_mode_(direction_mode),
        rnn_mode_(rnn_mode),
        data_type_(data_type),
        compute_type_(compute_type),
        algorithm_config_(algorithm_config) {
    // Create the dropout handle.
    cudnn_dropout_desc_.reset(
        new CudnnDropoutDescriptor(cudnn, dropout, seed, state_allocator));
    if (!cudnn_dropout_desc_->ok()) {
      SetFailure(cudnn_dropout_desc_->Status());
      return;
    }

    // Create the RNN handle
    cudnnStatus_t status = cudnnCreateRNNDescriptor(&rnn_desc_);
    CUDNN_RETURN_IF_FAIL(status, "Unable to create RNN descriptor");
#if CUDNN_VERSION >= 6000
    // TODO: allow the user to choose an algorithm.
    rnn_algo_ = ToCudnnRNNAlgo(algorithm_config_.algorithm());
    status = cudnnSetRNNDescriptor_v6(
        cudnn.handle(), /*rnnDesc=*/rnn_desc_, /*hiddenSize=*/hidden_size,
        /*numLayers=*/num_layers, /*dropoutDesc=*/dropout_handle(),
        /*inputMode=*/input_mode, /*direction=*/direction_mode,
        /*mode=*/rnn_mode, /*algo=*/rnn_algo_, /*dataType=*/compute_type);
    CUDNN_RETURN_IF_FAIL(status, ::tensorflow::strings::Printf(
                                     "Unable to update RNN descriptor with "
                                     "algo_id: %d and compute_type: %d",
                                     static_cast<int>(rnn_algo_),
                                     static_cast<int>(compute_type)));

    if (rnn_algo_ == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
      CHECK_GE(batch_size_, 0);
      status = cudnnCreatePersistentRNNPlan(rnn_desc_, batch_size_, data_type_,
                                            &rnn_plan_);
      CUDNN_RETURN_IF_FAIL(status, "Unable to create persistent RNN plan.");
      status = cudnnSetPersistentRNNPlan(rnn_desc_, rnn_plan_);
      CUDNN_RETURN_IF_FAIL(status, "Unable to update persistent RNN plan.");
    }
#else
    CHECK(algorithm_config_.is_default())
        << "Non-default algorithm not supported for CUDA version < 6.0";
    status = cudnnSetRNNDescriptor(
        /*rnnDesc=*/rnn_desc_, /*hiddenSize=*/hidden_size,
        /*numLayers=*/num_layers, /*dropoutDesc=*/dropout_handle(),
        /*inputMode=*/input_mode, /*direction=*/direction_mode,
        /*mode=*/rnn_mode, /*dataType=*/compute_type);
    CUDNN_RETURN_IF_FAIL(status, "Unable to update RNN descriptor");
#endif

    // Create the params handle.
    cudnn_params_desc_.reset(new CudnnRnnParamsDescriptor(cudnn, *this));
    if (!cudnn_params_desc_->ok()) {
      SetFailure(cudnn_params_desc_->Status());
      return;
    }
    set_use_tensor_op_math(algorithm_config_.algorithm().tensor_ops_enabled());
  }
  ~CudnnRnnDescriptor() override {
    if (rnn_desc_) {
      cudnnStatus_t status;
#if CUDNN_VERSION >= 6000
      if (rnn_algo_ == CUDNN_RNN_ALGO_PERSIST_DYNAMIC && rnn_plan_) {
        status = cudnnDestroyPersistentRNNPlan(rnn_plan_);
        CUDNN_RETURN_IF_FAIL(status, "Unable to destroy persistent RNN plan.");
      }
#endif
      status = cudnnDestroyRNNDescriptor(rnn_desc_);
      CUDNN_RETURN_IF_FAIL(status, "Unable to destroy RNN descriptor");
    }
  }
  void set_use_tensor_op_math(bool use_tensor_op_math) {
#if CUDNN_VERSION >= 7000
    cudnnMathType_t math_type =
        (use_tensor_op_math ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH);
    if (RnnTensorOpMathEnabled()) {
      cudnnStatus_t status = cudnnSetRNNMatrixMathType(rnn_desc_, math_type);
      if (status != CUDNN_STATUS_SUCCESS) {
        LOG(FATAL) << "could not set cudnn RNN math type: " << ToString(status);
      }
    }
#endif
  }
  cudnnRNNDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return rnn_desc_;
  }
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
    return cudnn_params_desc_->params_size_in_bytes();
  }
  cudnnDropoutDescriptor_t dropout_handle() const {
    if (!cudnn_dropout_desc_) return nullptr;
    return cudnn_dropout_desc_->handle();
  }
  cudnnFilterDescriptor_t params_handle() const {
    if (!cudnn_params_desc_) return nullptr;
    return cudnn_params_desc_->handle();
  }
  ParamsRegions ParamsWeightRegions() const override {
    if (!ok()) return ParamsRegions();
    return cudnn_params_desc_->params_weights();
  }
  ParamsRegions ParamsBiasRegions() const override {
    if (!ok()) return ParamsRegions();
    return cudnn_params_desc_->params_biases();
  }

 private:
  cudnnRNNDescriptor_t rnn_desc_;
  int num_layers_;
  int hidden_size_;
  int input_size_;
  // batch_size_ is set to -1 when not using CUDNN_RNN_ALGO_PERSIST_DYNAMIC
  // algorithm.
  int batch_size_;
#if CUDNN_VERSION >= 6000
  cudnnRNNAlgo_t rnn_algo_;
  cudnnPersistentRNNPlan_t rnn_plan_;
#endif
  cudnnRNNInputMode_t input_mode_;
  cudnnDirectionMode_t direction_mode_;
  cudnnRNNMode_t rnn_mode_;
  cudnnDataType_t data_type_;
  cudnnDataType_t compute_type_;
  dnn::AlgorithmConfig algorithm_config_;
  std::unique_ptr<CudnnDropoutDescriptor> cudnn_dropout_desc_;
  std::unique_ptr<CudnnRnnParamsDescriptor> cudnn_params_desc_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnDescriptor);
};

namespace {

CudnnRnnParamsDescriptor::CudnnRnnParamsDescriptor(
    const CudnnHandle& cudnn, const CudnnRnnDescriptor& rnn_desc)
    : handle_(nullptr), rnn_desc_(&rnn_desc), params_size_in_bytes_(0) {
  cudnnTensorDescriptor_t input_desc = nullptr;
  {
    // Query the params size.
    auto status = cudnnCreateTensorDescriptor(&input_desc);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to create tensor descriptor");
    int dims[] = {1, rnn_desc.input_size(), 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    status = cudnnSetTensorNdDescriptor(
        /*tensorDesc=*/input_desc, rnn_desc.data_type() /*dataType*/,
        sizeof(dims) / sizeof(dims[0]) /*nbDims*/, /*dimA=*/dims,
        /*strideA=*/strides);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to set tensor descriptor");

    size_t params_size = 0;
    status = cudnnGetRNNParamsSize(
        cudnn.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        /*xDesc=*/input_desc, /*sizeInBytes=*/&params_size,
        rnn_desc.data_type() /*dataType*/);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to get RNN parameter size");
    params_size_in_bytes_ = static_cast<int64>(params_size);
  }

  {
    // Create the params descriptor.
    auto status = cudnnCreateFilterDescriptor(&handle_);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to create RNN filter descriptor");
    int dims[] = {static_cast<int>(params_size_in_bytes_), 1, 1};
    status = cudnnSetFilterNdDescriptor(
        /*filterDesc=*/handle_, rnn_desc.data_type() /*dataType*/,
        /*format=*/CUDNN_TENSOR_NCHW, sizeof(dims) / sizeof(dims[0]) /*nbDims*/,
        /*filterDimA=*/dims);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to update RNN filter descriptor");
  }

  {
    // Create the weights and biases into the params buffer
    int region_count_per_layer = GetRegionCountPerLayer();
    cudnnFilterDescriptor_t region_desc_handle = nullptr;
    auto status = cudnnCreateFilterDescriptor(&region_desc_handle);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to create filter descriptor");
    const int layer_count = rnn_desc.direction_mode() == CUDNN_UNIDIRECTIONAL
                                ? rnn_desc.num_layers()
                                : 2 * rnn_desc.num_layers();
    for (int layer = 0; layer < layer_count; layer++) {
      for (int region = 0; region < region_count_per_layer; region++) {
        for (int type = 0; type < 2; type++) {
          void* offset = nullptr;
          if (type == 0) {
            status = cudnnGetRNNLinLayerMatrixParams(
                cudnn.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
                /*layer=*/layer, /*xDesc=*/input_desc, /*wDesc=*/handle_,
                /*w=*/nullptr, /*linLayerID=*/region,
                /*linLayerMatDesc=*/region_desc_handle,
                /*linLayerMat=*/&offset);
            CUDNN_RETURN_IF_FAIL(
                status, "Cudnn fails to call cudnnGetRNNLinLayerMatrixParams");
          } else {
            status = cudnnGetRNNLinLayerBiasParams(
                cudnn.handle() /*rnnDesc*/, rnn_desc.handle() /*rnnDesc*/,
                /*layer=*/layer, /*xDesc=*/input_desc, /*wDesc=*/handle_,
                /*w=*/nullptr, /*linLayerID=*/region,
                /*linLayerBiasDesc=*/region_desc_handle,
                /*linLayerBias=*/&offset);
            CUDNN_RETURN_IF_FAIL(
                status, "Cudnn fails to call cudnnGetRNNLinLayerBiasParams");
          }
          int dims[] = {1, 1, 1};
          cudnnDataType_t data_type;
          cudnnTensorFormat_t tensor_format;
          int n_dims;
          status = cudnnGetFilterNdDescriptor(
              /*filterDesc=*/region_desc_handle,
              sizeof(dims) / sizeof(dims[0]) /*nbDimsRequested*/,
              /*dataType=*/&data_type, /*format=*/&tensor_format,
              /*nbDims=*/&n_dims, /*filterDimA=*/dims);
          CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to get filter description");
          int64 size = dims[0] * dims[1] * dims[2] *
                       CudnnDataTypeToByteSize(rnn_desc.data_type());
          ParamsRegion region = {reinterpret_cast<int64>(offset), size};
          if (type == 0) {
            weights_.push_back(region);
          } else {
            biases_.push_back(region);
          }
        }
      }
    }
    status = cudnnDestroyFilterDescriptor(region_desc_handle);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to destroy filter descriptor");
  }

  {
    // Release the dummy input tensor descriptor.
    auto status = cudnnDestroyTensorDescriptor(input_desc);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to destroy tensor descriptor");
  }
}

int CudnnRnnParamsDescriptor::GetRegionCountPerLayer() const {
  auto rnn_mode = rnn_desc_->rnn_mode();
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
  }
}

}  // namespace

class CudnnRnnSequenceTensorDescriptor
    : public CudnnDescriptorCommon<dnn::RnnSequenceTensorDescriptor> {
 public:
  CudnnRnnSequenceTensorDescriptor(CUDAExecutor* parent, int seq_length,
                                   int batch_size, int data_size,
                                   cudnnDataType_t data_type)
      : parent_(parent),
        seq_length_(seq_length),
        batch_size_(batch_size),
        data_size_(data_size),
        data_type_(data_type) {
    cudnnTensorDescriptor_t handle = nullptr;
    if (seq_length <= 0) {
      string error_msg =
          port::StrCat("sequence length must be positive: ", seq_length);
      LOG(ERROR) << error_msg;
      SetFailure(port::Status(port::error::UNKNOWN, error_msg));
      return;
    }
    cudnnStatus_t status = cudnnCreateTensorDescriptor(&handle);
    CUDNN_RETURN_IF_FAIL(status, "Failed to create tensor descriptor");
    int dims[] = {batch_size, data_size, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    status = cudnnSetTensorNdDescriptor(
        /*tensorDesc=*/handle, /*dataType=*/data_type,
        sizeof(dims) / sizeof(dims[0]) /*nbDims*/, /*dimA=*/dims,
        /*strideA=*/strides);
    CUDNN_RETURN_IF_FAIL(status, "Failed to update tensor descriptor");
    // Replicate handle across the number of steps.
    handles_.assign(seq_length, handle);
  }

  ~CudnnRnnSequenceTensorDescriptor() override {
    // Only the first one needs to be destroyed. All others are the same.
    cudnnStatus_t status = cudnnDestroyTensorDescriptor(handles_[0]);
    CUDNN_RETURN_IF_FAIL(status,
                         "Failed to destroy sequence tensor descriptor");
  }

  const cudnnTensorDescriptor_t* handles() const {
    if (!ok()) return nullptr;
    CHECK(!handles_.empty()) << "handles cannot be empty";
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
  std::vector<cudnnTensorDescriptor_t> handles_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnSequenceTensorDescriptor);
};

class CudnnRnnStateTensorDescriptor
    : public CudnnDescriptorCommon<dnn::RnnStateTensorDescriptor> {
 public:
  CudnnRnnStateTensorDescriptor(CUDAExecutor* parent, int num_layers,
                                int batch_size, int data_size,
                                cudnnDataType_t data_type)
      : parent_(parent),
        handle_(nullptr),
        num_layers_(num_layers),
        batch_size_(batch_size),
        data_size_(data_size),
        data_type_(data_type) {
    cudnnStatus_t status = cudnnCreateTensorDescriptor(&handle_);
    CUDNN_RETURN_IF_FAIL(status, "Failed to create tensor descriptor");
    int dims[] = {num_layers, batch_size, data_size};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    status = cudnnSetTensorNdDescriptor(
        /*tensorDesc=*/handle_, /*dataType=*/data_type,
        sizeof(dims) / sizeof(dims[0]) /*nbDims*/, /*dimA=*/dims,
        /*strideA=*/strides);
    CUDNN_RETURN_IF_FAIL(status, "Failed to update tensor descriptor");
  }

  ~CudnnRnnStateTensorDescriptor() override {
    if (!handle_) {
      cudnnStatus_t status = cudnnDestroyTensorDescriptor(handle_);
      CUDNN_RETURN_IF_FAIL(status, "Unable to destroy RNN state tensor");
    }
  }

  cudnnTensorDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return handle_;
  }
  int num_layers() const { return num_layers_; }
  int batch_size() const { return batch_size_; }
  int data_size() const { return data_size_; }

 private:
  CUDAExecutor* parent_;
  cudnnTensorDescriptor_t handle_;
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
bool ExtractAndCheckRnnForward(
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
    const DeviceMemory<T>& output_c_data, RnnModelDims* model_dims) {
  // extract model parameters
  model_dims->num_layers = rnn_desc.num_layers();
  model_dims->batch_size = input_desc.batch_size();
  model_dims->seq_length = input_desc.seq_length();
  model_dims->hidden_size = rnn_desc.hidden_size();
  model_dims->input_size = input_desc.data_size();
  model_dims->dir_count =
      (rnn_desc.direction_mode() == CUDNN_BIDIRECTIONAL) ? 2 : 1;

  // check parameters
  if (!(input_h_desc.num_layers() ==
            model_dims->num_layers * model_dims->dir_count &&
        input_h_desc.batch_size() == model_dims->batch_size &&
        input_h_desc.data_size() == model_dims->hidden_size)) {
    LOG(ERROR) << "Invalid input_h shape";
    return false;
  }
  if (!(input_h_desc.num_layers() == input_c_desc.num_layers() &&
        input_h_desc.batch_size() == input_c_desc.batch_size() &&
        input_h_desc.data_size() == input_c_desc.data_size())) {
    LOG(ERROR) << "Invalid input_c shape";
    return false;
  }
  if (!(output_desc.seq_length() == model_dims->seq_length &&
        output_desc.batch_size() == model_dims->batch_size &&
        output_desc.data_size() ==
            model_dims->hidden_size * model_dims->dir_count)) {
    LOG(ERROR) << "Invalid output shape";
    return false;
  }
  if (!(input_h_desc.num_layers() == output_h_desc.num_layers() &&
        input_h_desc.batch_size() == output_h_desc.batch_size() &&
        input_h_desc.data_size() == output_h_desc.data_size())) {
    LOG(ERROR) << "Invalid output_h shape";
    return false;
  }
  if (!(input_h_desc.num_layers() == output_c_desc.num_layers() &&
        input_h_desc.batch_size() == output_c_desc.batch_size() &&
        input_h_desc.data_size() == output_c_desc.data_size())) {
    LOG(ERROR) << "Invalid output_h shape";
    return false;
  }

  return true;
}

bool CheckRNNParameterSize(const CudnnHandle& cudnn,
                           const CudnnRnnDescriptor& rnn_desc,
                           const CudnnRnnSequenceTensorDescriptor& input_desc) {
  size_t params_size_in_bytes = 0;
  cudnnStatus_t status = cudnnGetRNNParamsSize(
      /*handle=*/cudnn.handle(), rnn_desc.handle() /*rnnDesc*/,
      input_desc.handles()[0] /*xDesc*/, /*sizeInBytes=*/&params_size_in_bytes,
      rnn_desc.data_type() /*dataType*/);
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "Unable to check RNN param size: " << ToString(status);
    return false;
  }
  return static_cast<int64>(params_size_in_bytes) ==
         rnn_desc.ParamsSizeInBytes();
}

bool CreateRnnWorkspace(Stream* stream, const CudnnHandle& cudnn,
                        const CudnnRnnDescriptor& rnn_desc,
                        const CudnnRnnSequenceTensorDescriptor& input_desc,
                        ScratchAllocator* workspace_allocator,
                        DeviceMemory<uint8>* workspace) {
  // Query the workspace size.
  size_t workspace_size_in_bytes = 0;
  cudnnStatus_t status = cudnnGetRNNWorkspaceSize(
      /*handle=*/cudnn.handle(), /*rnnDesc=*/rnn_desc.handle(),
      /*seqLength=*/input_desc.seq_length(), /*xDesc=*/input_desc.handles(),
      /*sizeInBytes=*/&workspace_size_in_bytes);
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "Unable to query workspace size: " << ToString(status);
    return false;
  }
  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (*workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << port::StrCat("Failed to allocate RNN workspace of ",
                                 workspace_size_in_bytes, " bytes.");
      return false;
    }
  } else {
    *workspace = DeviceMemory<uint8>();
  }
  return true;
}

}  // namespace

template <class T>
bool CudnnSupport::DoRnnForwardImpl(
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
  // extract model parameters
  RnnModelDims model_dims;
  bool res = ExtractAndCheckRnnForward(
      rnn_desc, input_desc, input_data, input_h_desc, input_h_data,
      input_c_desc, input_c_data, params, output_desc, *output_data,
      output_h_desc, *output_h_data, output_c_desc, *output_c_data,
      &model_dims);
  if (!res) {
    LOG(ERROR) << "Invalid parameters for RNN Model";
    return false;
  }

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  // check params size
  if (!CheckRNNParameterSize(cudnn, rnn_desc, input_desc)) {
    LOG(ERROR) << "Invalid parameters";
    return false;
  }

  // create the workspace
  DeviceMemory<uint8> workspace;
  if (!CreateRnnWorkspace(stream, cudnn, rnn_desc, input_desc,
                          workspace_allocator, &workspace)) {
    LOG(ERROR) << "Unable to create rnn workspace";
    return false;
  }

  // query the reserve space size
  // allocate the reserve space
  DeviceMemory<uint8> reserve_space;
  if (is_training) {
    size_t reserve_space_size_in_bytes = 0;
    cudnnStatus_t status = cudnnGetRNNTrainingReserveSize(
        cudnn.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        /*seqLength=*/model_dims.seq_length, input_desc.handles() /*xDesc*/,
        /*sizeInBytes=*/&reserve_space_size_in_bytes);
    if (status != CUDNN_STATUS_SUCCESS) {
      LOG(ERROR) << "Unable to query reserve space size: " << ToString(status);
      return false;
    }

    if (reserve_space_size_in_bytes > 0) {
      auto allocated = reserve_space_allocator->AllocateBytes(
          stream, reserve_space_size_in_bytes);
      if (!allocated.ok() ||
          (reserve_space = allocated.ValueOrDie()) == nullptr) {
        LOG(ERROR) << "Failed to allocate RNN reserve space of "
                   << reserve_space_size_in_bytes << " bytes.";
        return false;
      }
    }
  }

  std::unique_ptr<CUDATimer, TimerDeleter> timer;
  const bool is_profiling = output_profile_result != nullptr;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));
    if (!timer->Init()) {
      return false;
    }
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Start(AsCUDAStream(stream))) {
      return false;
    }
  }
  // make the forward call
  cudnnStatus_t status;
  if (!is_training) {
    status = cudnnRNNForwardInference(
        cudnn.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        model_dims.seq_length /*seqLength*/, input_desc.handles() /*xDesc*/,
        input_data.opaque() /*x*/, input_h_desc.handle() /*hxDesc*/,
        input_h_data.opaque() /*hx*/, input_c_desc.handle() /*cxDesc*/,
        input_c_data.opaque() /*cx*/, rnn_desc.params_handle() /*wDesc*/,
        params.opaque() /*w*/, output_desc.handles() /*yDesc*/,
        output_data->opaque() /*y*/, output_h_desc.handle() /*hyDesc*/,
        output_h_data->opaque() /*hy*/, output_c_desc.handle() /*cyDesc*/,
        output_c_data->opaque() /*cy*/, workspace.opaque() /*workspace*/,
        workspace.size() /*workSpaceSizeInBytes*/);
  } else {
    status = cudnnRNNForwardTraining(
        cudnn.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        model_dims.seq_length /*seqLength*/, input_desc.handles() /*xDesc*/,
        input_data.opaque() /*x*/, input_h_desc.handle() /*hxDesc*/,
        input_h_data.opaque() /*hx*/, input_c_desc.handle() /*cxDesc*/,
        input_c_data.opaque() /*cx*/, rnn_desc.params_handle() /*wDesc*/,
        params.opaque() /*w*/, output_desc.handles() /*yDesc*/,
        output_data->opaque() /*y*/, output_h_desc.handle() /*hyDesc*/,
        output_h_data->opaque() /*hy*/, output_c_desc.handle() /*cyDesc*/,
        output_c_data->opaque() /*cy*/, workspace.opaque() /*workspace*/,
        workspace.size() /*workSpaceSizeInBytes*/,
        reserve_space.opaque() /*reserveSpace*/,
        reserve_space.size() /*reserveSpaceSizeInBytes*/);
  }
  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      return false;
    }
    if (status == CUDNN_STATUS_SUCCESS) {
      auto algo_desc = rnn_desc.algorithm_config().algorithm();
      output_profile_result->set_algorithm(algo_desc);
      output_profile_result->set_elapsed_time_in_ms(
          timer->GetElapsedMilliseconds());
    }
  }
  if (status != CUDNN_STATUS_SUCCESS) {
    // Silently return when we are profiling.
    if (!is_profiling) {
      LOG(ERROR) << "Failed to call "
                 << (is_training ? "cudnnRNNForwardTraining "
                                 : "cudnnRNNForwardInference ")
                 << ToString(status);
      return false;
    }
  }

  return true;
}

template <class T>
bool CudnnSupport::DoRnnBackwardImpl(
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
  // extract model parameters
  RnnModelDims model_dims;
  bool res = ExtractAndCheckRnnForward(
      rnn_desc, input_desc, input_data, input_h_desc, input_h_data,
      input_c_desc, input_c_data, params, output_desc, output_data,
      output_h_desc, output_h_data, output_c_desc, output_c_data, &model_dims);
  if (!res) {
    LOG(ERROR) << "Invalid parameters for RNN Model";
    return false;
  }

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  // check params size
  if (!CheckRNNParameterSize(cudnn, rnn_desc, input_desc)) {
    LOG(ERROR) << "Invalid parameters";
    return false;
  }

  // create the workspace
  DeviceMemory<uint8> workspace;
  if (!CreateRnnWorkspace(stream, cudnn, rnn_desc, input_desc,
                          workspace_allocator, &workspace)) {
    LOG(ERROR) << "Unable to create rnn workspace";
    return false;
  }

  std::unique_ptr<CUDATimer, TimerDeleter> timer;
  const bool is_profiling = output_profile_result != nullptr;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));
    if (!timer->Init()) {
      return false;
    }
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Start(AsCUDAStream(stream))) {
      return false;
    }
  }
  // make the backward data call
  cudnnStatus_t status = cudnnRNNBackwardData(
      cudnn.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
      model_dims.seq_length /*seqLength*/, output_desc.handles() /*yDesc*/,
      output_data.opaque() /*y*/, output_desc.handles() /*dyDesc*/,
      output_backprop_data.opaque() /*dy*/, output_h_desc.handle() /*dhyDesc*/,
      output_h_backprop_data.opaque() /*dhy*/,
      output_c_desc.handle() /*dcyDesc*/,
      output_c_backprop_data.opaque() /*dcy*/,
      rnn_desc.params_handle() /*wDesc*/, params.opaque() /*w*/,
      input_h_desc.handle() /*hxDesc*/, input_h_data.opaque() /*hx*/,
      input_c_desc.handle() /*cxDesc*/, input_c_data.opaque() /*cx*/,
      input_desc.handles() /*dxDesc*/, input_backprop_data->opaque() /*dx*/,
      input_h_desc.handle() /*dhxDesc*/,
      input_h_backprop_data->opaque() /*dhx*/,
      input_c_desc.handle() /*dcxDesc*/,
      input_c_backprop_data->opaque() /*dcx*/, workspace.opaque() /*workspace*/,
      workspace.size() /*workSpaceSizeInBytes*/,
      reserve_space_data->opaque() /*reserveSpace*/,
      reserve_space_data->size() /*reserveSpaceSizeInBytes*/);

  if (status != CUDNN_STATUS_SUCCESS) {
    if (is_profiling) {
      timer->Stop(AsCUDAStream(stream));
    }
    LOG(ERROR) << "Failed to call cudnnRNNBackwardData: " << ToString(status);
    return false;
  }

  if (params_backprop_data != nullptr) {
    // Clear the dw to zeros.
    stream->ThenMemZero(params_backprop_data, params_backprop_data->size());
    // make the backward weight call
    status = cudnnRNNBackwardWeights(
        cudnn.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        model_dims.seq_length /*seqLength*/, input_desc.handles() /*xDesc*/,
        input_data.opaque() /*x*/, input_h_desc.handle() /*hxDesc*/,
        input_h_data.opaque() /*hx*/, output_desc.handles() /*yDesc*/,
        output_data.opaque() /*y*/, workspace.opaque() /*workspace*/,
        workspace.size() /*workSpaceSizeInBytes*/,
        rnn_desc.params_handle() /*dwDesc*/,
        params_backprop_data->opaque() /*dw*/,
        reserve_space_data->opaque() /*reserveSpace*/,
        reserve_space_data->size() /*reserveSpaceSizeInBytes*/);
    if (status != CUDNN_STATUS_SUCCESS) {
      if (is_profiling) {
        timer->Stop(AsCUDAStream(stream));
      }
      LOG(ERROR) << "Failed to call cudnnRNNBackwardWeights: "
                 << ToString(status);
      return false;
    }
  }
  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      return false;
    }
    auto algo_desc = rnn_desc.algorithm_config().algorithm();
    output_profile_result->set_algorithm(algo_desc);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }

  return true;
}

#endif  // CUDNN_VERSION

port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
CudnnSupport::createRnnDescriptor(
    int num_layers, int hidden_size, int input_size, int batch_size,
    dnn::RnnInputMode input_mode, dnn::RnnDirectionMode direction_mode,
    dnn::RnnMode rnn_mode, dnn::DataType data_type,
    const dnn::AlgorithmConfig& algorithm_config, float dropout, uint64 seed,
    ScratchAllocator* state_allocator) {
#if CUDNN_VERSION >= 5000
  // Setting up a cudnnRNNDescriptor requires a cuDNN handle, but because it's
  // not enqueueing anything into a stream, we pass in the null stream.
  auto cudnn = cudnn_->GetHandle(parent_, /*stream=*/nullptr);
  std::unique_ptr<CudnnRnnDescriptor> rnn_desc(new CudnnRnnDescriptor(
      cudnn, num_layers, hidden_size, input_size, batch_size,
      ToCudnnRnnInputMode(input_mode), ToCudnnRnnDirectionMode(direction_mode),
      ToCudnnRnnMode(rnn_mode), ToCudnnDataType(data_type),
      GetRnnComputeType(data_type), algorithm_config, dropout, seed,
      state_allocator));
  if (!rnn_desc->ok()) {
    return rnn_desc->Status();
  }
  return port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>(
      std::move(rnn_desc));
#else
  string error_msg =
      port::StrCat("createRnnDescriptor needs at least Cudnn 5.0 to work. ",
                   "Current Cudnn version: ", CUDNN_VERSION, ". ");
  LOG(ERROR) << error_msg;
  return port::Status(port::error::UNIMPLEMENTED, error_msg);
#endif  // CUDNN_VERSION
}

port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
CudnnSupport::createRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                                int data_size,
                                                dnn::DataType data_type) {
#if CUDNN_VERSION >= 5000
  std::unique_ptr<CudnnRnnSequenceTensorDescriptor> seq_desc(
      new CudnnRnnSequenceTensorDescriptor(parent_, seq_length, batch_size,
                                           data_size,
                                           ToCudnnDataType(data_type)));
  if (!seq_desc->ok()) {
    return seq_desc->Status();
  }
  return port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>(
      std::move(seq_desc));
#else
  string error_msg = port::StrCat(
      "createRnnSequenceTensorDescriptor needs at least Cudnn 5.0 to work. ",
      "Current Cudnn version: ", CUDNN_VERSION, ". ");
  LOG(ERROR) << error_msg;
  return port::Status(port::error::UNIMPLEMENTED, error_msg);
#endif  // CUDNN_VERSION
}

port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
CudnnSupport::createRnnStateTensorDescriptor(int num_layer, int batch_size,
                                             int data_size,
                                             dnn::DataType data_type) {
#if CUDNN_VERSION >= 5000
  std::unique_ptr<CudnnRnnStateTensorDescriptor> state_desc(
      new CudnnRnnStateTensorDescriptor(parent_, num_layer, batch_size,
                                        data_size, ToCudnnDataType(data_type)));
  if (!state_desc->ok()) {
    return state_desc->Status();
  }
  return port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>(
      std::move(state_desc));
#else
  string error_msg = port::StrCat(
      "createRnnStateTensorDescriptor needs at least Cudnn 5.0 to work. ",
      "Current Cudnn version: ", CUDNN_VERSION, ". ");
  LOG(ERROR) << error_msg;
  return port::Status(port::error::UNIMPLEMENTED, error_msg);
#endif  // CUDNN_VERSION
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
#if CUDNN_VERSION >= 5000
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

  return DoRnnForwardImpl<Eigen::half>(
      stream, cudnn_rnn_desc, cudnn_input_desc, input_data, cudnn_input_h_desc,
      input_h_data, cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
      output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
      output_c_data, is_training, reserve_space_allocator, workspace_allocator,
      output_profile_result);
#else
  return false;
#endif  // CUDNN_VERSION
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
#if CUDNN_VERSION >= 5000
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

  return DoRnnForwardImpl<float>(
      stream, cudnn_rnn_desc, cudnn_input_desc, input_data, cudnn_input_h_desc,
      input_h_data, cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
      output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
      output_c_data, is_training, reserve_space_allocator, workspace_allocator,
      output_profile_result);
#else
  return false;
#endif  // CUDNN_VERSION
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
#if CUDNN_VERSION >= 5000
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

  return DoRnnForwardImpl<double>(
      stream, cudnn_rnn_desc, cudnn_input_desc, input_data, cudnn_input_h_desc,
      input_h_data, cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
      output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
      output_c_data, is_training, reserve_space_allocator, workspace_allocator,
      output_profile_result);
#else
  return false;
#endif  // CUDNN_VERSION
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
#if CUDNN_VERSION >= 5000
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

  return DoRnnBackwardImpl<Eigen::half>(
      stream, cudnn_rnn_desc, cudnn_input_desc, input_data, cudnn_input_h_desc,
      input_h_data, cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
      output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
      output_c_data, output_backprop_data, output_h_backprop_data,
      output_c_backprop_data, input_backprop_data, input_h_backprop_data,
      input_c_backprop_data, params_backprop_data, reserve_space_data,
      workspace_allocator, output_profile_result);
#else
  return false;
#endif  // CUDNN_VERSION
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
#if CUDNN_VERSION >= 5000
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

  return DoRnnBackwardImpl<float>(
      stream, cudnn_rnn_desc, cudnn_input_desc, input_data, cudnn_input_h_desc,
      input_h_data, cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
      output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
      output_c_data, output_backprop_data, output_h_backprop_data,
      output_c_backprop_data, input_backprop_data, input_h_backprop_data,
      input_c_backprop_data, params_backprop_data, reserve_space_data,
      workspace_allocator, output_profile_result);
#else
  return false;
#endif  // CUDNN_VERSION
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
#if CUDNN_VERSION >= 5000
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

  return DoRnnBackwardImpl<double>(
      stream, cudnn_rnn_desc, cudnn_input_desc, input_data, cudnn_input_h_desc,
      input_h_data, cudnn_input_c_desc, input_c_data, params, cudnn_output_desc,
      output_data, cudnn_output_h_desc, output_h_data, cudnn_output_c_desc,
      output_c_data, output_backprop_data, output_h_backprop_data,
      output_c_backprop_data, input_backprop_data, input_h_backprop_data,
      input_c_backprop_data, params_backprop_data, reserve_space_data,
      workspace_allocator, output_profile_result);
#else
  return false;
#endif  // CUDNN_VERSION
}

namespace {

inline cudnnConvolutionFwdAlgo_t GetCudnnConvolutionForwardAlgo(
    const CudnnHandle& cudnn, const ScopedTensorDescriptor& input_nd,
    const ScopedFilterDescriptor& filter,
    const ScopedConvolutionDescriptor& conv,
    const ScopedTensorDescriptor& output_nd, bool specify_workspace_limit,
    size_t memory_limit_bytes) {
  cudnnConvolutionFwdPreference_t preference =
      specify_workspace_limit ? CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
                              : CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;

  cudnnConvolutionFwdAlgo_t algo_to_use;
  auto status = cudnnGetConvolutionForwardAlgorithm(
      cudnn.handle(), input_nd.handle(), filter.handle(), conv.handle(),
      output_nd.handle(), preference, memory_limit_bytes, &algo_to_use);
  CHECK_EQ(status, CUDNN_STATUS_SUCCESS)
      << "Unable to find a suitable algorithm for doing forward convolution";
  return algo_to_use;
}

dnn::AlgorithmDesc GetCudnnConvolutionForwardAlgorithm(
    Stream* stream, const CudnnHandle& cudnn,
    const dnn::AlgorithmConfig& algorithm_config, bool is_profiling,
    const ScopedTensorDescriptor& input_nd,
    const ScopedFilterDescriptor& filter,
    const ScopedConvolutionDescriptor& conv,
    const ScopedTensorDescriptor& output_nd,
    ScratchAllocator* scratch_allocator, DeviceMemory<uint8>* scratch) {
  cudnnConvolutionFwdAlgo_t algo;
  bool use_tensor_ops;
  if (algorithm_config.algorithm().is_default()) {
    use_tensor_ops = true;

    auto memory_limit_bytes =
        scratch_allocator == nullptr
            ? 0
            : scratch_allocator->GetMemoryLimitInBytes(stream);
    if (memory_limit_bytes < 0) {
      memory_limit_bytes = 0;
    }

    algo = GetCudnnConvolutionForwardAlgo(
        cudnn, input_nd, filter, conv, output_nd,
        /*specify_workspace_limit=*/scratch_allocator != nullptr,
        memory_limit_bytes);
  } else {
    use_tensor_ops = algorithm_config.algorithm().tensor_ops_enabled();
    algo = ToConvForwardAlgo(algorithm_config.algorithm());
  }
  size_t size_in_bytes;
  auto status = cudnnGetConvolutionForwardWorkspaceSize(
      cudnn.handle(),
      /*xDesc=*/input_nd.handle(),
      /*wDesc=*/filter.handle(), /*convDesc=*/conv.handle(),
      /*yDesc=*/output_nd.handle(), /*algo=*/algo,
      /*sizeInBytes=*/&size_in_bytes);
  int64 size_in_bytes_int64 = size_in_bytes;
  if (TF_PREDICT_FALSE(status != CUDNN_STATUS_SUCCESS)) {
    CHECK(is_profiling) << "Cannot query the size of workspace needed "
                           "for the specified algorithm: "
                        << algorithm_config.algorithm().algo_id() << " "
                        << ToString(status);
    // Silently return when we are profiling.
    return dnn::AlgorithmDesc();
  }
  if (TF_PREDICT_FALSE(size_in_bytes_int64 < 0)) {
    LOG(WARNING) << "cudnnGetConvolutionForwardWorkspaceSize() returned "
                    "negative sizeInBytes value. This could be a cudnn bug.";
    if (TF_PREDICT_TRUE(is_profiling)) {
      return dnn::AlgorithmDesc();
    }
  } else if (size_in_bytes_int64 > 0) {
    port::StatusOr<DeviceMemory<uint8>> allocated;
    if (TF_PREDICT_TRUE(scratch_allocator)) {
      allocated = scratch_allocator->AllocateBytes(stream, size_in_bytes);
      if (TF_PREDICT_TRUE(allocated.ok())) {
        *scratch = allocated.ValueOrDie();
      } else {
        if (TF_PREDICT_TRUE(is_profiling)) {
          // Silently return when we are profiling.
          return dnn::AlgorithmDesc();
        }
        LOG(WARNING) << allocated.status().error_message();
        // For the int8 case, we fail at this point since the no_scratch
        // algorithm should be set to dnn::kDefaultAlgorithm.
        CHECK(!algorithm_config.algorithm_no_scratch().is_default())
            << "The primary convolution algorithm failed memory allocation, "
               "while a secondary algorithm is not provided.";
      }
    }
    if (TF_PREDICT_FALSE(!allocated.ok())) {
      if (algorithm_config.algorithm_no_scratch().is_default()) {
        use_tensor_ops = true;
        algo = GetCudnnConvolutionForwardAlgo(
            cudnn, input_nd, filter, conv, output_nd,
            /*specify_workspace_limit=*/false, 0);
      } else {
        use_tensor_ops = algorithm_config.algorithm().tensor_ops_enabled();
        algo = ToConvForwardAlgo(algorithm_config.algorithm_no_scratch());
      }
    }
  }

  return dnn::AlgorithmDesc(algo, use_tensor_ops);
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
// forward convolution. Before cudnn v5.1 it works fine but since cudnn v5.1
// it is turned off due to memory corruption caused by some shapes with this
// algorithm.
// Before NVIDIA fixes the memory corruption bug, users can explicitly
// enable the algorithm through an env-var "TF_ENABLE_FFT_TILING_FORWARD=1".
struct FftTilingForward {
  static constexpr const char* kName = "TF_ENABLE_FFT_TILING_FORWARD";
  // TODO(yangzihao): turn the default to True when the memory corruption bug
  // is fixed.
  static constexpr bool kDefaultFlag = CUDNN_VERSION < 5100;
};

// A helper struct to decide whether to enable the WINOGRAD_NONFUSED algorithms.
// By default it is turned on, users can explicitly disable them through an
// env-var "TF_ENABLE_WINOGRAD_NONFUSED=0".
// https://github.com/tensorflow/tensorflow/pull/4901
struct WinogradNonfused {
  static constexpr const char* kName = "TF_ENABLE_WINOGRAD_NONFUSED";
  // NVIDIA has fixed winograd nonfused bug for cudnn v>=7.
  // For cudnn v>=5.1, we have a workaround and for any lower version, we
  // disable it by default.
  static constexpr bool kDefaultFlag = CUDNN_VERSION >= 5100;
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

// A helper function to return the internal compute type for
// RNNs in cudnn.
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

}  // namespace

template <class T>
bool CudnnSupport::DoConvolveImpl(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<T>& input_data,
    const FilterDescriptor& filter_descriptor,
    const DeviceMemory<T>& filter_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& output_descriptor, DeviceMemory<T>* output_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  cudnnDataType_t cudnn_type = GetCudnnDataType<T>();
  ScopedTensorDescriptor input_nd(input_descriptor, cudnn_type);
  ScopedTensorDescriptor output_nd(output_descriptor, cudnn_type);
  ScopedFilterDescriptor filter(filter_descriptor, cudnn_type);
  ScopedConvolutionDescriptor conv(convolution_descriptor,
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
  cudnnConvolutionFwdAlgo_t algo;
  bool use_tensor_ops;
  DeviceMemory<uint8> scratch;

  // TODO(pauldonnelly): Replace the following code with a call to
  //   GetCudnnConvolutionForwardAlgorithm().
  if (algorithm_config.algorithm().is_default()) {
    // With the default algorithm, use Cudnn's heuristics.
    auto get_algorithm = [&](bool specify_limit) {
      cudnnConvolutionFwdPreference_t preference =
          specify_limit ? CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
                        : CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;

      auto memory_limit_bytes =
          scratch_allocator == nullptr
              ? 0
              : scratch_allocator->GetMemoryLimitInBytes(stream);
      if (memory_limit_bytes < 0) {
        memory_limit_bytes = 0;
      }

      cudnnConvolutionFwdAlgo_t algo_to_use;
      auto status = cudnnGetConvolutionForwardAlgorithm(
          cudnn.handle(), input_nd.handle(), filter.handle(), conv.handle(),
          output_nd.handle(),
          /*preference=*/preference,
          /*memoryLimitInBytes=*/memory_limit_bytes,
          /*algo=*/&algo_to_use);
      CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << "Unable to find a suitable "
                                                "algorithm for doing forward "
                                                "convolution";
      return algo_to_use;
    };

    algo = get_algorithm(/*specify_limit=*/scratch_allocator != nullptr);
    use_tensor_ops = true;
    if (scratch_allocator != nullptr) {
      size_t size_in_bytes;
      auto status = cudnnGetConvolutionForwardWorkspaceSize(
          cudnn.handle(),
          /*xDesc=*/input_nd.handle(),
          /*wDesc=*/filter.handle(), /*convDesc=*/conv.handle(),
          /*yDesc=*/output_nd.handle(), /*algo=*/algo,
          /*sizeInBytes=*/&size_in_bytes);
      int64 size_in_bytes_int64 = size_in_bytes;
      if (status == CUDNN_STATUS_SUCCESS && size_in_bytes_int64 != 0) {
        if (size_in_bytes_int64 > 0) {
          auto allocated =
              scratch_allocator->AllocateBytes(stream, size_in_bytes);
          if (allocated.ok()) {
            scratch = allocated.ValueOrDie();
          } else {
            LOG(WARNING) << allocated.status().error_message();
          }
        } else {
          LOG(WARNING)
              << "cudnnGetConvolutionForwardWorkspaceSize() returned "
                 "negative sizeInBytes value. This could be a cudnn bug.";
        }
      }
    }

    // If we didn't allocate any scratch space (perhaps because of failed
    // allocation), we force a switch back to the "no workspace" algorithm.
    if (scratch == nullptr) {
      algo = get_algorithm(/*specify_limit=*/false);
    }
  } else {
    // An algorithm has been specified.
    dnn::AlgorithmDesc algotype = algorithm_config.algorithm();
    algo = ToConvForwardAlgo(algotype);
    use_tensor_ops = algotype.tensor_ops_enabled();
    conv.set_use_tensor_op_math(use_tensor_ops);
    size_t size_in_bytes;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
        cudnn.handle(),
        /*xDesc=*/input_nd.handle(),
        /*wDesc=*/filter.handle(), /*convDesc=*/conv.handle(),
        /*yDesc=*/output_nd.handle(), /*algo=*/algo,
        /*sizeInBytes=*/&size_in_bytes);
    if (status != CUDNN_STATUS_SUCCESS) {
      if (is_profiling) {
        // Silently return when we are profiling.
        return false;
      }
      LOG(FATAL) << "Cannot query the size of workspace needed for the given "
                    "algorithm: "
                 << algorithm_config.algorithm().algo_id();
    }
    int64 size_in_bytes_int64 = size_in_bytes;
    if (size_in_bytes_int64 > 0) {
      if (scratch_allocator == nullptr) {
        LOG(FATAL) << "An allocator must be specified when scratch memory is "
                      "needed";
      }
      auto allocated = scratch_allocator->AllocateBytes(stream, size_in_bytes);
      if (is_profiling && !allocated.ok()) {
        // Silently return when we are profiling.
        return false;
      }
      if (allocated.ok()) {
        scratch = allocated.ValueOrDie();
      } else {
        LOG(WARNING) << allocated.status().error_message();
      }
      if (scratch == nullptr) {
        CHECK(!algorithm_config.algorithm_no_scratch().is_default())
            << "The primary convolution algorithm failed memory allocation, "
               "while a secondary algorithm is not provided.";
        dnn::AlgorithmDesc algotype = algorithm_config.algorithm_no_scratch();
        algo = ToConvForwardAlgo(algotype);
        use_tensor_ops = algotype.tensor_ops_enabled();
        conv.set_use_tensor_op_math(use_tensor_ops);
      }
    } else if (size_in_bytes_int64 < 0) {
      LOG(WARNING) << "cudnnGetConvolutionForwardWorkspaceSize() returned "
                      "negative sizeInBytes value. This could be a cudnn bug.";
    }
  }
  std::unique_ptr<CUDATimer> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));  // NOLINT
    if (!timer->Init()) {
      return false;
    }
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Start(AsCUDAStream(stream))) {
      timer->Destroy();
      return false;
    }
  }
  auto status = cudnnConvolutionForward(
      cudnn.handle(),
      /*alpha=*/alpha, /*srcDesc=*/input_nd.handle(),
      /*srcData=*/input_data.opaque(), /*filterDesc=*/filter.handle(),
      /*filterData=*/filter_data.opaque(), /*convDesc=*/conv.handle(),
      /*algo=*/algo, /*workSpace=*/scratch.opaque(),
      /*workSpaceSizeInBytes=*/scratch.size(), /*beta=*/beta,
      /*destDesc=*/output_nd.handle(), /*destData=*/output_data->opaque());

  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      timer->Destroy();
      return false;
    }
    if (status == CUDNN_STATUS_SUCCESS) {
      dnn::AlgorithmDesc algotype(algo, use_tensor_ops);
      output_profile_result->set_algorithm(algotype);
      output_profile_result->set_elapsed_time_in_ms(
          timer->GetElapsedMilliseconds());
    }
    timer->Destroy();
  }

  if (status != CUDNN_STATUS_SUCCESS) {
    // Silently return when we are profiling.
    if (!is_profiling) {
      LOG(ERROR) << "failed to enqueue convolution on stream: "
                 << ToString(status);
    }
    return false;
  }

  return true;
}

template <typename Type, typename BiasType, typename ScaleType,
          int cudnn_data_type, int cudnn_compute_type>
bool CudnnSupport::DoFusedConvolveImpl(
    Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
    const DeviceMemory<Type>& conv_input_data, ScaleType conv_input_scale,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<Type>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const DeviceMemory<Type>& side_input_data, ScaleType side_input_scale,
    const dnn::BatchDescriptor& bias_descriptor,
    const DeviceMemory<BiasType>& biases, dnn::ActivationMode activation_mode,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<Type>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
#if CUDNN_VERSION < 6000
  LOG(ERROR) << "cudnnConvolutionBiasActivationForward() is only "
                "supported for cuDNN version >= 6";
  return false;
#else
  ScopedTensorDescriptor conv_input_nd(
      conv_input_descriptor, static_cast<cudnnDataType_t>(cudnn_data_type));
  ScopedTensorDescriptor output_nd(
      output_descriptor, static_cast<cudnnDataType_t>(cudnn_data_type));
  ScopedFilterDescriptor filter(filter_descriptor,
                                static_cast<cudnnDataType_t>(cudnn_data_type));
  ScopedTensorDescriptor bias_nd(bias_descriptor, CUDNN_DATA_FLOAT);
  ScopedConvolutionDescriptor conv(
      convolution_descriptor, static_cast<cudnnDataType_t>(cudnn_compute_type));

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  const bool is_profiling = output_profile_result != nullptr;
  DeviceMemory<uint8> scratch;
  dnn::AlgorithmDesc algotype = GetCudnnConvolutionForwardAlgorithm(
      stream, cudnn, algorithm_config, is_profiling, conv_input_nd, filter,
      conv, output_nd, scratch_allocator, &scratch);
  if (algotype.is_default()) {
    if (!is_profiling) {
      LOG(ERROR) << "No suitable algorithm found";
    }
    return false;
  }
  auto algo = static_cast<cudnnConvolutionFwdAlgo_t>(algotype.algo_id());
  conv.set_use_tensor_op_math(algotype.tensor_ops_enabled());

  if (activation_mode != dnn::ActivationMode::kRelu) {
    LOG(ERROR) << "cudnnConvolutionBiasActivationForward() only supports Relu "
                  "activation.";
    return false;
  }

  std::unique_ptr<CUDATimer> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));  // NOLINT
    if (!timer->Init()) {
      return false;
    }
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Start(AsCUDAStream(stream))) {
      timer->Destroy();
      return false;
    }
  }
  // CUDNN v6 only supports CUDNN_NOT_PROPAGATE_NAN as the reluNanOpt for
  // activation descriptor. Note that this will change the nan propagation
  // behavior from separate conv, bias, and relu (which by default is
  // CUDNN_PROPAGATE_NAN.
  ScopedActivationDescriptor activation_desc(
      activation_mode, CUDNN_NOT_PROPAGATE_NAN, output_descriptor.value_max());
  auto side_input_data_ptr = (side_input_scale == 0) ? output_data->opaque()
                                                     : side_input_data.opaque();

  VLOG(2) << "\nconv_input_scale = " << conv_input_scale
          << "\nconv_input_nd.handle() = " << conv_input_nd.handle()
          << "\nconv_input_data.opaque() = " << conv_input_data.opaque()
          << "\nfilter.handle() = " << filter.handle()
          << "\nfilter_data.opaque() = " << filter_data.opaque()
          << "\nconv.handle() = " << conv.handle() << "\nalgo = " << algo
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

  auto status = cudnnConvolutionBiasActivationForward(
      cudnn.handle(),
      /*alpha1=*/&conv_input_scale,
      /*srcDesc=*/conv_input_nd.handle(), /*srcData=*/conv_input_data.opaque(),
      /*filterDesc=*/filter.handle(), /*filterData=*/filter_data.opaque(),
      /*convDesc=*/conv.handle(), algo, /*workSpace=*/scratch.opaque(),
      /*workSpaceSizeInBytes=*/scratch.size(), /*alpha2=*/&side_input_scale,
      /*zDesc=*/output_nd.handle(), /*z=*/side_input_data_ptr,
      /*biasDesc=*/bias_nd.handle(), /*bias=*/biases.opaque(),
      /*activationDesc=*/activation_desc.handle(),
      /*destDesc=*/output_nd.handle(), /*destData=*/output_data->opaque());

  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      timer->Destroy();
      return false;
    }
    if (status == CUDNN_STATUS_SUCCESS) {
      output_profile_result->set_algorithm(algotype);
      output_profile_result->set_elapsed_time_in_ms(
          timer->GetElapsedMilliseconds());
    }
    timer->Destroy();
  }

  if (status != CUDNN_STATUS_SUCCESS) {
    // Silently return when we are profiling.
    if (!is_profiling) {
      LOG(ERROR) << "failed to enqueue convolution on stream: "
                 << ToString(status);
    }
    return false;
  }

  return true;
#endif  // CUDNN_VERSION < 6000
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
#if CUDNN_VERSION >= 5000
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
#endif
    // clang-format on
  };
  if (CudnnEnvVar<FftTilingForward>::IsEnabled()) {
    algo_types.push_back(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING);
  }
#if CUDNN_VERSION >= 5100
  if (CudnnEnvVar<WinogradNonfused>::IsEnabled() && with_winograd_nonfused) {
    algo_types.push_back(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
  }
#endif

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
#if CUDNN_VERSION >= 6000
    CUDNN_RNN_ALGO_STANDARD,
    CUDNN_RNN_ALGO_PERSIST_STATIC,
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
#endif
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
#if CUDNN_VERSION >= 5000
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
#endif
    // clang-format on
  };
#if CUDNN_VERSION >= 5100
  if (CudnnEnvVar<WinogradNonfused>::IsEnabled() && with_winograd_nonfused) {
    algo_types.push_back(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);
  }
#endif

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
      // clang-format on
  };
#if CUDNN_VERSION >= 5100
  if (CudnnEnvVar<WinogradNonfused>::IsEnabled() && with_winograd_nonfused) {
    algo_types.push_back(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED);
  }
#endif

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
  return DoBatchNormalizationForwardImpl<float, float>(
      stream, dnn::DataType::kFloat, dnn::DataType::kFloat, x, scale, offset,
      estimated_mean, estimated_variance, x_desc, scale_offset_desc, epsilon, y,
      batch_mean, batch_var, saved_mean, saved_inv_var, is_training,
      std::move(var_to_inv_var), std::move(inv_var_to_var));
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
  return DoBatchNormalizationForwardImpl<Eigen::half, float>(
      stream, dnn::DataType::kHalf, dnn::DataType::kFloat, x, scale, offset,
      estimated_mean, estimated_variance, x_desc, scale_offset_desc, epsilon, y,
      batch_mean, batch_var, saved_mean, saved_inv_var, is_training,
      std::move(var_to_inv_var), std::move(inv_var_to_var));
}

template <class T, class U>
bool CudnnSupport::DoBatchNormalizationForwardImpl(
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
  ScopedTensorDescriptor x_descriptor(x_desc, ToCudnnDataType(input_data_type));
  ScopedTensorDescriptor scale_offset_descriptor(
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

  auto status = CUDNN_STATUS_SUCCESS;
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

    status = cudnnBatchNormalizationForwardTraining(
        cudnn.handle(), mode, &one, &zero, x_descriptor.handle(), x.opaque(),
        x_descriptor.handle(), y->opaque(), scale_offset_descriptor.handle(),
        scale.opaque(), offset.opaque(), 1.0, batch_mean_opaque,
        batch_var_opaque, epsilon, saved_mean->opaque(),
        saved_inv_var->opaque());
#if CUDNN_VERSION < 5000
    CHECK(inv_var_to_var);
    inv_var_to_var();
#endif
  } else {
#if CUDNN_VERSION < 5000
    CHECK(var_to_inv_var);
    const void* maybe_inv_var = var_to_inv_var().opaque();
#else
    const void* maybe_inv_var = estimated_variance.opaque();
#endif
    status = cudnnBatchNormalizationForwardInference(
        cudnn.handle(), mode, &one, &zero, x_descriptor.handle(), x.opaque(),
        x_descriptor.handle(), y->opaque(), scale_offset_descriptor.handle(),
        scale.opaque(), offset.opaque(), estimated_mean.opaque(), maybe_inv_var,
        epsilon);
  }
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to enqueue forward batch normalization on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<float>& y_backprop,
    const DeviceMemory<float>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& mean, const DeviceMemory<float>& inv_var,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<float>* x_backprop, DeviceMemory<float>* scale_backprop,
    DeviceMemory<float>* offset_backprop) {
  return DoBatchNormalizationBackwardImpl(
      stream, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, y_backprop, x, scale, mean,
      inv_var, x_desc, scale_offset_desc, epsilon, x_backprop, scale_backprop,
      offset_backprop);
}

bool CudnnSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<Eigen::half>& y_backprop,
    const DeviceMemory<Eigen::half>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& mean, const DeviceMemory<float>& inv_var,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<Eigen::half>* x_backprop, DeviceMemory<float>* scale_backprop,
    DeviceMemory<float>* offset_backprop) {
  return DoBatchNormalizationBackwardImpl(
      stream, CUDNN_DATA_HALF, CUDNN_DATA_FLOAT, y_backprop, x, scale, mean,
      inv_var, x_desc, scale_offset_desc, epsilon, x_backprop, scale_backprop,
      offset_backprop);
}

template <class T, class U>
bool CudnnSupport::DoBatchNormalizationBackwardImpl(
    Stream* stream, int cudnn_input_type, int cudnn_scale_type,
    const DeviceMemory<T>& y_backprop, const DeviceMemory<T>& x,
    const DeviceMemory<U>& scale, const DeviceMemory<U>& mean,
    const DeviceMemory<U>& inv_var, const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<T>* x_backprop, DeviceMemory<U>* scale_backprop,
    DeviceMemory<U>* offset_backprop) {
  ScopedTensorDescriptor x_descriptor(
      x_desc, static_cast<cudnnDataType_t>(cudnn_input_type));
  ScopedTensorDescriptor scale_offset_descriptor(
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

  auto status = cudnnBatchNormalizationBackward(
      cudnn.handle(), mode, &one, &zero, &one, &zero, x_descriptor.handle(),
      x.opaque(), x_descriptor.handle(), y_backprop.opaque(),
      x_descriptor.handle(), x_backprop->opaque(),
      scale_offset_descriptor.handle(), scale.opaque(),
      scale_backprop->opaque(), offset_backprop->opaque(), epsilon,
      mean.opaque(), inv_var.opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to enqueue backward batch normalization on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoConvolve(
    Stream* stream, const BatchDescriptor& batch_descriptor,
    const DeviceMemory<float>& input_data,
    const FilterDescriptor& filter_descriptor,
    const DeviceMemory<float>& filter_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& output_descriptor, DeviceMemory<float>* output_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveImpl<float>(
      stream, batch_descriptor, input_data, filter_descriptor, filter_data,
      convolution_descriptor, output_descriptor, output_data, scratch_allocator,
      algorithm_config, output_profile_result);
}

bool CudnnSupport::DoConvolve(
    Stream* stream, const BatchDescriptor& batch_descriptor,
    const DeviceMemory<double>& input_data,
    const FilterDescriptor& filter_descriptor,
    const DeviceMemory<double>& filter_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& output_descriptor, DeviceMemory<double>* output_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveImpl<double>(
      stream, batch_descriptor, input_data, filter_descriptor, filter_data,
      convolution_descriptor, output_descriptor, output_data, scratch_allocator,
      algorithm_config, output_profile_result);
}

bool CudnnSupport::DoConvolve(
    Stream* stream, const BatchDescriptor& batch_descriptor,
    const DeviceMemory<Eigen::half>& input_data,
    const FilterDescriptor& filter_descriptor,
    const DeviceMemory<Eigen::half>& filter_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& output_descriptor,
    DeviceMemory<Eigen::half>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveImpl<Eigen::half>(
      stream, batch_descriptor, input_data, filter_descriptor, filter_data,
      convolution_descriptor, output_descriptor, output_data, scratch_allocator,
      algorithm_config, output_profile_result);
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
  return DoFusedConvolveImpl<double, double, double, CUDNN_DATA_DOUBLE,
                             CUDNN_DATA_DOUBLE>(
      stream, conv_input_descriptor, conv_input_data, conv_input_scale,
      filter_descriptor, filter_data, convolution_descriptor, side_input_data,
      side_input_scale, bias_descriptor, biases, activation_mode,
      output_descriptor, output_data, scratch_allocator, algorithm_config,
      output_profile_result);
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
  return DoFusedConvolveImpl<float, float, float, CUDNN_DATA_FLOAT,
                             CUDNN_DATA_FLOAT>(
      stream, conv_input_descriptor, conv_input_data, conv_input_scale,
      filter_descriptor, filter_data, convolution_descriptor, side_input_data,
      side_input_scale, bias_descriptor, biases, activation_mode,
      output_descriptor, output_data, scratch_allocator, algorithm_config,
      output_profile_result);
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
  return DoFusedConvolveImpl<Eigen::half, Eigen::half, float, CUDNN_DATA_HALF,
                             CUDNN_DATA_FLOAT>(
      stream, conv_input_descriptor, conv_input_data, conv_input_scale,
      filter_descriptor, filter_data, convolution_descriptor, side_input_data,
      side_input_scale, bias_descriptor, biases, activation_mode,
      output_descriptor, output_data, scratch_allocator, algorithm_config,
      output_profile_result);
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
#if CUDNN_VERSION < 6000
  LOG(WARNING) << "cudnnConvolutionBiasActivationForward() is only "
                  "supported for cuDNN version >= 6";
  return false;
#else
  int cc_major, cc_minor;
  stream->parent()->GetDeviceDescription().cuda_compute_capability(&cc_major,
                                                                   &cc_minor);
  if (cc_major < 6 || (cc_major == 6 && cc_minor < 1)) {
    LOG(WARNING) << "cudnnConvolutionBiasActivationForward() for int8 is only "
                    "supported on GPUs with compute capability 6.1 or later.";
    return false;
  }
  return DoFusedConvolveImpl<int8, float, float, CUDNN_DATA_INT8x4,
                             CUDNN_DATA_INT32>(
      stream, conv_input_descriptor, conv_input_data, conv_input_scale,
      filter_descriptor, filter_data, convolution_descriptor, side_input_data,
      side_input_scale, bias_descriptor, biases, activation_mode,
      output_descriptor, output_data, scratch_allocator, algorithm_config,
      output_profile_result);
#endif
}

namespace {
// NOTE(keveman): Temporary data layout transformation until cuDNN supports
// kBatchYXDepth for backward pass. This function allocates temporary memory,
// lays out the source data into the temporary but in the kBatchDepthXY
// layout, and returns the temporary memory. The caller is responsible for
// deallocating the temporary. Since the allocation is done using Stream's
// AllocateTemporaryMemory, a later BlockHostUntilDone could be used for
// deallocation.
//
// transform_scratch is populated with a legitimate temporary allocation iff
// the original output data needs to be transformed.
template <class T>
DeviceMemory<T> MaybeTransformLayout(
    Stream* stream, const CudnnHandle& cudnn,
    BatchDescriptor* output_descriptor, DeviceMemory<T> backward_output_data,
    std::unique_ptr<TemporaryDeviceMemory<T>>* transform_scratch) {
  if (output_descriptor->layout() == dnn::DataLayout::kBatchDepthYX) {
    return backward_output_data;
  }
  CHECK(output_descriptor->layout() == dnn::DataLayout::kBatchYXDepth);
  *transform_scratch =
      stream->AllocateTemporaryArray<T>(backward_output_data.ElementCount())
          .ConsumeValueOrDie();
  BatchDescriptor transformed_output_descriptor;
  transformed_output_descriptor.CloneFrom(*output_descriptor);
  transformed_output_descriptor.set_layout(dnn::DataLayout::kBatchDepthYX);
  cudnnDataType_t cudnn_type = GetCudnnDataType<T>();
  ScopedTensorDescriptor orig_out_back_nd(*output_descriptor, cudnn_type);
  ScopedTensorDescriptor transformed_out_back_nd(transformed_output_descriptor,
                                                 cudnn_type);

  float alpha = 1.0f;
  float beta = 0.0f;
  auto status = cudnnTransformTensor(
      cudnn.handle(), &alpha, orig_out_back_nd.handle(),
      backward_output_data.opaque(), &beta, transformed_out_back_nd.handle(),
      (*transform_scratch)->mutable_device_memory()->opaque());

  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(FATAL) << "Failed to transform the data layout.";
  }
  output_descriptor->set_layout(dnn::DataLayout::kBatchDepthYX);
  return (*transform_scratch)->device_memory();
}
}  // namespace

bool CudnnSupport::DoTransformTensor(Stream* stream,
                                     const dnn::BatchDescriptor& input_desc,
                                     dnn::DataType input_type,
                                     const DeviceMemoryBase& input_data,
                                     const dnn::BatchDescriptor& output_desc,
                                     dnn::DataType output_type, float scale,
                                     DeviceMemoryBase* output_data) {
  float beta = 0.0f;
  ScopedTensorDescriptor input_tensor_desc(
      input_desc, ToCudnnDataType(input_type, input_desc.layout()));
  ScopedTensorDescriptor output_tensor_desc(
      output_desc, ToCudnnDataType(output_type, output_desc.layout()));
  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = cudnnTransformTensor(
      cudnn.handle(), &scale, input_tensor_desc.handle(), input_data.opaque(),
      &beta, output_tensor_desc.handle(), output_data->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "Could not transform a tensor with layout "
               << input_desc.ToString() << " and data type "
               << static_cast<int>(input_type) << " to another with layout "
               << output_desc.ToString() << " and data type "
               << static_cast<int>(output_type) << ": " << ToString(status);
    return false;
  }
  return true;
}

template <class T>
bool CudnnSupport::DoConvolveBackwardDataImpl(
    Stream* stream, const FilterDescriptor& filter_descriptor,
    const DeviceMemory<T>& filter_data,
    const BatchDescriptor& output_descriptor_in,
    DeviceMemory<T> backward_output_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& input_descriptor,
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

  // TBD(keveman): remove once cuDNN supports kBatchYXDepth for backward pass.
  BatchDescriptor output_descriptor;
  output_descriptor.CloneFrom(output_descriptor_in);
  std::unique_ptr<TemporaryDeviceMemory<T>> transform_scratch;
  backward_output_data =
      MaybeTransformLayout(stream, cudnn, &output_descriptor,
                           backward_output_data, &transform_scratch);

  ScopedTensorDescriptor out_back_nd(output_descriptor, cudnn_type);
  ScopedTensorDescriptor in_back_nd(input_descriptor, cudnn_type);
  ScopedFilterDescriptor filter(filter_descriptor, cudnn_type);
  ScopedConvolutionDescriptor conv(convolution_descriptor,
                                   GetConvComputeType<T>());

  const bool is_profiling = output_profile_result != nullptr;
  cudnnConvolutionBwdDataAlgo_t algo;
  DeviceMemory<uint8> scratch;

  if (algorithm_config.algorithm().is_default()) {
    // With the default algorithm, use Cudnn's heuristics.
    auto get_algorithm =
        [&](bool specify_limit) -> cudnnConvolutionBwdDataAlgo_t {
      cudnnConvolutionBwdDataPreference_t preference =
          specify_limit ? CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
                        : CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;

      auto memory_limit_bytes =
          scratch_allocator == nullptr
              ? 0
              : scratch_allocator->GetMemoryLimitInBytes(stream);
      if (memory_limit_bytes < 0) {
        memory_limit_bytes = 0;
      }
      cudnnConvolutionBwdDataAlgo_t algo_to_use;
      cudnnStatus_t status = cudnnGetConvolutionBackwardDataAlgorithm(
          cudnn.handle(),
          /*filterDesc=*/filter.handle(),
          /*diffDesc=*/out_back_nd.handle(),
          /*convDesc=*/conv.handle(),
          /*gradDesc=*/in_back_nd.handle(),
          /*preference=*/preference,
          /*memoryLimitInBytes=*/memory_limit_bytes,
          /*algo=*/&algo_to_use);
      CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << "Unable to find a suitable "
                                                "algorithm for doing backward "
                                                "data convolution";
      return algo_to_use;
    };

    algo = get_algorithm(/*specify_limit=*/scratch_allocator != nullptr);

    if (scratch_allocator != nullptr) {
      size_t size_in_bytes;
      auto status = cudnnGetConvolutionBackwardDataWorkspaceSize(
          cudnn.handle(),
          /*filterDesc=*/filter.handle(),
          /*diffDesc=*/out_back_nd.handle(),
          /*convDesc=*/conv.handle(),
          /*gradDesc=*/in_back_nd.handle(),
          /*algo=*/algo,
          /*sizeInBytes=*/&size_in_bytes);
      int64 size_in_bytes_int64 = size_in_bytes;
      if (status == CUDNN_STATUS_SUCCESS && size_in_bytes_int64 != 0) {
        if (size_in_bytes_int64 > 0) {
          auto allocated =
              scratch_allocator->AllocateBytes(stream, size_in_bytes);
          if (allocated.ok()) {
            scratch = allocated.ValueOrDie();
          } else {
            LOG(WARNING) << allocated.status().error_message();
          }
        } else {
          LOG(WARNING)
              << "cudnnGetConvolutionBackwardDataWorkspaceSize() returned "
                 "negative sizeInBytes value. This could be a cudnn bug.";
        }
      }
    }

    // If we didn't allocate any scratch space (perhaps because of failed
    // allocation), we force a switch back to the "no workspace" algorithm.
    if (scratch == nullptr) {
      algo = get_algorithm(/*specify_limit=*/false);
    }
  } else {
    // An algorithm has been specified.
    dnn::AlgorithmDesc algotype = algorithm_config.algorithm();
    algo = ToConvBackwardDataAlgo(algotype);
    conv.set_use_tensor_op_math(algotype.tensor_ops_enabled());
    size_t size_in_bytes;
    auto status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn.handle(),
        /*filterDesc=*/filter.handle(),
        /*diffDesc=*/out_back_nd.handle(),
        /*convDesc=*/conv.handle(),
        /*gradDesc=*/in_back_nd.handle(),
        /*algo=*/algo,
        /*sizeInBytes=*/&size_in_bytes);
    if (status != CUDNN_STATUS_SUCCESS) {
      if (is_profiling) {
        // Silently return when we are profiling.
        return false;
      }
      LOG(FATAL) << "Cannot query the size of workspace needed for the given "
                    "algorithm: "
                 << algorithm_config.algorithm().algo_id();
    }
    int64 size_in_bytes_int64 = size_in_bytes;
    if (size_in_bytes_int64 > 0) {
      if (scratch_allocator == nullptr) {
        LOG(FATAL) << "An allocator must be specified when scratch memory is "
                      "needed";
      }
      auto allocated = scratch_allocator->AllocateBytes(stream, size_in_bytes);
      if (is_profiling && !allocated.ok()) {
        // Silently return when we are profiling.
        return false;
      }
      if (allocated.ok()) {
        scratch = allocated.ValueOrDie();
      } else {
        LOG(WARNING) << allocated.status().error_message();
      }
      if (scratch == nullptr) {
        CHECK(!algorithm_config.algorithm_no_scratch().is_default())
            << "The primary convolution algorithm failed memory allocation, "
               "while a secondary algorithm is not provided.";
        dnn::AlgorithmDesc algotype = algorithm_config.algorithm_no_scratch();
        algo = ToConvBackwardDataAlgo(algotype);
        conv.set_use_tensor_op_math(algotype.tensor_ops_enabled());
      }
    } else if (size_in_bytes_int64 < 0) {
      LOG(WARNING) << "cudnnGetConvolutionBackwardDataWorkspaceSize() returned "
                      "negative sizeInBytes value. This could be a cudnn bug.";
    }
  }

  std::unique_ptr<CUDATimer> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));  // NOLINT
    timer->Init();
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    timer->Start(AsCUDAStream(stream));
  }

#if CUDNN_VERSION >= 5000
  auto status =
      cudnnConvolutionBackwardData(cudnn.handle(),
#else
  auto status =
      cudnnConvolutionBackwardData_v3(cudnn.handle(),
#endif
                                   /*alpha=*/alpha,
                                   /*wDesc=*/filter.handle(),
                                   /*w=*/filter_data.opaque(),
                                   /*dyDesc=*/out_back_nd.handle(),
                                   /*dy=*/backward_output_data.opaque(),
                                   /*convDesc=*/conv.handle(),
                                   /*algo=*/algo,
                                   /*workSpace=*/scratch.opaque(),
                                   /*workSpaceSizeInBytes=*/scratch.size(),
                                   /*beta=*/beta,
                                   /*dxDesc=*/in_back_nd.handle(),
                                   /*dx=*/backward_input_data->opaque());
  if (is_profiling) {
    timer->Stop(AsCUDAStream(stream));
    if (status == CUDNN_STATUS_SUCCESS) {
      bool use_tensor_ops = algorithm_config.algorithm().tensor_ops_enabled();
      dnn::AlgorithmDesc algotype(algo, use_tensor_ops);
      output_profile_result->set_algorithm(algotype);
      output_profile_result->set_elapsed_time_in_ms(
          timer->GetElapsedMilliseconds());
    }
    timer->Destroy();
  }
  if (status != CUDNN_STATUS_SUCCESS) {
    // Silently return when we are profiling.
    if (!is_profiling) {
      LOG(ERROR) << "failed to enqueue convolution on stream: "
                 << ToString(status);
    }
    return false;
  }
  return true;
}

bool CudnnSupport::DoConvolveBackwardData(
    Stream* stream, const FilterDescriptor& filter_descriptor,
    const DeviceMemory<double>& filter_data,
    const BatchDescriptor& output_descriptor,
    DeviceMemory<double> backward_output_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& input_descriptor,
    DeviceMemory<double>* backward_input_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveBackwardDataImpl(stream, filter_descriptor, filter_data,
                                    output_descriptor, backward_output_data,
                                    convolution_descriptor, input_descriptor,
                                    backward_input_data, scratch_allocator,
                                    algorithm_config, output_profile_result);
}

bool CudnnSupport::DoConvolveBackwardData(
    Stream* stream, const FilterDescriptor& filter_descriptor,
    const DeviceMemory<float>& filter_data,
    const BatchDescriptor& output_descriptor,
    DeviceMemory<float> backward_output_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& input_descriptor,
    DeviceMemory<float>* backward_input_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveBackwardDataImpl(stream, filter_descriptor, filter_data,
                                    output_descriptor, backward_output_data,
                                    convolution_descriptor, input_descriptor,
                                    backward_input_data, scratch_allocator,
                                    algorithm_config, output_profile_result);
}

bool CudnnSupport::DoConvolveBackwardData(
    Stream* stream, const FilterDescriptor& filter_descriptor,
    const DeviceMemory<Eigen::half>& filter_data,
    const BatchDescriptor& output_descriptor,
    DeviceMemory<Eigen::half> backward_output_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& input_descriptor,
    DeviceMemory<Eigen::half>* backward_input_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveBackwardDataImpl(stream, filter_descriptor, filter_data,
                                    output_descriptor, backward_output_data,
                                    convolution_descriptor, input_descriptor,
                                    backward_input_data, scratch_allocator,
                                    algorithm_config, output_profile_result);
}

template <class T>
bool CudnnSupport::DoConvolveBackwardFilterImpl(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<T>& input_data,
    const dnn::BatchDescriptor& output_descriptor_in,
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

  // TBD(keveman): remove once cuDNN supports kBatchYXDepth for backward pass.
  BatchDescriptor output_descriptor;
  output_descriptor.CloneFrom(output_descriptor_in);
  std::unique_ptr<TemporaryDeviceMemory<T>> transform_scratch;
  backward_output_data =
      MaybeTransformLayout(stream, cudnn, &output_descriptor,
                           backward_output_data, &transform_scratch);

  ScopedTensorDescriptor out_back_nd(output_descriptor, cudnn_type);
  ScopedTensorDescriptor input_nd(input_descriptor, cudnn_type);
  ScopedFilterDescriptor filter(filter_descriptor, cudnn_type);
  ScopedConvolutionDescriptor conv(convolution_descriptor,
                                   GetConvComputeType<T>());

  const bool is_profiling = output_profile_result != nullptr;
  cudnnConvolutionBwdFilterAlgo_t algo;
  DeviceMemory<uint8> scratch;

  if (algorithm_config.algorithm().is_default()) {
    // With the default algorithm, use Cudnn's heuristics.

    // Lambda that retrieves the algorithm.
    // specify_limit will occur when we have a scratch allocator and it succeeds
    // in allocating; otherwise, we'll fall back to the "no workspace" version.
    auto get_algorithm = [&](bool specify_limit) {
      cudnnConvolutionBwdFilterPreference_t preference =
          specify_limit ? CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
                        : CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;

      auto memory_limit_bytes =
          scratch_allocator == nullptr
              ? 0
              : scratch_allocator->GetMemoryLimitInBytes(stream);
      if (memory_limit_bytes < 0) {
        memory_limit_bytes = 0;
      }

      cudnnConvolutionBwdFilterAlgo_t algo_to_use;
      cudnnStatus_t status = cudnnGetConvolutionBackwardFilterAlgorithm(
          cudnn.handle(),
          /*srcDesc=*/input_nd.handle(),
          /*diffDesc=*/out_back_nd.handle(),
          /*convDesc=*/conv.handle(),
          /*gradDesc=*/filter.handle(),
          /*preference=*/preference,
          /*memoryLimitInBytes=*/memory_limit_bytes,
          /*algo=*/&algo_to_use);
      CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << "Unable to find a suitable "
                                                "algorithm for doing backward "
                                                "filter convolution";
      return algo_to_use;
    };

    algo = get_algorithm(/*specify_limit=*/scratch_allocator != nullptr);

    if (scratch_allocator != nullptr) {
      size_t size_in_bytes;
      auto status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
          cudnn.handle(),
          /*xDesc=*/input_nd.handle(),
          /*dyDesc=*/out_back_nd.handle(), /*convDesc=*/conv.handle(),
          /*gradDesc=*/filter.handle(), /*algo=*/algo,
          /*sizeInBytes=*/&size_in_bytes);
      int64 size_in_bytes_int64 = size_in_bytes;
      if (status == CUDNN_STATUS_SUCCESS && size_in_bytes_int64 != 0) {
        if (size_in_bytes_int64 > 0) {
          auto allocated =
              scratch_allocator->AllocateBytes(stream, size_in_bytes);
          if (allocated.ok()) {
            scratch = allocated.ValueOrDie();
          } else {
            LOG(WARNING) << allocated.status().error_message();
          }
        } else {
          LOG(WARNING)
              << "cudnnGetConvolutionBackwardFilterWorkspaceSize() returned "
                 "negative sizeInBytes value. This could be a cudnn bug.";
        }
      }
    }

    // If we didn't allocate any scratch space (perhaps because of failed
    // allocation), we force a switch back to the "no workspace" algorithm.
    if (scratch == nullptr) {
      algo = get_algorithm(/*specify_limit=*/false);
    }
  } else {
    // An algorithm has been specified.
    dnn::AlgorithmDesc algotype = algorithm_config.algorithm();
    algo = ToConvBackwardFilterAlgo(algotype);
    conv.set_use_tensor_op_math(algotype.tensor_ops_enabled());

    size_t size_in_bytes;
    auto status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn.handle(),
        /*xDesc=*/input_nd.handle(),
        /*dyDesc=*/out_back_nd.handle(), /*convDesc=*/conv.handle(),
        /*gradDesc=*/filter.handle(), /*algo=*/algo,
        /*sizeInBytes=*/&size_in_bytes);
    if (status != CUDNN_STATUS_SUCCESS) {
      if (is_profiling) {
        // Silently return when we are profiling.
        return false;
      }
      LOG(FATAL) << "Cannot query the size of workspace needed for the given "
                    "algorithm: "
                 << algorithm_config.algorithm().algo_id();
    }
    int64 size_in_bytes_int64 = size_in_bytes;
    if (size_in_bytes_int64 > 0) {
      if (scratch_allocator == nullptr) {
        LOG(FATAL) << "An allocator must be specified when scratch memory is "
                      "needed";
      }
      auto allocated = scratch_allocator->AllocateBytes(stream, size_in_bytes);
      if (is_profiling && !allocated.ok()) {
        // Silently return when we are profiling.
        return false;
      }
      if (allocated.ok()) {
        scratch = allocated.ValueOrDie();
      } else {
        LOG(WARNING) << allocated.status().error_message();
      }
      if (scratch == nullptr) {
        CHECK(!algorithm_config.algorithm_no_scratch().is_default())
            << "The primary convolution algorithm failed memory allocation, "
               "while a secondary algorithm is not provided.";
        dnn::AlgorithmDesc algotype = algorithm_config.algorithm_no_scratch();
        algo = ToConvBackwardFilterAlgo(algotype);
        conv.set_use_tensor_op_math(algotype.tensor_ops_enabled());
      }
    } else if (size_in_bytes_int64 < 0) {
      LOG(WARNING)
          << "cudnnGetConvolutionBackwardFilterWorkspaceSize() returned "
             "negative sizeInBytes value. This could be a cudnn bug.";
    }
  }

  std::unique_ptr<CUDATimer> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));  // NOLINT
    timer->Init();
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    timer->Start(AsCUDAStream(stream));
  }

#if CUDNN_VERSION >= 5000
  auto status = cudnnConvolutionBackwardFilter(
      cudnn.handle(),
#else
  auto status = cudnnConvolutionBackwardFilter_v3(
      cudnn.handle(),
#endif
      /*alpha=*/alpha,
      /*srcDesc=*/input_nd.handle(),
      /*srcData=*/input_data.opaque(),
      /*diffDesc=*/out_back_nd.handle(),
      /*diffData=*/backward_output_data.opaque(),
      /*convDesc=*/conv.handle(),
      /*algo=*/algo,
      /*workSpace=*/scratch.opaque(),
      /*workSpaceSizeInBytes=*/scratch.size(),
      /*beta=*/beta,
      /*gradDesc=*/filter.handle(),
      /*gradData=*/backward_filter_data->opaque());

  if (is_profiling) {
    timer->Stop(AsCUDAStream(stream));
    if (status == CUDNN_STATUS_SUCCESS) {
      bool use_tensor_ops = algorithm_config.algorithm().tensor_ops_enabled();
      dnn::AlgorithmDesc algotype(algo, use_tensor_ops);
      output_profile_result->set_algorithm(algotype);
      output_profile_result->set_elapsed_time_in_ms(
          timer->GetElapsedMilliseconds());
    }
    timer->Destroy();
  }
  if (status != CUDNN_STATUS_SUCCESS) {
    // Silently return when we are profiling.
    if (!is_profiling) {
      LOG(ERROR) << "failed to enqueue convolution on stream: "
                 << ToString(status);
    }
    return false;
  }
  return true;
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
  return DoConvolveBackwardFilterImpl(stream, input_descriptor, input_data,
                                      output_descriptor, backward_output_data,
                                      convolution_descriptor, filter_descriptor,
                                      backward_filter_data, scratch_allocator,
                                      algorithm_config, output_profile_result);
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
  return DoConvolveBackwardFilterImpl(stream, input_descriptor, input_data,
                                      output_descriptor, backward_output_data,
                                      convolution_descriptor, filter_descriptor,
                                      backward_filter_data, scratch_allocator,
                                      algorithm_config, output_profile_result);
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
  return DoConvolveBackwardFilterImpl(stream, input_descriptor, input_data,
                                      output_descriptor, backward_output_data,
                                      convolution_descriptor, filter_descriptor,
                                      backward_filter_data, scratch_allocator,
                                      algorithm_config, output_profile_result);
}

template <class T>
bool CudnnSupport::DoConvolveBackwardBiasImpl(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<T>& input_data,
    const dnn::BatchDescriptor& bias_descriptor,
    DeviceMemory<T>* backward_bias_data) {
  cudnnDataType_t cudnn_type = GetCudnnDataType<T>();
  ScopedTensorDescriptor input_nd(input_descriptor, cudnn_type);
  ScopedTensorDescriptor bias_nd(bias_descriptor, cudnn_type);

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = cudnnConvolutionBackwardBias(
      cudnn.handle(), &alpha, input_nd.handle(), input_data.opaque(), &beta,
      bias_nd.handle(), backward_bias_data->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to enqueue backward convolution on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<double>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<double>* backward_bias_data) {
  return DoConvolveBackwardBiasImpl(stream, input_descriptor, input_data,
                                    bias_descriptor, backward_bias_data);
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<float>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<float>* backward_bias_data) {
  return DoConvolveBackwardBiasImpl(stream, input_descriptor, input_data,
                                    bias_descriptor, backward_bias_data);
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<Eigen::half>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<Eigen::half>* backward_bias_data) {
  return DoConvolveBackwardBiasImpl(stream, input_descriptor, input_data,
                                    bias_descriptor, backward_bias_data);
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
  ScopedTensorDescriptor input_descriptor(dimensions, CUDNN_DATA_FLOAT);

  BatchDescriptor bias_dimensions;
  bias_dimensions.set_count(1)
      .set_feature_map_count(dimensions.feature_map_count())
      .set_height(1)
      .set_width(1)
      .set_layout(dnn::DataLayout::kBatchYXDepth);
  ScopedTensorDescriptor bias_descriptor(bias_dimensions, CUDNN_DATA_FLOAT);

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

#if CUDNN_VERSION >= 5000
  auto status = cudnnAddTensor(
#else
  auto status = cudnnAddTensor_v3(
#endif
      cudnn.handle(), &alpha, bias_descriptor.handle(), biases.opaque(), &beta,
      input_descriptor.handle(), output_data->opaque());

  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "stream " << stream << " could not enqueue bias addition.";
    return false;
  }

  return true;
}

bool CudnnSupport::DoActivate(Stream* stream,
                              dnn::ActivationMode activation_mode,
                              const dnn::BatchDescriptor& dimensions,
                              const DeviceMemory<float>& input_data,
                              DeviceMemory<float>* output_data,
                              uint64 options) {
#if CUDNN_VERSION >= 5000
  ScopedActivationDescriptor activation_desc(
      activation_mode, CUDNN_PROPAGATE_NAN, dimensions.value_max());
#else
  cudnnActivationMode_t mode;
  switch (activation_mode) {
    case dnn::ActivationMode::kRelu6:
      // TODO(leary) should probably do a post-pass to clip at 6?
      LOG(WARNING) << "user requested Relu6, but providing Relu instead";
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case dnn::ActivationMode::kReluX:
      // TODO(broune) should probably do a post-pass to clip at X?
      LOG(WARNING) << "user requested ReluX, but providing Relu instead";
      mode = CUDNN_ACTIVATION_RELU;
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
      LOG(ERROR) << "unrecognized activation mode: "
                 << static_cast<int>(activation_mode);
      return false;
  }
#endif

  ScopedTensorDescriptor input_nd(dimensions, CUDNN_DATA_FLOAT);
  // Alpha is the input scaling factor.
  float alpha = 1.0;
  // Beta is the output scaling factor.
  float beta = 0.0;

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status =
      cudnnActivationForward(cudnn.handle(),
#if CUDNN_VERSION >= 5000
                             activation_desc.handle(),
#else
                             mode,
#endif
                             &alpha, input_nd.handle(), input_data.opaque(),
                             &beta, input_nd.handle(), output_data->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "stream " << stream
               << " could not enqueue activation: " << ToString(status);
    return false;
  }

  return true;
}

bool CudnnSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<double>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<double>* output_data) {
  // Alpha is the scaling factor for input.
  double alpha = 1.0;
  // Beta is the scaling factor for output.
  double beta = 0.0;

  ScopedTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_DOUBLE);
  ScopedTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_DOUBLE);
  ScopedPoolingDescriptor pooling_desc(pooling_dimensions);

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = cudnnPoolingForward(
      cudnn.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
      input_data.opaque(), &beta, dest_desc.handle(), output_data->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to enqueue forward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<float>* output_data) {
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_FLOAT);
  ScopedTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_FLOAT);
  ScopedPoolingDescriptor pooling_desc(pooling_dimensions);

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = cudnnPoolingForward(
      cudnn.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
      input_data.opaque(), &beta, dest_desc.handle(), output_data->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to enqueue forward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<Eigen::half>* output_data) {
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_HALF);
  ScopedTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_HALF);
  ScopedPoolingDescriptor pooling_desc(pooling_dimensions);
  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = cudnnPoolingForward(
      cudnn.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
      input_data.opaque(), &beta, dest_desc.handle(), output_data->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to enqueue forward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<double>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<double>& output_data,
    const DeviceMemory<double>& input_diff_data,
    DeviceMemory<double>* output_diff_data) {
  // Alpha is the scaling factor for input.
  double alpha = 1.0;
  // Beta is the scaling factor for output.
  double beta = 0.0;

  ScopedTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_DOUBLE);
  ScopedTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_DOUBLE);
  ScopedPoolingDescriptor pooling_desc(pooling_dimensions);

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = cudnnPoolingBackward(
      cudnn.handle(), pooling_desc.handle(), &alpha, dest_desc.handle(),
      output_data.opaque(), dest_desc.handle(), input_diff_data.opaque(),
      src_desc.handle(), input_data.opaque(), &beta, src_desc.handle(),
      output_diff_data->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to enqueue backward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<float>& output_data,
    const DeviceMemory<float>& input_diff_data,
    DeviceMemory<float>* output_diff_data) {
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_FLOAT);
  ScopedTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_FLOAT);
  ScopedPoolingDescriptor pooling_desc(pooling_dimensions);

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = cudnnPoolingBackward(
      cudnn.handle(), pooling_desc.handle(), &alpha, dest_desc.handle(),
      output_data.opaque(), dest_desc.handle(), input_diff_data.opaque(),
      src_desc.handle(), input_data.opaque(), &beta, src_desc.handle(),
      output_diff_data->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to enqueue backward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<Eigen::half>& output_data,
    const DeviceMemory<Eigen::half>& input_diff_data,
    DeviceMemory<Eigen::half>* output_diff_data) {
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc(input_dimensions, CUDNN_DATA_HALF);
  ScopedTensorDescriptor dest_desc(output_dimensions, CUDNN_DATA_HALF);
  ScopedPoolingDescriptor pooling_desc(pooling_dimensions);

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = cudnnPoolingBackward(
      cudnn.handle(), pooling_desc.handle(), &alpha, dest_desc.handle(),
      output_data.opaque(), dest_desc.handle(), input_diff_data.opaque(),
      src_desc.handle(), input_data.opaque(), &beta, src_desc.handle(),
      output_diff_data->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to enqueue backward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
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

  ScopedTensorDescriptor dims(dimensions, CUDNN_DATA_FLOAT);
  ScopedNormalizeDescriptor normalize(normalize_descriptor);

  // Alpha is the scaling factor for input.
  float alpha = 1.0f;
  // Beta is the scaling factor for output.
  float beta = 0.0f;

  auto cudnn = cudnn_->GetHandle(parent_, stream);

  // Launch the normalization.
  auto status = cudnnLRNCrossChannelForward(
      cudnn.handle(), normalize.handle(), CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha,
      dims.handle(), input_data.opaque(), &beta, dims.handle(),
      output_data->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to run cudnnLRNCrossChannelForward";
    return false;
  }
  return true;
}

bool CudnnSupport::DoNormalizeBackwardWithDimensions(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const dnn::BatchDescriptor& dimensions, const DeviceMemory<float>& raw_data,
    const DeviceMemory<float>& normalized_data,
    const DeviceMemory<float>& normalized_variable_gradient,
    DeviceMemory<float>* raw_variable_gradient) {
  // Check for unsupported modes.
  if (normalize_descriptor.wrap_around()) {
    LOG(ERROR) << "CUDA LRN does not support cudnn-around mode";
    return false;
  }
  if (normalize_descriptor.segment_size()) {
    LOG(ERROR) << "CUDA LRN does not support segmentation";
    return false;
  }

  ScopedTensorDescriptor dims(dimensions, CUDNN_DATA_FLOAT);
  ScopedNormalizeDescriptor normalize(normalize_descriptor);

  float alpha = 1.0f;
  float beta = 0.0f;

  auto cudnn = cudnn_->GetHandle(parent_, stream);
  auto status = cudnnLRNCrossChannelBackward(
      cudnn.handle(), normalize.handle(), CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha,
      dims.handle(), normalized_data.opaque(), dims.handle(),
      normalized_variable_gradient.opaque(), dims.handle(), raw_data.opaque(),
      &beta, dims.handle(), raw_variable_gradient->opaque());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to run cudnnLRNCrossChannelBackward";
    return false;
  }
  return true;
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
    stream->ThenMemcpyD2H<float>(*input_data[i], &tmp);
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
    const BatchDescriptor& batch_descriptor,
    const FilterDescriptor& filter_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::BatchDescriptor* output_batch_descriptor) {
  ScopedTensorDescriptor input_nd(batch_descriptor, CUDNN_DATA_FLOAT);
  ScopedFilterDescriptor filter(filter_descriptor, CUDNN_DATA_FLOAT);
  ScopedConvolutionDescriptor conv(convolution_descriptor, CUDNN_DATA_FLOAT);

  int dn = batch_descriptor.ndims() + 2;
  std::vector<int> dims(dn);  // in BDYX
  auto status = cudnnGetConvolutionNdForwardOutputDim(
      conv.handle(), input_nd.handle(), filter.handle(), dn, dims.data());
  if (status != CUDNN_STATUS_SUCCESS) {
    LOG(ERROR) << "could not get output tensor for convolution: "
               << ToString(status);
    return false;
  }

  output_batch_descriptor->set_count(dims[0])
      .set_feature_map_count(dims[1])
      .set_layout(batch_descriptor.layout());

  for (int i = 0; i < batch_descriptor.ndims(); i++) {
    output_batch_descriptor->set_spatial_dim(static_cast<dnn::DimIndex>(i),
                                             dims.rbegin()[i]);
  }

  return true;
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
              LOG(ERROR)
                  << "Attempting to initialize an instance of the cuBLAS "
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
