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

#include "tensorflow/stream_executor/rocm/rocm_dnn.h"

#include <functional>
#include <memory>

#include "absl/strings/str_cat.h"
#include "rocm/include/miopen/miopen.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/rocm/rocm_diagnostics.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "third_party/eigen3/Eigen/Core"

namespace {

// Converts (via narrowing) a type T value to a type U, and checks that the
// value has no value change due to the conversion.
template <typename WideT, typename NarrowT>
NarrowT CheckedNarrowing(const WideT& wide) {
  NarrowT narrow = wide;
  CHECK_EQ(narrow, wide)
      << "checked narrowing failed; values not equal post-conversion";
  return narrow;
}

}  // namespace

namespace stream_executor {

using dnn::BatchDescriptor;
using dnn::ConvolutionDescriptor;
using dnn::FilterDescriptor;
using dnn::NormalizeDescriptor;
using dnn::PoolingDescriptor;

namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kMIOpenPlugin);

string ToString(miopenStatus_t status) {
  switch (status) {
    case miopenStatusSuccess:
      return "miopenStatusSuccess";
    case miopenStatusNotInitialized:
      return "miopenStatusNotInitialized";
    case miopenStatusAllocFailed:
      return "miopenStatusAllocFailed";
    case miopenStatusBadParm:
      return "miopenStatusBadParm";
    case miopenStatusInternalError:
      return "miopenStatusInternalError";
    case miopenStatusInvalidValue:
      return "miopenStatusInvalidValue";
    case miopenStatusNotImplemented:
      return "miopenStatusNotImplemented";
    case miopenStatusUnknownError:
      return "miopenStatusUnknownError";
    default:
      return absl::StrCat("<unknown miopen status: ", static_cast<int>(status),
                          ">");
  }
}

// RAII wrapper for all calls to MIOpen with a MIOpen handle argument.
//
// See MIOpenAccess::GetHandle() for details.
class MIOpenHandle {
 public:
  // Takes ownership of the executor context and the lock to access MIOpen
  // using handle.
  MIOpenHandle(gpu::ScopedActivateExecutorContext context, mutex_lock lock,
               miopenHandle_t handle)
      : context_(std::move(context)), lock_(std::move(lock)), handle_(handle) {}

  // Returns MIOpen handle. To be passed directly to MIOpen APIs, don't keep
  // a copy.
  miopenHandle_t handle() const { return handle_; }

 private:
  gpu::ScopedActivateExecutorContext context_;
  mutex_lock lock_;
  miopenHandle_t handle_;  // Not owned.
};

namespace wrap {

#ifdef PLATFORM_GOOGLE

#define STREAM_EXECUTOR_MIOPEN_WRAP(__name)      \
  struct WrapperShim__##__name {                 \
    template <typename... Args>                  \
    miopenStatus_t operator()(Args... args) {    \
      miopenStatus_t retval = ::__name(args...); \
      return retval;                             \
    }                                            \
  } __name;

#else

#define STREAM_EXECUTOR_MIOPEN_WRAP(__name)                               \
  struct DynLoadShim__##__name {                                          \
    static const char* kName;                                             \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;          \
    static void* GetDsoHandle() {                                         \
      auto s = internal::CachedDsoLoader::GetMiopenDsoHandle();           \
      return s.ValueOrDie();                                              \
    }                                                                     \
    static FuncPtrT LoadOrDie() {                                         \
      void* f;                                                            \
      auto s = port::Env::Default()->GetSymbolFromLibrary(GetDsoHandle(), \
                                                          kName, &f);     \
      CHECK(s.ok()) << "could not find " << kName                         \
                    << " in miopen DSO; dlerror: " << s.error_message();  \
      return reinterpret_cast<FuncPtrT>(f);                               \
    }                                                                     \
    static FuncPtrT DynLoad() {                                           \
      static FuncPtrT f = LoadOrDie();                                    \
      return f;                                                           \
    }                                                                     \
    template <typename... Args>                                           \
    miopenStatus_t operator()(Args... args) {                             \
      return DynLoad()(args...);                                          \
    }                                                                     \
  } __name;                                                               \
  const char* DynLoadShim__##__name::kName = #__name;

#endif

// clang-format off
#define MIOPEN_DNN_ROUTINE_EACH(__macro)                   \
  __macro(miopenBatchNormalizationBackward)                \
  __macro(miopenBatchNormalizationForwardInference)        \
  __macro(miopenBatchNormalizationForwardTraining)         \
  __macro(miopenGetConvolutionForwardOutputDim)            \
  __macro(miopenFindConvolutionForwardAlgorithm)           \
  __macro(miopenCreateTensorDescriptor)                    \
  __macro(miopenDestroyTensorDescriptor)                   \
  __macro(miopenSet2dPoolingDescriptor)                    \
  __macro(miopenSetLRNDescriptor)                          \
  __macro(miopenLRNGetWorkSpaceSize)                       \
  __macro(miopenCreateConvolutionDescriptor)               \
  __macro(miopenCreatePoolingDescriptor)                   \
  __macro(miopenDestroyPoolingDescriptor)                  \
  __macro(miopenCreateLRNDescriptor)                       \
  __macro(miopenDestroyLRNDescriptor)                      \
  __macro(miopenDestroyConvolutionDescriptor)              \
  __macro(miopenCreateWithStream)                          \
  __macro(miopenDestroy)                                   \
  __macro(miopenSetStream)                                 \
  __macro(miopenSetAllocator)                              \
  __macro(miopenActivationForward)                         \
  __macro(miopenConvolutionForward)                        \
  __macro(miopenConvolutionBackwardBias)                   \
  __macro(miopenConvolutionForwardGetWorkSpaceSize)        \
  __macro(miopenInitConvolutionDescriptor)                 \
  __macro(miopenGetConvolutionDescriptor)                  \
  __macro(miopenSetConvolutionGroupCount)                  \
  __macro(miopenSet4dTensorDescriptor)                     \
  __macro(miopenGetTensorDescriptor)                       \
  __macro(miopenSetTensorDescriptor)                       \
  __macro(miopenGetTensorDescriptorSize)                   \
  __macro(miopenPoolingForward)                            \
  __macro(miopenPoolingGetWorkSpaceSize)                   \
  __macro(miopenPoolingBackward)                           \
  __macro(miopenLRNForward)                                \
  __macro(miopenLRNBackward)                               \
  __macro(miopenOpTensor)                                  \
  __macro(miopenConvolutionBackwardData)                   \
  __macro(miopenConvolutionBackwardWeights)                \
  __macro(miopenConvolutionBackwardWeightsGetWorkSpaceSize)\
  __macro(miopenFindConvolutionBackwardDataAlgorithm)      \
  __macro(miopenFindConvolutionBackwardWeightsAlgorithm)   \
  __macro(miopenConvolutionBackwardDataGetWorkSpaceSize)   \
  __macro(miopenCreateRNNDescriptor)                       \
  __macro(miopenSetRNNDescriptor)                          \
  __macro(miopenDestroyRNNDescriptor)                      \
  __macro(miopenGetRNNParamsSize)                          \
  __macro(miopenGetRNNLayerParam)                          \
  __macro(miopenGetRNNLayerBias)                           \
  __macro(miopenGetRNNWorkspaceSize)                       \
  __macro(miopenGetRNNTrainingReserveSize)                 \
  __macro(miopenRNNForwardInference)                       \
  __macro(miopenRNNForwardTraining)                        \
  __macro(miopenRNNBackwardData)                           \
  __macro(miopenRNNBackwardWeights)                        \
  __macro(miopenGetRNNLayerParamOffset)                    \
  __macro(miopenGetRNNLayerParamSize)                      \
  __macro(miopenGetRNNLayerBiasOffset)                     \
  __macro(miopenGetRNNLayerBiasSize)                       \
  __macro(miopenGetRNNParamsDescriptor)			   \
  __macro(miopenCreateActivationDescriptor)		   \
  __macro(miopenSetActivationDescriptor)		   \
  __macro(miopenGetActivationDescriptor)		   \
  __macro(miopenDestroyActivationDescriptor)		   \
  __macro(miopenCreateFusionPlan)                          \
  __macro(miopenCreateOpConvForward)                       \
  __macro(miopenCreateOpBiasForward)                       \
  __macro(miopenCreateOpActivationForward)                 \
  __macro(miopenCreateOpActivationBackward)		   \
  __macro(miopenCreateOpBatchNormInference)		   \
  __macro(miopenCreateOpBatchNormForward)		   \
  __macro(miopenCreateOpBatchNormBackward)		   \
  __macro(miopenCompileFusionPlan)                         \
  __macro(miopenFusionPlanGetOp)                           \
  __macro(miopenCreateOperatorArgs)                        \
  __macro(miopenSetOpArgsConvForward)                      \
  __macro(miopenSetOpArgsBiasForward)                      \
  __macro(miopenSetOpArgsActivForward)			   \
  __macro(miopenSetOpArgsActivBackward)			   \
  __macro(miopenSetOpArgsBatchNormInference)		   \
  __macro(miopenSetOpArgsBatchNormForward)		   \
  __macro(miopenSetOpArgsBatchNormBackward)		   \
  __macro(miopenExecuteFusionPlan)                         \
  __macro(miopenDestroyOperatorArgs)                       \
  __macro(miopenDestroyFusionPlan)

// clang-format on

MIOPEN_DNN_ROUTINE_EACH(STREAM_EXECUTOR_MIOPEN_WRAP)

#undef MIOPEN_DNN_ROUTINE_EACH

}  // namespace wrap

namespace {

// These routines should ideally be provided as an MIOpen API.
// They are called for *every* _ROCMmFusedOp*::Compute call, and they need to be
// efficient! Instead of calculating the hash value by quering the MIOpen Get*
// APIs for the descriptor components, it would be a lot more efficient if,
// MIOpen calculated the hash value when creating the descriptor, stored it on
// the descriptor datastructure, and provided an API routine to query it.

const int kMaxMIOpenTensorSize = 5;

uint64 GetHashValue(miopenTensorDescriptor_t tensor_desc) {
  miopenDataType_t datatype = miopenFloat;
  int dims[kMaxMIOpenTensorSize] = {0};
  int strides[kMaxMIOpenTensorSize] = {0};
  wrap::miopenGetTensorDescriptor(tensor_desc, &datatype, dims, strides);

  uint64 hash_value = tensorflow::hash<int>()(datatype);
  for (int dim : dims)
    hash_value =
        tensorflow::Hash64Combine(hash_value, tensorflow::hash<int>()(dim));
  for (int stride : strides)
    hash_value =
        tensorflow::Hash64Combine(hash_value, tensorflow::hash<int>()(stride));

  return hash_value;
}

uint64 GetHashValue(miopenConvolutionDescriptor_t conv_desc) {
  miopenConvolutionMode_t c_mode = miopenConvolution;
  int pad_h = 0, pad_w = 0, u = 0, v = 0, dilation_h = 0, dilation_w = 0;
  wrap::miopenGetConvolutionDescriptor(conv_desc, &c_mode, &pad_h, &pad_w, &u,
                                       &v, &dilation_h, &dilation_w);

  uint64 hash_value = tensorflow::hash<int>()(c_mode);
  hash_value =
      tensorflow::Hash64Combine(hash_value, tensorflow::hash<int>()(pad_h));
  hash_value =
      tensorflow::Hash64Combine(hash_value, tensorflow::hash<int>()(pad_w));
  hash_value =
      tensorflow::Hash64Combine(hash_value, tensorflow::hash<int>()(u));
  hash_value =
      tensorflow::Hash64Combine(hash_value, tensorflow::hash<int>()(v));
  hash_value = tensorflow::Hash64Combine(hash_value,
                                         tensorflow::hash<int>()(dilation_h));
  hash_value = tensorflow::Hash64Combine(hash_value,
                                         tensorflow::hash<int>()(dilation_w));

  return hash_value;
}

// Class to implement a cache of compiled fusion plans.
class CachedFusionPlans {
 public:
  // Check if we already have a fusion_plan corresponding to the given hash
  // value.
  // If we do, then
  //   return true (+ the cached fusion plan via given pointer)
  // Else
  //   create a new fusion plan descriptor,
  //   associate it with the given hash value in the cache
  //   return false (+ newly created fusion plan via given pointer)
  static bool FindOrCreate(uint64 hash,
                           miopenFusionPlanDescriptor_t* fusion_plan,
                           miopenFusionDirection_t fusion_direction,
                           miopenTensorDescriptor_t input_descriptor) {
    mutex_lock lock{cached_plans_mutex};

    bool found_cached_plan = false;

    auto it = cached_plans.find(hash);
    if (it != cached_plans.end()) {
      *fusion_plan = it->second;
      found_cached_plan = true;
    } else {
      auto status = wrap::miopenCreateFusionPlan(fusion_plan, fusion_direction,
                                                 input_descriptor);
      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenCreateFusionPlan failed: "
                   << ToString(status);
      } else {
        cached_plans[hash] = *fusion_plan;
      }
    }

    return found_cached_plan;
  }

  // Need to figure out the right place to call this routine.
  static void Clear() {
    mutex_lock lock{cached_plans_mutex};

    for (auto it : cached_plans) {
      auto status = wrap::miopenDestroyFusionPlan(it.second);
      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenDestroyFusionPlan failed: "
                   << ToString(status);
      }
    }

    cached_plans.clear();

    unsupported_plans.clear();
  }

  // Is the Fusion plan corresponding to this hash unsupported.
  static bool IsUnsupportedFusionPlan(uint64 hash) {
    mutex_lock lock{cached_plans_mutex};
    return unsupported_plans.count(hash) > 0;
  }

  // Mark the given hash value as corresponding to an unsupported fusion plan.
  static void MarkFusionPlanUnsupported(uint64 hash) {
    mutex_lock lock{cached_plans_mutex};
    unsupported_plans.insert(hash);
  }

 private:
  // Mutex to guard access to all data within this class.
  static mutex cached_plans_mutex;

  // Map of hash-value to MIOpen Fusion plan descriptors.
  // Need to be able share this across more than one stream and hence static.
  static std::map<uint64, miopenFusionPlanDescriptor_t> cached_plans;

  // Set of hash-values that correspond to MIOpen Fusion plans that will fail
  // compile and hence are not supported.
  static std::set<uint64> unsupported_plans;
};

mutex CachedFusionPlans::cached_plans_mutex;
std::map<uint64, miopenFusionPlanDescriptor_t> CachedFusionPlans::cached_plans;
std::set<uint64> CachedFusionPlans::unsupported_plans;

miopenHandle_t ToHandle(void* opaque_handle) {
  return static_cast<miopenHandle_t>(opaque_handle);
}

miopenConvFwdAlgorithm_t ToConvForwardAlgo(dnn::AlgorithmDesc algorithm) {
  miopenConvFwdAlgorithm_t algo = miopenConvFwdAlgorithm_t(algorithm.algo_id());
  switch (algo) {
    case miopenConvolutionFwdAlgoGEMM:
    case miopenConvolutionFwdAlgoDirect:
    case miopenConvolutionFwdAlgoFFT:
    case miopenConvolutionFwdAlgoWinograd:
      return algo;
    default:
      LOG(FATAL) << "Unsupported MIOpen convolution forward algorithm: "
                 << algorithm.algo_id();
  }
}

miopenConvBwdDataAlgorithm_t ToConvBackwardDataAlgo(
    dnn::AlgorithmDesc algorithm) {
  miopenConvBwdDataAlgorithm_t algo =
      miopenConvBwdDataAlgorithm_t(algorithm.algo_id());
  switch (algo) {
    case miopenConvolutionBwdDataAlgoGEMM:
    case miopenConvolutionBwdDataAlgoDirect:
    case miopenConvolutionBwdDataAlgoFFT:
    case miopenConvolutionBwdDataAlgoWinograd:
      return algo;
    default:
      LOG(FATAL)
          << "Unsupported MIOpen convolution backward algorithm for data: "
          << algorithm.algo_id();
  }
}

miopenConvBwdWeightsAlgorithm_t ToConvBackwardFilterAlgo(
    dnn::AlgorithmDesc algorithm) {
  miopenConvBwdWeightsAlgorithm_t algo =
      miopenConvBwdWeightsAlgorithm_t(algorithm.algo_id());
  switch (algo) {
    case miopenConvolutionBwdWeightsAlgoGEMM:
    case miopenConvolutionBwdWeightsAlgoDirect:
      return algo;
    default:
      LOG(FATAL)
          << "Unsupported MIOpen convolution backward algorithm for filter: "
          << algorithm.algo_id();
  }
}

}  // namespace

// Wraps a MIOpen handle and provides access to it through miopenHandle_t
// instances, which also locks a mutex, acquires the ROCm context, and sets
// the stream that MIOpen should use to enqueue any work.
//
// Note: MIOpenSupport::miopen_ should be the only instantiation of this class.
class MIOpenAccess {
 public:
  // Takes ownership of the handle.
  explicit MIOpenAccess(miopenHandle_t handle) : handle_(handle) {}

  ~MIOpenAccess() {
    mutex_lock lock(mutex_);
    wrap::miopenDestroy(handle_);
  }

  // Creates a MIOpenHandle instance for stream.
  //
  // MIOpen API calls using the same handle instance need to be serialized
  // across threads. This is guaranteed by MIOpenHandle instances locking the
  // mutex owned by this class.
  //
  // Most MIOpen APIs taking a handle perform work on a HIP stream. The
  // MIOpenHandle instance acquires the executor's ROCm context and sets MIOpen
  // to use the provided stream.
  //
  // The stream argument may be null, which translates to the null stream.
  // The null stream synchronizes with all other streams and it is
  // therefore a bad idea (performance wise) to call any MIOpen APIs that
  // enqueue work in the stream.
  MIOpenHandle GetHandle(GpuExecutor* executor, Stream* stream) {
    mutex_lock lock(mutex_);
    gpu::ScopedActivateExecutorContext context(executor);
    hipStream_t hip_stream = stream ? AsGpuStreamValue(stream) : nullptr;
    auto status = wrap::miopenSetStream(handle_, hip_stream);
    CHECK_EQ(status, miopenStatusSuccess) << "Failed to set MIOpen stream.";
    return MIOpenHandle(std::move(context), std::move(lock), handle_);
  }

 private:
  // Guards the enqueueing of MIOpen operations via the handle_ below.
  mutex mutex_;

  // MIOpen library handle.
  miopenHandle_t handle_ GUARDED_BY(mutex_);  // Owned.
};

MIOpenSupport::MIOpenSupport(GpuExecutor* parent) : parent_(parent) {}

port::Status MIOpenSupport::Init() {
  ScopedActivateExecutorContext context(parent_);
  miopenHandle_t miopen_handle = nullptr;
  auto status = wrap::miopenCreateWithStream(
      reinterpret_cast<miopenHandle_t*>(&miopen_handle), (hipStream_t)(0));
  if (status == miopenStatusSuccess) {
    miopen_.reset(new MIOpenAccess(miopen_handle));
    return port::Status::OK();
  }

  CHECK_EQ(miopen_handle, nullptr);
  LOG(ERROR) << "could not create miopen handle: " << ToString(status);
  if (status == miopenStatusNotInitialized) {
    auto result = rocm::Diagnostician::FindKernelDriverVersion();
    if (!result.ok()) {
      LOG(ERROR) << "error retrieving driver version: "
                 << rocm::DriverVersionStatusToString(result);
    } else {
      const auto& version = result.ValueOrDie();
      LOG(INFO) << "possibly insufficient driver version: "
                << rocm::DriverVersionToString(version);
    }
  }

  return port::Status{port::error::INTERNAL,
                      absl::StrCat("miopen library could not create a handle: ",
                                   ToString(status))};
}

port::StatusOr<perftools::gputools::dnn::VersionInfo>
MIOpenSupport::GetVersion() {
  // ROCM TODO: retrieve MIOpen version with its API
  return perftools::gputools::dnn::VersionInfo(1, 3, 0);
}

// Turns a BatchDescriptor structure into a miopen tensor handle within a scope.
class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor(const BatchDescriptor& batch_descriptor,
                         miopenDataType_t elem_type)
      : handle_(nullptr) {
    auto status = wrap::miopenCreateTensorDescriptor(&handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen tensor descriptor: "
                 << ToString(status);
    }

    switch (batch_descriptor.layout()) {
      case dnn::DataLayout::kBatchYXDepth:
      case dnn::DataLayout::kBatchDepthYX: {
        const int nd = batch_descriptor.ndims() + 2;
        if (nd != 4) {
          LOG(FATAL) << "miopen only supports 4D tensors, dim=" << nd
                     << " not allowed";
        }

        // MIOpen requires the strides and dims to be ordered as BDYX.
        std::vector<int64> strides64 =
            batch_descriptor.full_strides(dnn::DataLayout::kBatchDepthYX);
        std::vector<int64> dims64 =
            batch_descriptor.full_dims(dnn::DataLayout::kBatchDepthYX);

        // MIOpen requires arrays of ints.
        std::vector<int> strides(nd);
        std::vector<int> dims(nd);
        std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                       &CheckedNarrowing<int64, int>);
        std::transform(dims64.cbegin(), dims64.cend(), dims.begin(),
                       &CheckedNarrowing<int64, int>);
        status = wrap::miopenSet4dTensorDescriptor(handle_, elem_type, dims[0],
                                                   dims[1], dims[2], dims[3]);

        if (status != miopenStatusSuccess) {
          LOG(FATAL) << "could not convert BatchDescriptor "
                     << batch_descriptor.ToString()
                     << " to miopen tensor descriptor: " << ToString(status);
        }
      } break;
      default:
        LOG(FATAL) << "Unsupported tensor format "
                   << DataLayoutString(batch_descriptor.layout());
        break;
    }
  }

  ~ScopedTensorDescriptor() {
    auto status = wrap::miopenDestroyTensorDescriptor(handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen tensor descriptor: "
                 << ToString(status);
    }
  }

  miopenTensorDescriptor_t handle() const { return handle_; }

 private:
  miopenTensorDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

// Turns a FilterDescriptor structure into a miopen filter handle within a
// scope.
class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor(const FilterDescriptor& filter_descriptor,
                         const BatchDescriptor& batch_descriptor,
                         miopenDataType_t elem_type)
      : handle_(nullptr) {
    auto status = wrap::miopenCreateTensorDescriptor(&handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen filter descriptor: "
                 << ToString(status);
    }

    const int nd = batch_descriptor.ndims() + 2;

    if (nd != 4) {
      LOG(FATAL) << "miopen only supports 4D filters, dim=" << nd
                 << "not allowed" << ToString(status);
    }

    std::vector<int> dims(2 + filter_descriptor.ndims());
    dims[0] = filter_descriptor.output_feature_map_count();
    dims[1] = filter_descriptor.input_feature_map_count();
    const auto& spatial_dims = filter_descriptor.input_filter_dims();
    std::copy(spatial_dims.begin(), spatial_dims.end(), dims.begin() + 2);

    status = wrap::miopenSet4dTensorDescriptor(handle_, elem_type, dims[0],
                                               dims[1], dims[2], dims[3]);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen filter descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedFilterDescriptor() {
    auto status = wrap::miopenDestroyTensorDescriptor(handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen filter descriptor: "
                 << ToString(status);
    }
  }

  miopenTensorDescriptor_t handle() const { return handle_; }

 private:
  // miopen filter descriptor this object creates. Owned.
  miopenTensorDescriptor_t handle_;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

// Turns a ConvolutionDescriptor structure into a miopen convolution handle
// within a scope.
class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor(
      const ConvolutionDescriptor& convolution_descriptor,
      miopenDataType_t data_type)
      : handle_(nullptr) {
    auto status = wrap::miopenCreateConvolutionDescriptor(&handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen convolution descriptor: "
                 << ToString(status);
    }
    const auto& strides64 = convolution_descriptor.strides();
    const auto& padding64 = convolution_descriptor.padding();
    if (convolution_descriptor.pad_alignment() ==
        dnn::PadAlignment::kTensorFlowPadding) {
      LOG(ERROR) << "TensorFlow padding alignment is not supported.";
    }

    // MIOpen requires arrays of ints.
    std::vector<int> strides(convolution_descriptor.ndims());
    std::vector<int> padding(convolution_descriptor.ndims());
    std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                   &CheckedNarrowing<int64, int>);
    std::transform(padding64.cbegin(), padding64.cend(), padding.begin(),
                   &CheckedNarrowing<int64, int>);
    std::vector<int> upscale(convolution_descriptor.ndims(), 1);

    status = wrap::miopenInitConvolutionDescriptor(
        handle_, miopenConvolution, padding[0], padding[1], strides[0],
        strides[1], upscale[0], upscale[1]);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen convolution descriptor: "
                 << ToString(status);
    }

    VLOG(2) << "Requesting grouped convolution: "
            << convolution_descriptor.group_count();
    status = wrap::miopenSetConvolutionGroupCount(
        handle_, convolution_descriptor.group_count());
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen convolution group count: "
                 << ToString(status);
    }
  }
  ~ScopedConvolutionDescriptor() {
    auto status = wrap::miopenDestroyConvolutionDescriptor(handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen convolution descriptor: "
                 << ToString(status);
    }
  }

  miopenConvolutionDescriptor_t handle() const { return handle_; }

 private:
  miopenConvolutionDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedConvolutionDescriptor);
};

// Turns a PoolingDescriptor structure into a miopen pooling descriptor handle
// within a scope.
class ScopedPoolingDescriptor {
 public:
  ScopedPoolingDescriptor(const PoolingDescriptor& pooling_descriptor)
      : handle_(nullptr) {
    auto status = wrap::miopenCreatePoolingDescriptor(&handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen pooling descriptor: "
                 << ToString(status);
    }

    absl::Span<const int64> strides64 = pooling_descriptor.strides();
    absl::Span<const int64> padding64 = pooling_descriptor.padding();
    absl::Span<const int64> shape64 = pooling_descriptor.window();

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

    if (nd != 2) {
      LOG(FATAL) << "miopen requires pooling dimensions be 2"
                 << ToString(status);
    }

    status = wrap::miopenSet2dPoolingDescriptor(
        handle_,
        (pooling_descriptor.mode() == dnn::PoolingMode::kMaximum
             ? miopenPoolingMax
             : miopenPoolingAverage),
        shape[0], shape[1], padding[0], padding[1], strides[0], strides[1]);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen pooling descriptor: "
                 << ToString(status);
    }
  }
  ~ScopedPoolingDescriptor() {
    auto status = wrap::miopenDestroyPoolingDescriptor(handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen pooling descriptor: "
                 << ToString(status);
    }
  }

  miopenPoolingDescriptor_t handle() const { return handle_; }

 private:
  miopenPoolingDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

// Turns a NormalizeDescriptor structure into a miopen LRN descriptor handle.
class ScopedNormalizeDescriptor {
 public:
  ScopedNormalizeDescriptor(const NormalizeDescriptor& normalize_descriptor)
      : handle_(nullptr) {
    auto status = wrap::miopenCreateLRNDescriptor(&handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen LRN descriptor: "
                 << ToString(status);
    }

    // The range specifies that the indices in the closed range
    // [i - range, i + range] should be included in the normalization for index
    // i. The lrnN value is the total number of elements in the range, so
    // lrnN = 2*range + 1.
    unsigned lrn_N = 2 * normalize_descriptor.range() + 1;

    // Note that SE defines the normalization operation as
    //
    //  U_i = V_i / ((bias +  alpha      * (sum_j V_j^2)) ^ beta)
    //
    // but MIOpen defines it as
    //
    //  U_i = V_i / ((bias + (alpha / n) * (sum_j V_j^2)) ^ beta)
    //
    // i.e. there is a factor of n difference between the meaning of the alphas
    // in the two contexts. The MIOpen alpha is n times the SE alpha.
    double lrn_alpha = lrn_N * normalize_descriptor.alpha();

    double lrn_beta = normalize_descriptor.beta();
    double lrn_k = normalize_descriptor.bias();
    status = wrap::miopenSetLRNDescriptor(handle_, miopenLRNCrossChannel, lrn_N,
                                          lrn_alpha, lrn_beta, lrn_k);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen LRN descriptor: " << ToString(status);
    }
  }

  ~ScopedNormalizeDescriptor() {
    auto status = wrap::miopenDestroyLRNDescriptor(handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen LRN descriptor: "
                 << ToString(status);
    }
  }

  miopenLRNDescriptor_t handle() const { return handle_; }

 private:
  miopenLRNDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedNormalizeDescriptor);
};

// Turns a activation mode into a miopen activation mode descriptor with a scope
// around it
class ScopedActivationDescriptor {
 public:
  ScopedActivationDescriptor(dnn::ActivationMode activation_mode)
      : handle_(nullptr),
        miopen_activation_mode_(miopenActivationPASTHRU),
        alpha_(0.0),
        beta_(0.0),
        gamma_(0.0) {
    auto status = wrap::miopenCreateActivationDescriptor(&handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenCreateActivationDescriptor failed: "
                 << ToString(status);
    } else {
      switch (activation_mode) {
        case dnn::ActivationMode::kNone:
          miopen_activation_mode_ = miopenActivationPASTHRU;
          break;

        case dnn::ActivationMode::kSigmoid:
          miopen_activation_mode_ = miopenActivationLOGISTIC;
          break;

        case dnn::ActivationMode::kRelu:
          miopen_activation_mode_ = miopenActivationRELU;
          break;

        case dnn::ActivationMode::kRelu6:
          miopen_activation_mode_ = miopenActivationRELU;
          alpha_ = 6.0;
          break;

        case dnn::ActivationMode::kTanh:
          miopen_activation_mode_ = miopenActivationTANH;
          break;

        default:
          LOG(FATAL) << "Activation mode ("
                     << dnn::ActivationModeString(activation_mode)
                     << ") not yet implemented";
          break;
      }

      status = wrap::miopenSetActivationDescriptor(
          handle_, miopen_activation_mode_, alpha_, beta_, gamma_);
      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenSetActivationDescriptor failed: "
                   << ToString(status);
      }
    }
  }

  ~ScopedActivationDescriptor() {
    auto status = wrap::miopenDestroyActivationDescriptor(handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenDestroyActivationDescriptor failed: "
                 << ToString(status);
    }
  }

  miopenActivationDescriptor_t handle() const { return handle_; }

  uint64 GetHashValue() {
    uint64 hash_value = tensorflow::hash<int>()(miopen_activation_mode_);
    hash_value = tensorflow::Hash64Combine(hash_value,
                                           tensorflow::hash<double>()(alpha_));
    hash_value = tensorflow::Hash64Combine(hash_value,
                                           tensorflow::hash<double>()(beta_));
    hash_value = tensorflow::Hash64Combine(hash_value,
                                           tensorflow::hash<double>()(gamma_));

    return hash_value;
  }

 private:
  miopenActivationDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedActivationDescriptor);

 public:
  // caching these values here to avoid calling miopenGetActivationDescriptor
  // to do the same. miopenGetActivationDescriptor gets called twice during each
  // call to execute a fusion plan (that involves the activation op)...once call
  // during calculating hashvalue for the fusion op, and another before calling
  // SetOpArgs for the activation op
  miopenActivationMode_t miopen_activation_mode_;
  double alpha_;
  double beta_;
  double gamma_;
};

// base class for all fusion plan implementations to derive from
class ScopedFusionPlanBase {
 public:
  ScopedFusionPlanBase(miopenHandle_t miopen_handle,
                       const miopenFusionDirection_t fuse_direction,
                       const miopenTensorDescriptor_t input_descriptor)
      : miopen_handle_(miopen_handle),
        fusion_plan_(nullptr),
        fusion_args_(nullptr),
        fusion_plan_compiled_(false) {
    auto status = wrap::miopenCreateOperatorArgs(&fusion_args_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenCreateOperatorArgs failed: "
                 << ToString(status);
    }
  }

  virtual ~ScopedFusionPlanBase() {
    auto status = wrap::miopenDestroyOperatorArgs(fusion_args_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenDestroyoperatorArgs failed: "
                 << ToString(status);
    }
  }

  miopenStatus_t Execute(miopenTensorDescriptor_t input_descriptor,
                         const void* input_data,
                         miopenTensorDescriptor_t output_descriptor,
                         void* output_data) {
    auto status = wrap::miopenExecuteFusionPlan(
        miopen_handle_, fusion_plan_, input_descriptor, input_data,
        output_descriptor, output_data, fusion_args_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenExecuteFusionPlan failed: "
                 << ToString(status);
    }

    return status;
  }

  bool CompilationSucceeded() { return fusion_plan_compiled_; }

 protected:
  miopenStatus_t SetConvolutionArgs(const int op_idx, const float* alpha,
                                    const float* beta, const void* data) {
    miopenFusionOpDescriptor_t conv_op;
    auto status = wrap::miopenFusionPlanGetOp(fusion_plan_, op_idx, &conv_op);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenFusionPlanGetOp failed: "
                 << ToString(status);
    }

    status = wrap::miopenSetOpArgsConvForward(fusion_args_, conv_op, alpha,
                                              beta, data);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenSetOpArgsConvForward failed: "
                 << ToString(status);
    }
    return status;
  }

  miopenStatus_t SetBiasArgs(const int op_idx, const float* alpha,
                             const float* beta, const void* data) {
    miopenFusionOpDescriptor_t bias_op;
    auto status = wrap::miopenFusionPlanGetOp(fusion_plan_, op_idx, &bias_op);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenFusionPlanGetOp failed: "
                 << ToString(status);
    }

    status = wrap::miopenSetOpArgsBiasForward(fusion_args_, bias_op, alpha,
                                              beta, data);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenSetOpArgsBiasForward failed: "
                 << ToString(status);
    }
    return status;
  }

  miopenStatus_t SetBatchNormInferenceArgs(const int op_idx, const float* alpha,
                                           const float* beta, const void* scale,
                                           const void* offset, const void* mean,
                                           const void* variance,
                                           double epsilon) {
    miopenFusionOpDescriptor_t batchnorm_op;
    auto status =
        wrap::miopenFusionPlanGetOp(fusion_plan_, op_idx, &batchnorm_op);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenFusionPlanGetOp failed: "
                 << ToString(status);
    }

    status = wrap::miopenSetOpArgsBatchNormInference(fusion_args_, batchnorm_op,
                                                     alpha, beta, scale, offset,
                                                     mean, variance, epsilon);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenSetOpArgsBatchNormInference failed: "
                 << ToString(status);
    }
    return status;
  }

  miopenStatus_t SetBatchNormForwardArgs(const int op_idx, const float* alpha,
                                         const float* beta, const void* scale,
                                         const void* offset, void* running_mean,
                                         void* running_variance, void* saved_mean,
                                         void* saved_inv_variance,
                                         double epsilon) {
    miopenFusionOpDescriptor_t batchnorm_op;
    auto status =
        wrap::miopenFusionPlanGetOp(fusion_plan_, op_idx, &batchnorm_op);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenFusionPlanGetOp failed: "
                 << ToString(status);
    }

    double exp_avg_factor = 1.0;

    status = wrap::miopenSetOpArgsBatchNormForward(
        fusion_args_, batchnorm_op, alpha, beta, scale, offset, saved_mean,
        saved_inv_variance, running_mean, running_variance, exp_avg_factor, epsilon);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenSetOpArgsBatchNormForward failed: "
                 << ToString(status);
    }
    return status;
  }

  miopenStatus_t SetBatchNormBackwardArgs(const int op_idx, const float* alpha,
                                          const float* beta, const void* x,
                                          const void* scale, const void* offset,
                                          void* scale_grad, void* offset_grad,
                                          const void* saved_mean,
                                          const void* saved_inv_variance) {
    miopenFusionOpDescriptor_t batchnorm_op;
    auto status =
        wrap::miopenFusionPlanGetOp(fusion_plan_, op_idx, &batchnorm_op);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenFusionPlanGetOp failed: "
                 << ToString(status);
    }

    status = wrap::miopenSetOpArgsBatchNormBackward(
        fusion_args_, batchnorm_op, alpha, beta, x, scale, offset, scale_grad,
        offset_grad, saved_mean, saved_inv_variance);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenSetOpArgsBatchNormBackward failed: "
                 << ToString(status);
    }
    return status;
  }

  miopenStatus_t SetActivationForwardArgs(const int op_idx, const float* alpha,
                                          const float* beta, double activ_alpha,
                                          double activ_beta,
                                          double activ_gamma) {
    miopenFusionOpDescriptor_t actv_op;
    auto status = wrap::miopenFusionPlanGetOp(fusion_plan_, op_idx, &actv_op);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenFusionPlanGetOp failed: "
                 << ToString(status);
    }

    status =
        wrap::miopenSetOpArgsActivForward(fusion_args_, actv_op, alpha, beta,
                                          activ_alpha, activ_beta, activ_gamma);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenSetOpArgsActivForward failed: "
                 << ToString(status);
    }
    return status;
  }

  miopenStatus_t SetActivationBackwardArgs(const int op_idx, const float* alpha,
                                           const float* beta, const void* y,
                                           double activ_alpha,
                                           double activ_beta,
                                           double activ_gamma) {
    miopenFusionOpDescriptor_t actv_op;
    auto status = wrap::miopenFusionPlanGetOp(fusion_plan_, op_idx, &actv_op);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenFusionPlanGetOp failed: "
                 << ToString(status);
    }

    status = wrap::miopenSetOpArgsActivBackward(fusion_args_, actv_op, alpha,
                                                beta, y, nullptr, activ_alpha,
                                                activ_beta, activ_gamma);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenSetOpArgsActivBackward failed: "
                 << ToString(status);
    }
    return status;
  }

  miopenHandle_t miopen_handle_;
  miopenFusionPlanDescriptor_t fusion_plan_;
  miopenOperatorArgs_t fusion_args_;  // Owned.
  bool fusion_plan_compiled_;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedFusionPlanBase);
};

// class to represent the Convolution+Bias+Activation fusion plan
class ScopedFusionPlanConvolutionBiasActivation : public ScopedFusionPlanBase {
 public:
  ScopedFusionPlanConvolutionBiasActivation(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t filter_descriptor,
      miopenConvolutionDescriptor_t conv_descriptor,
      miopenTensorDescriptor_t bias_descriptor,
      ScopedActivationDescriptor& activation_descriptor)
      : ScopedFusionPlanBase(miopen_handle, miopenVerticalFusion,
                             input_descriptor) {
    uint64 hash = GetFusionOpHashValue(miopen_handle, input_descriptor,
                                       filter_descriptor, conv_descriptor,
                                       bias_descriptor, activation_descriptor);

    bool is_compiled = CachedFusionPlans::FindOrCreate(
        hash, &fusion_plan_, miopenVerticalFusion, input_descriptor);
    if (!is_compiled) {
      miopenFusionOpDescriptor_t conv_op;
      auto status = wrap::miopenCreateOpConvForward(
          fusion_plan_, &conv_op, conv_descriptor, filter_descriptor);
      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenCreateOpConvForward failed: "
                   << ToString(status);
      }

      miopenFusionOpDescriptor_t bias_op;
      status = wrap::miopenCreateOpBiasForward(fusion_plan_, &bias_op,
                                               bias_descriptor);
      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenCreateOpBiasForward failed: "
                   << ToString(status);
      }

      miopenFusionOpDescriptor_t actv_op;
      status = wrap::miopenCreateOpActivationForward(
          fusion_plan_, &actv_op,
          activation_descriptor.miopen_activation_mode_);
      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenCreateOpActivationForward failed: "
                   << ToString(status);
      }

      status = wrap::miopenCompileFusionPlan(miopen_handle_, fusion_plan_);
      if (status != miopenStatusSuccess) {
        VLOG(2) << "call to miopenCompileFusionPlan (CBA) failed: "
                << ToString(status);

        CachedFusionPlans::MarkFusionPlanUnsupported(hash);
      } else {
        VLOG(2) << "Fusion Plan compile succedded (CBA) ";
        fusion_plan_compiled_ = true;
      }
    } else {
      // fusion plan was already compiled...check whether it failed to compile
      fusion_plan_compiled_ = !CachedFusionPlans::IsUnsupportedFusionPlan(hash);
    }
  }

  miopenStatus_t SetConvolutionArgs(const void* filter_data) {
    float alpha = 1.0;
    float beta = 0.0;
    return ScopedFusionPlanBase::SetConvolutionArgs(k_conv_op_idx, &alpha,
                                                    &beta, filter_data);
  }

  miopenStatus_t SetBiasArgs(const void* bias_data) {
    float alpha = 1.0;
    float beta = 0.0;
    return ScopedFusionPlanBase::SetBiasArgs(k_bias_op_idx, &alpha, &beta,
                                             bias_data);
  }

  miopenStatus_t SetActivationForwardArgs(
      ScopedActivationDescriptor& activation_descriptor) {
    float alpha = 1.0;
    float beta = 0.0;

    return ScopedFusionPlanBase::SetActivationForwardArgs(
        k_actv_op_idx, &alpha, &beta, activation_descriptor.alpha_,
        activation_descriptor.beta_, activation_descriptor.gamma_);
  }

  uint64 GetFusionOpHashValue(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t filter_descriptor,
      miopenConvolutionDescriptor_t conv_descriptor,
      miopenTensorDescriptor_t bias_descriptor,
      ScopedActivationDescriptor& activation_descriptor) {
    uint64 hash_value = tensorflow::Hash64("ConvolutionBiasActivation");

    hash_value = tensorflow::Hash64Combine(
        hash_value, tensorflow::hash<miopenHandle_t>()(miopen_handle));

    hash_value =
        tensorflow::Hash64Combine(hash_value, GetHashValue(input_descriptor));
    hash_value =
        tensorflow::Hash64Combine(hash_value, GetHashValue(filter_descriptor));
    hash_value =
        tensorflow::Hash64Combine(hash_value, GetHashValue(conv_descriptor));
    hash_value =
        tensorflow::Hash64Combine(hash_value, GetHashValue(bias_descriptor));
    hash_value = tensorflow::Hash64Combine(
        hash_value, activation_descriptor.GetHashValue());
    return hash_value;
  }

 private:
  const int k_conv_op_idx = 0;
  const int k_bias_op_idx = 1;
  const int k_actv_op_idx = 2;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedFusionPlanConvolutionBiasActivation);
};

// class to represent the BatchNorm+Activation (inference) fusion plan
class ScopedFusionPlanBatchNormActivationInference
    : public ScopedFusionPlanBase {
 public:
  ScopedFusionPlanBatchNormActivationInference(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t scale_offset_mean_variance_descriptor,
      ScopedActivationDescriptor& activation_descriptor)
      : ScopedFusionPlanBase(miopen_handle, miopenVerticalFusion,
                             input_descriptor) {
    uint64 hash = GetFusionOpHashValue(miopen_handle, input_descriptor,
                                       scale_offset_mean_variance_descriptor,
                                       activation_descriptor);

    bool is_compiled = CachedFusionPlans::FindOrCreate(
        hash, &fusion_plan_, miopenVerticalFusion, input_descriptor);

    if (!is_compiled) {
      miopenFusionOpDescriptor_t batchnorm_op;
      auto status = wrap::miopenCreateOpBatchNormInference(
          fusion_plan_, &batchnorm_op, miopenBNSpatial,
          scale_offset_mean_variance_descriptor);

      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenCreateOpBatchNormInference failed: "
                   << ToString(status);
      }

      miopenFusionOpDescriptor_t actv_op;
      status = wrap::miopenCreateOpActivationForward(
          fusion_plan_, &actv_op,
          activation_descriptor.miopen_activation_mode_);
      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenCreateOpActivationForward failed: "
                   << ToString(status);
      }

      status = wrap::miopenCompileFusionPlan(miopen_handle_, fusion_plan_);
      if (status != miopenStatusSuccess) {
        VLOG(2) << "call to miopenCompileFusionPlan (BnA inference) failed: "
                << ToString(status);

        CachedFusionPlans::MarkFusionPlanUnsupported(hash);
      } else {
        VLOG(2) << "Fusion Plan compile succedded (BnA inference) ";
        fusion_plan_compiled_ = true;
      }
    } else {
      // fusion plan was already compiled...check whether it failed to compile
      fusion_plan_compiled_ = !CachedFusionPlans::IsUnsupportedFusionPlan(hash);
    }
  }

  miopenStatus_t SetBatchNormInferenceArgs(const void* scale,
                                           const void* offset, const void* mean,
                                           const void* variance,
                                           double epsilon) {
    float alpha = 1.0;
    float beta = 0.0;
    return ScopedFusionPlanBase::SetBatchNormInferenceArgs(
        k_batchnorm_op_idx, &alpha, &beta, scale, offset, mean, variance,
        epsilon);
  }

  miopenStatus_t SetActivationForwardArgs(
      ScopedActivationDescriptor& activation_descriptor) {
    float alpha = 1.0;
    float beta = 0.0;

    return ScopedFusionPlanBase::SetActivationForwardArgs(
        k_actv_op_idx, &alpha, &beta, activation_descriptor.alpha_,
        activation_descriptor.beta_, activation_descriptor.gamma_);
  }

  uint64 GetFusionOpHashValue(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t scale_offset_mean_variance_descriptor,
      ScopedActivationDescriptor& activation_descriptor) {
    uint64 hash_value = tensorflow::Hash64("BatchNormActivationInference");

    hash_value = tensorflow::Hash64Combine(
        hash_value, tensorflow::hash<miopenHandle_t>()(miopen_handle));

    hash_value =
        tensorflow::Hash64Combine(hash_value, GetHashValue(input_descriptor));

    hash_value = tensorflow::Hash64Combine(
        hash_value, GetHashValue(scale_offset_mean_variance_descriptor));

    hash_value = tensorflow::Hash64Combine(
        hash_value, activation_descriptor.GetHashValue());
    return hash_value;
  }

 private:
  const int k_batchnorm_op_idx = 0;
  const int k_actv_op_idx = 1;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedFusionPlanBatchNormActivationInference);
};

// class to represent the BatchNorm+Activation (training-forward) fusion plan
class ScopedFusionPlanBatchNormActivationForward : public ScopedFusionPlanBase {
 public:
  ScopedFusionPlanBatchNormActivationForward(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t scale_offset_mean_variance_descriptor,
      ScopedActivationDescriptor& activation_descriptor)
      : ScopedFusionPlanBase(miopen_handle, miopenVerticalFusion,
                             input_descriptor) {
    uint64 hash = GetFusionOpHashValue(miopen_handle, input_descriptor,
                                       scale_offset_mean_variance_descriptor,
                                       activation_descriptor);

    bool is_compiled = CachedFusionPlans::FindOrCreate(
        hash, &fusion_plan_, miopenVerticalFusion, input_descriptor);

    if (!is_compiled) {
      miopenFusionOpDescriptor_t batchnorm_op;
      auto status = wrap::miopenCreateOpBatchNormForward(
          fusion_plan_, &batchnorm_op, miopenBNSpatial,
          true /* runningMeanVariance */);

      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenCreateOpBatchNormForward failed: "
                   << ToString(status);
      }

      miopenFusionOpDescriptor_t actv_op;
      status = wrap::miopenCreateOpActivationForward(
          fusion_plan_, &actv_op,
          activation_descriptor.miopen_activation_mode_);
      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenCreateOpActivationForward failed: "
                   << ToString(status);
      }

      status = wrap::miopenCompileFusionPlan(miopen_handle_, fusion_plan_);
      if (status != miopenStatusSuccess) {
        VLOG(2) << "call to miopenCompileFusionPlan (BnA forward) failed: "
                << ToString(status);

        CachedFusionPlans::MarkFusionPlanUnsupported(hash);
      } else {
        VLOG(2) << "Fusion Plan compile succedded (BnA forward) ";
        fusion_plan_compiled_ = true;
      }
    } else {
      // fusion plan was already compiled...check whether it failed to compile
      fusion_plan_compiled_ = !CachedFusionPlans::IsUnsupportedFusionPlan(hash);
    }
  }

  miopenStatus_t SetBatchNormForwardArgs(const void* scale, const void* offset,
                                         void* batch_mean, void* batch_var,
                                         void* saved_mean, void* saved_var,
                                         double epsilon) {
    float alpha = 1.0;
    float beta = 0.0;
    return ScopedFusionPlanBase::SetBatchNormForwardArgs(
        k_batchnorm_op_idx, &alpha, &beta, scale, offset, batch_mean, batch_var,
        saved_mean, saved_var, epsilon);
  }

  miopenStatus_t SetActivationForwardArgs(
      ScopedActivationDescriptor& activation_descriptor) {
    float alpha = 1.0;
    float beta = 0.0;

    return ScopedFusionPlanBase::SetActivationForwardArgs(
        k_actv_op_idx, &alpha, &beta, activation_descriptor.alpha_,
        activation_descriptor.beta_, activation_descriptor.gamma_);
  }

  uint64 GetFusionOpHashValue(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t scale_offset_mean_variance_descriptor,
      ScopedActivationDescriptor& activation_descriptor) {
    uint64 hash_value = tensorflow::Hash64("BatchNormActivationForward");

    hash_value = tensorflow::Hash64Combine(
        hash_value, tensorflow::hash<miopenHandle_t>()(miopen_handle));

    hash_value =
        tensorflow::Hash64Combine(hash_value, GetHashValue(input_descriptor));

    hash_value = tensorflow::Hash64Combine(
        hash_value, GetHashValue(scale_offset_mean_variance_descriptor));

    hash_value = tensorflow::Hash64Combine(
        hash_value, activation_descriptor.GetHashValue());
    return hash_value;
  }

 private:
  const int k_batchnorm_op_idx = 0;
  const int k_actv_op_idx = 1;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedFusionPlanBatchNormActivationForward);
};

// class to represent the BatchNorm+Activation (training-backward) fusion plan
class ScopedFusionPlanBatchNormActivationBackward
    : public ScopedFusionPlanBase {
 public:
  ScopedFusionPlanBatchNormActivationBackward(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t scale_offset_mean_variance_descriptor,
      ScopedActivationDescriptor& activation_descriptor)
      : ScopedFusionPlanBase(miopen_handle, miopenVerticalFusion,
                             input_descriptor) {
    uint64 hash = GetFusionOpHashValue(miopen_handle, input_descriptor,
                                       scale_offset_mean_variance_descriptor,
                                       activation_descriptor);

    bool is_compiled = CachedFusionPlans::FindOrCreate(
        hash, &fusion_plan_, miopenVerticalFusion, input_descriptor);

    if (!is_compiled) {
      miopenFusionOpDescriptor_t batchnorm_op;
      auto status = wrap::miopenCreateOpBatchNormBackward(
          fusion_plan_, &batchnorm_op, miopenBNSpatial);

      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenCreateOpBatchNormBackward failed: "
                   << ToString(status);
      }

      miopenFusionOpDescriptor_t actv_op;
      status = wrap::miopenCreateOpActivationBackward(
          fusion_plan_, &actv_op,
          activation_descriptor.miopen_activation_mode_);
      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenCreateOpActivationBackward failed: "
                   << ToString(status);
      }

      status = wrap::miopenCompileFusionPlan(miopen_handle_, fusion_plan_);
      if (status != miopenStatusSuccess) {
        VLOG(2) << "call to miopenCompileFusionPlan (BnA backward) failed: "
                << ToString(status);

        CachedFusionPlans::MarkFusionPlanUnsupported(hash);
      } else {
        VLOG(2) << "Fusion Plan compile succedded (BnA backward) ";
        fusion_plan_compiled_ = true;
      }
    } else {
      // fusion plan was already compiled...check whether it failed to compile
      fusion_plan_compiled_ = !CachedFusionPlans::IsUnsupportedFusionPlan(hash);
    }
  }

  miopenStatus_t SetBatchNormBackwardArgs(const void* x, const void* scale,
                                          const void* offset,
                                          const void* saved_mean,
                                          const void* saved_var,
                                          void* scale_grad, void* offset_grad) {
    float alpha = 1.0;
    float beta = 0.0;

    return ScopedFusionPlanBase::SetBatchNormBackwardArgs(
        k_batchnorm_op_idx, &alpha, &beta, x, scale, offset, scale_grad,
        offset_grad, saved_mean, saved_var);
  }

  miopenStatus_t SetActivationBackwardArgs(
      ScopedActivationDescriptor& activation_descriptor, const void* y) {
    float alpha = 1.0;
    float beta = 0.0;

    return ScopedFusionPlanBase::SetActivationBackwardArgs(
        k_actv_op_idx, &alpha, &beta, y, activation_descriptor.alpha_,
        activation_descriptor.beta_, activation_descriptor.gamma_);
  }

  uint64 GetFusionOpHashValue(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t scale_offset_mean_variance_descriptor,
      ScopedActivationDescriptor& activation_descriptor) {
    uint64 hash_value = tensorflow::Hash64("BatchNormActivationBackward");

    hash_value = tensorflow::Hash64Combine(
        hash_value, tensorflow::hash<miopenHandle_t>()(miopen_handle));

    hash_value =
        tensorflow::Hash64Combine(hash_value, GetHashValue(input_descriptor));

    hash_value = tensorflow::Hash64Combine(
        hash_value, GetHashValue(scale_offset_mean_variance_descriptor));

    hash_value = tensorflow::Hash64Combine(
        hash_value, activation_descriptor.GetHashValue());
    return hash_value;
  }

 private:
  const int k_batchnorm_op_idx = 0;
  const int k_actv_op_idx = 1;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedFusionPlanBatchNormActivationBackward);
};

namespace {
miopenDataType_t ToMIOpenDataType(
    dnn::DataType data_type,
    dnn::DataLayout data_layout = dnn::DataLayout::kBatchDepthYX) {
  switch (data_type) {
    case dnn::DataType::kFloat:
      return miopenFloat;
    case dnn::DataType::kHalf:
      return miopenHalf;
    case dnn::DataType::kDouble:
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

miopenDataType_t ToMIOpenDataType(dnn::DataType data_type,
                                  dnn::FilterLayout filter_layout) {
  return ToMIOpenDataType(data_type);
}

miopenRNNInputMode_t ToMIOpenRnnInputMode(dnn::RnnInputMode input_mode) {
  switch (input_mode) {
    case dnn::RnnInputMode::kRnnLinearSkip:
      return miopenRNNlinear;
    case dnn::RnnInputMode::kRnnSkipInput:
      return miopenRNNskip;
    default:
      LOG(FATAL) << "Invalid RNN input mode: " << static_cast<int>(input_mode);
  }
}

miopenRNNDirectionMode_t ToMIOpenRnnDirectionMode(
    dnn::RnnDirectionMode direction_mode) {
  switch (direction_mode) {
    case dnn::RnnDirectionMode::kRnnUnidirectional:
      return miopenRNNunidirection;
    case dnn::RnnDirectionMode::kRnnBidirectional:
      return miopenRNNbidirection;
    default:
      LOG(FATAL) << "Invalid RNN direction mode: "
                 << static_cast<int>(direction_mode);
  }
}

miopenRNNMode_t ToMIOpenRnnMode(dnn::RnnMode rnn_mode) {
  switch (rnn_mode) {
    case dnn::RnnMode::kRnnRelu:
      return miopenRNNRELU;
    case dnn::RnnMode::kRnnTanh:
      return miopenRNNTANH;
    case dnn::RnnMode::kRnnLstm:
      return miopenLSTM;
    case dnn::RnnMode::kRnnGru:
      return miopenGRU;
    default:
      LOG(FATAL) << "Invalid RNN Mode: " << static_cast<int>(rnn_mode);
  }
}

int MIOpenDataTypeToByteSize(miopenDataType_t data_type) {
  switch (data_type) {
    case miopenFloat:
      return sizeof(float);
    case miopenHalf:
      return sizeof(Eigen::half);
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

template <typename Base>
class MixinBase : public Base {};
template <>
class MixinBase<void> {};

dnn::DataType GetConvAccumulatorType(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kFloat:
    case dnn::DataType::kDouble:
      return data_type;
    case dnn::DataType::kHalf:
      // FIXME: Check if MIOpen can switch dynamically change accumulator type
      return dnn::DataType::kFloat;
    case dnn::DataType::kInt8:
    case dnn::DataType::kInt32:
      return dnn::DataType::kInt32;
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

}  // namespace

#define RETURN_IF_MIOPEN_ERROR(STATUS, ...)                              \
  if (!SE_PREDICT_TRUE((STATUS) == miopenStatusSuccess)) {               \
    string error_msg = absl::StrCat(ToString(STATUS), " ", __VA_ARGS__); \
    SetFailure(port::Status(port::error::UNKNOWN, error_msg));           \
    LOG(ERROR) << error_msg;                                             \
    return;                                                              \
  }

template <typename Base>
class MIOpenDescriptorCommon : public MixinBase<Base> {
 public:
  bool ok() const { return status_.ok(); }
  port::Status Status() const { return status_; }

 protected:
  void SetFailure(const port::Status& status) { status_.Update(status); }
  port::Status status_;
};

class MIOpenRnnParamsDescriptor : public MIOpenDescriptorCommon<void> {
 public:
  typedef dnn::RnnDescriptor::ParamsRegion ParamsRegion;
  typedef dnn::RnnDescriptor::ParamsRegions ParamsRegions;
  MIOpenRnnParamsDescriptor(miopenHandle_t miopen_handle,
                            const MIOpenRnnDescriptor& rnn_desc);
  ~MIOpenRnnParamsDescriptor() {
    auto status = wrap::miopenDestroyTensorDescriptor(handle_);
    RETURN_IF_MIOPEN_ERROR(status, "Failed to destroy RNN tensor descriptor");
  }
  miopenTensorDescriptor_t handle() const {
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
  miopenTensorDescriptor_t handle_;
  const MIOpenRnnDescriptor* rnn_desc_;
  int64 params_size_in_bytes_;
  ParamsRegions weights_;
  ParamsRegions biases_;
  port::Status status_;
  SE_DISALLOW_COPY_AND_ASSIGN(MIOpenRnnParamsDescriptor);
};

class MIOpenRnnDescriptor : public MIOpenDescriptorCommon<dnn::RnnDescriptor> {
 public:
  MIOpenRnnDescriptor(miopenHandle_t miopen_handle, int num_layers,
                      int hidden_size, int input_size,
                      miopenRNNInputMode_t input_mode,
                      miopenRNNDirectionMode_t direction_mode,
                      miopenRNNMode_t rnn_mode, miopenDataType_t data_type,
                      float dropout, uint64 seed,
                      ScratchAllocator* state_allocator)
      : rnn_desc_(nullptr),
        num_layers_(num_layers),
        hidden_size_(hidden_size),
        input_size_(input_size),
        input_mode_(input_mode),
        direction_mode_(direction_mode),
        rnn_mode_(rnn_mode),
        data_type_(data_type) {
    // Create the RNN handle
    auto status = wrap::miopenCreateRNNDescriptor(&rnn_desc_);
    RETURN_IF_MIOPEN_ERROR(status, "Unable to create RNN descriptor");
    status = wrap::miopenSetRNNDescriptor(
        rnn_desc_ /*rnnDesc*/, hidden_size /*hiddenSize*/,
        num_layers /*numLayers*/, input_mode /*inputMode*/,
        direction_mode /*direction*/, rnn_mode /*mode*/,
        miopenRNNwithBias /*biasMode*/, miopenRNNdefault /*algo*/,
        data_type /*dataType*/);
    RETURN_IF_MIOPEN_ERROR(status, "Unable to update RNN descriptor");
    // Create the params handle.
    miopen_params_desc_.reset(
        new MIOpenRnnParamsDescriptor(miopen_handle, *this));
    if (!miopen_params_desc_->ok()) {
      SetFailure(miopen_params_desc_->Status());
      return;
    }
  }
  ~MIOpenRnnDescriptor() override {
    if (rnn_desc_) {
      auto status = wrap::miopenDestroyRNNDescriptor(rnn_desc_);
      RETURN_IF_MIOPEN_ERROR(status, "Unable to destroy RNN descriptor");
    }
  }
  miopenRNNDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return rnn_desc_;
  }
  int num_layers() const { return num_layers_; }
  int hidden_size() const { return hidden_size_; }
  int input_size() const { return input_size_; }
  miopenRNNInputMode_t input_mode() const { return input_mode_; }
  miopenRNNDirectionMode_t direction_mode() const { return direction_mode_; }
  miopenRNNMode_t rnn_mode() const { return rnn_mode_; }
  miopenDataType_t data_type() const { return data_type_; }
  int64 ParamsSizeInBytes() const override {
    return miopen_params_desc_->params_size_in_bytes();
  }
  miopenTensorDescriptor_t params_handle() const {
    if (!miopen_params_desc_) return nullptr;
    return miopen_params_desc_->handle();
  }
  ParamsRegions ParamsWeightRegions() const override {
    if (!ok()) return ParamsRegions();
    return miopen_params_desc_->params_weights();
  }
  ParamsRegions ParamsBiasRegions() const override {
    if (!ok()) return ParamsRegions();
    return miopen_params_desc_->params_biases();
  }

 private:
  miopenRNNDescriptor_t rnn_desc_;
  int num_layers_;
  int hidden_size_;
  int input_size_;
  miopenRNNInputMode_t input_mode_;
  miopenRNNDirectionMode_t direction_mode_;
  miopenRNNMode_t rnn_mode_;
  miopenDataType_t data_type_;
  port::Status status_;
  // no dropout in MIOpen.
  // std::unique_ptr<miopenDropoutDescriptor> miopen_dropout_desc_;
  std::unique_ptr<MIOpenRnnParamsDescriptor> miopen_params_desc_;
  SE_DISALLOW_COPY_AND_ASSIGN(MIOpenRnnDescriptor);
};

// Get ID of the internal parameter tensor.
//
int MIOpenRnnParamsDescriptor::GetRegionCountPerLayer() const {
  auto rnn_mode = rnn_desc_->rnn_mode();
  switch (rnn_mode) {
    case miopenRNNRELU:
    case miopenRNNTANH:
      return 2;
    case miopenLSTM:
      return 8;
    case miopenGRU:
      return 6;
    default:
      LOG(FATAL) << "Invalid RNN Mode: " << static_cast<int>(rnn_mode);
  }
}

class MIOpenRnnSequenceTensorDescriptor
    : public MIOpenDescriptorCommon<dnn::RnnSequenceTensorDescriptor> {
 public:
  MIOpenRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                    int data_size, miopenDataType_t data_type)
      : seq_length_(seq_length),
        batch_size_(batch_size),
        data_size_(data_size),
        data_type_(data_type) {
    miopenTensorDescriptor_t handle = nullptr;
    if (seq_length <= 0) {
      string error_msg =
          absl::StrCat("sequence length must be positive: ", seq_length);
      LOG(ERROR) << error_msg;
      SetFailure(port::Status(port::error::UNKNOWN, error_msg));
      return;
    }
    auto status = wrap::miopenCreateTensorDescriptor(&handle);
    RETURN_IF_MIOPEN_ERROR(status, "Failed to create tensor descriptor");
    std::array<int, 2> dims = {{batch_size, data_size}};
    status = wrap::miopenSetTensorDescriptor(
        handle /*tensorDesc*/, data_type /*dataType*/, 2 /*nbDims*/,
        dims.data() /*dimA*/, nullptr /*strideA*/);
    RETURN_IF_MIOPEN_ERROR(status, "Failed to update tensor descriptor");
    // Replicate handle across the number of steps.
    handles_.assign(seq_length, handle);
  }

  ~MIOpenRnnSequenceTensorDescriptor() override {
    // Only the first one needs to be destroyed. All others are the same.
    auto status = wrap::miopenDestroyTensorDescriptor(handles_[0]);
    RETURN_IF_MIOPEN_ERROR(status,
                           "Failed to destroy sequence tensor descriptor");
  }

  const miopenTensorDescriptor_t* handles() const {
    if (!ok()) return nullptr;
    CHECK(!handles_.empty()) << "handles cannot be empty";
    return handles_.data();
  }

  int seq_length() const { return seq_length_; }
  int batch_size() const { return batch_size_; }
  int data_size() const { return data_size_; }

 private:
  int seq_length_;
  int batch_size_;
  int data_size_;
  miopenDataType_t data_type_;
  std::vector<miopenTensorDescriptor_t> handles_;
  port::Status status_;
  SE_DISALLOW_COPY_AND_ASSIGN(MIOpenRnnSequenceTensorDescriptor);
};

class MIOpenRnnStateTensorDescriptor
    : public MIOpenDescriptorCommon<dnn::RnnStateTensorDescriptor> {
 public:
  MIOpenRnnStateTensorDescriptor(int num_layers, int batch_size, int data_size,
                                 miopenDataType_t data_type)
      : handle_(nullptr),
        num_layers_(num_layers),
        batch_size_(batch_size),
        data_size_(data_size),
        data_type_(data_type) {
    auto status = wrap::miopenCreateTensorDescriptor(&handle_);
    RETURN_IF_MIOPEN_ERROR(status, "Failed to create tensor descriptor");
    std::array<int, 3> dims = {{num_layers, batch_size, data_size}};
    status = wrap::miopenSetTensorDescriptor(
        handle_ /*tensorDesc*/, data_type /*dataType*/, 3 /*nbDims*/,
        dims.data() /*dimA*/, nullptr /*strideA*/);
    RETURN_IF_MIOPEN_ERROR(status, "Failed to update tensor descriptor");
  }

  ~MIOpenRnnStateTensorDescriptor() override {
    if (!handle_) {
      auto status = wrap::miopenDestroyTensorDescriptor(handle_);
      RETURN_IF_MIOPEN_ERROR(status, "Unable to destroy RNN state tensor");
    }
  }

  miopenTensorDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return handle_;
  }
  int num_layers() const { return num_layers_; }
  int batch_size() const { return batch_size_; }
  int data_size() const { return data_size_; }

 private:
  miopenTensorDescriptor_t handle_;
  int num_layers_;
  int batch_size_;
  int data_size_;
  port::Status status_;
  miopenDataType_t data_type_;
  SE_DISALLOW_COPY_AND_ASSIGN(MIOpenRnnStateTensorDescriptor);
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
    const MIOpenRnnDescriptor& rnn_desc,
    const MIOpenRnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<T>& input_data,
    const MIOpenRnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<T>& input_h_data,
    const MIOpenRnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
    const MIOpenRnnSequenceTensorDescriptor& output_desc,
    const DeviceMemory<T>& output_data,
    const MIOpenRnnStateTensorDescriptor& output_h_desc,
    const DeviceMemory<T>& output_h_data,
    const MIOpenRnnStateTensorDescriptor& output_c_desc,
    const DeviceMemory<T>& output_c_data, RnnModelDims* model_dims) {
  // extract model parameters
  model_dims->num_layers = rnn_desc.num_layers();
  model_dims->batch_size = input_desc.batch_size();
  model_dims->seq_length = input_desc.seq_length();
  model_dims->hidden_size = rnn_desc.hidden_size();
  model_dims->input_size = input_desc.data_size();
  model_dims->dir_count =
      (rnn_desc.direction_mode() == miopenRNNbidirection) ? 2 : 1;

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

bool CheckRNNParameterSize(
    miopenHandle_t miopen_handle, const MIOpenRnnDescriptor& rnn_desc,
    const MIOpenRnnSequenceTensorDescriptor& input_desc) {
  size_t params_size_in_bytes = 0;
  auto status = wrap::miopenGetRNNParamsSize(
      miopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
      input_desc.handles()[0] /*xDesc*/, &params_size_in_bytes /*sizeInBytes*/,
      rnn_desc.data_type() /*dataType*/);
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "Unable to check RNN param size: " << ToString(status);
    return false;
  }
  return static_cast<int64>(params_size_in_bytes) ==
         rnn_desc.ParamsSizeInBytes();
}

bool CreateRnnWorkspace(Stream* stream, miopenHandle_t miopen_handle,
                        const MIOpenRnnDescriptor& rnn_desc,
                        const MIOpenRnnSequenceTensorDescriptor& input_desc,
                        ScratchAllocator* workspace_allocator,
                        DeviceMemory<uint8>* workspace) {
  // Query the workspace size.
  size_t workspace_size_in_bytes = 0;
  auto status = wrap::miopenGetRNNWorkspaceSize(
      miopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
      input_desc.seq_length() /*seqLength*/, input_desc.handles() /*xDesc*/,
      &workspace_size_in_bytes /*sizeInBytes*/);
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "Unable to query workspace size: " << ToString(status);
    return false;
  }
  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (*workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate RNN workspace";

      return false;
    }
    stream->ThenMemZero(workspace, workspace_size_in_bytes);
  } else {
    *workspace = DeviceMemory<uint8>();
  }
  return true;
}

}  // namespace

template <class T>
bool MIOpenSupport::DoRnnForwardImpl(
    Stream* stream, const MIOpenRnnDescriptor& rnn_desc,
    const MIOpenRnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<T>& input_data,
    const MIOpenRnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<T>& input_h_data,
    const MIOpenRnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
    const MIOpenRnnSequenceTensorDescriptor& output_desc,
    DeviceMemory<T>* output_data,
    const MIOpenRnnStateTensorDescriptor& output_h_desc,
    DeviceMemory<T>* output_h_data,
    const MIOpenRnnStateTensorDescriptor& output_c_desc,
    DeviceMemory<T>* output_c_data, bool is_training,
    ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator) {
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

  auto miopen = miopen_->GetHandle(parent_, stream);

  // check params size

  if (!CheckRNNParameterSize(miopen.handle(), rnn_desc, input_desc)) {
    LOG(ERROR) << "Invalid parameters";
    return false;
  }

  // create the workspace
  DeviceMemory<uint8> workspace;
  if (!CreateRnnWorkspace(stream, miopen.handle(), rnn_desc, input_desc,
                          workspace_allocator, &workspace)) {
    LOG(ERROR) << "Unable to create rnn workspace";

    return false;
  }

  // query the reserve space size
  // allocate the reserve space
  DeviceMemory<uint8> reserve_space;
  if (is_training) {
    size_t reserve_space_size_in_bytes = 0;
    auto status = wrap::miopenGetRNNTrainingReserveSize(
        miopen.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        model_dims.seq_length /*seqLength*/, input_desc.handles() /*xDesc*/,
        &reserve_space_size_in_bytes /*sizeInBytes*/);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "Unable to query reserve space size: " << ToString(status);
      return false;
    }

    if (reserve_space_size_in_bytes > 0) {
      auto allocated = reserve_space_allocator->AllocateBytes(
          stream, reserve_space_size_in_bytes);
      if (!allocated.ok() ||
          (reserve_space = allocated.ValueOrDie()) == nullptr) {
        LOG(ERROR) << "Fail to allocate RNN reserve space";
        return false;
      }
      stream->ThenMemZero(&reserve_space, reserve_space_size_in_bytes);
    }
  }

  // make the forward call
  if (!is_training) {
    auto status = wrap::miopenRNNForwardInference(
        miopen.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        model_dims.seq_length /*seqLength*/, input_desc.handles() /*xDesc*/,
        input_data.opaque() /*x*/, input_h_desc.handle() /*hxDesc*/,
        input_h_data.opaque() /*hx*/, input_c_desc.handle() /*cxDesc*/,
        input_c_data.opaque() /*cx*/, rnn_desc.params_handle() /*wDesc*/,
        params.opaque() /*w*/, output_desc.handles() /*yDesc*/,
        output_data->opaque() /*y*/, output_h_desc.handle() /*hyDesc*/,
        output_h_data->opaque() /*hy*/, output_c_desc.handle() /*cyDesc*/,
        output_c_data->opaque() /*cy*/, workspace.opaque() /*workspace*/,
        workspace.size() /*workSpaceSizeInBytes*/);

    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "Failed to call miopenRNNForwardInference: "
                 << ToString(status);
      return false;
    }
  } else {
    auto status = wrap::miopenRNNForwardTraining(
        miopen.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
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
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "Failed to call miopenRNNForwardTraining"
                 << ToString(status);
      return false;
    }
  }
  return true;
}

template <class T>
bool MIOpenSupport::DoRnnBackwardImpl(
    Stream* stream, const MIOpenRnnDescriptor& rnn_desc,
    const MIOpenRnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<T>& input_data,
    const MIOpenRnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<T>& input_h_data,
    const MIOpenRnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
    const MIOpenRnnSequenceTensorDescriptor& output_desc,
    const DeviceMemory<T>& output_data,
    const MIOpenRnnStateTensorDescriptor& output_h_desc,
    const DeviceMemory<T>& output_h_data,
    const MIOpenRnnStateTensorDescriptor& output_c_desc,
    const DeviceMemory<T>& output_c_data,
    const DeviceMemory<T>& output_backprop_data,
    const DeviceMemory<T>& output_h_backprop_data,
    const DeviceMemory<T>& output_c_backprop_data,
    DeviceMemory<T>* input_backprop_data,
    DeviceMemory<T>* input_h_backprop_data,
    DeviceMemory<T>* input_c_backprop_data,
    DeviceMemory<T>* params_backprop_data,
    DeviceMemory<uint8>* reserve_space_data,
    ScratchAllocator* workspace_allocator) {
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

  auto miopen = miopen_->GetHandle(parent_, stream);

  // check params size

  if (!CheckRNNParameterSize(miopen.handle(), rnn_desc, input_desc)) {
    LOG(ERROR) << "Invalid parameters";
    return false;
  }

  // create the workspace
  DeviceMemory<uint8> workspace;
  if (!CreateRnnWorkspace(stream, miopen.handle(), rnn_desc, input_desc,
                          workspace_allocator, &workspace)) {
    LOG(ERROR) << "Unable to create rnn workspace";
    return false;
  }

  // workaround for missing initialization support in MIOpen.
  // TODO: remove this when MIOpen is ready.
  long size_data = input_desc.seq_length() * input_desc.batch_size() *
                   input_desc.data_size();
  if ((size_data > 0) && (input_backprop_data->opaque() != nullptr))
    stream->ThenMemZero(input_backprop_data, size_data * sizeof(float));

  size_data = input_h_desc.num_layers() * input_h_desc.batch_size() *
              input_h_desc.data_size();
  if ((size_data > 0) && (input_h_backprop_data->opaque() != nullptr))
    stream->ThenMemZero(input_h_backprop_data, size_data * sizeof(float));

  size_data = input_c_desc.num_layers() * input_c_desc.batch_size() *
              input_c_desc.data_size();
  if ((size_data > 0) && (input_c_backprop_data->opaque() != nullptr))
    stream->ThenMemZero(input_c_backprop_data, size_data * sizeof(float));

  // make the backward data call
  auto status = wrap::miopenRNNBackwardData(
      miopen.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
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
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "Failed to call miopenRNNBackwardData: " << ToString(status);
    return false;
  }

  if (params_backprop_data != nullptr) {
    // Clear the dw to zeros.
    stream->ThenMemZero(params_backprop_data, params_backprop_data->size());
    // make the backward weight call
    status = wrap::miopenRNNBackwardWeights(
        miopen.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        model_dims.seq_length /*seqLength*/, input_desc.handles() /*xDesc*/,
        input_data.opaque() /*x*/, input_h_desc.handle() /*hxDesc*/,
        input_h_data.opaque() /*hx*/, output_desc.handles() /*yDesc*/,
        output_data.opaque() /*y*/, rnn_desc.params_handle() /*dwDesc*/,
        params_backprop_data->opaque() /*dw*/, workspace.opaque() /*workspace*/,
        workspace.size() /*workSpaceSizeInBytes*/,
        reserve_space_data->opaque() /*reserveSpace*/,
        reserve_space_data->size() /*reserveSpaceSizeInBytes*/);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "Failed to call miopenRNNBackwardWeights: "
                 << ToString(status);
      return false;
    }
  }

  return true;
}

MIOpenRnnParamsDescriptor::MIOpenRnnParamsDescriptor(
    miopenHandle_t miopen_handle, const MIOpenRnnDescriptor& rnn_desc)
    : handle_(nullptr), rnn_desc_(&rnn_desc), params_size_in_bytes_(0) {
  miopenTensorDescriptor_t input_desc = nullptr;
  {
    // Query the params size.
    auto status = wrap::miopenCreateTensorDescriptor(&input_desc);
    RETURN_IF_MIOPEN_ERROR(status, "MIOpen fails to create tensor descriptor");
    std::array<int, 2> dims = {{1, rnn_desc.input_size()}};
    status = wrap::miopenSetTensorDescriptor(
        input_desc /*tensorDesc*/, rnn_desc.data_type() /*dataType*/,
        2 /*nbDims*/, dims.data() /*dimA*/, nullptr /*strideA*/);
    RETURN_IF_MIOPEN_ERROR(status, "MIOpen fails to set tensor descriptor");

    size_t params_size = 0;
    status = wrap::miopenGetRNNParamsSize(
        miopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        input_desc /*xDesc*/, &params_size /*sizeInBytes*/,
        rnn_desc.data_type() /*dataType*/);
    RETURN_IF_MIOPEN_ERROR(status, "MIOpen fails to get RNN parameter size");
    params_size_in_bytes_ = static_cast<int64>(params_size);
  }

  {
    // Create the params descriptor.
    auto status = wrap::miopenCreateTensorDescriptor(&handle_);
    RETURN_IF_MIOPEN_ERROR(status,
                           "MIOpen fails to create RNN params descriptor");
    status = wrap::miopenGetRNNParamsDescriptor(miopen_handle,
                                                rnn_desc.handle(), input_desc,
                                                handle_, rnn_desc.data_type());
    RETURN_IF_MIOPEN_ERROR(status,
                           "MIOpen fails to update RNN filter descriptor");
  }
  {
    // Release the dummy input tensor descriptor.
    auto status = wrap::miopenDestroyTensorDescriptor(input_desc);
    RETURN_IF_MIOPEN_ERROR(status, "MIOpen fails to destroy tensor descriptor");
  }
}

port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
MIOpenSupport::createRnnDescriptor(
    int num_layers, int hidden_size, int input_size, int batch_size,
    dnn::RnnInputMode input_mode, dnn::RnnDirectionMode direction_mode,
    dnn::RnnMode rnn_mode, dnn::DataType data_type,
    const dnn::AlgorithmConfig& algorithm_config, float dropout, uint64 seed,
    ScratchAllocator* state_allocator) {
  // ROCM TODO: batch_size is ignored for now

  auto miopen = miopen_->GetHandle(parent_, nullptr);
  std::unique_ptr<MIOpenRnnDescriptor> rnn_desc(new MIOpenRnnDescriptor(
      miopen.handle(), num_layers, hidden_size, input_size,
      ToMIOpenRnnInputMode(input_mode),
      ToMIOpenRnnDirectionMode(direction_mode), ToMIOpenRnnMode(rnn_mode),
      ToMIOpenDataType(data_type), dropout, seed, state_allocator));
  if (!rnn_desc->ok()) {
    return rnn_desc->Status();
  }
  return port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>(
      std::move(rnn_desc));
}

port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
MIOpenSupport::createRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                                 int data_size,
                                                 dnn::DataType data_type) {
  std::unique_ptr<MIOpenRnnSequenceTensorDescriptor> seq_desc(
      new MIOpenRnnSequenceTensorDescriptor(seq_length, batch_size, data_size,
                                            ToMIOpenDataType(data_type)));
  if (!seq_desc->ok()) {
    return seq_desc->Status();
  }
  return port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>(
      std::move(seq_desc));
}

port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
MIOpenSupport::createRnnStateTensorDescriptor(int num_layer, int batch_size,
                                              int data_size,
                                              dnn::DataType data_type) {
  std::unique_ptr<MIOpenRnnStateTensorDescriptor> state_desc(
      new MIOpenRnnStateTensorDescriptor(num_layer, batch_size, data_size,
                                         ToMIOpenDataType(data_type)));
  if (!state_desc->ok()) {
    return state_desc->Status();
  }
  return port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>(
      std::move(state_desc));
}

bool MIOpenSupport::DoRnnForward(
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
  // ROCM TODO: output_profile_result is ignore for now

  const MIOpenRnnDescriptor& miopen_rnn_desc =
      static_cast<const MIOpenRnnDescriptor&>(rnn_desc);
  const MIOpenRnnSequenceTensorDescriptor& miopen_input_desc =
      static_cast<const MIOpenRnnSequenceTensorDescriptor&>(input_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_input_h_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(input_h_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_input_c_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(input_c_desc);
  const MIOpenRnnSequenceTensorDescriptor& miopen_output_desc =
      static_cast<const MIOpenRnnSequenceTensorDescriptor&>(output_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_output_h_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(output_h_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_output_c_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(output_c_desc);

  return DoRnnForwardImpl<Eigen::half>(
      stream, miopen_rnn_desc, miopen_input_desc, input_data,
      miopen_input_h_desc, input_h_data, miopen_input_c_desc, input_c_data,
      params, miopen_output_desc, output_data, miopen_output_h_desc,
      output_h_data, miopen_output_c_desc, output_c_data, is_training,
      reserve_space_allocator, workspace_allocator);
}

bool MIOpenSupport::DoRnnForward(
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
  // ROCM TODO: output_profile_result is ignore for now

  const MIOpenRnnDescriptor& miopen_rnn_desc =
      static_cast<const MIOpenRnnDescriptor&>(rnn_desc);
  const MIOpenRnnSequenceTensorDescriptor& miopen_input_desc =
      static_cast<const MIOpenRnnSequenceTensorDescriptor&>(input_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_input_h_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(input_h_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_input_c_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(input_c_desc);
  const MIOpenRnnSequenceTensorDescriptor& miopen_output_desc =
      static_cast<const MIOpenRnnSequenceTensorDescriptor&>(output_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_output_h_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(output_h_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_output_c_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(output_c_desc);

  return DoRnnForwardImpl<float>(
      stream, miopen_rnn_desc, miopen_input_desc, input_data,
      miopen_input_h_desc, input_h_data, miopen_input_c_desc, input_c_data,
      params, miopen_output_desc, output_data, miopen_output_h_desc,
      output_h_data, miopen_output_c_desc, output_c_data, is_training,
      reserve_space_allocator, workspace_allocator);
}

bool MIOpenSupport::DoRnnForward(
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
  LOG(ERROR) << "miopen does not support double type RNN fwd yet";
  return false;
}

bool MIOpenSupport::DoRnnBackward(
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
  // ROCM TODO: output_profile_result is ignore for now

  const MIOpenRnnDescriptor& miopen_rnn_desc =
      static_cast<const MIOpenRnnDescriptor&>(rnn_desc);
  const MIOpenRnnSequenceTensorDescriptor& miopen_input_desc =
      static_cast<const MIOpenRnnSequenceTensorDescriptor&>(input_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_input_h_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(input_h_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_input_c_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(input_c_desc);
  const MIOpenRnnSequenceTensorDescriptor& miopen_output_desc =
      static_cast<const MIOpenRnnSequenceTensorDescriptor&>(output_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_output_h_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(output_h_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_output_c_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(output_c_desc);

  return DoRnnBackwardImpl<Eigen::half>(
      stream, miopen_rnn_desc, miopen_input_desc, input_data,
      miopen_input_h_desc, input_h_data, miopen_input_c_desc, input_c_data,
      params, miopen_output_desc, output_data, miopen_output_h_desc,
      output_h_data, miopen_output_c_desc, output_c_data, output_backprop_data,
      output_h_backprop_data, output_c_backprop_data, input_backprop_data,
      input_h_backprop_data, input_c_backprop_data, params_backprop_data,
      reserve_space_data, workspace_allocator);
}

bool MIOpenSupport::DoRnnBackward(
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
  // ROCM TODO: output_profile_result is ignore for now

  const MIOpenRnnDescriptor& miopen_rnn_desc =
      static_cast<const MIOpenRnnDescriptor&>(rnn_desc);
  const MIOpenRnnSequenceTensorDescriptor& miopen_input_desc =
      static_cast<const MIOpenRnnSequenceTensorDescriptor&>(input_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_input_h_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(input_h_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_input_c_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(input_c_desc);
  const MIOpenRnnSequenceTensorDescriptor& miopen_output_desc =
      static_cast<const MIOpenRnnSequenceTensorDescriptor&>(output_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_output_h_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(output_h_desc);
  const MIOpenRnnStateTensorDescriptor& miopen_output_c_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(output_c_desc);

  return DoRnnBackwardImpl<float>(
      stream, miopen_rnn_desc, miopen_input_desc, input_data,
      miopen_input_h_desc, input_h_data, miopen_input_c_desc, input_c_data,
      params, miopen_output_desc, output_data, miopen_output_h_desc,
      output_h_data, miopen_output_c_desc, output_c_data, output_backprop_data,
      output_h_backprop_data, output_c_backprop_data, input_backprop_data,
      input_h_backprop_data, input_c_backprop_data, params_backprop_data,
      reserve_space_data, workspace_allocator);
}

bool MIOpenSupport::DoRnnBackward(
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
  LOG(ERROR) << "miopen does not support half type RNN bwd yet";
  return false;
}

// This is the context required to use the TF scratch allocator:
struct MIOpenAllocatorContext {
  MIOpenAllocatorContext(ScratchAllocator* scratch_allocator, Stream* stream)
      : scratch_allocator_(scratch_allocator), stream_(stream){};

  ScratchAllocator* scratch_allocator_;
  Stream* stream_;
};

void* MIOpenAllocatorCallback(void* ctx, size_t size_in_bytes) {
  auto* mac = static_cast<MIOpenAllocatorContext*>(ctx);
  auto allocated =
      mac->scratch_allocator_->AllocateBytes(mac->stream_, size_in_bytes);

  DeviceMemory<uint8> scratch;
  if (allocated.ok()) {
    scratch = allocated.ValueOrDie();
    return scratch.opaque();
  } else {
    return nullptr;
  }
}

void MIOpenDeallocatorCallback(void* ctx, void* mem) {
  // Don't need dealloactor since the TensorFlow heap will automatically reclaim
  // the memory
}

port::Status MIOpenSupport::DoPrepareForConvolution(
    dnn::ConvolutionKind kind, dnn::DataType element_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::AlgorithmConfig& algorithm_config,
    ScratchAllocator* scratch_allocator, dnn::AlgorithmDesc* algorithm_desc,
    DeviceMemory<uint8>* scratch_memory) {
  ScopedTensorDescriptor input_nd{
      input_descriptor,
      ToMIOpenDataType(element_type, input_descriptor.layout())};
  ScopedFilterDescriptor filter{
      filter_descriptor, input_descriptor,
      ToMIOpenDataType(element_type, filter_descriptor.layout())};
  ScopedTensorDescriptor output_nd{
      output_descriptor,
      ToMIOpenDataType(element_type, output_descriptor.layout())};
  ScopedConvolutionDescriptor conv{
      convolution_descriptor,
      ToMIOpenDataType(GetConvAccumulatorType(element_type))};

  auto miopen = miopen_->GetHandle(parent_, stream);

  absl::optional<dnn::AlgorithmDesc> algo_desc = algorithm_config.algorithm();
  size_t scratch_memory_size;

  if (!algo_desc.has_value()) {
    // With the default algorithm, use MIOpen's heuristics.
    assert(scratch_allocator);

    DeviceMemory<uint8> scratch_memory_temp;
    MIOpenAllocatorContext mac(scratch_allocator, stream);
    wrap::miopenSetAllocator(miopen.handle(), MIOpenAllocatorCallback,
                             MIOpenDeallocatorCallback, &mac);
    size_t size_in_bytes;
    miopenStatus_t status = miopenStatusSuccess;

    switch (kind) {
      case dnn::ConvolutionKind::FORWARD: {
        status = wrap::miopenConvolutionForwardGetWorkSpaceSize(
            miopen.handle(), /*filterDesc=*/filter.handle(),
            /*srcDesc=*/input_nd.handle(), /*convDesc=*/conv.handle(),
            /*destDesc=*/output_nd.handle(), /*sizeInBytes=*/&size_in_bytes);
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_DATA: {
        status = wrap::miopenConvolutionBackwardDataGetWorkSpaceSize(
            miopen.handle(), /*diffDesc=*/output_nd.handle(),
            /*filterDesc=*/filter.handle(), /*convDesc=*/conv.handle(),
            /*gradDesc=*/input_nd.handle(), /*sizeInBytes=*/&size_in_bytes);
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_FILTER: {
        status = wrap::miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            miopen.handle(), /*diffDesc=*/output_nd.handle(),
            /*srcDesc=*/input_nd.handle(), /*convDesc=*/conv.handle(),
            /*gradDesc=*/filter.handle(), /*sizeInBytes=*/&size_in_bytes);
        break;
      }
      default:
        return port::InternalError(absl::StrCat("Unexpected convolution kind ",
                                                static_cast<int>(kind)));
    }

    if (status == miopenStatusSuccess && size_in_bytes != 0) {
      auto allocated = scratch_allocator->AllocateBytes(stream, size_in_bytes);
      if (allocated.ok()) {
        scratch_memory_temp = allocated.ValueOrDie();
      }
    }

    miopenConvAlgoPerf_t preference;
    int returnedAlgoCount;

    switch (kind) {
      case dnn::ConvolutionKind::FORWARD: {
        auto status = wrap::miopenFindConvolutionForwardAlgorithm(
            miopen.handle(), input_nd.handle(), input_data.opaque(),
            filter.handle(), filter_data.opaque(), conv.handle(),
            output_nd.handle(), output_data.opaque(),
            /*requestAlgoCount=*/1, &returnedAlgoCount,
            /*preference=*/&preference,
            /*workspace*/ scratch_memory_temp.opaque(),
            /*WorkSpaceSize*/ scratch_memory_temp.size(),
            /*exhaustiveSearch*/ false);
        CHECK_EQ(status, miopenStatusSuccess) << "Unable to find a suitable "
                                                 "algorithm for doing forward "
                                                 "convolution";
        *algorithm_desc = dnn::AlgorithmDesc(preference.fwd_algo, false);
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_DATA: {
        auto status = wrap::miopenFindConvolutionBackwardDataAlgorithm(
            miopen.handle(),
            /*diffDesc=*/output_nd.handle(), output_data.opaque(),
            /*filterDesc=*/filter.handle(), filter_data.opaque(),
            /*convDesc=*/conv.handle(),
            /*gradDesc=*/input_nd.handle(), input_data.opaque(),
            /*requestCount=*/1, /*returnedAlgoCount=*/&returnedAlgoCount,
            /*preference=*/&preference,
            /*WorkSpace=*/scratch_memory_temp.opaque(),
            /*WorkSpaceSize=*/scratch_memory_temp.size(),
            /*exhaustiveSearch=*/false);
        CHECK_EQ(status, miopenStatusSuccess) << "Unable to find a suitable "
                                                 "algorithm for doing backward "
                                                 "data convolution";
        *algorithm_desc = dnn::AlgorithmDesc(preference.bwd_data_algo, false);
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_FILTER: {
        auto status = wrap::miopenFindConvolutionBackwardWeightsAlgorithm(
            miopen.handle(),
            /*diffDesc=*/output_nd.handle(), output_data.opaque(),
            /*srcDesc=*/input_nd.handle(), input_data.opaque(),
            /*convDesc=*/conv.handle(),
            /*gradDesc=*/filter.handle(), filter_data.opaque(),
            /*requestAlgoCount=*/1, /*returnedAlgoCount=*/&returnedAlgoCount,
            /*preference=*/&preference,
            /*WorkSpace=*/scratch_memory_temp.opaque(),
            /*WorkSpaceSize=*/scratch_memory_temp.size(),
            /*exhaustiveSearch=*/false);
        CHECK_EQ(status, miopenStatusSuccess) << "Unable to find a suitable "
                                                 "algorithm for doing backward "
                                                 "filter convolution";
        *algorithm_desc =
            dnn::AlgorithmDesc(preference.bwd_weights_algo, false);
        break;
      }
      default:
        return port::InternalError(absl::StrCat("Unexpected convolution kind ",
                                                static_cast<int>(kind)));
    }

    // Restore default allocator, note mac is stack temp
    wrap::miopenSetAllocator(miopen.handle(), nullptr, nullptr, nullptr);

    scratch_memory_size = preference.memory;
  } else {
    // An algorithm has been specified.
    *algorithm_desc = *algo_desc;
    // commenting this line out for the upstream repo, since
    // AlgorithmConfig::scratch_size_ has been removed in the upstream repo but
    // is still used in the ROCM develop-upstream repo
    //
    // scratch_memory_size = *(algorithm_config.scratch_size());
    //
  }

  // allocate scratch memory
  if (scratch_memory_size != 0) {
    if (scratch_allocator == nullptr) {
      return port::InternalError(
          absl::StrCat("An allocator must be specified when scratch memory is "
                       "needed"));
    }
    auto allocated =
        scratch_allocator->AllocateBytes(stream, scratch_memory_size);
    if (!allocated.ok()) {
      return port::InternalError(absl::StrCat(
          "Failed to allocate scratch memory of size: ", scratch_memory_size));
    }
    if (allocated.ok()) {
      *scratch_memory = allocated.ValueOrDie();
    }
  }

  return port::Status::OK();
}

// NOTE(keveman): Temporary data layout transformation until MIOpen supports
// kBatchYXDepth for backward pass. This function allocates temporary memory,
// lays out the source data into the temporary but in the kBatchDepthXY
// layout, and returns the temporary memory. The caller is responsible for
// deallocating the temporary. Since the allocation is done using Stream's
// AllocateTemporaryMemory, a later BlockHostUntilDone could be used for
// deallocation.
//
// transform_scratch is populated with a legitimate temporary allocation iff
// the original output data needs to be transformed.
static DeviceMemoryBase MaybeTransformLayout(
    Stream* stream, miopenHandle_t handle_,
    int miopen_type,  // Actually miopenDataType_t.
    BatchDescriptor* output_descriptor, DeviceMemoryBase backward_output_data,
    std::unique_ptr<TemporaryDeviceMemory<uint8>>* transform_scratch) {
  if (output_descriptor->layout() == dnn::DataLayout::kBatchDepthYX) {
    return backward_output_data;
  }
  CHECK(output_descriptor->layout() == dnn::DataLayout::kBatchYXDepth);
  *transform_scratch =
      stream->AllocateTemporaryArray<uint8>(backward_output_data.size())
          .ConsumeValueOrDie();
  BatchDescriptor transformed_output_descriptor;
  transformed_output_descriptor.CloneFrom(*output_descriptor);
  transformed_output_descriptor.set_layout(dnn::DataLayout::kBatchDepthYX);
  ScopedTensorDescriptor orig_out_back_nd{
      *output_descriptor, static_cast<miopenDataType_t>(miopen_type)};
  ScopedTensorDescriptor transformed_out_back_nd{
      transformed_output_descriptor,
      static_cast<miopenDataType_t>(miopen_type)};

  float alpha1 = 1.0f;
  float alpha2 = 0.0f;
  float beta = 0.0f;
  auto status = wrap::miopenOpTensor(
      handle_, miopenTensorOpAdd, &alpha1, orig_out_back_nd.handle(),
      backward_output_data.opaque(), &alpha2, orig_out_back_nd.handle(),
      backward_output_data.opaque(), &beta, transformed_out_back_nd.handle(),
      (*transform_scratch)->mutable_device_memory()->opaque());

  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "Failed to transform the data layout.";
  }
  output_descriptor->set_layout(dnn::DataLayout::kBatchDepthYX);
  return (*transform_scratch)->device_memory();
}

port::Status MIOpenSupport::DoConvolve(
    dnn::ConvolutionKind kind, dnn::DataType element_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::AlgorithmDesc algorithm_desc, DeviceMemory<uint8> scratch_memory,
    dnn::ProfileResult* output_profile_result) {
  auto miopen = miopen_->GetHandle(parent_, stream);
  ScopedTensorDescriptor input_nd{input_descriptor,
                                  ToMIOpenDataType(element_type)};
  ScopedTensorDescriptor output_nd{output_descriptor,
                                   ToMIOpenDataType(element_type)};
  ScopedFilterDescriptor filter{filter_descriptor, input_descriptor,
                                ToMIOpenDataType(element_type)};
  ScopedConvolutionDescriptor conv{convolution_descriptor,
                                   ToMIOpenDataType(element_type)};

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  const bool is_profiling = output_profile_result != nullptr;

  std::unique_ptr<GpuTimer> timer;
  if (is_profiling) {
    timer.reset(new GpuTimer(parent_));
    if (!timer->Init()) {
      return port::Status(port::error::INTERNAL, "Failed to init timer");
    }
    // The start and stop of the timer should be as close to the MIOpen call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Start(AsGpuStream(stream))) {
      timer->Destroy();
      return port::Status(port::error::INTERNAL, "Failed to start timer");
    }
  }

  miopenStatus_t status = miopenStatusSuccess;
  switch (kind) {
    case dnn::ConvolutionKind::FORWARD: {
      status = wrap::miopenConvolutionForward(
          miopen.handle(),
          /*alpha=*/&alpha, /*srcDesc=*/input_nd.handle(),
          /*srcData=*/input_data.opaque(), /*filterDesc=*/filter.handle(),
          /*filterData=*/filter_data.opaque(), /*convDesc=*/conv.handle(),
          /*algo=*/
          static_cast<miopenConvFwdAlgorithm_t>(algorithm_desc.algo_id()),
          /*beta=*/&beta, /*destDesc=*/output_nd.handle(),
          /*destData=*/output_data.opaque(),
          /*workSpace=*/scratch_memory.opaque(),
          /*workSpaceSizeInBytes=*/scratch_memory.size());
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_DATA: {
      // TBD: remove once MIOpen supports kBatchYXDepth for backward pass.
      BatchDescriptor output_back_descriptor;
      output_back_descriptor.CloneFrom(output_descriptor);
      std::unique_ptr<TemporaryDeviceMemory<uint8>> transform_scratch;
      output_data = MaybeTransformLayout(
          stream, miopen.handle(), ToMIOpenDataType(element_type),
          &output_back_descriptor, output_data, &transform_scratch);

      status = wrap::miopenConvolutionBackwardData(
          miopen.handle(),
          /*alpha=*/&alpha,
          /*diffDesc=*/output_nd.handle(),
          /*diffData=*/output_data.opaque(),
          /*filterDesc=*/filter.handle(),
          /*filterData=*/filter_data.opaque(),
          /*convDesc=*/conv.handle(),
          /*algo=*/
          static_cast<miopenConvBwdDataAlgorithm_t>(algorithm_desc.algo_id()),
          /*beta=*/&beta,
          /*gradDesc=*/input_nd.handle(),
          /*gradData=*/input_data.opaque(),
          /*workSpace=*/scratch_memory.opaque(),
          /*workSpaceSizeInBytes=*/scratch_memory.size());
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_FILTER: {
      // TBD: remove once MIOpen supports kBatchYXDepth for backward pass.
      BatchDescriptor output_back_descriptor;
      output_back_descriptor.CloneFrom(output_descriptor);
      std::unique_ptr<TemporaryDeviceMemory<uint8>> transform_scratch;
      output_data = MaybeTransformLayout(
          stream, miopen.handle(), ToMIOpenDataType(element_type),
          &output_back_descriptor, output_data, &transform_scratch);

      status = wrap::miopenConvolutionBackwardWeights(
          miopen.handle(),
          /*alpha=*/&alpha,
          /*diffDesc=*/output_nd.handle(),
          /*diffData=*/output_data.opaque(),
          /*srcDesc=*/input_nd.handle(),
          /*srcData=*/input_data.opaque(),
          /*convDesc=*/conv.handle(),
          /*algo=*/
          static_cast<miopenConvBwdWeightsAlgorithm_t>(
              algorithm_desc.algo_id()),
          /*beta=*/&beta,
          /*gradDesc=*/filter.handle(),
          /*gradData=*/filter_data.opaque(),
          /*workSpace=*/scratch_memory.opaque(),
          /*workSpaceSizeInBytes=*/scratch_memory.size());
      break;
    }
    default:
      return port::InternalError(
          absl::StrCat("Unexpected convolution kind ", static_cast<int>(kind)));
  }

  if (is_profiling) {
    if (!timer->Stop(AsGpuStream(stream))) {
      timer->Destroy();
      return port::Status(port::error::INTERNAL, "Failed to stop timer");
    }
    if (status == miopenStatusSuccess) {
      dnn::AlgorithmDesc algotype(algorithm_desc.algo_id(), false);
      output_profile_result->set_algorithm(algotype);
      output_profile_result->set_elapsed_time_in_ms(
          timer->GetElapsedMilliseconds());
      output_profile_result->set_scratch_size(scratch_memory.size());
    }
    timer->Destroy();
  }

  if (status != miopenStatusSuccess) {
    return port::InternalError(absl::StrCat(
        "Failed to euqueue convolution on stream: ", ToString(status)));
  }

  return port::Status::OK();
}

bool MIOpenSupport::GetConvolveAlgorithms(
    // ROCM TODO: refactor cc_major / cc_minor
    bool with_winograd_nonfused, int cc_major, int cc_minor,
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  out_algorithms->assign({
      // clang-format off
      dnn::AlgorithmDesc(miopenConvolutionFwdAlgoGEMM, false),
      dnn::AlgorithmDesc(miopenConvolutionFwdAlgoDirect, false),
      dnn::AlgorithmDesc(miopenConvolutionFwdAlgoFFT, false),
      dnn::AlgorithmDesc(miopenConvolutionFwdAlgoWinograd, false),
      // clang-format on
  });
  return true;
}

bool MIOpenSupport::GetRnnAlgorithms(
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  // ROCM TODO: implement this with proper MIOpen API
  return true;
}

bool MIOpenSupport::GetConvolveBackwardDataAlgorithms(
    // ROCM TODO: refactor cc_major / cc_minor
    bool with_winograd_nonfused, int cc_major, int cc_minor,
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  out_algorithms->assign({
      // clang-format off
      dnn::AlgorithmDesc(miopenConvolutionBwdDataAlgoGEMM, false),
      dnn::AlgorithmDesc(miopenConvolutionBwdDataAlgoDirect, false),
      dnn::AlgorithmDesc(miopenConvolutionBwdDataAlgoFFT, false),
      dnn::AlgorithmDesc(miopenConvolutionBwdDataAlgoWinograd, false),
      // clang-format on
  });
  return true;
}

bool MIOpenSupport::GetConvolveBackwardFilterAlgorithms(
    // ROCM TODO: refactor cc_major / cc_minor
    bool with_winograd_nonfused, int cc_major, int cc_minor,
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  out_algorithms->assign({
      // clang-format off
      dnn::AlgorithmDesc(miopenConvolutionBwdWeightsAlgoGEMM, false),
      dnn::AlgorithmDesc(miopenConvolutionBwdWeightsAlgoDirect, false),
      // clang-format on
  });
  return true;
}

bool MIOpenSupport::DoBatchNormalizationForward(
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

bool MIOpenSupport::DoBatchNormalizationForward(
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

template <class T, class U>
bool MIOpenSupport::DoBatchNormalizationForwardImpl(
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
  auto miopen = miopen_->GetHandle(parent_, stream);

  ScopedTensorDescriptor x_descriptor{x_desc,
                                      ToMIOpenDataType(input_data_type)};
  ScopedTensorDescriptor scale_offset_descriptor{
      scale_offset_desc, ToMIOpenDataType(scale_data_type)};
  miopenBatchNormMode_t mode = miopenBNSpatial;
  float one = 1.0;
  float zero = 0.0;

  auto status = miopenStatusInvalidValue;
  if (is_training) {
    stream->ThenMemZero(batch_mean, batch_mean->size());
    stream->ThenMemZero(batch_var, batch_var->size());
    status = wrap::miopenBatchNormalizationForwardTraining(
        miopen.handle(), mode, &one, &zero, x_descriptor.handle(), x.opaque(),
        x_descriptor.handle(), y->opaque(), scale_offset_descriptor.handle(),
        const_cast<void*>(scale.opaque()), const_cast<void*>(offset.opaque()),
        1.0, batch_mean->opaque(), batch_var->opaque(), epsilon,
        saved_mean->opaque(), saved_inv_var->opaque());
  } else {
    const void* maybe_inv_var = estimated_variance.opaque();
    status = wrap::miopenBatchNormalizationForwardInference(
        miopen.handle(), mode, &one, &zero, x_descriptor.handle(), x.opaque(),
        x_descriptor.handle(), y->opaque(), scale_offset_descriptor.handle(),
        const_cast<void*>(scale.opaque()), const_cast<void*>(offset.opaque()),
        const_cast<void*>(estimated_mean.opaque()),
        const_cast<void*>(maybe_inv_var), epsilon);
  }
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue forward batch normalization on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool MIOpenSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<Eigen::half>& y_backprop,
    const DeviceMemory<Eigen::half>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& mean, const DeviceMemory<float>& inv_var,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<Eigen::half>* x_backprop, DeviceMemory<float>* scale_backprop,
    DeviceMemory<float>* offset_backprop) {
  return DoBatchNormalizationBackwardImpl<Eigen::half, float>(
      stream, miopenHalf, miopenFloat, y_backprop, x, scale, mean, inv_var,
      x_desc, scale_offset_desc, epsilon, x_backprop, scale_backprop,
      offset_backprop);
}

bool MIOpenSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<float>& y_backprop,
    const DeviceMemory<float>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& mean, const DeviceMemory<float>& variance,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<float>* x_backprop, DeviceMemory<float>* scale_backprop,
    DeviceMemory<float>* offset_backprop) {
  return DoBatchNormalizationBackwardImpl<float, float>(
      stream, miopenFloat, miopenFloat, y_backprop, x, scale, mean, variance,
      x_desc, scale_offset_desc, epsilon, x_backprop, scale_backprop,
      offset_backprop);
}

template <class T, class U>
bool MIOpenSupport::DoBatchNormalizationBackwardImpl(
    Stream* stream, int miopen_input_type, int miopen_scale_type,
    const DeviceMemory<T>& y_backprop, const DeviceMemory<T>& x,
    const DeviceMemory<U>& scale, const DeviceMemory<U>& mean,
    const DeviceMemory<U>& variance, const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<T>* x_backprop, DeviceMemory<U>* scale_backprop,
    DeviceMemory<U>* offset_backprop) {
  auto miopen = miopen_->GetHandle(parent_, stream);
  ScopedTensorDescriptor x_descriptor{
      x_desc, static_cast<miopenDataType_t>(miopen_input_type)};
  ScopedTensorDescriptor scale_offset_descriptor{
      scale_offset_desc, static_cast<miopenDataType_t>(miopen_scale_type)};
  miopenBatchNormMode_t mode = miopenBNSpatial;
  float one = 1.0;
  float zero = 0.0;

  auto status = wrap::miopenBatchNormalizationBackward(
      miopen.handle(), mode, &one, &zero, &one, &zero, x_descriptor.handle(),
      x.opaque(), x_descriptor.handle(), y_backprop.opaque(),
      x_descriptor.handle(), x_backprop->opaque(),
      scale_offset_descriptor.handle(), scale.opaque(),
      scale_backprop->opaque(), offset_backprop->opaque(), epsilon,
      mean.opaque(), variance.opaque());
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue backward batch normalization on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool MIOpenSupport::DoFusedConvolve(
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
  LOG(ERROR) << "fused convolve not implemented yet";
  return false;
}

bool MIOpenSupport::DoFusedConvolve(
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
  LOG(ERROR) << "fused convolve not implemented yet";
  return false;
}

bool MIOpenSupport::DoFusedConvolve(
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
  LOG(ERROR) << "fused convolve not implemented yet";
  return false;
}

bool MIOpenSupport::DoFusedConvolve(
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
  LOG(ERROR) << "fused convolve not implemented yet";
  return false;
}

bool MIOpenSupport::DoTransformTensor(Stream* stream,
                                      const dnn::BatchDescriptor& input_desc,
                                      dnn::DataType input_type,
                                      const DeviceMemoryBase& input_data,
                                      const dnn::BatchDescriptor& output_desc,
                                      dnn::DataType output_type, float scale,
                                      DeviceMemoryBase* output_data) {
  // ROCM TODO implement this operation
  LOG(ERROR) << "transform tensor not implemented yet";
  return false;
}

template <class T>
bool MIOpenSupport::DoConvolveBackwardBiasImpl(
    Stream* stream, int miopen_type,  // Actually miopenDataType_t.
    const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<T>& input_data,
    const dnn::BatchDescriptor& bias_descriptor,
    DeviceMemory<T>* backward_bias_data) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  ScopedTensorDescriptor input_nd{input_descriptor,
                                  static_cast<miopenDataType_t>(miopen_type)};
  ScopedTensorDescriptor bias_nd{bias_descriptor,
                                 static_cast<miopenDataType_t>(miopen_type)};

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  auto status = wrap::miopenConvolutionBackwardBias(
      miopen.handle(), &alpha, input_nd.handle(), input_data.opaque(), &beta,
      bias_nd.handle(), backward_bias_data->opaque());
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "failed to enqueue backward convolution on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool MIOpenSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<double>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<double>* backward_bias_data) {
  LOG(ERROR) << "miopen does not support double bwd bias yet";
  return false;
}

bool MIOpenSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<float>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<float>* backward_bias_data) {
  return DoConvolveBackwardBiasImpl(stream, miopenFloat, input_descriptor,
                                    input_data, bias_descriptor,
                                    backward_bias_data);
}

bool MIOpenSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<Eigen::half>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<Eigen::half>* backward_bias_data) {
  return DoConvolveBackwardBiasImpl(stream, miopenHalf, input_descriptor,
                                    input_data, bias_descriptor,
                                    backward_bias_data);
}

bool MIOpenSupport::DoMatMul(Stream* stream,
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

bool MIOpenSupport::DoBiasAdd(Stream* stream,
                              const DeviceMemory<float>& input_data,
                              const DeviceMemory<float>& biases,
                              const dnn::BatchDescriptor& dimensions,
                              DeviceMemory<float>* output_data) {
  ScopedTensorDescriptor input_descriptor{dimensions, miopenFloat};

  BatchDescriptor bias_dimensions;
  bias_dimensions.set_count(1)
      .set_feature_map_count(dimensions.feature_map_count())
      .set_height(1)
      .set_width(1)
      .set_layout(dnn::DataLayout::kBatchYXDepth);
  ScopedTensorDescriptor bias_descriptor{bias_dimensions, miopenFloat};

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

  auto miopen = miopen_->GetHandle(parent_, stream);

  const float alpha1 = 1.0f;
  const float alpha2 = 0.0f;
  const float beta = 1.0f;

  auto status = wrap::miopenOpTensor(
      miopen.handle(), miopenTensorOpAdd, &alpha1, bias_descriptor.handle(),
      biases.opaque(), &alpha2, bias_descriptor.handle(), biases.opaque(),
      &beta, input_descriptor.handle(), output_data->opaque());

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "stream " << stream << " could not enqueue bias addition.";
    return false;
  }

  return true;
}

bool MIOpenSupport::DoActivate(Stream* stream,
                               dnn::ActivationMode activation_mode,
                               const dnn::BatchDescriptor& dimensions,
                               const DeviceMemory<float>& input_data,
                               DeviceMemory<float>* output_data,
                               uint64 options) {
  LOG(ERROR) << "miopen does not support activation yet";
  return false;
}

bool MIOpenSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<double>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<double>* output_data, ScratchAllocator* workspace_allocator) {
  LOG(ERROR) << "miopen does not support pooling for dobule type yet";
  return false;
}

bool MIOpenSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<float>* output_data, ScratchAllocator* workspace_allocator) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc{input_dimensions, miopenFloat};
  ScopedTensorDescriptor dest_desc{output_dimensions, miopenFloat};
  ScopedPoolingDescriptor pooling_desc{pooling_dimensions};

  auto status = wrap::miopenPoolingForward(
      miopen.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
      input_data.opaque(), &beta, dest_desc.handle(), output_data->opaque(),
      false, nullptr, 0);
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue forward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool MIOpenSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<Eigen::half>* output_data,
    ScratchAllocator* workspace_allocator) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc{input_dimensions, miopenHalf};
  ScopedTensorDescriptor dest_desc{output_dimensions, miopenHalf};
  ScopedPoolingDescriptor pooling_desc{pooling_dimensions};

  auto status = wrap::miopenPoolingForward(
      miopen.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
      input_data.opaque(), &beta, dest_desc.handle(), output_data->opaque(),
      false, nullptr, 0);
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue forward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool MIOpenSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<double>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<double>& output_data,
    const DeviceMemory<double>& input_diff_data,
    DeviceMemory<double>* output_diff_data,
    ScratchAllocator* workspace_allocator) {
  LOG(ERROR) << "miopen does not support backward pooling on double type yet";
  return false;
}

bool MIOpenSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<float>& output_data,
    const DeviceMemory<float>& input_diff_data,
    DeviceMemory<float>* output_diff_data,
    ScratchAllocator* workspace_allocator) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc{input_dimensions, miopenFloat};
  ScopedTensorDescriptor dest_desc{output_dimensions, miopenFloat};
  ScopedPoolingDescriptor pooling_desc{pooling_dimensions};

  DeviceMemory<uint8> workspace;
  size_t workspace_size_in_bytes = 0;
  auto status = wrap::miopenPoolingGetWorkSpaceSize(dest_desc.handle(),
                                                    &workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR)
        << "failed to obtain workspace size for backward pooling on stream: "
        << ToString(status);
    return false;
  }

  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    assert(workspace_allocator);
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  }

  DeviceMemory<uint8> dest2;  // duplicated dest from forward:
  int dest2_size = 0;

  // miopen requires the strides and dims to be ordered as BDYX.
  std::vector<int64> dims64 =
      output_dimensions.full_dims(dnn::DataLayout::kBatchDepthYX);

  // miopen does not use strides and must have 4D tensor.
  std::vector<int> dims(4);

  std::transform(dims64.cbegin(), dims64.cend(), dims.begin(),
                 &CheckedNarrowing<int64, int>);

  dest2_size = dims[0] * dims[1] * dims[2] * dims[3] * sizeof(float);

  if (dest2_size > 0) {
    assert(workspace_allocator);
    auto allocated = workspace_allocator->AllocateBytes(stream, dest2_size);
    if (!allocated.ok() || (dest2 = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  } else {
    LOG(ERROR) << "Failed to calcuate tensor size to chain forward and "
                  "backward pooling";
  }

  status = wrap::miopenPoolingForward(
      miopen.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
      input_data.opaque(), &beta, dest_desc.handle(), dest2.opaque(), true,
      workspace.opaque(), workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR)
        << "failed to enqueue forward pooling (before backward) on stream: "
        << ToString(status);
    return false;
  }

  status = wrap::miopenPoolingBackward(
      miopen.handle(), pooling_desc.handle(), &alpha, dest_desc.handle(),
      dest2.opaque(), dest_desc.handle(), input_diff_data.opaque(),
      src_desc.handle(), input_data.opaque(), &beta, src_desc.handle(),
      output_diff_data->opaque(), workspace.opaque());

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue backward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool MIOpenSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<Eigen::half>& output_data,
    const DeviceMemory<Eigen::half>& input_diff_data,
    DeviceMemory<Eigen::half>* output_diff_data,
    ScratchAllocator* workspace_allocator) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc{input_dimensions, miopenHalf};
  ScopedTensorDescriptor dest_desc{output_dimensions, miopenHalf};
  ScopedPoolingDescriptor pooling_desc{pooling_dimensions};

  DeviceMemory<uint8> workspace;
  size_t workspace_size_in_bytes = 0;
  auto status = wrap::miopenPoolingGetWorkSpaceSize(dest_desc.handle(),
                                                    &workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR)
        << "failed to obtain workspace size for backward pooling on stream: "
        << ToString(status);
    return false;
  }

  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    assert(workspace_allocator);
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  }

  DeviceMemory<uint8> dest2;  // duplicated dest from forward:
  int dest2_size = 0;

  // miopen requires the strides and dims to be ordered as BDYX.
  std::vector<int64> dims64 =
      output_dimensions.full_dims(dnn::DataLayout::kBatchDepthYX);

  // miopen does not use strides and must have 4D tensor.
  std::vector<int> dims(4);

  std::transform(dims64.cbegin(), dims64.cend(), dims.begin(),
                 &CheckedNarrowing<int64, int>);

  dest2_size = dims[0] * dims[1] * dims[2] * dims[3] * sizeof(float);

  if (dest2_size > 0) {
    assert(workspace_allocator);
    auto allocated = workspace_allocator->AllocateBytes(stream, dest2_size);
    if (!allocated.ok() || (dest2 = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  } else {
    LOG(ERROR) << "Failed to calcuate tensor size to chain forward and "
                  "backward pooling";
  }

  status = wrap::miopenPoolingForward(
      miopen.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
      input_data.opaque(), &beta, dest_desc.handle(), dest2.opaque(), true,
      workspace.opaque(), workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR)
        << "failed to enqueue forward pooling (before backward) on stream: "
        << ToString(status);
    return false;
  }

  status = wrap::miopenPoolingBackward(
      miopen.handle(), pooling_desc.handle(), &alpha, dest_desc.handle(),
      dest2.opaque(), dest_desc.handle(), input_diff_data.opaque(),
      src_desc.handle(), input_data.opaque(), &beta, src_desc.handle(),
      output_diff_data->opaque(), workspace.opaque());

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue backward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool MIOpenSupport::DoNormalizeWithDimensions(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const dnn::BatchDescriptor& dimensions,
    const DeviceMemory<float>& input_data, DeviceMemory<float>* output_data) {
  // Check for unsupported modes.
  if (normalize_descriptor.wrap_around()) {
    LOG(ERROR) << "MIOpen LRN does not support wrap-around mode";
    return false;
  }
  if (normalize_descriptor.segment_size()) {
    LOG(ERROR) << "MIOpen LRN does not support segmentation";
    return false;
  }

  auto miopen = miopen_->GetHandle(parent_, stream);

  // Launch the normalization.
  ScopedTensorDescriptor dims{dimensions, miopenFloat};
  ScopedNormalizeDescriptor normalize{normalize_descriptor};

  // Alpha is the scaling factor for input.
  float alpha = 1.0f;
  // Beta is the scaling factor for output.
  float beta = 0.0f;

  auto status = wrap::miopenLRNForward(
      miopen.handle(), normalize.handle(), &alpha, dims.handle(),
      input_data.opaque(), &beta, dims.handle(), output_data->opaque(), false,
      nullptr);
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to run miopenLRNForward";
    return false;
  }
  return true;
}

bool MIOpenSupport::DoNormalizeBackwardWithDimensions(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const dnn::BatchDescriptor& dimensions, const DeviceMemory<float>& raw_data,
    const DeviceMemory<float>& normalized_data,
    const DeviceMemory<float>& normalized_variable_gradient,
    DeviceMemory<float>* raw_variable_gradient,
    ScratchAllocator* workspace_allocator) {
  // Check for unsupported modes.
  if (normalize_descriptor.wrap_around()) {
    LOG(ERROR) << "MIOpen LRN does not support wrap-around mode";
    return false;
  }
  if (normalize_descriptor.segment_size()) {
    LOG(ERROR) << "MIOpen LRN does not support segmentation";
    return false;
  }

  auto miopen = miopen_->GetHandle(parent_, stream);

  ScopedTensorDescriptor dims{dimensions, miopenFloat};
  ScopedNormalizeDescriptor normalize{normalize_descriptor};

  float alpha = 1.0f;
  float beta = 0.0f;

  DeviceMemory<uint8> workspace;
  size_t workspace_size_in_bytes = 0;
  auto status =
      wrap::miopenLRNGetWorkSpaceSize(dims.handle(), &workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to obtain workspace size for miopenLRNBackward";
    return false;
  }

  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    assert(workspace_allocator);
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  }

  DeviceMemory<uint8> dest2;  // duplicated dest from forward:
  int dest2_size = 0;

  // miopen requires the strides and dims to be ordered as BDYX.
  std::vector<int64> dims64 =
      dimensions.full_dims(dnn::DataLayout::kBatchDepthYX);

  // miopen does not use strides and must have 4D tensor.
  std::vector<int> dimsint(4);

  std::transform(dims64.cbegin(), dims64.cend(), dimsint.begin(),
                 &CheckedNarrowing<int64, int>);

  dest2_size =
      dimsint[0] * dimsint[1] * dimsint[2] * dimsint[3] * sizeof(float);

  if (dest2_size > 0) {
    assert(workspace_allocator);
    auto allocated = workspace_allocator->AllocateBytes(stream, dest2_size);
    if (!allocated.ok() || (dest2 = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR)
          << "Failed to allocate tensor to chain forward and backward LRN";
      return false;
    }
  } else {
    LOG(ERROR)
        << "Failed to calcuate tensor size to chain forward and backward LRN";
  }

  status = wrap::miopenLRNForward(miopen.handle(), normalize.handle(), &alpha,
                                  dims.handle(), raw_data.opaque(), &beta,
                                  dims.handle(), dest2.opaque(), true,
                                  workspace.opaque());

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to run miopenLRNForward";
    return false;
  }

  status = wrap::miopenLRNBackward(
      miopen.handle(), normalize.handle(), &alpha, dims.handle(),
      normalized_data.opaque(), dims.handle(),
      normalized_variable_gradient.opaque(), dims.handle(), raw_data.opaque(),
      &beta, dims.handle(), raw_variable_gradient->opaque(),
      workspace.opaque());

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to run miopenLRNBackward";
    return false;
  }
  return true;
}

bool MIOpenSupport::DoDepthConcatenate(
    Stream* stream, port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float>*> input_data,
    DeviceMemory<float>* output_data) {
  CHECK_EQ(input_dimensions.size(), input_data.size());

  for (const auto& dimensions : input_dimensions) {
    if (dimensions.layout() != dnn::DataLayout::kBatchDepthYX) {
      LOG(ERROR) << "MIOpenSupport::DoDepthConcatenate currently only "
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

bool MIOpenSupport::DoElementwiseOperate(
    Stream* stream, dnn::ElementwiseOperation operation,
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float>*> input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
  return false;
}

bool MIOpenSupport::DoXYPad(Stream* stream,
                            const dnn::BatchDescriptor& dimensions,
                            const DeviceMemory<float>& input_data,
                            int64 left_pad, int64 right_pad, int64 top_pad,
                            int64 bottom_pad,
                            DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
  return false;
}

bool MIOpenSupport::DoXYSlice(Stream* stream,
                              const dnn::BatchDescriptor& dimensions,
                              const DeviceMemory<float>& input_data,
                              int64 left_trim, int64 right_trim, int64 top_trim,
                              int64 bottom_trim,
                              DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
  return false;
}

bool MIOpenSupport::DoMemcpyD2HQuantized(
    Stream* stream, const DeviceMemory<float>& gpu_unquantized_src,
    dnn::QuantizedActivationMode mode, void* host_dst, int64 size) {
  LOG(ERROR) << "quantized memcpy not supported by MIOpen";
  return false;
}

bool MIOpenSupport::DoMemcpyH2DQuantized(
    Stream* stream, const void* host_src, int64 size,
    dnn::QuantizedActivationMode mode,
    DeviceMemory<float>* gpu_unquantized_dst) {
  LOG(ERROR) << "quantized memcpy not supported by MIOpen";
  return false;
}

bool MIOpenSupport::DeriveOutputBatchDescriptor(
    const BatchDescriptor& batch_descriptor,
    const FilterDescriptor& filter_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::BatchDescriptor* output_batch_descriptor) {
  ScopedTensorDescriptor input_nd{batch_descriptor, miopenFloat};
  ScopedFilterDescriptor filter{filter_descriptor, batch_descriptor,
                                miopenFloat};
  ScopedConvolutionDescriptor conv{convolution_descriptor, miopenFloat};

  int dn = batch_descriptor.ndims() + 2;
  std::vector<int> dims(dn);  // in BDYX
  auto status = wrap::miopenGetConvolutionForwardOutputDim(
      conv.handle(), input_nd.handle(), filter.handle(), &dims[0], &dims[1],
      &dims[2], &dims[3]);
  if (status != miopenStatusSuccess) {
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

template <typename T>
bool MIOpenSupport::DoFusedConvolutionBiasActivationImpl(
    Stream* stream,
    int miopen_type,  // Actually miopenDataType_t.
    const dnn::BatchDescriptor& conv_input_descriptor,
    const DeviceMemory<T>& conv_input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<T>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const DeviceMemory<T>& bias_data, dnn::ActivationMode activation_mode,
    const dnn::BatchDescriptor& output_descriptor, DeviceMemory<T>* output_data,
    dnn::ProfileResult* output_profile_result) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  ScopedTensorDescriptor conv_input_nd{
      conv_input_descriptor, static_cast<miopenDataType_t>(miopen_type)};

  ScopedTensorDescriptor bias_nd{bias_descriptor,
                                 static_cast<miopenDataType_t>(miopen_type)};

  ScopedTensorDescriptor output_nd{output_descriptor,
                                   static_cast<miopenDataType_t>(miopen_type)};

  ScopedConvolutionDescriptor conv{convolution_descriptor,
                                   static_cast<miopenDataType_t>(miopen_type)};

  ScopedFilterDescriptor filter{filter_descriptor, conv_input_descriptor,
                                static_cast<miopenDataType_t>(miopen_type)};

  ScopedActivationDescriptor activation_desc{activation_mode};

  ScopedFusionPlanConvolutionBiasActivation fusion_plan{
      miopen.handle(), conv_input_nd.handle(), filter.handle(),
      conv.handle(),   bias_nd.handle(),       activation_desc};

  bool retval = false;

  if (fusion_plan.CompilationSucceeded()) {
    const bool is_profiling = output_profile_result != nullptr;

    std::unique_ptr<GpuTimer> timer;
    if (is_profiling) {
      timer.reset(new GpuTimer(parent_));
      timer->Init();
      timer->Start(AsGpuStream(stream));
    }

    miopenStatus_t status = miopenStatusSuccess;

    if (status == miopenStatusSuccess) {
      fusion_plan.SetConvolutionArgs(filter_data.opaque());
    }

    if (status == miopenStatusSuccess) {
      status = fusion_plan.SetBiasArgs(bias_data.opaque());
    }

    if (status == miopenStatusSuccess) {
      status = fusion_plan.SetActivationForwardArgs(activation_desc);
    }

    if (status == miopenStatusSuccess) {
      status =
          fusion_plan.Execute(conv_input_nd.handle(), conv_input_data.opaque(),
                              output_nd.handle(), output_data->opaque());
    }

    if (is_profiling) {
      timer->Stop(AsGpuStream(stream));
      if (status == miopenStatusSuccess) {
        output_profile_result->set_elapsed_time_in_ms(
            timer->GetElapsedMilliseconds());
      }
      timer->Destroy();
    }

    if (status != miopenStatusSuccess) {
      // Silently return when we are profiling.
      if (!is_profiling) {
        LOG(FATAL) << "failed to enqueue fused-convolution on stream: "
                   << ToString(status);
      }
    }

    retval = true;
  }

  return retval;
}

bool MIOpenSupport::DoFusedConvolutionBiasActivation(
    Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
    const DeviceMemory<float>& conv_input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    const DeviceMemory<float>& filter_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const DeviceMemory<float>& bias_data, dnn::ActivationMode activation_mode,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceMemory<float>* output_data,
    dnn::ProfileResult* output_profile_result) {
  return DoFusedConvolutionBiasActivationImpl<float>(
      stream, miopenFloat, conv_input_descriptor, conv_input_data,
      filter_descriptor, filter_data, convolution_descriptor, bias_descriptor,
      bias_data, activation_mode, output_descriptor, output_data,
      output_profile_result);
}

template <typename T, typename U>
bool MIOpenSupport::DoFusedBatchNormActivationInferenceImpl(
    Stream* stream,
    int miopen_type,  // Actually miopenDataType_t.
    const dnn::BatchDescriptor& x_descriptor, const DeviceMemory<T>& x_data,
    const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
    const DeviceMemory<U>& scale_data, const DeviceMemory<U>& offset_data,
    const DeviceMemory<U>& mean_data, const DeviceMemory<U>& variance_data,
    double epsilon, dnn::ActivationMode activation_mode,
    DeviceMemory<T>* y_data, dnn::ProfileResult* output_profile_result) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  ScopedTensorDescriptor x_nd{x_descriptor,
                              static_cast<miopenDataType_t>(miopen_type)};

  ScopedTensorDescriptor scale_offset_mean_variance_nd{
      scale_offset_mean_variance_descriptor,
      static_cast<miopenDataType_t>(miopen_type)};

  ScopedActivationDescriptor activation_desc{activation_mode};

  ScopedFusionPlanBatchNormActivationInference fusion_plan{
      miopen.handle(), x_nd.handle(), scale_offset_mean_variance_nd.handle(),
      activation_desc};

  bool retval = false;

  if (fusion_plan.CompilationSucceeded()) {
    const bool is_profiling = output_profile_result != nullptr;

    std::unique_ptr<GpuTimer> timer;
    if (is_profiling) {
      timer.reset(new GpuTimer(parent_));
      timer->Init();
      timer->Start(AsGpuStream(stream));
    }

    miopenStatus_t status = miopenStatusSuccess;

    if (status == miopenStatusSuccess) {
      fusion_plan.SetBatchNormInferenceArgs(
          scale_data.opaque(), offset_data.opaque(), mean_data.opaque(),
          variance_data.opaque(), epsilon);
    }

    if (status == miopenStatusSuccess) {
      status = fusion_plan.SetActivationForwardArgs(activation_desc);
    }

    if (status == miopenStatusSuccess) {
      status = fusion_plan.Execute(x_nd.handle(), x_data.opaque(),
                                   x_nd.handle(), y_data->opaque());
    }

    if (is_profiling) {
      timer->Stop(AsGpuStream(stream));
      if (status == miopenStatusSuccess) {
        output_profile_result->set_elapsed_time_in_ms(
            timer->GetElapsedMilliseconds());
      }
      timer->Destroy();
    }

    if (status != miopenStatusSuccess) {
      // Silently return when we are profiling.
      if (!is_profiling) {
        LOG(FATAL) << "failed to enqueue fused-convolution on stream: "
                   << ToString(status);
      }
    }

    retval = true;
  }

  return retval;
}

bool MIOpenSupport::DoFusedBatchNormActivationInference(
    Stream* stream, const dnn::BatchDescriptor& x_descriptor,
    const DeviceMemory<float>& x_data,
    const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
    const DeviceMemory<float>& scale_data,
    const DeviceMemory<float>& offset_data,
    const DeviceMemory<float>& mean_data,
    const DeviceMemory<float>& variance_data, double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<float>* y_data,
    dnn::ProfileResult* output_profile_result) {
  return DoFusedBatchNormActivationInferenceImpl<float, float>(
      stream, miopenFloat, x_descriptor, x_data,
      scale_offset_mean_variance_descriptor, scale_data, offset_data, mean_data,
      variance_data, epsilon, activation_mode, y_data, output_profile_result);
}

bool MIOpenSupport::DoFusedBatchNormActivationInference(
    Stream* stream, const dnn::BatchDescriptor& x_descriptor,
    const DeviceMemory<Eigen::half>& x_data,
    const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
    const DeviceMemory<float>& scale_data,
    const DeviceMemory<float>& offset_data,
    const DeviceMemory<float>& mean_data,
    const DeviceMemory<float>& variance_data, double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half>* y_data,
    dnn::ProfileResult* output_profile_result) {
  return DoFusedBatchNormActivationInferenceImpl<Eigen::half, float>(
      stream, miopenHalf, x_descriptor, x_data,
      scale_offset_mean_variance_descriptor, scale_data, offset_data, mean_data,
      variance_data, epsilon, activation_mode, y_data, output_profile_result);
}

template <typename T, typename U>
bool MIOpenSupport::DoFusedBatchNormActivationForwardImpl(
    Stream* stream,
    int miopen_type,  // Actually miopenDataType_t.
    const dnn::BatchDescriptor& x_descriptor, const DeviceMemory<T>& x_data,
    const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
    const DeviceMemory<U>& scale_data, const DeviceMemory<U>& offset_data,
    double epsilon, dnn::ActivationMode activation_mode,
    DeviceMemory<T>* y_data, DeviceMemory<U>* batch_mean_data,
    DeviceMemory<U>* batch_var_data, DeviceMemory<U>* saved_mean_data,
    DeviceMemory<U>* saved_var_data,
    dnn::ProfileResult* output_profile_result) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  ScopedTensorDescriptor x_nd{x_descriptor,
                              static_cast<miopenDataType_t>(miopen_type)};

  ScopedTensorDescriptor scale_offset_mean_variance_nd{
      scale_offset_mean_variance_descriptor,
      static_cast<miopenDataType_t>(miopen_type)};

  ScopedActivationDescriptor activation_desc{activation_mode};

  ScopedFusionPlanBatchNormActivationForward fusion_plan{
      miopen.handle(), x_nd.handle(), scale_offset_mean_variance_nd.handle(),
      activation_desc};

  bool retval = false;

  if (fusion_plan.CompilationSucceeded()) {
    const bool is_profiling = output_profile_result != nullptr;

    std::unique_ptr<GpuTimer> timer;
    if (is_profiling) {
      timer.reset(new GpuTimer(parent_));
      timer->Init();
      timer->Start(AsGpuStream(stream));
    }

    miopenStatus_t status = miopenStatusSuccess;

    if (status == miopenStatusSuccess) {
      fusion_plan.SetBatchNormForwardArgs(
          scale_data.opaque(), offset_data.opaque(), batch_mean_data->opaque(),
          batch_var_data->opaque(), saved_mean_data->opaque(),
          saved_var_data->opaque(), epsilon);
    }

    if (status == miopenStatusSuccess) {
      status = fusion_plan.SetActivationForwardArgs(activation_desc);
    }

    if (status == miopenStatusSuccess) {
      status = fusion_plan.Execute(x_nd.handle(), x_data.opaque(),
                                   x_nd.handle(), y_data->opaque());
    }

    if (is_profiling) {
      timer->Stop(AsGpuStream(stream));
      if (status == miopenStatusSuccess) {
        output_profile_result->set_elapsed_time_in_ms(
            timer->GetElapsedMilliseconds());
      }
      timer->Destroy();
    }

    if (status != miopenStatusSuccess) {
      // Silently return when we are profiling.
      if (!is_profiling) {
        LOG(FATAL) << "failed to enqueue fused-convolution on stream: "
                   << ToString(status);
      }
    }

    retval = true;
  }

  return retval;
}

bool MIOpenSupport::DoFusedBatchNormActivationForward(
    Stream* stream, const dnn::BatchDescriptor& x_descriptor,
    const DeviceMemory<float>& x_data,
    const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
    const DeviceMemory<float>& scale_data,
    const DeviceMemory<float>& offset_data, double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<float>* y_data,
    DeviceMemory<float>* batch_mean_data, DeviceMemory<float>* batch_var_data,
    DeviceMemory<float>* saved_mean_data, DeviceMemory<float>* saved_var_data,
    dnn::ProfileResult* output_profile_result) {
  return DoFusedBatchNormActivationForwardImpl<float, float>(
      stream, miopenFloat, x_descriptor, x_data,
      scale_offset_mean_variance_descriptor, scale_data, offset_data, epsilon,
      activation_mode, y_data, batch_mean_data, batch_var_data, saved_mean_data,
      saved_var_data, output_profile_result);
}

bool MIOpenSupport::DoFusedBatchNormActivationForward(
    Stream* stream, const dnn::BatchDescriptor& x_descriptor,
    const DeviceMemory<Eigen::half>& x_data,
    const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
    const DeviceMemory<float>& scale_data,
    const DeviceMemory<float>& offset_data, double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half>* y_data,
    DeviceMemory<float>* batch_mean_data, DeviceMemory<float>* batch_var_data,
    DeviceMemory<float>* saved_mean_data, DeviceMemory<float>* saved_var_data,
    dnn::ProfileResult* output_profile_result) {
  return DoFusedBatchNormActivationForwardImpl<Eigen::half, float>(
      stream, miopenHalf, x_descriptor, x_data,
      scale_offset_mean_variance_descriptor, scale_data, offset_data, epsilon,
      activation_mode, y_data, batch_mean_data, batch_var_data, saved_mean_data,
      saved_var_data, output_profile_result);
}

template <typename T, typename U>
bool MIOpenSupport::DoFusedBatchNormActivationBackwardImpl(
    Stream* stream,
    int miopen_type,  // Actually miopenDataType_t.
    const dnn::BatchDescriptor& y_act_backprop_descriptor,
    const DeviceMemory<T>& y_act_backprop_data,
    const DeviceMemory<T>& y_act_data, dnn::ActivationMode activation_mode,
    const DeviceMemory<T>& x_bn_data,
    const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
    const DeviceMemory<U>& scale_data, const DeviceMemory<U>& offset_data,
    const DeviceMemory<U>& saved_mean_data,
    const DeviceMemory<U>& saved_var_data, DeviceMemory<T>* x_bn_backprop_data,
    DeviceMemory<U>* scale_backprop_data, DeviceMemory<U>* offset_backprop_data,
    dnn::ProfileResult* output_profile_result) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  ScopedTensorDescriptor y_act_backprop_nd{
      y_act_backprop_descriptor, static_cast<miopenDataType_t>(miopen_type)};

  ScopedTensorDescriptor scale_offset_mean_variance_nd{
      scale_offset_mean_variance_descriptor,
      static_cast<miopenDataType_t>(miopen_type)};

  ScopedActivationDescriptor activation_desc{activation_mode};

  ScopedFusionPlanBatchNormActivationBackward fusion_plan{
      miopen.handle(), y_act_backprop_nd.handle(),
      scale_offset_mean_variance_nd.handle(), activation_desc};

  bool retval = false;

  if (fusion_plan.CompilationSucceeded()) {
    const bool is_profiling = output_profile_result != nullptr;

    std::unique_ptr<GpuTimer> timer;
    if (is_profiling) {
      timer.reset(new GpuTimer(parent_));
      timer->Init();
      timer->Start(AsGpuStream(stream));
    }

    miopenStatus_t status = miopenStatusSuccess;

    if (status == miopenStatusSuccess) {
      fusion_plan.SetBatchNormBackwardArgs(
          x_bn_data.opaque(), scale_data.opaque(), offset_data.opaque(),
          saved_mean_data.opaque(), saved_var_data.opaque(),
          scale_backprop_data->opaque(), offset_backprop_data->opaque());
    }

    if (status == miopenStatusSuccess) {
      status = fusion_plan.SetActivationBackwardArgs(activation_desc,
                                                     y_act_data.opaque());
    }

    if (status == miopenStatusSuccess) {
      status = fusion_plan.Execute(
          y_act_backprop_nd.handle(), y_act_backprop_data.opaque(),
          y_act_backprop_nd.handle(), x_bn_backprop_data->opaque());
    }

    if (is_profiling) {
      timer->Stop(AsGpuStream(stream));
      if (status == miopenStatusSuccess) {
        output_profile_result->set_elapsed_time_in_ms(
            timer->GetElapsedMilliseconds());
      }
      timer->Destroy();
    }

    if (status != miopenStatusSuccess) {
      // Silently return when we are profiling.
      if (!is_profiling) {
        LOG(FATAL) << "failed to enqueue fused-convolution on stream: "
                   << ToString(status);
      }
    }

    retval = true;
  }

  return retval;
}

bool MIOpenSupport::DoFusedBatchNormActivationBackward(
    Stream* stream, const dnn::BatchDescriptor& y_act_backprop_descriptor,
    const DeviceMemory<float>& y_act_backprop_data,
    const DeviceMemory<float>& y_act_data, dnn::ActivationMode activation_mode,
    const DeviceMemory<float>& x_bn_data,
    const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
    const DeviceMemory<float>& scale_data,
    const DeviceMemory<float>& offset_data,
    const DeviceMemory<float>& saved_mean_data,
    const DeviceMemory<float>& saved_var_data,
    DeviceMemory<float>* x_bn_backprop_data,
    DeviceMemory<float>* scale_backprop_data,
    DeviceMemory<float>* offset_backprop_data,
    dnn::ProfileResult* output_profile_result) {
  return DoFusedBatchNormActivationBackwardImpl<float, float>(
      stream, miopenFloat, y_act_backprop_descriptor, y_act_backprop_data,
      y_act_data, activation_mode, x_bn_data,
      scale_offset_mean_variance_descriptor, scale_data, offset_data,
      saved_mean_data, saved_var_data, x_bn_backprop_data, scale_backprop_data,
      offset_backprop_data, output_profile_result);
}

bool MIOpenSupport::DoFusedBatchNormActivationBackward(
    Stream* stream, const dnn::BatchDescriptor& y_act_backprop_descriptor,
    const DeviceMemory<Eigen::half>& y_act_backprop_data,
    const DeviceMemory<Eigen::half>& y_act_data,
    dnn::ActivationMode activation_mode,
    const DeviceMemory<Eigen::half>& x_bn_data,
    const dnn::BatchDescriptor& scale_offset_mean_variance_descriptor,
    const DeviceMemory<float>& scale_data,
    const DeviceMemory<float>& offset_data,
    const DeviceMemory<float>& saved_mean_data,
    const DeviceMemory<float>& saved_var_data,
    DeviceMemory<Eigen::half>* x_bn_backprop_data,
    DeviceMemory<float>* scale_backprop_data,
    DeviceMemory<float>* offset_backprop_data,
    dnn::ProfileResult* output_profile_result) {
  return DoFusedBatchNormActivationBackwardImpl<Eigen::half, float>(
      stream, miopenHalf, y_act_backprop_descriptor, y_act_backprop_data,
      y_act_data, activation_mode, x_bn_data,
      scale_offset_mean_variance_descriptor, scale_data, offset_data,
      saved_mean_data, saved_var_data, x_bn_backprop_data, scale_backprop_data,
      offset_backprop_data, output_profile_result);
}

}  // namespace gpu

void initialize_miopen() {
  auto miopenAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      rocm::kROCmPlatformId, PluginKind::kDnn, gpu::kMIOpenPlugin);

  if (!miopenAlreadyRegistered) {
    port::Status status =
        PluginRegistry::Instance()->RegisterFactory<PluginRegistry::DnnFactory>(
            rocm::kROCmPlatformId, gpu::kMIOpenPlugin, "MIOpen",
            [](internal::StreamExecutorInterface* parent) -> dnn::DnnSupport* {
              gpu::GpuExecutor* rocm_executor =
                  dynamic_cast<gpu::GpuExecutor*>(parent);
              if (rocm_executor == nullptr) {
                LOG(ERROR)
                    << "Attempting to initialize an instance of the MIOpen "
                    << "support library with a non-ROCM StreamExecutor";
                return nullptr;
              }

              gpu::MIOpenSupport* dnn = new gpu::MIOpenSupport(rocm_executor);
              if (!dnn->Init().ok()) {
                // Note: Init() will log a more specific error.
                delete dnn;
                return nullptr;
              }
              return dnn;
            });

    if (!status.ok()) {
      LOG(ERROR) << "Unable to register MIOpen factory: "
                 << status.error_message();
    }

    PluginRegistry::Instance()->SetDefaultFactory(
        rocm::kROCmPlatformId, PluginKind::kDnn, gpu::kMIOpenPlugin);
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_miopen,
                            { stream_executor::initialize_miopen(); });
