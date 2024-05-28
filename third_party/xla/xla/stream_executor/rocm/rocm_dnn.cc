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

#include "xla/stream_executor/rocm/rocm_dnn.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <functional>
#include <memory>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "Eigen/Core"  // from @eigen_archive
#include "rocm/include/miopen/miopen.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/gpu_activation.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_timer.h"
#include "xla/stream_executor/platform/dso_loader.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/rocm/rocm_diagnostics.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/determinism.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/hash.h"
#include "tsl/platform/logging.h"

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

const int kConvDebugVlogLevel = 3;

}  // namespace

namespace stream_executor {

using dnn::AlgorithmDesc;
using dnn::BatchDescriptor;
using dnn::ConvolutionDescriptor;
using dnn::FilterDescriptor;
using dnn::NormalizeDescriptor;
using dnn::PoolingDescriptor;

namespace gpu {

// Populates the profile result if not empty.
static absl::Status PopulateProfileFromTimer(
    std::optional<GpuTimer>& timer, const dnn::AlgorithmDesc& algorithm,
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

std::string ToString(miopenStatus_t status) {
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
    case miopenStatusUnsupportedOp:
      return "miopenStatusUnsupportedOp";
    default:
      return absl::StrCat("<unknown miopen status: ", static_cast<int>(status),
                          ">");
  }
}

std::string ToString(miopenConvFwdAlgorithm_t algorithm) {
  std::string s;
  switch (algorithm) {
    case miopenConvolutionFwdAlgoGEMM:
      s = "GEMM";
      break;
    case miopenConvolutionFwdAlgoDirect:
      s = "Direct";
      break;
    case miopenConvolutionFwdAlgoFFT:
      s = "FFT";
      break;
    case miopenConvolutionFwdAlgoWinograd:
      s = "Winograd";
      break;
    case miopenConvolutionFwdAlgoImplicitGEMM:
      s = "Implicit GEMM";
      break;
  }
  return s;
}

std::string ToString(miopenConvBwdWeightsAlgorithm_t algorithm) {
  std::string s;
  switch (algorithm) {
    case miopenConvolutionBwdWeightsAlgoGEMM:
      s = "GEMM";
      break;
    case miopenConvolutionBwdWeightsAlgoDirect:
      s = "Direct";
      break;
    case miopenConvolutionBwdWeightsAlgoWinograd:
      s = "Winograd";
      break;
    case miopenConvolutionBwdWeightsAlgoImplicitGEMM:
      s = "Implicit GEMM";
      break;
  }
  return s;
}

std::string ToString(miopenConvBwdDataAlgorithm_t algorithm) {
  std::string s;
  switch (algorithm) {
    case miopenConvolutionBwdDataAlgoGEMM:
      s = "GEMM";
      break;
    case miopenConvolutionBwdDataAlgoDirect:
      s = "Direct";
      break;
    case miopenConvolutionBwdDataAlgoFFT:
      s = "FFT";
      break;
    case miopenConvolutionBwdDataAlgoWinograd:
      s = "Winograd";
      break;
    case miopenTransposeBwdDataAlgoGEMM:
      s = "Transpose GEMM";
      break;
    case miopenConvolutionBwdDataAlgoImplicitGEMM:
      s = "Implicit GEMM";
      break;
  }
  return s;
}

std::string ToString(miopenConvAlgorithm_t algorithm) {
  std::string s;
  switch (algorithm) {
    case miopenConvolutionAlgoGEMM:
      s = "GEMM";
      break;
    case miopenConvolutionAlgoDirect:
      s = "Direct";
      break;
    case miopenConvolutionAlgoFFT:
      s = "FFT";
      break;
    case miopenConvolutionAlgoWinograd:
      s = "Winograd";
      break;
    case miopenConvolutionAlgoImplicitGEMM:
      s = "Implicit GEMM";
      break;
  }
  return s;
}

// RAII wrapper for all calls to MIOpen with a MIOpen handle argument.
//
// See MIOpenAccess::GetHandle() for details.
class MIOpenHandle {
 public:
  // Takes ownership of the executor context and the lock to access MIOpen
  // using handle.
  MIOpenHandle(GpuExecutor* executor, std::unique_ptr<absl::MutexLock> lock,
               miopenHandle_t handle)
      : context_(executor), lock_(std::move(lock)), handle_(handle) {}

  // Returns MIOpen handle. To be passed directly to MIOpen APIs, don't keep
  // a copy.
  miopenHandle_t handle() const { return handle_; }

 private:
  gpu::ScopedActivateExecutorContext context_;
  std::unique_ptr<absl::MutexLock> lock_;
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

#define STREAM_EXECUTOR_MIOPEN_WRAP(__name)                              \
  struct DynLoadShim__##__name {                                         \
    static const char* kName;                                            \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;         \
    static void* GetDsoHandle() {                                        \
      auto s = internal::CachedDsoLoader::GetMiopenDsoHandle();          \
      return s.value();                                                  \
    }                                                                    \
    static FuncPtrT LoadOrDie() {                                        \
      void* f;                                                           \
      auto s = tsl::Env::Default()->GetSymbolFromLibrary(GetDsoHandle(), \
                                                         kName, &f);     \
      CHECK(s.ok()) << "could not find " << kName                        \
                    << " in miopen DSO; dlerror: " << s.message();       \
      return reinterpret_cast<FuncPtrT>(f);                              \
    }                                                                    \
    static FuncPtrT DynLoad() {                                          \
      static FuncPtrT f = LoadOrDie();                                   \
      return f;                                                          \
    }                                                                    \
    template <typename... Args>                                          \
    miopenStatus_t operator()(Args... args) {                            \
      return DynLoad()(args...);                                         \
    }                                                                    \
  } __name;                                                              \
  const char* DynLoadShim__##__name::kName = #__name;

#endif

#if (TF_ROCM_VERSION >= 50000)
// clang-format off
#define MIOPEN_DNN_ROUTINE_EACH(__macro)                             \
  __macro(miopenBatchNormalizationBackward)                          \
  __macro(miopenBatchNormalizationForwardInference)                  \
  __macro(miopenBatchNormalizationForwardTraining)                   \
  __macro(miopenGetConvolutionForwardOutputDim)                      \
  __macro(miopenGetConvolutionNdForwardOutputDim)                    \
  __macro(miopenFindConvolutionForwardAlgorithm)                     \
  __macro(miopenCreateTensorDescriptor)                              \
  __macro(miopenDestroyTensorDescriptor)                             \
  __macro(miopenSetNdPoolingDescriptor)                              \
  __macro(miopenSetPoolingIndexType)                                 \
  __macro(miopenSetLRNDescriptor)                                    \
  __macro(miopenLRNGetWorkSpaceSize)                                 \
  __macro(miopenCreateConvolutionDescriptor)                         \
  __macro(miopenCreatePoolingDescriptor)                             \
  __macro(miopenDestroyPoolingDescriptor)                            \
  __macro(miopenCreateLRNDescriptor)                                 \
  __macro(miopenDestroyLRNDescriptor)                                \
  __macro(miopenDestroyConvolutionDescriptor)                        \
  __macro(miopenCreateWithStream)                                    \
  __macro(miopenDestroy)                                             \
  __macro(miopenSetStream)                                           \
  __macro(miopenSetAllocator)                                        \
  __macro(miopenActivationForward)                                   \
  __macro(miopenConvolutionForward)                                  \
  __macro(miopenConvolutionBackwardBias)                             \
  __macro(miopenConvolutionForwardGetWorkSpaceSize)                  \
  __macro(miopenInitConvolutionDescriptor)                           \
  __macro(miopenInitConvolutionNdDescriptor)                         \
  __macro(miopenGetConvolutionDescriptor)                            \
  __macro(miopenGetConvolutionNdDescriptor)                          \
  __macro(miopenSetConvolutionGroupCount)                            \
  __macro(miopenSet4dTensorDescriptor)                               \
  __macro(miopenGetTensorDescriptor)                                 \
  __macro(miopenSetTensorDescriptor)                                 \
  __macro(miopenGetTensorDescriptorSize)                             \
  __macro(miopenPoolingForward)                                      \
  __macro(miopenPoolingGetWorkSpaceSizeV2)                           \
  __macro(miopenPoolingBackward)                                     \
  __macro(miopenLRNForward)                                          \
  __macro(miopenLRNBackward)                                         \
  __macro(miopenOpTensor)                                            \
  __macro(miopenConvolutionBackwardData)                             \
  __macro(miopenConvolutionBackwardWeights)                          \
  __macro(miopenConvolutionBackwardWeightsGetWorkSpaceSize)          \
  __macro(miopenFindConvolutionBackwardDataAlgorithm)                \
  __macro(miopenFindConvolutionBackwardWeightsAlgorithm)             \
  __macro(miopenConvolutionBackwardDataGetWorkSpaceSize)             \
  __macro(miopenCreateRNNDescriptor)                                 \
  __macro(miopenSetRNNDescriptor)                                    \
  __macro(miopenSetRNNDescriptor_V2)                                 \
  __macro(miopenDestroyRNNDescriptor)                                \
  __macro(miopenGetRNNParamsSize)                                    \
  __macro(miopenGetRNNLayerParam)                                    \
  __macro(miopenGetRNNLayerBias)                                     \
  __macro(miopenGetRNNWorkspaceSize)                                 \
  __macro(miopenGetRNNTrainingReserveSize)                           \
  __macro(miopenRNNForwardInference)                                 \
  __macro(miopenRNNForwardTraining)                                  \
  __macro(miopenRNNBackwardData)                                     \
  __macro(miopenRNNBackwardWeights)                                  \
  __macro(miopenGetRNNLayerParamOffset)                              \
  __macro(miopenGetRNNLayerParamSize)                                \
  __macro(miopenGetRNNLayerBiasOffset)                               \
  __macro(miopenGetRNNLayerBiasSize)                                 \
  __macro(miopenGetRNNParamsDescriptor)                              \
  __macro(miopenCreateDropoutDescriptor)                             \
  __macro(miopenSetDropoutDescriptor)                                \
  __macro(miopenGetDropoutDescriptor)                                \
  __macro(miopenDestroyDropoutDescriptor)                            \
  __macro(miopenRestoreDropoutDescriptor)                            \
  __macro(miopenDropoutGetReserveSpaceSize)                          \
  __macro(miopenDropoutGetStatesSize)                                \
  __macro(miopenDropoutForward)                                      \
  __macro(miopenDropoutBackward)                                     \
  __macro(miopenCreateActivationDescriptor)                          \
  __macro(miopenSetActivationDescriptor)                             \
  __macro(miopenGetActivationDescriptor)                             \
  __macro(miopenDestroyActivationDescriptor)                         \
  __macro(miopenCreateFusionPlan)                                    \
  __macro(miopenCreateOpConvForward)                                 \
  __macro(miopenCreateOpBiasForward)                                 \
  __macro(miopenCreateOpActivationForward)                           \
  __macro(miopenCreateOpActivationBackward)                          \
  __macro(miopenCreateOpBatchNormInference)                          \
  __macro(miopenCreateOpBatchNormForward)                            \
  __macro(miopenCreateOpBatchNormBackward)                           \
  __macro(miopenCompileFusionPlan)                                   \
  __macro(miopenFusionPlanGetOp)                                     \
  __macro(miopenCreateOperatorArgs)                                  \
  __macro(miopenSetOpArgsConvForward)                                \
  __macro(miopenSetOpArgsBiasForward)                                \
  __macro(miopenSetOpArgsActivForward)                               \
  __macro(miopenSetOpArgsActivBackward)                              \
  __macro(miopenSetOpArgsBatchNormInference)                         \
  __macro(miopenSetOpArgsBatchNormForward)                           \
  __macro(miopenSetOpArgsBatchNormBackward)                          \
  __macro(miopenExecuteFusionPlan)                                   \
  __macro(miopenDestroyOperatorArgs)                                 \
  __macro(miopenDestroyFusionPlan)                                   \
  __macro(miopenConvolutionForwardGetSolutionCount)                  \
  __macro(miopenConvolutionForwardGetSolution)                       \
  __macro(miopenConvolutionForwardGetSolutionWorkspaceSize)          \
  __macro(miopenConvolutionForwardCompileSolution)                   \
  __macro(miopenConvolutionForwardImmediate)                         \
  __macro(miopenConvolutionForwardBias)                              \
  __macro(miopenConvolutionBiasActivationForward)                    \
  __macro(miopenConvolutionBackwardDataGetSolutionCount)             \
  __macro(miopenConvolutionBackwardDataGetSolution)                  \
  __macro(miopenConvolutionBackwardDataGetSolutionWorkspaceSize)     \
  __macro(miopenConvolutionBackwardDataCompileSolution)              \
  __macro(miopenConvolutionBackwardDataImmediate)                    \
  __macro(miopenConvolutionBackwardWeightsGetSolutionCount)          \
  __macro(miopenConvolutionBackwardWeightsGetSolution)               \
  __macro(miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize)  \
  __macro(miopenConvolutionBackwardWeightsCompileSolution)           \
  __macro(miopenConvolutionBackwardWeightsImmediate)                 \
  __macro(miopenCreateCTCLossDescriptor)                             \
  __macro(miopenSetCTCLossDescriptor)                                \
  __macro(miopenGetCTCLossWorkspaceSize)                             \
  __macro(miopenCTCLoss)                                             \
  __macro(miopenDestroyCTCLossDescriptor)                            \
  __macro(miopenSetConvolutionAttribute)  // clang-format on
#else
// clang-format off
#define MIOPEN_DNN_ROUTINE_EACH(__macro)                             \
  __macro(miopenBatchNormalizationBackward)                          \
  __macro(miopenBatchNormalizationForwardInference)                  \
  __macro(miopenBatchNormalizationForwardTraining)                   \
  __macro(miopenGetConvolutionForwardOutputDim)                      \
  __macro(miopenGetConvolutionNdForwardOutputDim)                    \
  __macro(miopenFindConvolutionForwardAlgorithm)                     \
  __macro(miopenCreateTensorDescriptor)                              \
  __macro(miopenDestroyTensorDescriptor)                             \
  __macro(miopenSetNdPoolingDescriptor)                              \
  __macro(miopenSetPoolingIndexType)                                 \
  __macro(miopenSetLRNDescriptor)                                    \
  __macro(miopenLRNGetWorkSpaceSize)                                 \
  __macro(miopenCreateConvolutionDescriptor)                         \
  __macro(miopenCreatePoolingDescriptor)                             \
  __macro(miopenDestroyPoolingDescriptor)                            \
  __macro(miopenCreateLRNDescriptor)                                 \
  __macro(miopenDestroyLRNDescriptor)                                \
  __macro(miopenDestroyConvolutionDescriptor)                        \
  __macro(miopenCreateWithStream)                                    \
  __macro(miopenDestroy)                                             \
  __macro(miopenSetStream)                                           \
  __macro(miopenSetAllocator)                                        \
  __macro(miopenActivationForward)                                   \
  __macro(miopenConvolutionForward)                                  \
  __macro(miopenConvolutionBackwardBias)                             \
  __macro(miopenConvolutionForwardGetWorkSpaceSize)                  \
  __macro(miopenInitConvolutionDescriptor)                           \
  __macro(miopenInitConvolutionNdDescriptor)                         \
  __macro(miopenGetConvolutionDescriptor)                            \
  __macro(miopenGetConvolutionNdDescriptor)                          \
  __macro(miopenSetConvolutionGroupCount)                            \
  __macro(miopenSet4dTensorDescriptor)                               \
  __macro(miopenGetTensorDescriptor)                                 \
  __macro(miopenSetTensorDescriptor)                                 \
  __macro(miopenGetTensorDescriptorSize)                             \
  __macro(miopenPoolingForward)                                      \
  __macro(miopenPoolingGetWorkSpaceSizeV2)                           \
  __macro(miopenPoolingBackward)                                     \
  __macro(miopenLRNForward)                                          \
  __macro(miopenLRNBackward)                                         \
  __macro(miopenOpTensor)                                            \
  __macro(miopenConvolutionBackwardData)                             \
  __macro(miopenConvolutionBackwardWeights)                          \
  __macro(miopenConvolutionBackwardWeightsGetWorkSpaceSize)          \
  __macro(miopenFindConvolutionBackwardDataAlgorithm)                \
  __macro(miopenFindConvolutionBackwardWeightsAlgorithm)             \
  __macro(miopenConvolutionBackwardDataGetWorkSpaceSize)             \
  __macro(miopenCreateRNNDescriptor)                                 \
  __macro(miopenSetRNNDescriptor)                                    \
  __macro(miopenSetRNNDescriptor_V2)                                 \
  __macro(miopenDestroyRNNDescriptor)                                \
  __macro(miopenGetRNNParamsSize)                                    \
  __macro(miopenGetRNNLayerParam)                                    \
  __macro(miopenGetRNNLayerBias)                                     \
  __macro(miopenGetRNNWorkspaceSize)                                 \
  __macro(miopenGetRNNTrainingReserveSize)                           \
  __macro(miopenRNNForwardInference)                                 \
  __macro(miopenRNNForwardTraining)                                  \
  __macro(miopenRNNBackwardData)                                     \
  __macro(miopenRNNBackwardWeights)                                  \
  __macro(miopenGetRNNLayerParamOffset)                              \
  __macro(miopenGetRNNLayerParamSize)                                \
  __macro(miopenGetRNNLayerBiasOffset)                               \
  __macro(miopenGetRNNLayerBiasSize)                                 \
  __macro(miopenGetRNNParamsDescriptor)                              \
  __macro(miopenCreateDropoutDescriptor)                             \
  __macro(miopenSetDropoutDescriptor)                                \
  __macro(miopenGetDropoutDescriptor)                                \
  __macro(miopenDestroyDropoutDescriptor)                            \
  __macro(miopenRestoreDropoutDescriptor)                            \
  __macro(miopenDropoutGetReserveSpaceSize)                          \
  __macro(miopenDropoutGetStatesSize)                                \
  __macro(miopenDropoutForward)                                      \
  __macro(miopenDropoutBackward)                                     \
  __macro(miopenCreateActivationDescriptor)                          \
  __macro(miopenSetActivationDescriptor)                             \
  __macro(miopenGetActivationDescriptor)                             \
  __macro(miopenDestroyActivationDescriptor)                         \
  __macro(miopenCreateFusionPlan)                                    \
  __macro(miopenCreateOpConvForward)                                 \
  __macro(miopenCreateOpBiasForward)                                 \
  __macro(miopenCreateOpActivationForward)                           \
  __macro(miopenCreateOpActivationBackward)                          \
  __macro(miopenCreateOpBatchNormInference)                          \
  __macro(miopenCreateOpBatchNormForward)                            \
  __macro(miopenCreateOpBatchNormBackward)                           \
  __macro(miopenCompileFusionPlan)                                   \
  __macro(miopenFusionPlanGetOp)                                     \
  __macro(miopenCreateOperatorArgs)                                  \
  __macro(miopenSetOpArgsConvForward)                                \
  __macro(miopenSetOpArgsBiasForward)                                \
  __macro(miopenSetOpArgsActivForward)                               \
  __macro(miopenSetOpArgsActivBackward)                              \
  __macro(miopenSetOpArgsBatchNormInference)                         \
  __macro(miopenSetOpArgsBatchNormForward)                           \
  __macro(miopenSetOpArgsBatchNormBackward)                          \
  __macro(miopenExecuteFusionPlan)                                   \
  __macro(miopenDestroyOperatorArgs)                                 \
  __macro(miopenDestroyFusionPlan)                                   \
  __macro(miopenConvolutionBiasActivationForward)                    \
  __macro(miopenConvolutionForwardGetSolutionCount)                  \
  __macro(miopenConvolutionForwardGetSolution)                       \
  __macro(miopenConvolutionForwardGetSolutionWorkspaceSize)          \
  __macro(miopenConvolutionForwardCompileSolution)                   \
  __macro(miopenConvolutionForwardImmediate)                         \
  __macro(miopenConvolutionForwardBias)                              \
  __macro(miopenConvolutionBackwardDataGetSolutionCount)             \
  __macro(miopenConvolutionBackwardDataGetSolution)                  \
  __macro(miopenConvolutionBackwardDataGetSolutionWorkspaceSize)     \
  __macro(miopenConvolutionBackwardDataCompileSolution)              \
  __macro(miopenConvolutionBackwardDataImmediate)                    \
  __macro(miopenConvolutionBackwardWeightsGetSolutionCount)          \
  __macro(miopenConvolutionBackwardWeightsGetSolution)               \
  __macro(miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize)  \
  __macro(miopenConvolutionBackwardWeightsCompileSolution)           \
  __macro(miopenConvolutionBackwardWeightsImmediate)                 \
  __macro(miopenCreateCTCLossDescriptor)                             \
  __macro(miopenSetCTCLossDescriptor)                                \
  __macro(miopenGetCTCLossWorkspaceSize)                             \
  __macro(miopenCTCLoss)                                             \
  __macro(miopenDestroyCTCLossDescriptor)
// clang-format on
#endif

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

uint64_t GetHashValue(miopenTensorDescriptor_t tensor_desc) {
  miopenDataType_t datatype = miopenFloat;
  int dims[kMaxMIOpenTensorSize] = {0};
  int strides[kMaxMIOpenTensorSize] = {0};
  wrap::miopenGetTensorDescriptor(tensor_desc, &datatype, dims, strides);

  uint64_t hash_value = tsl::hash<int>()(datatype);
  for (int dim : dims)
    hash_value = tsl::Hash64Combine(hash_value, tsl::hash<int>()(dim));
  for (int stride : strides)
    hash_value = tsl::Hash64Combine(hash_value, tsl::hash<int>()(stride));

  return hash_value;
}

uint64_t GetHashValue(miopenConvolutionDescriptor_t conv_desc) {
  miopenConvolutionMode_t c_mode = miopenConvolution;
  int nd = 0;
  wrap::miopenGetConvolutionNdDescriptor(conv_desc, 0, &nd, nullptr, nullptr,
                                         nullptr, &c_mode);

  std::vector<int> stride(nd);
  std::vector<int> pad(nd);
  std::vector<int> dilation(nd);

  wrap::miopenGetConvolutionNdDescriptor(
      conv_desc, nd, &nd, pad.data(), stride.data(), dilation.data(), &c_mode);

  uint64_t hash_value = tsl::hash<int>()(c_mode);
  auto hash64Combine = [&hash_value](int element) {
    hash_value = tsl::Hash64Combine(hash_value, tsl::hash<int>()(element));
  };
  std::for_each(pad.begin(), pad.end(), hash64Combine);
  std::for_each(stride.begin(), stride.end(), hash64Combine);
  std::for_each(dilation.begin(), dilation.end(), hash64Combine);

  return hash_value;
}

bool RequireMIOpenDeterminism() { return tsl::OpDeterminismRequired(); }

// Class to implement a cache of compiled fusion plans
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
  static bool FindOrCreate(uint64_t hash,
                           miopenFusionPlanDescriptor_t* fusion_plan,
                           miopenFusionDirection_t fusion_direction,
                           miopenTensorDescriptor_t input_descriptor) {
    absl::MutexLock lock{&cached_plans_mutex};

    bool found_cached_plan = false;

    auto it = cached_plans.find(hash);
    if (it != cached_plans.end()) {
      VLOG(2) << "Found a cached plan for " << hash;
      *fusion_plan = it->second;
      found_cached_plan = true;
    } else {
      VLOG(2) << "Creating a new plan for " << hash;
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

  // Need to figure out the right place to call this routine
  static void Clear() {
    absl::MutexLock lock{&cached_plans_mutex};

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

  // Is the Fusion plan corresponding to this hash unsupported
  static bool IsUnsupportedFusionPlan(uint64_t hash) {
    absl::MutexLock lock{&cached_plans_mutex};
    return unsupported_plans.count(hash) > 0;
  }

  // Mark the given hash value as corresponding to an unsupported fusion plan
  static void MarkFusionPlanUnsupported(uint64_t hash) {
    absl::MutexLock lock{&cached_plans_mutex};
    unsupported_plans.insert(hash);
  }

 private:
  // Mutex to guard access to all data within this class
  static absl::Mutex cached_plans_mutex;

  // Map of hash-value to MIOpen Fusion plan descriptors
  // Need to be able share this across more than one stream and hence static
  static std::map<uint64_t, miopenFusionPlanDescriptor_t> cached_plans;

  // Set of hash-values that correspond to MIOpen Fusion plans that will fail
  // compile and hence are not supported.
  static std::set<uint64_t> unsupported_plans;
};

absl::Mutex CachedFusionPlans::cached_plans_mutex;
std::map<uint64_t, miopenFusionPlanDescriptor_t>
    CachedFusionPlans::cached_plans;
std::set<uint64_t> CachedFusionPlans::unsupported_plans;

dnn::ProfileResult GetProfileResultFromConvSolution(
    miopenConvSolution_t solution) {
  dnn::ProfileResult profile_result;
  profile_result.set_algorithm({(dnn::AlgorithmDesc::Index)solution.solution_id,
                                false, solution.workspace_size});
  profile_result.set_elapsed_time_in_ms(solution.time);
  profile_result.set_scratch_size(solution.workspace_size);
  return profile_result;
}

dnn::ProfileResult GetProfileResultFromConvAlgoPerf(
    dnn::ConvolutionKind kind, miopenConvAlgoPerf_t algorithm) {
  int64_t algo_id;
  switch (kind) {
    case dnn::ConvolutionKind::FORWARD:
    case dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION:
      algo_id = algorithm.fwd_algo;
      break;
    case dnn::ConvolutionKind::BACKWARD_DATA:
      algo_id = algorithm.bwd_data_algo;
      break;
    case dnn::ConvolutionKind::BACKWARD_FILTER:
      algo_id = algorithm.bwd_weights_algo;
      break;
    default:
      LOG(FATAL) << "Unexpected convolution kind " << static_cast<int>(kind);
      break;
  }

  dnn::ProfileResult profile_result;
  profile_result.set_algorithm({algo_id, false, algorithm.memory});
  profile_result.set_elapsed_time_in_ms(algorithm.time);
  profile_result.set_scratch_size(algorithm.memory);
  return profile_result;
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
    absl::MutexLock lock(&mutex_);
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
    auto lock = std::make_unique<absl::MutexLock>(&mutex_);
    mutex_.AssertHeld();
    hipStream_t hip_stream = stream ? AsGpuStreamValue(stream) : nullptr;
    auto status = wrap::miopenSetStream(handle_, hip_stream);
    CHECK_EQ(status, miopenStatusSuccess) << "Failed to set MIOpen stream.";
    return MIOpenHandle(executor, std::move(lock), handle_);
  }

 private:
  // Guards the enqueueing of MIOpen operations via the handle_ below.
  absl::Mutex mutex_;

  // MIOpen library handle.
  miopenHandle_t handle_ ABSL_GUARDED_BY(mutex_);  // Owned.
};

MIOpenSupport::MIOpenSupport(GpuExecutor* parent) : parent_(parent) {
  // by default, the Get*Algorithm API will return the list of all applicable
  // algorithms
  return_best_algo_only_ = false;
  // but if the env var TF_ROCM_RETURN_BEST_ALGO_ONLY is set, only the best
  // (i.e. most efficient) algorithm will be returned
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_RETURN_BEST_ALGO_ONLY", false,
                                      &return_best_algo_only_));

  // by default, use Find Mode APIs for convolution
  use_immediate_mode_ = false;
  // swich to Find Mode if env var TF_ROCM_USE_IMMEDIATE_MODE is set

  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_USE_IMMEDIATE_MODE", false,
                                      &use_immediate_mode_));

  bool enable_pooling_cache = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_BW_POOL_CACHE", false,
                                      &enable_pooling_cache));
  if (enable_pooling_cache) m_pooling_cache_allowed = true;
}

absl::Status MIOpenSupport::Init() {
  ScopedActivateExecutorContext context(parent_);
  miopenHandle_t miopen_handle = nullptr;
  auto status = wrap::miopenCreateWithStream(
      reinterpret_cast<miopenHandle_t*>(&miopen_handle), (hipStream_t)(0));
  if (status == miopenStatusSuccess) {
    miopen_.reset(new MIOpenAccess(miopen_handle));
    return absl::OkStatus();
  }

  CHECK_EQ(miopen_handle, nullptr);
  LOG(ERROR) << "could not create miopen handle: " << ToString(status);
  if (status == miopenStatusNotInitialized) {
    auto result = rocm::Diagnostician::FindKernelDriverVersion();
    if (!result.ok()) {
      LOG(ERROR) << "error retrieving driver version: "
                 << rocm::DriverVersionStatusToString(result);
    } else {
      const auto& version = result.value();
      LOG(INFO) << "possibly insufficient driver version: "
                << rocm::DriverVersionToString(version);
    }
  }

  return absl::Status{absl::StatusCode::kInternal,
                      absl::StrCat("miopen library could not create a handle: ",
                                   ToString(status))};
}

absl::StatusOr<stream_executor::dnn::VersionInfo> MIOpenSupport::GetVersion() {
  // ROCM TODO: retrieve MIOpen version with its API
  return stream_executor::dnn::VersionInfo(1, 3, 0);
}

template <typename T>
miopenStatus_t miDestroyObject(T obj) {
  return miopenStatusSuccess;
}

template <>
miopenStatus_t miDestroyObject(miopenTensorDescriptor_t obj) {
  return wrap::miopenDestroyTensorDescriptor(obj);
}

template <>
miopenStatus_t miDestroyObject(miopenConvolutionDescriptor_t obj) {
  return wrap::miopenDestroyConvolutionDescriptor(obj);
}

template <>
miopenStatus_t miDestroyObject(miopenPoolingDescriptor_t obj) {
  return wrap::miopenDestroyPoolingDescriptor(obj);
}

template <>
miopenStatus_t miDestroyObject(miopenLRNDescriptor_t obj) {
  return wrap::miopenDestroyLRNDescriptor(obj);
}

template <typename T>
struct ScopedDescriptor {
  ScopedDescriptor() : handle_(nullptr) {}

  ScopedDescriptor(ScopedDescriptor<T>&& other) {
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }

  ~ScopedDescriptor() {
    if (handle_ != nullptr) return;

    auto status = miDestroyObject(
        handle_);  // wrap::miopenDestroyTensorDescriptor(handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen tensor descriptor: "
                 << ToString(status);
    }
  }

  T handle() const { return handle_; }

  T handle_;  // Owned.

  ScopedDescriptor(const ScopedDescriptor<T>&) = delete;
  void operator=(const ScopedDescriptor<T>&) = delete;
};

using ScopedTensorDescriptor = ScopedDescriptor<miopenTensorDescriptor_t>;
using ScopedFilterDescriptor = ScopedDescriptor<miopenTensorDescriptor_t>;
using ScopedConvolutionDescriptor =
    ScopedDescriptor<miopenConvolutionDescriptor_t>;
using ScopedPoolingDescriptor = ScopedDescriptor<miopenPoolingDescriptor_t>;
using ScopedNormalizeDescriptor = ScopedDescriptor<miopenLRNDescriptor_t>;

absl::StatusOr<ScopedTensorDescriptor> scope(
    const BatchDescriptor& batch_descriptor, miopenDataType_t data_type) {
  ScopedTensorDescriptor obj;
  auto status = wrap::miopenCreateTensorDescriptor(&obj.handle_);
  if (status != miopenStatusSuccess) {
    return absl::InternalError("could not create miopen tensor descriptor: " +
                               ToString(status));
  }

  switch (batch_descriptor.layout()) {
    case dnn::DataLayout::kBatchYXDepth:
    case dnn::DataLayout::kBatchDepthYX: {
      const int nd = batch_descriptor.ndims() + 2;

      // MIOpen requires the strides and dims to be ordered as BDYX.
      std::vector<int64_t> strides64 =
          batch_descriptor.full_strides(dnn::DataLayout::kBatchDepthYX);
      std::vector<int64_t> dims64 =
          batch_descriptor.full_dims(dnn::DataLayout::kBatchDepthYX);

      // MIOpen requires arrays of ints.
      std::vector<int> strides(nd);
      std::vector<int> dims(nd);
      std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                     &CheckedNarrowing<int64_t, int>);
      std::transform(dims64.cbegin(), dims64.cend(), dims.begin(),
                     &CheckedNarrowing<int64_t, int>);
      status = wrap::miopenSetTensorDescriptor(obj.handle_, data_type, nd,
                                               dims.data(), strides.data());

      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "could not convert BatchDescriptor " + batch_descriptor.ToString() +
            " to miopen tensor descriptor: " + ToString(status));
      }
    } break;
    default:
      return absl::InternalError("Unsupported tensor format " +
                                 DataLayoutString(batch_descriptor.layout()));
      break;
  }
  return obj;
}

absl::StatusOr<ScopedFilterDescriptor> scope(
    const FilterDescriptor& filter_descriptor, miopenDataType_t data_type) {
  ScopedFilterDescriptor obj;
  auto status = wrap::miopenCreateTensorDescriptor(&obj.handle_);
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "could not create miopen filter descriptor: "
               << ToString(status);
  }

  // We need to pass two vectors to the miopenSetTensorDescriptor routine
  // "dims" (length == number of dims, elem value == dimension size)
  // "strides" (length == number of dims, elem value == stride size)
  //
  // Irrespective of the actual filter layout, the indexing of both those
  // vectors must be the following (coz that is what MIOpen expects)
  // dims[0] = strides[0] = N or output
  // dims[1] = strides[1] = C or input
  // dims[2] = strides[2] = H or spatial dim 0
  // dims[3] = strides[3] = W or spatial dim 1
  //
  // assume you have a tensor with dimensions
  // batch descriptor name    filter descriptor name    value
  //   N (batch size)            O (output features)    256
  //   C (channels)              I (input features)       3
  //   H (height)                H (height)               7
  //   W (width)                 W (width)                5
  //
  // The content of "dims" will be the same irrespective of layout
  // layout (NCHW or NHWC), and MIOpen expects it should be
  //                           NCHW layout   NHWC layout
  // dims[0] = size of N dim =    256           256
  // dims[1] = size of C dim =      3             3
  // dims[2] = size of H dim =      7             7
  // dims[3] = size of W dim =      5             5
  //
  // The content of "strides" will be different based on layout
  //                                  NCHW layout   NHWC layout
  //  strides[0] = stride of N dim =     7x5x3       7x5x3
  //  strides[1] = stride of C dim =     7x5         1
  //  strides[2] = stride of H dim =     5           5x3
  //  strides[3] = stride of W dim =     1           3

  switch (filter_descriptor.layout()) {
    case dnn::FilterLayout::kOutputYXInput:
    case dnn::FilterLayout::kOutputInputYX: {
      const int nd = filter_descriptor.ndims() + 2;

      // MIOpen requires the strides and dims to be ordered as BDYX.
      std::vector<int64_t> strides64 =
          filter_descriptor.full_strides(dnn::FilterLayout::kOutputInputYX);
      std::vector<int64_t> dims64 =
          filter_descriptor.full_dims(dnn::FilterLayout::kOutputInputYX);

      // MIOpen requires arrays of ints.
      std::vector<int> strides;
      std::vector<int> dims;
      absl::c_transform(strides64, std::back_inserter(strides),
                        &CheckedNarrowing<int64_t, int>);
      absl::c_transform(dims64, std::back_inserter(dims),
                        &CheckedNarrowing<int64_t, int>);
      status = wrap::miopenSetTensorDescriptor(obj.handle_, data_type, nd,
                                               dims.data(), strides.data());

      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "could not convert FilterDescriptor "
                   << filter_descriptor.ToString()
                   << " to miopen tensor descriptor: " << ToString(status);
      }
    } break;
    default:
      LOG(FATAL) << "Unsupported tensor format "
                 << FilterLayoutString(filter_descriptor.layout());
      break;
  }
  return obj;
}

absl::StatusOr<ScopedConvolutionDescriptor> scope(
    const ConvolutionDescriptor& convolution_descriptor) {
  ScopedConvolutionDescriptor obj;
  auto status = wrap::miopenCreateConvolutionDescriptor(&obj.handle_);
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
                 &CheckedNarrowing<int64_t, int>);
  std::transform(padding64.cbegin(), padding64.cend(), padding.begin(),
                 &CheckedNarrowing<int64_t, int>);

  std::vector<int> upscale(convolution_descriptor.ndims());
  const auto& dilations64 = convolution_descriptor.dilations();
  std::transform(dilations64.cbegin(), dilations64.cend(), upscale.begin(),
                 &CheckedNarrowing<int64_t, int>);

  status = wrap::miopenInitConvolutionNdDescriptor(
      obj.handle_, convolution_descriptor.ndims(), padding.data(),
      strides.data(), upscale.data(), miopenConvolution);
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "could not set miopen convolution descriptor: "
               << ToString(status);
  }

  VLOG(2) << "Requesting grouped convolution: "
          << convolution_descriptor.group_count();
  status = wrap::miopenSetConvolutionGroupCount(
      obj.handle_, convolution_descriptor.group_count());
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "could not set miopen convolution group count: "
               << ToString(status);
  }

#if (TF_ROCM_VERSION >= 50300)
  if (RequireMIOpenDeterminism()) {
    status = wrap::miopenSetConvolutionAttribute(
        obj.handle_, MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen convolution attribute: "
                 << ToString(status);
    }
  }
#endif
  return obj;
}

absl::StatusOr<ScopedPoolingDescriptor> scope(
    const PoolingDescriptor& pooling_descriptor) {
  ScopedPoolingDescriptor obj;
  auto status = wrap::miopenCreatePoolingDescriptor(&obj.handle_);
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "could not create miopen pooling descriptor: "
               << ToString(status);
  }

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

  status = wrap::miopenSetNdPoolingDescriptor(
      obj.handle_,
      (pooling_descriptor.mode() == dnn::PoolingMode::kMaximum
           ? miopenPoolingMax
           : miopenPoolingAverage),
      nd, shape.data(), padding.data(), strides.data());

  // Note: The index type has to be uint32 type for now because MIOpen
  // API assumes all input indexes to be the same type. Since a tensor
  // descriptor can only use int32 type, the index type here need to be
  // aligned with the tensor index type of the (input) tensor descritptor
  status = wrap::miopenSetPoolingIndexType(obj.handle_, miopenIndexUint32);

  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "could not set miopen pooling descriptor: "
               << ToString(status);
  }
  return obj;
}

absl::StatusOr<ScopedNormalizeDescriptor> scope(
    const NormalizeDescriptor& normalize_descriptor) {
  ScopedNormalizeDescriptor obj;
  auto status = wrap::miopenCreateLRNDescriptor(&obj.handle_);
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
  status = wrap::miopenSetLRNDescriptor(obj.handle_, miopenLRNCrossChannel,
                                        lrn_N, lrn_alpha, lrn_beta, lrn_k);
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "could not set miopen LRN descriptor: " << ToString(status);
  }
  return obj;
}

// Turns a activation mode into a miopen activation mode descriptor with a scope
// around it
struct ScopedActivationDescriptor
    : ScopedDescriptor<miopenActivationDescriptor_t> {
  static absl::StatusOr<ScopedActivationDescriptor> Create(
      dnn::ActivationMode activation_mode, double alpha = 0.0) {
    ScopedActivationDescriptor obj;
    obj.alpha_ = alpha;
    auto status = wrap::miopenCreateActivationDescriptor(&obj.handle_);
    if (status != miopenStatusSuccess) {
      return absl::InternalError(
          "call to miopenCreateActivationDescriptor failed: " +
          ToString(status));
    } else {
      switch (activation_mode) {
        case dnn::ActivationMode::kNone:
          obj.miopen_activation_mode_ = miopenActivationPASTHRU;
          break;

        case dnn::ActivationMode::kSigmoid:
          obj.miopen_activation_mode_ = miopenActivationLOGISTIC;
          break;

        case dnn::ActivationMode::kRelu:
        case dnn::ActivationMode::kReluX:
          obj.miopen_activation_mode_ = miopenActivationRELU;
          break;

        case dnn::ActivationMode::kRelu6:
          obj.miopen_activation_mode_ = miopenActivationRELU;
          obj.alpha_ = 6.0;
          break;

        case dnn::ActivationMode::kTanh:
          obj.miopen_activation_mode_ = miopenActivationTANH;
          break;

        case dnn::ActivationMode::kElu:
          obj.miopen_activation_mode_ = miopenActivationELU;
          break;

        case dnn::ActivationMode::kLeakyRelu:
          obj.miopen_activation_mode_ = miopenActivationLEAKYRELU;
          break;
          // Check with MIOpen re: support: kBandPass, kGeluExact

        default:
          VLOG(1) << "Activation mode ("
                  << dnn::ActivationModeString(activation_mode)
                  << ") not yet implemented";
          return absl::InternalError("Activation not implemented");
      }

      status = wrap::miopenSetActivationDescriptor(
          obj.handle_, obj.miopen_activation_mode_, obj.alpha_, obj.beta_,
          obj.gamma_);
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenSetActivationDescriptor failed: " +
            ToString(status));
      }
    }
    return obj;
  }
  ScopedActivationDescriptor(ScopedActivationDescriptor&& other)
      : ScopedDescriptor<miopenActivationDescriptor_t>(std::move(other)) {
    miopen_activation_mode_ = other.miopen_activation_mode_;
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    gamma_ = other.gamma_;
  }

  uint64_t GetHashValue() {
    uint64_t hash_value = tsl::hash<int>()(miopen_activation_mode_);
    hash_value = tsl::Hash64Combine(hash_value, tsl::hash<double>()(alpha_));
    hash_value = tsl::Hash64Combine(hash_value, tsl::hash<double>()(beta_));
    hash_value = tsl::Hash64Combine(hash_value, tsl::hash<double>()(gamma_));

    return hash_value;
  }

  ScopedActivationDescriptor()
      : miopen_activation_mode_(miopenActivationPASTHRU),
        alpha_(0.0),
        beta_(0.0),
        gamma_(0.0) {}

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
    if (fusion_args_ == nullptr) return;
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

  miopenStatus_t SetBatchNormForwardArgs(
      const int op_idx, const float* alpha, const float* beta,
      const void* scale, const void* offset, void* running_mean,
      void* running_variance, void* saved_mean, void* saved_inv_variance,
      double exponential_average_factor, double epsilon) {
    miopenFusionOpDescriptor_t batchnorm_op;
    auto status =
        wrap::miopenFusionPlanGetOp(fusion_plan_, op_idx, &batchnorm_op);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenFusionPlanGetOp failed: "
                 << ToString(status);
    }

    status = wrap::miopenSetOpArgsBatchNormForward(
        fusion_args_, batchnorm_op, alpha, beta, scale, offset, saved_mean,
        saved_inv_variance, running_mean, running_variance,
        exponential_average_factor, epsilon);
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

 public:
  miopenHandle_t miopen_handle_;
  miopenFusionPlanDescriptor_t fusion_plan_;
  miopenOperatorArgs_t fusion_args_;  // Owned.
  bool fusion_plan_compiled_;

  ScopedFusionPlanBase(ScopedFusionPlanBase&& other) {
    miopen_handle_ = other.miopen_handle_;
    fusion_plan_ = other.fusion_plan_;
    fusion_args_ = other.fusion_args_;
    other.fusion_args_ = nullptr;
    fusion_plan_compiled_ = other.fusion_plan_compiled_;
  }

  ScopedFusionPlanBase(const ScopedFusionPlanBase&) = delete;
  void operator=(const ScopedFusionPlanBase&) = delete;
};

// class to represent the Convolution+Bias+Activation fusion plan
class ScopedFusionPlanConvolutionBiasActivation : public ScopedFusionPlanBase {
 public:
  ScopedFusionPlanConvolutionBiasActivation(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor)
      : ScopedFusionPlanBase(miopen_handle, miopenVerticalFusion,
                             input_descriptor) {}

  ScopedFusionPlanConvolutionBiasActivation(
      ScopedFusionPlanConvolutionBiasActivation&& other)
      : ScopedFusionPlanBase(std::move(other)) {
    conv_op = other.conv_op;
    bias_op = other.bias_op;
    actv_op = other.actv_op;
  }

  static absl::StatusOr<ScopedFusionPlanConvolutionBiasActivation> Create(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t filter_descriptor,
      miopenConvolutionDescriptor_t conv_descriptor,
      miopenTensorDescriptor_t bias_descriptor,
      ScopedActivationDescriptor& act_descriptor) {
    ScopedFusionPlanConvolutionBiasActivation obj(miopen_handle,
                                                  input_descriptor);

    VLOG(2) << "Fusion Plan compile begin";

    uint64_t hash =
        GetFusionOpHashValue(miopen_handle, input_descriptor, filter_descriptor,
                             conv_descriptor, bias_descriptor, act_descriptor);

    bool is_compiled = CachedFusionPlans::FindOrCreate(
        hash, &obj.fusion_plan_, miopenVerticalFusion, input_descriptor);
    if (is_compiled) VLOG(2) << "Cache hit";
    if (!is_compiled) {
      auto status = wrap::miopenCreateOpConvForward(
          obj.fusion_plan_, &obj.conv_op, conv_descriptor, filter_descriptor);
      if (status != miopenStatusSuccess)
        return absl::InternalError("miopenCreateOpConvForward failed: " +
                                   ToString(status));

      status = wrap::miopenCreateOpBiasForward(obj.fusion_plan_, &obj.bias_op,
                                               bias_descriptor);
      if (status != miopenStatusSuccess)
        return absl::InternalError("miopenCreateOpBiasForward failed: " +
                                   ToString(status));

      if (act_descriptor.miopen_activation_mode_ != miopenActivationPASTHRU) {
        status = wrap::miopenCreateOpActivationForward(
            obj.fusion_plan_, &obj.actv_op,
            act_descriptor.miopen_activation_mode_);
        if (status != miopenStatusSuccess)
          return absl::InternalError(
              "miopenCreateOpActivationForward failed: " + ToString(status));
      }

      status = wrap::miopenCompileFusionPlan(miopen_handle, obj.fusion_plan_);
      if (status != miopenStatusSuccess) {
        VLOG(2) << "call to miopenCompileFusionPlan (CBA) failed: "
                << ToString(status);

        CachedFusionPlans::MarkFusionPlanUnsupported(hash);
      } else {
        VLOG(2) << "Fusion Plan compile succeeded (CBA) ";
        obj.fusion_plan_compiled_ = true;
      }
    } else {
      // fusion plan was already compiled...check whether it failed to compile
      obj.fusion_plan_compiled_ =
          !CachedFusionPlans::IsUnsupportedFusionPlan(hash);
    }
    return obj;
  }

  miopenStatus_t SetConvolutionArgs(const void* filter_data) {
    static const float alpha = 1.0;
    static const float beta = 0.0;
    return ScopedFusionPlanBase::SetConvolutionArgs(k_conv_op_idx, &alpha,
                                                    &beta, filter_data);
  }

  miopenStatus_t SetBiasArgs(const void* bias_data) {
    static const float alpha = 1.0;
    static const float beta = 0.0;
    return ScopedFusionPlanBase::SetBiasArgs(k_bias_op_idx, &alpha, &beta,
                                             bias_data);
  }

  miopenStatus_t SetActivationForwardArgs(
      ScopedActivationDescriptor& activation_descriptor) {
    static const float alpha = 1.0;
    static const float beta = 0.0;

    return ScopedFusionPlanBase::SetActivationForwardArgs(
        k_actv_op_idx, &alpha, &beta, activation_descriptor.alpha_,
        activation_descriptor.beta_, activation_descriptor.gamma_);
  }

  static uint64_t GetFusionOpHashValue(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t filter_descriptor,
      miopenConvolutionDescriptor_t conv_descriptor,
      miopenTensorDescriptor_t bias_descriptor,
      ScopedActivationDescriptor& activation_descriptor) {
    uint64_t hash_value = tsl::Hash64("ConvolutionBiasActivation");

    hash_value = tsl::Hash64Combine(hash_value,
                                    tsl::hash<miopenHandle_t>()(miopen_handle));

    hash_value = tsl::Hash64Combine(hash_value, GetHashValue(input_descriptor));
    hash_value =
        tsl::Hash64Combine(hash_value, GetHashValue(filter_descriptor));
    hash_value = tsl::Hash64Combine(hash_value, GetHashValue(conv_descriptor));
    hash_value = tsl::Hash64Combine(hash_value, GetHashValue(bias_descriptor));
    hash_value =
        tsl::Hash64Combine(hash_value, activation_descriptor.GetHashValue());
    return hash_value;
  }

 public:
  miopenFusionOpDescriptor_t conv_op;
  miopenFusionOpDescriptor_t bias_op;
  miopenFusionOpDescriptor_t actv_op;

 private:
  const int k_conv_op_idx = 0;
  const int k_bias_op_idx = 1;
  const int k_actv_op_idx = 2;

  ScopedFusionPlanConvolutionBiasActivation(
      const ScopedFusionPlanConvolutionBiasActivation&) = delete;
  void operator=(const ScopedFusionPlanConvolutionBiasActivation&) = delete;
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
    uint64_t hash = GetFusionOpHashValue(miopen_handle, input_descriptor,
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

  uint64_t GetFusionOpHashValue(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t scale_offset_mean_variance_descriptor,
      ScopedActivationDescriptor& activation_descriptor) {
    uint64_t hash_value = tsl::Hash64("BatchNormActivationInference");

    hash_value = tsl::Hash64Combine(hash_value,
                                    tsl::hash<miopenHandle_t>()(miopen_handle));

    hash_value = tsl::Hash64Combine(hash_value, GetHashValue(input_descriptor));

    hash_value = tsl::Hash64Combine(
        hash_value, GetHashValue(scale_offset_mean_variance_descriptor));

    hash_value =
        tsl::Hash64Combine(hash_value, activation_descriptor.GetHashValue());
    return hash_value;
  }

 private:
  const int k_batchnorm_op_idx = 0;
  const int k_actv_op_idx = 1;

  ScopedFusionPlanBatchNormActivationInference(
      const ScopedFusionPlanBatchNormActivationInference&) = delete;
  void operator=(const ScopedFusionPlanBatchNormActivationInference&) = delete;
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
    uint64_t hash = GetFusionOpHashValue(miopen_handle, input_descriptor,
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
    static const float alpha = 1.0;
    static const float beta = 0.0;
    return ScopedFusionPlanBase::SetBatchNormForwardArgs(
        k_batchnorm_op_idx, &alpha, &beta, scale, offset, batch_mean, batch_var,
        saved_mean, saved_var, /*exponential_average_factor=*/1.0, epsilon);
  }

  miopenStatus_t SetActivationForwardArgs(
      ScopedActivationDescriptor& activation_descriptor) {
    static const float alpha = 1.0;
    static const float beta = 0.0;

    return ScopedFusionPlanBase::SetActivationForwardArgs(
        k_actv_op_idx, &alpha, &beta, activation_descriptor.alpha_,
        activation_descriptor.beta_, activation_descriptor.gamma_);
  }

  uint64_t GetFusionOpHashValue(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t scale_offset_mean_variance_descriptor,
      ScopedActivationDescriptor& activation_descriptor) {
    uint64_t hash_value = tsl::Hash64("BatchNormActivationForward");

    hash_value = tsl::Hash64Combine(hash_value,
                                    tsl::hash<miopenHandle_t>()(miopen_handle));

    hash_value = tsl::Hash64Combine(hash_value, GetHashValue(input_descriptor));

    hash_value = tsl::Hash64Combine(
        hash_value, GetHashValue(scale_offset_mean_variance_descriptor));

    hash_value =
        tsl::Hash64Combine(hash_value, activation_descriptor.GetHashValue());
    return hash_value;
  }

 private:
  const int k_batchnorm_op_idx = 0;
  const int k_actv_op_idx = 1;

  ScopedFusionPlanBatchNormActivationForward(
      const ScopedFusionPlanBatchNormActivationForward&) = delete;
  void operator=(const ScopedFusionPlanBatchNormActivationForward&) = delete;
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
    uint64_t hash = GetFusionOpHashValue(miopen_handle, input_descriptor,
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

  uint64_t GetFusionOpHashValue(
      miopenHandle_t miopen_handle, miopenTensorDescriptor_t input_descriptor,
      miopenTensorDescriptor_t scale_offset_mean_variance_descriptor,
      ScopedActivationDescriptor& activation_descriptor) {
    uint64_t hash_value = tsl::Hash64("BatchNormActivationBackward");

    hash_value = tsl::Hash64Combine(hash_value,
                                    tsl::hash<miopenHandle_t>()(miopen_handle));

    hash_value = tsl::Hash64Combine(hash_value, GetHashValue(input_descriptor));

    hash_value = tsl::Hash64Combine(
        hash_value, GetHashValue(scale_offset_mean_variance_descriptor));

    hash_value =
        tsl::Hash64Combine(hash_value, activation_descriptor.GetHashValue());
    return hash_value;
  }

 private:
  const int k_batchnorm_op_idx = 0;
  const int k_actv_op_idx = 1;

  ScopedFusionPlanBatchNormActivationBackward(
      const ScopedFusionPlanBatchNormActivationBackward&) = delete;
  void operator=(const ScopedFusionPlanBatchNormActivationBackward&) = delete;
};

namespace {

const char* getTypeName(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kBF16:
      return "BF16";
    case dnn::DataType::kFloat:
      return "F32";
    case dnn::DataType::kHalf:
      return "F16";
    case dnn::DataType::kInt8:
      return "I8";
    case dnn::DataType::kDouble:
      return "F64";
    default:
      return "Unknown";
  }
}

miopenDataType_t ToMIOpenDataType(
    dnn::DataType data_type,
    dnn::DataLayout data_layout = dnn::DataLayout::kBatchDepthYX) {
  switch (data_type) {
    case dnn::DataType::kBF16:
      return miopenBFloat16;
    case dnn::DataType::kFloat:
      return miopenFloat;
    case dnn::DataType::kHalf:
      return miopenHalf;
    case dnn::DataType::kInt8:
      if (data_layout == dnn::DataLayout::kBatchDepthYX) return miopenInt8;
      LOG(FATAL)
          << "The kInt8 data type only supports the kBatchDepthYX data layout!";
      break;
    case dnn::DataType::kDouble:
      LOG(FATAL)
          << "Unsupported DNN data type: tf.float64 (dnn::DataType::kDouble)";
      break;
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
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

template <typename Base>
class MixinBase : public Base {};
template <>
class MixinBase<void> {};

}  // namespace

#define RETURN_IF_MIOPEN_ERROR(STATUS, ...)                                   \
  if (!SE_PREDICT_TRUE((STATUS) == miopenStatusSuccess)) {                    \
    std::string error_msg = absl::StrCat(ToString(STATUS), " ", __VA_ARGS__); \
    SetFailure(::absl::UnknownError(error_msg));                              \
    LOG(ERROR) << error_msg;                                                  \
    return;                                                                   \
  }

template <typename Base>
class MIOpenDescriptorCommon : public MixinBase<Base> {
 public:
  bool ok() const { return status_.ok(); }
  absl::Status Status() const { return status_; }

 protected:
  void SetFailure(const absl::Status& status) { status_.Update(status); }
  absl::Status status_;
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
  int64_t params_size_in_bytes() const { return params_size_in_bytes_; }
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
  int64_t params_size_in_bytes_;
  ParamsRegions weights_;
  ParamsRegions biases_;
  absl::Status status_;
  MIOpenRnnParamsDescriptor(const MIOpenRnnParamsDescriptor&) = delete;
  void operator=(const MIOpenRnnParamsDescriptor&) = delete;
};

class MIOpenDropoutDescriptor {
 public:
  MIOpenDropoutDescriptor(miopenHandle_t miopen_handle, float dropout,
                          uint64_t seed, ScratchAllocator* state_allocator)
      : dropout_desc_(nullptr) {
    auto status = wrap::miopenCreateDropoutDescriptor(&dropout_desc_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenCreateDropoutDescriptor failed: "
                 << ToString(status);
    }

    if (dropout > 0.0f) {
      DeviceMemory<uint8_t> state_memory;
      if (state_allocator) {
        size_t state_sizes_in_bytes = 0;
        status = wrap::miopenDropoutGetStatesSize(miopen_handle,
                                                  &state_sizes_in_bytes);
        if (status != miopenStatusSuccess) {
          LOG(FATAL) << "call to miopenDropoutGetStatesSize failed: "
                     << ToString(status);
        }
        if (state_sizes_in_bytes > 0) {
          auto allocated = state_allocator->AllocateBytes(state_sizes_in_bytes);
          if (!allocated.ok() ||
              (state_memory = allocated.value()) == nullptr) {
            LOG(FATAL) << "Failed to allocate dropout state space.";
          }
        }
      }

      bool state_evo = false;  // input placeholder, currently not enabled
      bool use_mask = true;
      status = wrap::miopenSetDropoutDescriptor(
          dropout_desc_ /*dropoutDesc*/, miopen_handle /*handle*/,
          dropout /*dropout*/, state_memory.opaque() /*states*/,
          state_memory.size() /*stateSizeInBytes*/, seed /*seed*/,
          use_mask /*use_mask*/, state_evo /*state_evo*/,
          MIOPEN_RNG_PSEUDO_XORWOW /*rng_mode*/);
      if (status != miopenStatusSuccess) {
        LOG(FATAL) << "call to miopenSetDropoutDescriptor failed: "
                   << ToString(status);
      }
    }
  }

  ~MIOpenDropoutDescriptor() {
    auto status = wrap::miopenDestroyDropoutDescriptor(dropout_desc_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenDestroyDropoutDescriptor failed: "
                 << ToString(status);
    }
  }

  miopenDropoutDescriptor_t handle() const { return dropout_desc_; }

 private:
  miopenDropoutDescriptor_t dropout_desc_;

  MIOpenDropoutDescriptor(const MIOpenDropoutDescriptor&) = delete;
  void operator=(const MIOpenDropoutDescriptor&) = delete;
};

class MIOpenRnnDescriptor : public MIOpenDescriptorCommon<dnn::RnnDescriptor> {
 public:
  MIOpenRnnDescriptor(miopenHandle_t miopen_handle, int num_layers,
                      int hidden_size, int input_size,
                      miopenRNNInputMode_t input_mode,
                      miopenRNNDirectionMode_t direction_mode,
                      miopenRNNMode_t rnn_mode, miopenDataType_t data_type,
                      float dropout, uint64_t seed,
                      const dnn::AlgorithmConfig& algorithm_config,
                      ScratchAllocator* state_allocator)
      : rnn_desc_(nullptr),
        num_layers_(num_layers),
        hidden_size_(hidden_size),
        input_size_(input_size),
        input_mode_(input_mode),
        direction_mode_(direction_mode),
        rnn_mode_(rnn_mode),
        data_type_(data_type),
        algorithm_config_(algorithm_config) {
    // Create the dropout handle
    miopen_dropout_desc_.reset(new MIOpenDropoutDescriptor(
        miopen_handle, dropout, seed, state_allocator));
    // Create the RNN handle
    auto status = wrap::miopenCreateRNNDescriptor(&rnn_desc_);
    RETURN_IF_MIOPEN_ERROR(status, "Unable to create RNN descriptor");
    status = wrap::miopenSetRNNDescriptor_V2(
        rnn_desc_ /*rnnDesc*/, hidden_size /*hiddenSize*/,
        num_layers /*numLayers*/,
        miopen_dropout_desc_->handle() /*dropoutDesc*/,
        input_mode /*inputMode*/, direction_mode /*direction*/,
        rnn_mode /*mode*/, miopenRNNwithBias /*biasMode*/,
        miopenRNNdefault /*algo*/, data_type /*dataType*/);
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
  const dnn::AlgorithmConfig& algorithm_config() const {
    return algorithm_config_;
  }
  int64_t ParamsSizeInBytes() const override {
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
  dnn::AlgorithmConfig algorithm_config_;
  absl::Status status_;
  std::unique_ptr<MIOpenDropoutDescriptor> miopen_dropout_desc_;
  std::unique_ptr<MIOpenRnnParamsDescriptor> miopen_params_desc_;
  MIOpenRnnDescriptor(const MIOpenRnnDescriptor&) = delete;
  void operator=(const MIOpenRnnDescriptor&) = delete;
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
      std::string error_msg =
          absl::StrCat("sequence length must be positive: ", seq_length);
      LOG(ERROR) << error_msg;
      SetFailure(absl::UnknownError(error_msg));
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
  absl::Status status_;
  MIOpenRnnSequenceTensorDescriptor(const MIOpenRnnSequenceTensorDescriptor&) =
      delete;
  void operator=(const MIOpenRnnSequenceTensorDescriptor&) = delete;
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
  absl::Status status_;
  miopenDataType_t data_type_;
  MIOpenRnnStateTensorDescriptor(const MIOpenRnnStateTensorDescriptor&) =
      delete;
  void operator=(const MIOpenRnnStateTensorDescriptor&) = delete;
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
  return static_cast<int64_t>(params_size_in_bytes) ==
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
        workspace_allocator->AllocateBytes(workspace_size_in_bytes);
    if (!allocated.ok() || (*workspace = allocated.value()) == nullptr) {
      LOG(ERROR) << "Failed to allocate RNN workspace";

      return false;
    }
    if (!stream->MemZero(workspace, workspace_size_in_bytes).ok()) {
      return false;
    }
  } else {
    *workspace = DeviceMemory<uint8>();
  }
  return true;
}

}  // namespace

template <class T>
absl::Status MIOpenSupport::DoRnnForwardImpl(
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
    return absl::InternalError("ExtractAndCheckRnnForward returned false");
  }

  auto miopen = miopen_->GetHandle(parent_, stream);

  // check params size

  if (!CheckRNNParameterSize(miopen.handle(), rnn_desc, input_desc)) {
    LOG(ERROR) << "Invalid parameters";
    return absl::InternalError("CheckRNNParameterSize returned false");
  }

  // create the workspace
  DeviceMemory<uint8> workspace;
  if (!CreateRnnWorkspace(stream, miopen.handle(), rnn_desc, input_desc,
                          workspace_allocator, &workspace)) {
    LOG(ERROR) << "Unable to create rnn workspace";
    return absl::InternalError("CreateRnnWorkspace returned false");
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
      return absl::InternalError(
          "miopenGetRNNTrainingReserveSize returned failure");
    }

    if (reserve_space_size_in_bytes > 0) {
      auto allocated =
          reserve_space_allocator->AllocateBytes(reserve_space_size_in_bytes);
      if (!allocated.ok() || (reserve_space = allocated.value()) == nullptr) {
        LOG(ERROR) << "Fail to allocate RNN reserve space";
        return absl::InternalError("AllocateBytes for RNN failed");
      }
      TF_RETURN_IF_ERROR(
          stream->MemZero(&reserve_space, reserve_space_size_in_bytes));
    }
  }

  const bool is_profiling = output_profile_result != nullptr;

  TF_ASSIGN_OR_RETURN(
      std::optional<GpuTimer> timer,
      GpuTimer::CreateIfNeeded(
          stream,
          output_profile_result && output_profile_result->warmup_run_executed(),
          is_profiling));

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
      return absl::InternalError("miopenRNNForwardInference failed");
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
      return absl::InternalError("miopenRNNForwardTraining failed");
    }
  }

  if (is_profiling) {
    TF_RETURN_IF_ERROR(PopulateProfileFromTimer(
        timer, *rnn_desc.algorithm_config().algorithm(),
        output_profile_result));
  }

  return absl::OkStatus();
}

template <class T>
absl::Status MIOpenSupport::DoRnnBackwardImpl(
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
    return absl::InternalError("ExtractAndCheckRnnForward failed");
  }

  auto miopen = miopen_->GetHandle(parent_, stream);

  // check params size

  if (!CheckRNNParameterSize(miopen.handle(), rnn_desc, input_desc)) {
    LOG(ERROR) << "Invalid parameters";
    return absl::InternalError("CheckRNNParameterSize failed");
  }

  // create the workspace
  DeviceMemory<uint8> workspace;
  if (!CreateRnnWorkspace(stream, miopen.handle(), rnn_desc, input_desc,
                          workspace_allocator, &workspace)) {
    LOG(ERROR) << "Unable to create rnn workspace";
    return absl::InternalError("CreateRnnWorkspace failed");
  }

  // workaround for missing initialization support in MIOpen.
  // TODO: remove this when MIOpen is ready.
  auto type_size = std::is_same<T, Eigen::half>::value ? 2 : sizeof(T);
  auto size_data = input_desc.seq_length() * input_desc.batch_size() *
                   input_desc.data_size();
  if ((size_data > 0) && (input_backprop_data->opaque() != nullptr))
    TF_RETURN_IF_ERROR(
        stream->MemZero(input_backprop_data, size_data * type_size));

  size_data = input_h_desc.num_layers() * input_h_desc.batch_size() *
              input_h_desc.data_size();
  if ((size_data > 0) && (input_h_backprop_data->opaque() != nullptr))
    TF_RETURN_IF_ERROR(
        stream->MemZero(input_h_backprop_data, size_data * type_size));

  size_data = input_c_desc.num_layers() * input_c_desc.batch_size() *
              input_c_desc.data_size();
  if ((size_data > 0) && (input_c_backprop_data->opaque() != nullptr))
    TF_RETURN_IF_ERROR(
        stream->MemZero(input_c_backprop_data, size_data * type_size));

  const bool is_profiling = output_profile_result != nullptr;

  TF_ASSIGN_OR_RETURN(
      std::optional<GpuTimer> timer,
      GpuTimer::CreateIfNeeded(
          stream,
          output_profile_result && output_profile_result->warmup_run_executed(),
          is_profiling));

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
    return absl::InternalError("miopenRNNBackwardData failed");
  }

  if (params_backprop_data != nullptr) {
    // Clear the dw to zeros.
    TF_RETURN_IF_ERROR(
        stream->MemZero(params_backprop_data, params_backprop_data->size()));
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
      return absl::InternalError("miopenRNNBackwardWeights failed");
    }
  }

  if (is_profiling) {
    TF_RETURN_IF_ERROR(PopulateProfileFromTimer(
        timer, *rnn_desc.algorithm_config().algorithm(),
        output_profile_result));
  }

  return absl::OkStatus();
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
    params_size_in_bytes_ = static_cast<int64_t>(params_size);
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

class MIOpenCTCLossDescriptor {
 public:
  explicit MIOpenCTCLossDescriptor(miopenDataType_t data_type) {
    auto status = wrap::miopenCreateCTCLossDescriptor(&handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenCreateCTCLossDescriptor failed: "
                 << ToString(status);
    }

    bool apply_softmax_layer = true;
    status = wrap::miopenSetCTCLossDescriptor(handle_, data_type, 0,
                                              apply_softmax_layer);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenSetCTCLossDescriptor failed: "
                 << ToString(status);
    }
  }

  ~MIOpenCTCLossDescriptor() {
    auto status = wrap::miopenDestroyCTCLossDescriptor(handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "call to miopenDestroyCTCLossDescriptor failed: "
                 << ToString(status);
    }
  }

  miopenCTCLossDescriptor_t handle() const { return handle_; }

 private:
  miopenCTCLossDescriptor_t handle_;  // Owned

  MIOpenCTCLossDescriptor(const MIOpenCTCLossDescriptor&) = delete;
  void operator=(const MIOpenCTCLossDescriptor&) = delete;
};

absl::Status MIOpenSupport::DoPrepareForCtcLoss(
    Stream* stream, dnn::DataType element_type,
    const dnn::RnnStateTensorDescriptor& probs_desc,
    const dnn::RnnStateTensorDescriptor& grads_desc,
    absl::Span<const int> labels_data,
    absl::Span<const int> labels_lengths_data,
    absl::Span<const int> input_lengths_data,
    const NumericOptions& numeric_options, ScratchAllocator* scratch_allocator,
    DeviceMemory<uint8>* scratch_memory, int* ctc_loss_algo_id) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  MIOpenCTCLossDescriptor miopen_ctc_loss_desc(ToMIOpenDataType(element_type));

  // Query the workspace size.
  size_t workspace_size_in_bytes = 0;

  const MIOpenRnnStateTensorDescriptor& miopen_probs_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(probs_desc);

  const MIOpenRnnStateTensorDescriptor& miopen_grads_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(grads_desc);

  auto status = wrap::miopenGetCTCLossWorkspaceSize(
      miopen.handle(), miopen_probs_desc.handle(), miopen_grads_desc.handle(),
      labels_data.data(), labels_lengths_data.data(), input_lengths_data.data(),
      MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC, miopen_ctc_loss_desc.handle(),
      &workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "call to miopenDestroyCTCLossDescriptor failed: "
               << ToString(status);
    return absl::InternalError(
        "Failed to determine scratch memory size for MIOpen CTC Loss");
  }

  *scratch_memory = DeviceMemory<uint8>();

  // Allocate the workspace.
  if (workspace_size_in_bytes != 0) {
    if (scratch_allocator == nullptr) {
      return absl::InternalError(
          "An allocator must be specified when scratch memory is needed");
    }
    auto scratch_or = scratch_allocator->AllocateBytes(workspace_size_in_bytes);
    if (scratch_or.ok()) {
      *scratch_memory = scratch_or.value();
    } else {
      LOG(ERROR)
          << "Failed to allocate scratch memory - "
          << scratch_or.status().message() << "\n"
          << "\tYou can set the env var TF_CUDNN_WORKSPACE_LIMIT_IN_MB to a "
             "larger number (e.g. 8192) to increase the max memory limit.\n"
          << "\tIncreasing the max memory limit might help resolve this "
             "error";
      return absl::InternalError(absl::StrCat(
          "Failed to allocate scratch memory for MIOpen CTC Loss, of size: ",
          workspace_size_in_bytes));
    }
  }

  return absl::OkStatus();
}

absl::Status MIOpenSupport::DoCtcLossImpl(
    Stream* stream, const MIOpenRnnStateTensorDescriptor& probs_desc,
    const DeviceMemoryBase probs_data, absl::Span<const int> labels_data,
    absl::Span<const int> labels_lengths_data,
    absl::Span<const int> input_lengths_data, DeviceMemoryBase costs_data,
    const MIOpenRnnStateTensorDescriptor& grads_desc,
    DeviceMemoryBase grads_data, const MIOpenCTCLossDescriptor& ctc_loss_desc,
    DeviceMemory<uint8> scratch_memory, int ctc_loss_algo_id) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  int kNumTimestamps = probs_desc.num_layers();
  int kBatchSize = probs_desc.batch_size();
  int kNumLabels = probs_desc.data_size();
  int total_size = kNumLabels * kNumTimestamps * kBatchSize;
  (void)total_size;

  auto status = wrap::miopenCTCLoss(
      miopen.handle(), probs_desc.handle(), probs_data.opaque(),
      labels_data.data(), labels_lengths_data.data(), input_lengths_data.data(),
      costs_data.opaque(), grads_desc.handle(), grads_data.opaque(),
      MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_loss_desc.handle(),
      scratch_memory.opaque(), scratch_memory.size());
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "call to miopenCTCLoss failed: " << ToString(status);
    return absl::InternalError("Failure during MIOpen CTC Loss");
  }

  return absl::OkStatus();
}

absl::Status MIOpenSupport::DoCtcLoss(
    Stream* stream, dnn::DataType element_type,
    const dnn::RnnStateTensorDescriptor& probs_desc,
    const DeviceMemoryBase probs_data, absl::Span<const int> labels_data,
    absl::Span<const int> labels_lengths_data,
    absl::Span<const int> input_lengths_data, DeviceMemoryBase costs_data,
    const dnn::RnnStateTensorDescriptor& grads_desc,
    DeviceMemoryBase grads_data, DeviceMemory<uint8> scratch_memory,
    int ctc_loss_algo_id) {
  // Current MIOPen CTC Loss only supports the float datatype
  if (element_type != dnn::DataType::kFloat) {
    return absl::InvalidArgumentError(
        "MIOpenCTCLossDescriptor is supported only when the "
        "DataType is float");
  }

  MIOpenCTCLossDescriptor miopen_ctc_loss_desc(ToMIOpenDataType(element_type));

  const MIOpenRnnStateTensorDescriptor& miopen_probs_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(probs_desc);

  const MIOpenRnnStateTensorDescriptor& miopen_grads_desc =
      static_cast<const MIOpenRnnStateTensorDescriptor&>(grads_desc);

  return DoCtcLossImpl(stream, miopen_probs_desc, probs_data, labels_data,
                       labels_lengths_data, input_lengths_data, costs_data,
                       miopen_grads_desc, grads_data, miopen_ctc_loss_desc,
                       scratch_memory, ctc_loss_algo_id);
}

absl::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
MIOpenSupport::CreateRnnDescriptor(
    int num_layers, int hidden_size, int input_size, int cell_size,
    int batch_size, dnn::RnnInputMode input_mode,
    dnn::RnnDirectionMode direction_mode, dnn::RnnMode rnn_mode,
    dnn::DataType data_type, const dnn::AlgorithmConfig& algorithm_config,
    const NumericOptions& numeric_options, float dropout, uint64_t seed,
    ScratchAllocator* state_allocator, bool use_padded_io) {
  // ROCM TODO: batch_size is used in dynamic persistent RNN algorithm and is
  // not supported by MIOpen now.
  if (use_padded_io) {
    return absl::InvalidArgumentError(
        "ROCm MIOpen only supports packed input output.");
  }

  bool use_projection = cell_size != 0 && hidden_size < cell_size;
  if (use_projection) {
    return absl::InvalidArgumentError(
        "ROCm MIOpen does not support RNN ProjectionLayers yet.");
  }

  auto miopen = miopen_->GetHandle(parent_, nullptr);
  std::unique_ptr<MIOpenRnnDescriptor> rnn_desc(new MIOpenRnnDescriptor(
      miopen.handle(), num_layers, hidden_size, input_size,
      ToMIOpenRnnInputMode(input_mode),
      ToMIOpenRnnDirectionMode(direction_mode), ToMIOpenRnnMode(rnn_mode),
      ToMIOpenDataType(data_type), dropout, seed, algorithm_config,
      state_allocator));
  if (!rnn_desc->ok()) {
    return rnn_desc->Status();
  }
  return absl::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>(
      std::move(rnn_desc));
}

absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
MIOpenSupport::CreateRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                                 int data_size,
                                                 dnn::DataType data_type) {
  std::unique_ptr<MIOpenRnnSequenceTensorDescriptor> seq_desc(
      new MIOpenRnnSequenceTensorDescriptor(seq_length, batch_size, data_size,
                                            ToMIOpenDataType(data_type)));
  if (!seq_desc->ok()) {
    return seq_desc->Status();
  }
  return absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>(
      std::move(seq_desc));
}

absl::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
MIOpenSupport::CreateRnnStateTensorDescriptor(int num_layer, int batch_size,
                                              int data_size,
                                              dnn::DataType data_type) {
  std::unique_ptr<MIOpenRnnStateTensorDescriptor> state_desc(
      new MIOpenRnnStateTensorDescriptor(num_layer, batch_size, data_size,
                                         ToMIOpenDataType(data_type)));
  if (!state_desc->ok()) {
    return state_desc->Status();
  }
  return absl::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>(
      std::move(state_desc));
}

bool MIOpenSupport::DoRnnForward(
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

  return IsStatusOk(
      DoRnnForwardImpl<Eigen::half>(
          stream, miopen_rnn_desc, miopen_input_desc, input_data,
          miopen_input_h_desc, input_h_data, miopen_input_c_desc, input_c_data,
          params, miopen_output_desc, output_data, miopen_output_h_desc,
          output_h_data, miopen_output_c_desc, output_c_data, is_training,
          reserve_space_allocator, workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool MIOpenSupport::DoRnnForward(
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

  return IsStatusOk(
      DoRnnForwardImpl<float>(
          stream, miopen_rnn_desc, miopen_input_desc, input_data,
          miopen_input_h_desc, input_h_data, miopen_input_c_desc, input_c_data,
          params, miopen_output_desc, output_data, miopen_output_h_desc,
          output_h_data, miopen_output_c_desc, output_c_data, is_training,
          reserve_space_allocator, workspace_allocator, output_profile_result),
      /*report_error=*/!output_profile_result);
}

bool MIOpenSupport::DoRnnForward(
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
  LOG(ERROR) << "miopen does not support double type RNN fwd yet";
  return false;
}

bool MIOpenSupport::DoRnnBackward(
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
    DeviceMemory<uint8>* reserve_space_data,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
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

  return IsStatusOk(
      DoRnnBackwardImpl<Eigen::half>(
          stream, miopen_rnn_desc, miopen_input_desc, input_data,
          miopen_input_h_desc, input_h_data, miopen_input_c_desc, input_c_data,
          params, miopen_output_desc, output_data, miopen_output_h_desc,
          output_h_data, miopen_output_c_desc, output_c_data,
          output_backprop_data, output_h_backprop_data, output_c_backprop_data,
          input_backprop_data, input_h_backprop_data, input_c_backprop_data,
          params_backprop_data, reserve_space_data, workspace_allocator,
          output_profile_result),
      /*report_error=*/true);
}

bool MIOpenSupport::DoRnnBackward(
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
    DeviceMemory<uint8>* reserve_space_data,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
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

  return IsStatusOk(
      DoRnnBackwardImpl<float>(
          stream, miopen_rnn_desc, miopen_input_desc, input_data,
          miopen_input_h_desc, input_h_data, miopen_input_c_desc, input_c_data,
          params, miopen_output_desc, output_data, miopen_output_h_desc,
          output_h_data, miopen_output_c_desc, output_c_data,
          output_backprop_data, output_h_backprop_data, output_c_backprop_data,
          input_backprop_data, input_h_backprop_data, input_c_backprop_data,
          params_backprop_data, reserve_space_data, workspace_allocator,
          output_profile_result),
      /*report_error=*/true);
}

bool MIOpenSupport::DoRnnBackward(
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
    DeviceMemory<uint8>* reserve_space_data,
    ScratchAllocator* workspace_allocator,
    dnn::ProfileResult* output_profile_result) {
  LOG(ERROR) << "miopen does not support half type RNN bwd yet";
  return false;
}

// This is the context required to use the TF scratch allocator:
struct MIOpenAllocatorContext {
  MIOpenAllocatorContext(ScratchAllocator* scratch_allocator, Stream* stream)
      : scratch_allocator_(scratch_allocator), stream_(stream) {}

  ScratchAllocator* scratch_allocator_;
  Stream* stream_;
};

void* MIOpenAllocatorCallback(void* ctx, size_t size_in_bytes) {
  auto* mac = static_cast<MIOpenAllocatorContext*>(ctx);
  auto allocated = mac->scratch_allocator_->AllocateBytes(size_in_bytes);

  DeviceMemory<uint8> scratch;
  if (allocated.ok()) {
    scratch = allocated.value();
    return scratch.opaque();
  } else {
    return nullptr;
  }
}

void MIOpenDeallocatorCallback(void* ctx, void* mem) {
  // Don't need deallocator since the TensorFlow heap will automatically
  // reclaim the memory
}

absl::Status MIOpenSupport::DoPrepareForConvolution(
    dnn::ConvolutionKind kind, dnn::DataType element_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::AlgorithmConfig& algorithm_config,
    ScratchAllocator* scratch_allocator, dnn::AlgorithmDesc* algorithm_desc,
    DeviceMemory<uint8>* scratch_memory) {
  std::optional<dnn::AlgorithmDesc> input_algo_desc =
      algorithm_config.algorithm();

  assert(input_algo_desc.has_value());

  // An algorithm has been specified.
  *algorithm_desc = *input_algo_desc;

  assert(algorithm_config.scratch_size().has_value());

  size_t scratch_memory_size = *(algorithm_config.scratch_size());

  // allocate scratch memory
  if (scratch_memory_size != 0) {
    if (scratch_allocator == nullptr) {
      return absl::InternalError(
          "An allocator must be specified when scratch memory is needed");
    }
    auto allocated = scratch_allocator->AllocateBytes(scratch_memory_size);
    if (allocated.ok()) {
      *scratch_memory = allocated.value();
    } else {
      LOG(ERROR)
          << "Failed to allocate scratch memory - "
          << allocated.status().message() << "\n"
          << "\tYou can set the env var TF_CUDNN_WORKSPACE_LIMIT_IN_MB to a "
             "larger number (e.g. 8192) to increase the max memory limit.\n"
          << "\tIncreasing the max memory limit might help resolve this "
             "error";
      return absl::InternalError(absl::StrCat(
          "Failed to allocate scratch memory of size: ", scratch_memory_size));
    }
  }

  return absl::OkStatus();
}

class RocmConvRunner : public dnn::ConvRunner {
 public:
  RocmConvRunner(GpuExecutor* parent, MIOpenAccess* miopen, int64_t algo_id,
                 size_t workspace_size, dnn::ConvolutionKind kind,
                 dnn::DataType input_type, dnn::DataType output_type,
                 bool use_immediate_mode,
                 ScopedTensorDescriptor& scoped_input_desc,
                 ScopedTensorDescriptor& scoped_output_desc,
                 ScopedFilterDescriptor& scoped_filter_desc,
                 ScopedConvolutionDescriptor& scoped_conv_desc)
      : parent_(parent),
        miopen_(miopen),
        algo_id_(algo_id),
        workspace_size_(workspace_size),
        kind_(kind),
        use_immediate_mode_(use_immediate_mode),
        input_desc_(std::move(scoped_input_desc)),
        output_desc_(std::move(scoped_output_desc)),
        filter_desc_(std::move(scoped_filter_desc)),
        conv_desc_(std::move(scoped_conv_desc)) {
    bool is_backprop = ((kind == dnn::ConvolutionKind::BACKWARD_DATA) ||
                        (kind == dnn::ConvolutionKind::BACKWARD_FILTER));
    // #if TF_ROCM_VERSION >= 50000
    if (is_backprop && (ToMIOpenDataType(input_type) == miopenHalf)) {
      wrap::miopenSetConvolutionAttribute(
          conv_desc_.handle(), MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL, 1);
    }
    // #endif
  }

  std::string ToString() const override {
    return dnn::AlgorithmDesc{algo_id_, false, workspace_size_}.ToString();
  }

  size_t GetWorkspaceSize() const override { return workspace_size_; }

  absl::StatusOr<AlgorithmDesc> ToAlgorithmDesc() const override {
    return {{algo_id_, false, workspace_size_}};
  }

  absl::Status operator()(Stream* stream,
                          dnn::ProfileResult* output_profile_result,
                          DeviceMemoryBase scratch_memory,
                          DeviceMemoryBase input_data,
                          DeviceMemoryBase filter_data,
                          DeviceMemoryBase output_data) const override {
    auto miopen = miopen_->GetHandle(parent_, stream);
    // Alpha is the scaling factor for input.
    float alpha = 1.0;
    // Beta is the scaling factor for output.
    float beta = 0.0;

    const bool is_profiling = output_profile_result != nullptr;
    TF_ASSIGN_OR_RETURN(std::optional<GpuTimer> timer,
                        GpuTimer::CreateIfNeeded(
                            stream,
                            output_profile_result &&
                                output_profile_result->warmup_run_executed(),
                            is_profiling));

    miopenStatus_t status = miopenStatusSuccess;
    switch (kind_) {
      case dnn::ConvolutionKind::FORWARD: {
        if (use_immediate_mode_) {
          status = wrap::miopenConvolutionForwardImmediate(
              miopen.handle(), filter_desc_.handle(), filter_data.opaque(),
              input_desc_.handle(), input_data.opaque(), conv_desc_.handle(),
              output_desc_.handle(), output_data.opaque(),
              scratch_memory.opaque(), scratch_memory.size(),
              static_cast<uint64_t>(algo_id_));
        } else {
          status = wrap::miopenConvolutionForward(
              miopen.handle(), &alpha, input_desc_.handle(),
              input_data.opaque(), filter_desc_.handle(), filter_data.opaque(),
              conv_desc_.handle(),
              static_cast<miopenConvFwdAlgorithm_t>(algo_id_), &beta,
              output_desc_.handle(), output_data.opaque(),
              scratch_memory.opaque(), scratch_memory.size());
        }

        break;
      }
      case dnn::ConvolutionKind::BACKWARD_DATA: {
        if (use_immediate_mode_) {
          status = wrap::miopenConvolutionBackwardDataImmediate(
              miopen.handle(), output_desc_.handle(), output_data.opaque(),
              filter_desc_.handle(), filter_data.opaque(), conv_desc_.handle(),
              input_desc_.handle(), input_data.opaque(),
              scratch_memory.opaque(), scratch_memory.size(),
              static_cast<uint64_t>(algo_id_));
        } else {
          status = wrap::miopenConvolutionBackwardData(
              miopen.handle(), &alpha, output_desc_.handle(),
              output_data.opaque(), filter_desc_.handle(), filter_data.opaque(),
              conv_desc_.handle(),
              static_cast<miopenConvBwdDataAlgorithm_t>(algo_id_), &beta,
              input_desc_.handle(), input_data.opaque(),
              scratch_memory.opaque(), scratch_memory.size());
        }
        break;
      }
      case dnn::ConvolutionKind::BACKWARD_FILTER: {
        if (use_immediate_mode_) {
          status = wrap::miopenConvolutionBackwardWeightsImmediate(
              miopen.handle(), output_desc_.handle(), output_data.opaque(),
              input_desc_.handle(), input_data.opaque(), conv_desc_.handle(),
              filter_desc_.handle(), filter_data.opaque(),
              scratch_memory.opaque(), scratch_memory.size(),
              static_cast<uint64_t>(algo_id_));
        } else {
          status = wrap::miopenConvolutionBackwardWeights(
              miopen.handle(), &alpha, output_desc_.handle(),
              output_data.opaque(), input_desc_.handle(), input_data.opaque(),
              conv_desc_.handle(),
              static_cast<miopenConvBwdWeightsAlgorithm_t>(algo_id_), &beta,
              filter_desc_.handle(), filter_data.opaque(),
              scratch_memory.opaque(), scratch_memory.size());
        }
        break;
      }
      default:
        return absl::InternalError(absl::StrCat("Unexpected convolution kind ",
                                                static_cast<int>(kind_)));
    }

    if (is_profiling) {
      if (status == miopenStatusSuccess) {
        TF_ASSIGN_OR_RETURN(absl::Duration elapsed,
                            timer->GetElapsedDuration());
        output_profile_result->set_elapsed_time_in_ms(
            absl::ToDoubleMilliseconds(elapsed));
        dnn::AlgorithmDesc algotype(algo_id_, false);
        output_profile_result->set_algorithm(algotype);
        output_profile_result->set_scratch_size(scratch_memory.size());
      }
    }

    if (status != miopenStatusSuccess) {
      return absl::InternalError(
          absl::StrCat("Failed to enqueue convolution on stream: ",
                       ::stream_executor::gpu::ToString(status)));
    }

    return absl::OkStatus();
  }

 private:
  GpuExecutor* parent_;
  MIOpenAccess* miopen_;
  int64_t algo_id_;
  size_t workspace_size_;
  dnn::ConvolutionKind kind_;
  bool use_immediate_mode_;

  ScopedTensorDescriptor input_desc_;
  ScopedTensorDescriptor output_desc_;
  ScopedFilterDescriptor filter_desc_;
  ScopedConvolutionDescriptor conv_desc_;
};

absl::Status MIOpenSupport::DoConvolve(
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::AlgorithmDesc algorithm_desc, DeviceMemory<uint8> scratch_memory,
    dnn::ProfileResult* output_profile_result) {
  TF_ASSIGN_OR_RETURN(
      auto runner,
      ConvolveRunnerFromDesc(stream, algorithm_desc, kind, element_type,
                             output_type, input_descriptor, filter_descriptor,
                             output_descriptor, convolution_descriptor));

  return (*runner)(stream, output_profile_result, scratch_memory, input_data,
                   filter_data, output_data);
}

absl::Status MIOpenSupport::GetConvolveRunners(
    bool use_cudnn_frontend, dnn::ConvolutionKind kind,
    dnn::DataType input_type, dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    ScratchAllocator* scratch_allocator, const NumericOptions& numeric_options,
    std::vector<std::unique_ptr<const dnn::ConvRunner>>* out_runners) {
  if (input_type != output_type) {
    return absl::UnimplementedError(
        absl::StrFormat("MIOpen backend does not support different input and "
                        "output types: %d != %d",
                        input_type, output_type));
  }

  std::vector<dnn::ProfileResult> profile_results;
  if (!GetMIOpenConvolveAlgorithms(
          kind, input_type, output_type, stream, input_descriptor, input_data,
          filter_descriptor, filter_data, output_descriptor, output_data,
          convolution_descriptor, scratch_allocator, &profile_results))
    return absl::InternalError("GetMIOpenConvolveAlgorithms failure");

  for (const auto& profile_result : profile_results) {
    TF_ASSIGN_OR_RETURN(
        auto runner, ConvolveRunnerFromDesc(
                         stream, profile_result.algorithm(), kind, input_type,
                         output_type, input_descriptor, filter_descriptor,
                         output_descriptor, convolution_descriptor));
    out_runners->push_back(std::move(runner));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>>
MIOpenSupport::ConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor) {
  auto workspace_size = algorithm_desc.workspace_size();
  TF_ASSIGN_OR_RETURN(auto scoped_input_desc,
                      scope(input_descriptor, ToMIOpenDataType(input_type)));
  TF_ASSIGN_OR_RETURN(auto scoped_output_desc,
                      scope(output_descriptor, ToMIOpenDataType(output_type)));
  TF_ASSIGN_OR_RETURN(auto scoped_filter_desc,
                      scope(filter_descriptor, ToMIOpenDataType(input_type)));
  TF_ASSIGN_OR_RETURN(auto scoped_conv_desc, scope(convolution_descriptor));

  return {std::make_unique<RocmConvRunner>(
      parent_, miopen_.get(), algorithm_desc.algo_id(), *workspace_size, kind,
      input_type, output_type, use_immediate_mode_, scoped_input_desc,
      scoped_output_desc, scoped_filter_desc, scoped_conv_desc)};
}

bool MIOpenSupport::GetMIOpenConvolveAlgorithms(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    ScratchAllocator* scratch_allocator,
    std::vector<dnn::ProfileResult>* out_algorithms) {
  return use_immediate_mode_
             ? GetMIOpenConvolveAlgorithmsImmediateMode(
                   kind, input_type, output_type, stream, input_descriptor,
                   input_data, filter_descriptor, filter_data,
                   output_descriptor, output_data, convolution_descriptor,
                   scratch_allocator, out_algorithms)
                   .ok()
             : GetMIOpenConvolveAlgorithmsFindMode(
                   kind, input_type, output_type, stream, input_descriptor,
                   input_data, filter_descriptor, filter_data,
                   output_descriptor, output_data, convolution_descriptor,
                   scratch_allocator, out_algorithms)
                   .ok();
}

absl::Status MIOpenSupport::GetMIOpenConvolveAlgorithmsImmediateMode(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    ScratchAllocator* scratch_allocator,
    std::vector<dnn::ProfileResult>* out_algorithms) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  TF_ASSIGN_OR_RETURN(auto input_nd,
                      scope(input_descriptor, ToMIOpenDataType(input_type)));
  TF_ASSIGN_OR_RETURN(auto output_nd,
                      scope(output_descriptor, ToMIOpenDataType(output_type)));
  TF_ASSIGN_OR_RETURN(auto filter,
                      scope(filter_descriptor, ToMIOpenDataType(input_type)));
  TF_ASSIGN_OR_RETURN(auto conv, scope(convolution_descriptor));

  bool is_backprop = ((kind == dnn::ConvolutionKind::BACKWARD_DATA) ||
                      (kind == dnn::ConvolutionKind::BACKWARD_FILTER));
  // bool is_backprop = (call_context == dnn::CallContext::kBackpropData) ||
  //                   (call_context == dnn::CallContext::kBackpropFilter);

#if TF_ROCM_VERSION >= 50000
  if (is_backprop && (ToMIOpenDataType(input_type) == miopenHalf)) {
    wrap::miopenSetConvolutionAttribute(
        conv.handle(), MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL, 1);
  }
#endif
  // First determine the number of algorithms available
  size_t maxSolutionCount = 0;

  switch (kind) {
    case dnn::ConvolutionKind::FORWARD:
    case dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION: {
      auto status = wrap::miopenConvolutionForwardGetSolutionCount(
          miopen.handle(), filter.handle(), input_nd.handle(), conv.handle(),
          output_nd.handle(), &maxSolutionCount);
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenConvolutionForwardGetSolutionCount failed: " +
            ToString(status));
      }
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_DATA: {
      auto status = wrap::miopenConvolutionBackwardDataGetSolutionCount(
          miopen.handle(), output_nd.handle(), filter.handle(), conv.handle(),
          input_nd.handle(), &maxSolutionCount);
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenConvolutionBackwardDataGetSolutionCount "
            "failed: " +
            ToString(status));
      }
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_FILTER: {
      auto status = wrap::miopenConvolutionBackwardWeightsGetSolutionCount(
          miopen.handle(), output_nd.handle(), input_nd.handle(), conv.handle(),
          filter.handle(), &maxSolutionCount);
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenConvolutionBackwardWeightsGetSolutionCount "
            "failed: " +
            ToString(status));
      }
      break;
    }
    default: {
      return absl::InternalError("Unexpected convolution kind " +
                                 std::to_string(static_cast<int>(kind)));
    }
  }

  VLOG(kConvDebugVlogLevel)
      << "Number of conv solutions max: " << maxSolutionCount;

  if (return_best_algo_only_) {
    VLOG(kConvDebugVlogLevel) << "TF_ROCM_RETURN_BEST_ALGO_ONLY is set, "
                              << "setting maxSolutionCount to 1";
    maxSolutionCount = 1;
  }

  size_t solutionCount = 0;
  std::unique_ptr<miopenConvSolution_t[]> solutions(
      new miopenConvSolution_t[maxSolutionCount]);

  switch (kind) {
    case dnn::ConvolutionKind::FORWARD: {
      auto status = wrap::miopenConvolutionForwardGetSolution(
          miopen.handle(), filter.handle(), input_nd.handle(), conv.handle(),
          output_nd.handle(), maxSolutionCount, &solutionCount,
          solutions.get());

      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenConvolutionForwardGetSolution failed: " +
            ToString(status));
      }

      VLOG(kConvDebugVlogLevel)
          << "Number of conv solutions actual: " << solutionCount;

      for (size_t i = 0; i < solutionCount; i++) {
        miopenConvSolution_t solution = solutions[i];

        VLOG(kConvDebugVlogLevel)
            << "solution " << i << " (time, mem, id, algo) =  " << solution.time
            << ", " << solution.workspace_size << ", " << solution.solution_id
            << ", " << ToString(solution.algorithm);

        status = wrap::miopenConvolutionForwardCompileSolution(
            miopen.handle(), filter.handle(), input_nd.handle(), conv.handle(),
            output_nd.handle(), solution.solution_id);

        if (status != miopenStatusSuccess) {
          return absl::InternalError(
              "call to miopenConvolutionForwardCompileSolution failed: " +
              ToString(status));
        }

        out_algorithms->emplace_back(
            GetProfileResultFromConvSolution(solution));
      }
      break;
    }

    case dnn::ConvolutionKind::BACKWARD_DATA: {
      auto status = wrap::miopenConvolutionBackwardDataGetSolution(
          miopen.handle(), output_nd.handle(), filter.handle(), conv.handle(),
          input_nd.handle(), maxSolutionCount, &solutionCount, solutions.get());
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenConvolutionBackwardDataGetSolution failed: " +
            ToString(status));
      }

      VLOG(kConvDebugVlogLevel)
          << "Number of conv solutions actual: " << solutionCount;

      for (size_t i = 0; i < solutionCount; i++) {
        miopenConvSolution_t solution = solutions[i];

        VLOG(kConvDebugVlogLevel)
            << "solution " << i << " (time, mem, id, algo) =  " << solution.time
            << ", " << solution.workspace_size << ", " << solution.solution_id
            << ", " << ToString(solution.algorithm);

        status = wrap::miopenConvolutionBackwardDataCompileSolution(
            miopen.handle(), output_nd.handle(), filter.handle(), conv.handle(),
            input_nd.handle(), solution.solution_id);

        if (status != miopenStatusSuccess) {
          return absl::InternalError(
              " call to miopenConvolutionBackwardDataCompileSolution "
              "failed: " +
              ToString(status));
        }

        out_algorithms->emplace_back(
            GetProfileResultFromConvSolution(solution));
      }
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_FILTER: {
      auto status = wrap::miopenConvolutionBackwardWeightsGetSolution(
          miopen.handle(), output_nd.handle(), input_nd.handle(), conv.handle(),
          filter.handle(), maxSolutionCount, &solutionCount, solutions.get());
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenConvolutionBackwardWeightsGetSolution failed: " +
            ToString(status));
      }

      VLOG(kConvDebugVlogLevel)
          << "Number of conv solutions actual: " << solutionCount;

      for (size_t i = 0; i < solutionCount; i++) {
        miopenConvSolution_t solution = solutions[i];

        VLOG(kConvDebugVlogLevel)
            << "solution " << i << " (time, mem, id, algo) =  " << solution.time
            << ", " << solution.workspace_size << ", " << solution.solution_id
            << ", " << ToString(solution.algorithm);

        status = wrap::miopenConvolutionBackwardWeightsCompileSolution(
            miopen.handle(), output_nd.handle(), input_nd.handle(),
            conv.handle(), filter.handle(), solution.solution_id);

        if (status != miopenStatusSuccess) {
          return absl::InternalError(
              "call to miopenConvolutionBackwardWeightsCompileSolution "
              "failed: " +
              ToString(status));
        }

        out_algorithms->emplace_back(
            GetProfileResultFromConvSolution(solution));
      }
      break;
    }
    default: {
      return absl::InternalError("Unexpected convolution kind " +
                                 std::to_string(static_cast<int>(kind)));
    }
  }

  return absl::OkStatus();
}

absl::Status MIOpenSupport::GetMIOpenConvolveAlgorithmsFindMode(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    ScratchAllocator* scratch_allocator,
    std::vector<dnn::ProfileResult>* out_algorithms) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  TF_ASSIGN_OR_RETURN(auto input_nd,
                      scope(input_descriptor, ToMIOpenDataType(input_type)));
  TF_ASSIGN_OR_RETURN(auto output_nd,
                      scope(output_descriptor, ToMIOpenDataType(output_type)));
  TF_ASSIGN_OR_RETURN(auto filter,
                      scope(filter_descriptor, ToMIOpenDataType(input_type)));
  TF_ASSIGN_OR_RETURN(auto conv, scope(convolution_descriptor));

  bool is_backprop = ((kind == dnn::ConvolutionKind::BACKWARD_DATA) ||
                      (kind == dnn::ConvolutionKind::BACKWARD_FILTER));
  // bool is_backprop = (call_context == dnn::CallContext::kBackpropData) ||
  //                    (call_context == dnn::CallContext::kBackpropFilter);

#if TF_ROCM_VERSION >= 50000
  if (is_backprop && (ToMIOpenDataType(input_type) == miopenHalf)) {
    wrap::miopenSetConvolutionAttribute(
        conv.handle(), MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL, 1);
  }
#endif

  // Determine the workspace memory size that will need by the call to Find
  size_t scratch_memory_size = 0;
  switch (kind) {
    case dnn::ConvolutionKind::FORWARD:
    case dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION: {
      auto status = wrap::miopenConvolutionForwardGetWorkSpaceSize(
          miopen.handle(), filter.handle(), input_nd.handle(), conv.handle(),
          output_nd.handle(), &scratch_memory_size);
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenConvolutionForwardGetWorkspaceSize failed: " +
            ToString(status));
      }
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_DATA: {
      auto status = wrap::miopenConvolutionBackwardDataGetWorkSpaceSize(
          miopen.handle(), output_nd.handle(), filter.handle(), conv.handle(),
          input_nd.handle(), &scratch_memory_size);
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenConvolutionBackwardDataGetWorkspaceSize failed: " +
            ToString(status));
      }
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_FILTER: {
      auto status = wrap::miopenConvolutionBackwardWeightsGetWorkSpaceSize(
          miopen.handle(), output_nd.handle(), input_nd.handle(), conv.handle(),
          filter.handle(), &scratch_memory_size);
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenConvolutionBackwardWeightsGetWorkspaceSize "
            "failed: " +
            ToString(status));
      }
      break;
    }
    default: {
      return absl::InternalError(absl::StrFormat(
          "Unexpected convolution kind %d", static_cast<int>(kind)));
      break;
    }
  }

  // allocate scratch memory
  DeviceMemory<uint8> scratch_memory;
  if (scratch_memory_size != 0) {
    if (scratch_allocator == nullptr) {
      return absl::InternalError(
          "An allocator must be specified "
          "when scratch memory is needed");
    }
    auto allocated = scratch_allocator->AllocateBytes(scratch_memory_size);
    if (allocated.ok()) {
      scratch_memory = allocated.value();
    } else {
      LOG(FATAL)
          << "Failed to allocate scratch memory - "
          << allocated.status().message() << "\n"
          << "\tYou can set the env var TF_CUDNN_WORKSPACE_LIMIT_IN_MB to a "
             "larger number (e.g. 8192) to increase the max memory limit.\n"
          << "\tIncreasing the max memory limit might help resolve this "
             "error";
      return absl::InternalError("Out of memory");
    }
  }

  // Only get the best algorithm for Find Mode
  size_t requestedAlgorithmCount = 1;

  VLOG(kConvDebugVlogLevel)
      << "Number of conv algortihms to request: " << requestedAlgorithmCount;

  miopenConvAlgoPerf_t returnedAlgorithm;

  int returnedAlgorithmCount = 0;
  bool exhaustiveSearch = false;

  switch (kind) {
    case dnn::ConvolutionKind::FORWARD:
    case dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION: {
      auto status = wrap::miopenFindConvolutionForwardAlgorithm(
          miopen.handle(), input_nd.handle(), input_data.opaque(),
          filter.handle(), filter_data.opaque(), conv.handle(),
          output_nd.handle(), output_data.opaque(), requestedAlgorithmCount,
          &returnedAlgorithmCount, &returnedAlgorithm, scratch_memory.opaque(),
          scratch_memory_size, exhaustiveSearch);
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenFindConvolutionForwardAlgorithm failed: " +
            ToString(status));
      }
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_DATA: {
      auto status = wrap::miopenFindConvolutionBackwardDataAlgorithm(
          miopen.handle(), output_nd.handle(), output_data.opaque(),
          filter.handle(), filter_data.opaque(), conv.handle(),
          input_nd.handle(), input_data.opaque(), requestedAlgorithmCount,
          &returnedAlgorithmCount, &returnedAlgorithm, scratch_memory.opaque(),
          scratch_memory_size, exhaustiveSearch);
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenFindConvolutionBackwardDataAlgorithm failed: " +
            ToString(status));
      }
      break;
    }
    case dnn::ConvolutionKind::BACKWARD_FILTER: {
      auto status = wrap::miopenFindConvolutionBackwardWeightsAlgorithm(
          miopen.handle(), output_nd.handle(), output_data.opaque(),
          input_nd.handle(), input_data.opaque(), conv.handle(),
          filter.handle(), filter_data.opaque(), requestedAlgorithmCount,
          &returnedAlgorithmCount, &returnedAlgorithm, scratch_memory.opaque(),
          scratch_memory_size, exhaustiveSearch);
      if (status != miopenStatusSuccess) {
        return absl::InternalError(
            "call to miopenConvolutionBackwardWeightsAlgorithm "
            "failed: " +
            ToString(status));
      }
      break;
    }
    default: {
      return absl::InternalError("Unexpected convolution kind " +
                                 std::to_string(static_cast<int>(kind)));
      break;
    }
  }

  out_algorithms->emplace_back(
      GetProfileResultFromConvAlgoPerf(kind, returnedAlgorithm));

  return absl::OkStatus();
}

bool MIOpenSupport::GetRnnAlgorithms(
    std::vector<dnn::AlgorithmDesc>* out_algorithms) {
  std::vector<dnn::AlgorithmDesc::Index> algo_types = {
      // clang-format off
    miopenRNNdefault,
      // clang-format on
  };

  out_algorithms->clear();
  for (auto i : algo_types) {
    out_algorithms->push_back({i, /*use_tensor_ops=*/false});
  }
  return true;
}

bool MIOpenSupport::DoBatchNormalizationForward(
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
  return DoBatchNormalizationForwardImpl<Eigen::bfloat16, float>(
             stream, dnn::DataType::kBF16, dnn::DataType::kFloat, x, scale,
             offset, estimated_mean, estimated_variance, side_input, x_desc,
             scale_offset_desc, epsilon, exponential_average_factor,
             activation_mode, y, batch_mean, batch_var, saved_mean,
             saved_inv_var, is_training)
      .ok();
}

bool MIOpenSupport::DoBatchNormalizationForward(
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
  return DoBatchNormalizationForwardImpl<Eigen::half, float>(
             stream, dnn::DataType::kHalf, dnn::DataType::kFloat, x, scale,
             offset, estimated_mean, estimated_variance, side_input, x_desc,
             scale_offset_desc, epsilon, exponential_average_factor,
             activation_mode, y, batch_mean, batch_var, saved_mean,
             saved_inv_var, is_training)
      .ok();
}

bool MIOpenSupport::DoBatchNormalizationForward(
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
  return DoBatchNormalizationForwardImpl<float, float>(
             stream, dnn::DataType::kFloat, dnn::DataType::kFloat, x, scale,
             offset, estimated_mean, estimated_variance, side_input, x_desc,
             scale_offset_desc, epsilon, exponential_average_factor,
             activation_mode, y, batch_mean, batch_var, saved_mean,
             saved_inv_var, is_training)
      .ok();
}

template <class T, class U>
absl::Status MIOpenSupport::DoBatchNormalizationForwardImpl(
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
    bool is_training) {
  auto miopen = miopen_->GetHandle(parent_, stream);

  TF_ASSIGN_OR_RETURN(auto x_descriptor,
                      scope(x_desc, ToMIOpenDataType(input_data_type)));
  TF_ASSIGN_OR_RETURN(
      auto scale_offset_descriptor,
      scope(scale_offset_desc, ToMIOpenDataType(scale_data_type)));
  miopenBatchNormMode_t mode = miopenBNSpatial;
  float one = 1.0;
  float zero = 0.0;

  auto status = miopenStatusInvalidValue;
  if (is_training) {
    status = wrap::miopenBatchNormalizationForwardTraining(
        miopen.handle(), mode, &one, &zero, x_descriptor.handle(), x.opaque(),
        x_descriptor.handle(), y->opaque(), scale_offset_descriptor.handle(),
        const_cast<void*>(scale.opaque()), const_cast<void*>(offset.opaque()),
        exponential_average_factor, batch_mean->opaque(), batch_var->opaque(),
        epsilon, saved_mean->opaque(), saved_inv_var->opaque());
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
    return absl::InternalError(
        "failed to enqueue forward batch normalization on stream: " +
        ToString(status));
  }
  return absl::OkStatus();
}

bool MIOpenSupport::DoBatchNormalizationBackward(
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
  return DoBatchNormalizationBackwardImpl<Eigen::bfloat16, float>(
             stream, miopenBFloat16, miopenFloat, y_backprop, x, scale, mean,
             inv_var, x_desc, scale_offset_desc, epsilon, x_backprop,
             scale_backprop, offset_backprop)
      .ok();
}

bool MIOpenSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<Eigen::half>& y_backprop,
    const DeviceMemory<Eigen::half>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
    const DeviceMemory<float>& inv_var, const DeviceMemory<Eigen::half>& y,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half>* x_backprop,
    DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
    DeviceMemory<Eigen::half>* side_input_backprop,
    DeviceMemory<uint8>* reserve_space_data,
    ScratchAllocator* workspace_allocator) {
  return DoBatchNormalizationBackwardImpl<Eigen::half, float>(
             stream, miopenHalf, miopenFloat, y_backprop, x, scale, mean,
             inv_var, x_desc, scale_offset_desc, epsilon, x_backprop,
             scale_backprop, offset_backprop)
      .ok();
}

bool MIOpenSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<float>& y_backprop,
    const DeviceMemory<float>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
    const DeviceMemory<float>& variance, const DeviceMemory<float>& y,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<float>* x_backprop,
    DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
    DeviceMemory<float>* side_input_backprop,
    DeviceMemory<uint8>* reserve_space_data,
    ScratchAllocator* workspace_allocator) {
  return DoBatchNormalizationBackwardImpl<float, float>(
             stream, miopenFloat, miopenFloat, y_backprop, x, scale, mean,
             variance, x_desc, scale_offset_desc, epsilon, x_backprop,
             scale_backprop, offset_backprop)
      .ok();
}

template <class T, class U>
absl::Status MIOpenSupport::DoBatchNormalizationBackwardImpl(
    Stream* stream, int miopen_input_type, int miopen_scale_type,
    const DeviceMemory<T>& y_backprop, const DeviceMemory<T>& x,
    const DeviceMemory<U>& scale, const DeviceMemory<U>& mean,
    const DeviceMemory<U>& variance, const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<T>* x_backprop, DeviceMemory<U>* scale_backprop,
    DeviceMemory<U>* offset_backprop) {
  auto miopen = miopen_->GetHandle(parent_, stream);
  TF_ASSIGN_OR_RETURN(
      auto x_descriptor,
      scope(x_desc, static_cast<miopenDataType_t>(miopen_input_type)));
  TF_ASSIGN_OR_RETURN(auto scale_offset_descriptor,
                      scope(scale_offset_desc,
                            static_cast<miopenDataType_t>(miopen_scale_type)));
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
    return absl::InternalError(
        "failed to enqueue backward batch normalization on stream: " +
        ToString(status));
  }
  return absl::OkStatus();
}

template <typename T, typename Tbias>
void launchInplaceBiasActivation(hipStream_t stream, void* c_data,
                                 const void* bias_data,
                                 const void* side_input_data,
                                 float side_input_scale, int activation_mode,
                                 uint64_t batch, uint64_t m, uint64_t n,
                                 int64_t ldc, float param);

class ROCmFusedMatmulRunner : public dnn::FusedMatmulRunner {
  template <typename T>
  absl::Status gemm(Stream*, DeviceMemoryBase /* a_data */,
                    DeviceMemoryBase /* b_data */,
                    DeviceMemoryBase /* c_data */) const;

  Stream* _stream;
  dnn::DataType _input_type, _bias_type, _output_type;
  bool _trans_a, _trans_b;
  uint64_t _m, _n, _k;
  int64_t _lda, _ldb, _ldc;
  dnn::ActivationMode _activation_mode;

 public:
  std::string ToString() const override;
  size_t GetWorkspaceSize() const override { return 0; }
  // Convert to an AlgorithmDesc for AoT compilation or autotuning
  absl::StatusOr<AlgorithmDesc> ToAlgorithmDesc() const override;
  // Launch the operation, with the signature determined by `Sig`.
  absl::Status operator()(Stream*, dnn::ProfileResult*,
                          DeviceMemoryBase scratch_memory,
                          DeviceMemoryBase /* a_data */,
                          DeviceMemoryBase /* b_data */,
                          DeviceMemoryBase /* bias_data */,
                          DeviceMemoryBase /* c_data */) const override;

  ROCmFusedMatmulRunner(Stream* stream, dnn::DataType input_type,
                        dnn::DataType bias_type, dnn::DataType output_type,
                        bool trans_a, bool trans_b, uint64_t m, uint64_t n,
                        uint64_t k, int64_t lda, int64_t ldb, int64_t ldc,
                        dnn::ActivationMode activation_mode);
};

ROCmFusedMatmulRunner::ROCmFusedMatmulRunner(
    Stream* stream, dnn::DataType input_type, dnn::DataType bias_type,
    dnn::DataType output_type, bool trans_a, bool trans_b, uint64_t m,
    uint64_t n, uint64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    dnn::ActivationMode activation_mode)
    : _stream(stream),
      _input_type(input_type),
      _bias_type(bias_type),
      _output_type(output_type),
      _trans_a(trans_a),
      _trans_b(trans_b),
      _m(m),
      _n(n),
      _k(k),
      _lda(lda),
      _ldb(ldb),
      _ldc(ldc),
      _activation_mode(activation_mode) {}

absl::StatusOr<AlgorithmDesc> ROCmFusedMatmulRunner::ToAlgorithmDesc() const {
  std::vector<std::pair<int64_t, int64_t>> knobs;
  knobs.emplace_back(0, static_cast<int64_t>(_input_type));
  knobs.emplace_back(1, static_cast<int64_t>(_bias_type));
  knobs.emplace_back(2, static_cast<int64_t>(_output_type));
  knobs.emplace_back(3, static_cast<int64_t>(_trans_a));
  knobs.emplace_back(4, static_cast<int64_t>(_trans_b));
  knobs.emplace_back(5, static_cast<int64_t>(_m));
  knobs.emplace_back(6, static_cast<int64_t>(_n));
  knobs.emplace_back(7, static_cast<int64_t>(_k));
  knobs.emplace_back(8, static_cast<int64_t>(_lda));
  knobs.emplace_back(9, static_cast<int64_t>(_ldb));
  knobs.emplace_back(10, static_cast<int64_t>(_ldc));
  return AlgorithmDesc(0, knobs, 0);
}

std::string ROCmFusedMatmulRunner::ToString() const {
  return ToAlgorithmDesc().value().ToString();
}

template <typename T>
absl::Status ROCmFusedMatmulRunner::gemm(Stream* stream,
                                         DeviceMemoryBase a_data,
                                         DeviceMemoryBase b_data,
                                         DeviceMemoryBase c_data) const {
  blas::Transpose ta =
      _trans_a ? blas::Transpose::kTranspose : blas::Transpose::kNoTranspose;
  blas::Transpose tb =
      _trans_b ? blas::Transpose::kTranspose : blas::Transpose::kNoTranspose;

  auto* blas = stream->parent()->AsBlas();
  if (blas == nullptr) {
    return absl::InternalError("No Blas support for stream");
  }
  return blas->BlasGemm<T, T>(stream, tb, ta, _n, _m, _k,
                              static_cast<DeviceMemory<T>>(b_data), _ldb,
                              static_cast<DeviceMemory<T>>(a_data), _lda,
                              static_cast<DeviceMemory<T>*>(&c_data), _ldc,
                              NumericOptions{}, blas::CallContext::kNone);
}

template <typename T, typename Tbias = T>
absl::Status InplaceBiasActivation(
    Stream* stream, DeviceMemoryBase c_data, DeviceMemoryBase bias_data,
    DeviceMemoryBase side_input_data, float side_input_scale,
    dnn::ActivationMode activation_mode, uint64_t batch, uint64_t m, uint64_t n,
    int64_t ldc, float param, bool transpose = false) {
  typedef typename std::conditional<
      std::is_same_v<T, Eigen::half>, __half,
      typename std::conditional<std::is_same_v<T, Eigen::bfloat16>,
                                hip_bfloat16, T>::type>::type CT;
  typedef typename std::conditional<
      std::is_same_v<Tbias, Eigen::half>, __half,
      typename std::conditional<std::is_same_v<Tbias, Eigen::bfloat16>,
                                hip_bfloat16, Tbias>::type>::type CTbias;
  launchInplaceBiasActivation<CT, CTbias>(
      AsGpuStreamValue(stream), c_data.opaque(), bias_data.opaque(),
      side_input_data.opaque(), side_input_scale,
      static_cast<int>(activation_mode) + (transpose ? 10 : 0), batch, m, n,
      ldc, param);
  return absl::OkStatus();
}

template <typename Ta, typename Tb, typename... Args>
absl::Status InplaceBiasActivation(Stream* stream, DeviceMemory<Ta> c_data,
                                   DeviceMemory<Tb> bias_data, Args... args) {
  return InplaceBiasActivation<Ta, Tb>(stream, DeviceMemoryBase(c_data),
                                       DeviceMemoryBase(bias_data), args...);
}

// Launch the operation, with the signature determined by `Sig`.
absl::Status ROCmFusedMatmulRunner::operator()(
    Stream* stream, dnn::ProfileResult* prof, DeviceMemoryBase scratch_memory,
    DeviceMemoryBase a_data, DeviceMemoryBase b_data,
    DeviceMemoryBase bias_data, DeviceMemoryBase c_data) const {
  absl::Status status;
  if (_input_type == dnn::DataType::kFloat)
    status = gemm<float>(stream, a_data, b_data, c_data);
  else if (_input_type == dnn::DataType::kHalf)
    status = gemm<Eigen::half>(stream, a_data, b_data, c_data);
  else if (_input_type == dnn::DataType::kBF16)
    status = gemm<Eigen::bfloat16>(stream, a_data, b_data, c_data);
  else if (_input_type == dnn::DataType::kDouble)
    status = gemm<double>(stream, a_data, b_data, c_data);
  else
    return absl::InvalidArgumentError("Unsupported input type");

  if (!status.ok()) return status;

  DeviceMemory<uint8_t> side_input;
  if (_input_type == dnn::DataType::kFloat)
    return InplaceBiasActivation<float>(stream, c_data, bias_data, side_input,
                                        0.0f, _activation_mode, 1, _m, _n, _ldc,
                                        0.0f);
  else if (_input_type == dnn::DataType::kHalf)
    return InplaceBiasActivation<Eigen::half>(
        stream, c_data, bias_data, side_input, 0.0f, _activation_mode, 1, _m,
        _n, _ldc, 0.0f);
  else if (_input_type == dnn::DataType::kBF16)
    return InplaceBiasActivation<Eigen::bfloat16>(
        stream, c_data, bias_data, side_input, 0.0f, _activation_mode, 1, _m,
        _n, _ldc, 0.0f);
  else if (_input_type == dnn::DataType::kDouble)
    return InplaceBiasActivation<double>(stream, c_data, bias_data, side_input,
                                         0.0f, _activation_mode, 1, _m, _n,
                                         _ldc, 0.0f);
  else
    return absl::InvalidArgumentError("Unsupported input type");
}

absl::Status MIOpenSupport::GetFusedMatmulRunners(
    bool use_cudnn_frontend, dnn::DataType input_type, dnn::DataType bias_type,
    dnn::DataType output_type, Stream* stream, bool trans_a, bool trans_b,
    uint64_t m, uint64_t n, uint64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    dnn::ActivationMode activation_mode, bool use_fallback,
    const NumericOptions& numeric_options,
    std::vector<std::unique_ptr<const dnn::FusedMatmulRunner>>*
        out_exec_plans) {
  out_exec_plans->clear();
  if (input_type != output_type)
    return absl::InvalidArgumentError(
        "ROCm fused matmul does not support input/output type mismatch");
  if (input_type != bias_type)
    return absl::InvalidArgumentError(
        "ROCm fused matmul does not support input/bias type mismatch");
  auto runner_ptr = new ROCmFusedMatmulRunner(
      stream, input_type, bias_type, output_type, trans_a, trans_b, m, n, k,
      lda, ldb, ldc, activation_mode);
  out_exec_plans->push_back(
      std::unique_ptr<const dnn::FusedMatmulRunner>(runner_ptr));
  return absl::OkStatus();
}

absl::Status MIOpenSupport::DoFusedConvolve(
    Stream* stream, dnn::DataType input_type, dnn::DataType side_input_type,
    dnn::DataType bias_type, dnn::DataType output_type,
    const dnn::BatchDescriptor& conv_input_descriptor,
    DeviceMemoryBase conv_input_data, double conv_input_scale,
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
  return absl::UnimplementedError("fused convolve not implemented yet");
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

absl::Status MIOpenSupport::DoPoolForward(
    dnn::DataType element_type, Stream* stream,
    const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
    const dnn::BatchDescriptor& output_dimensions, DeviceMemoryBase output_data,
    ScratchAllocator* workspace_allocator) {
  if (element_type == dnn::DataType::kDouble) {
    return absl::InvalidArgumentError(
        "MIOpen does not support pooling for double type yet");
  }

  auto miopen = miopen_->GetHandle(parent_, stream);
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  auto miopen_dtype =
      element_type == dnn::DataType::kFloat ? miopenFloat : miopenHalf;

  TF_ASSIGN_OR_RETURN(auto src_desc, scope(input_dimensions, miopen_dtype));
  TF_ASSIGN_OR_RETURN(auto dest_desc, scope(output_dimensions, miopen_dtype));
  TF_ASSIGN_OR_RETURN(auto pooling_desc, scope(pooling_dimensions));

  bool do_backward = false;
  uint8* workspace = nullptr;
  size_t workspace_size = 0;
  if (m_pooling_cache_enabled && element_type == dnn::DataType::kFloat) {
    do_backward = true;
    auto status = wrap::miopenPoolingGetWorkSpaceSizeV2(
        pooling_desc.handle(), dest_desc.handle(), &workspace_size);
    if (status != miopenStatusSuccess) {
      return absl::InternalError(absl::StrCat(
          "Failed to obtain workspace size for backward pooling on stream: ",
          ToString(status)));
    }
    if (workspace_size != 0) {
      PoolingWorkspaceDescriptor* pdesc = 0;
      bool cache_hit =
          m_pooling_cache_allowed &&
          m_pooling_cache.find(input_data.opaque(), input_dimensions,
                               output_dimensions, pooling_dimensions,
                               miopenFloat, pdesc);
      if (cache_hit) {
        // reusing the same buffer
        workspace = reinterpret_cast<uint8*>(pdesc->workspace.ptr()->opaque());
      } else {
        TF_ASSIGN_OR_RETURN(auto allocated,
                            workspace_allocator->AllocateBytes(workspace_size));
        workspace = reinterpret_cast<uint8*>(allocated.opaque());
      }
    }
  }

  auto status = wrap::miopenPoolingForward(
      miopen.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
      input_data.opaque(), &beta, dest_desc.handle(), output_data.opaque(),
      do_backward, workspace, workspace_size);
  if (status != miopenStatusSuccess) {
    return absl::InternalError(absl::StrCat(
        "Failed to enqueue forward pooling on stream: ", ToString(status)));
  }
  return absl::OkStatus();
}

bool PoolingWorkspaceDescriptor::IsSame(
    const dnn::BatchDescriptor& input_dimensions,
    const dnn::BatchDescriptor& output_dimensions,
    const dnn::PoolingDescriptor& pooling_dimensions, int _type) {
  return dtype == _type &&
         input_dims ==
             input_dimensions.full_dims(dnn::DataLayout::kBatchDepthYX) &&
         output_dims ==
             output_dimensions.full_dims(dnn::DataLayout::kBatchDepthYX) &&
         op.mode() == pooling_dimensions.mode() &&
         op.window() == pooling_dimensions.window() &&
         op.padding() == pooling_dimensions.padding() &&
         op.strides() == pooling_dimensions.strides();
}

bool PoolingWorkspaceCache::find(
    const void* p, const dnn::BatchDescriptor& input_dimensions,
    const dnn::BatchDescriptor& output_dimensions,
    const dnn::PoolingDescriptor& pooling_dimensions, int _type,
    PoolingWorkspaceDescriptor*& pdesc) {
  pdesc = 0;
  auto it = cache.find(p);
  if (it == cache.end()) {
    return false;
  }
  if (!it->second.IsSame(input_dimensions, output_dimensions,
                         pooling_dimensions, _type)) {
    return false;
  }
  pdesc = &it->second;
  return true;
}

void PoolingWorkspaceCache::insert(
    const void* p, const dnn::BatchDescriptor& input_dimensions,
    const dnn::BatchDescriptor& output_dimensions,
    const dnn::PoolingDescriptor& pooling_dimensions, int _type,
    ScopedDeviceMemory<uint8>& workspace, size_t wsp_size,
    hipStream_t hip_stream) {
  PoolingWorkspaceDescriptor* desc = 0;
  auto it = cache.find(p);
  if (it != cache.end()) {
    // replacing an entry with the same pointer but different attributes
    // (if everything matches, the caller is expected to reuse the entry)
    desc = &it->second;
    CHECK_EQ(hipStreamSynchronize(hip_stream), hipSuccess)
        << "Failed to sync hipStream";
    memory_used -= desc->workspace_size;
  } else {
    cache[p] = PoolingWorkspaceDescriptor();
    desc = &cache[p];
  }
  desc->input_dims = input_dimensions.full_dims(dnn::DataLayout::kBatchDepthYX);
  desc->output_dims =
      output_dimensions.full_dims(dnn::DataLayout::kBatchDepthYX);
  desc->op = pooling_dimensions;
  desc->dtype = _type;
  desc->timestamp = timestamp;
  timestamp++;
  desc->workspace = std::move(workspace);
  desc->workspace_size = wsp_size;
  memory_used += wsp_size;
  trim(hip_stream);
}

void PoolingWorkspaceCache::trim(hipStream_t hip_stream) {
  if (memory_used < memory_budget && cache.size() < trim_size) return;
  bool must_sync = true;
  while (true) {
    int new_size = cache.size() - (cache.size() >> 2);
    std::vector<const void*> old_entries;
    for (auto& x : cache)
      if (x.second.timestamp + new_size < timestamp)
        old_entries.push_back(x.first);
    if (old_entries.empty()) break;
    if (must_sync)
      CHECK_EQ(hipStreamSynchronize(hip_stream), hipSuccess)
          << "Failed to sync hipStream";
    must_sync = true;
    for (auto x : old_entries) {
      memory_used -= cache[x].workspace_size;
      cache.erase(x);
    }
    if (memory_used < memory_budget || cache.size() < 10) break;
  }
}

absl::Status MIOpenSupport::DoPoolBackward(
    dnn::DataType element_type, Stream* stream,
    const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
    const dnn::BatchDescriptor& output_dimensions, DeviceMemoryBase output_data,
    DeviceMemoryBase input_diff_data, DeviceMemoryBase output_diff_data,
    ScratchAllocator* workspace_allocator) {
  if (element_type == dnn::DataType::kDouble) {
    return absl::InvalidArgumentError(
        "MIOpen does not support pooling for double type yet");
  }

  auto miopen = miopen_->GetHandle(parent_, stream);
  if (m_pooling_cache_allowed) m_pooling_cache_enabled = true;
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  auto miopen_dtype =
      element_type == dnn::DataType::kFloat ? miopenFloat : miopenHalf;

  TF_ASSIGN_OR_RETURN(auto src_desc, scope(input_dimensions, miopen_dtype));
  TF_ASSIGN_OR_RETURN(auto dest_desc, scope(output_dimensions, miopen_dtype));
  TF_ASSIGN_OR_RETURN(auto pooling_desc, scope(pooling_dimensions));

  uint8* workspace_ptr = 0;
  DeviceMemory<uint8> workspace;
  PoolingWorkspaceDescriptor* pdesc = 0;

  size_t workspace_size_in_bytes = 0;
  auto status = wrap::miopenPoolingGetWorkSpaceSizeV2(
      pooling_desc.handle(), dest_desc.handle(), &workspace_size_in_bytes);
  if (status != miopenStatusSuccess) {
    return absl::InternalError(absl::StrCat(
        "Failed to obtain workspace size for backward pooling on stream: ",
        ToString(status)));
  }

  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    bool cache_hit = m_pooling_cache_allowed &&
                     m_pooling_cache.find(input_data.opaque(), input_dimensions,
                                          output_dimensions, pooling_dimensions,
                                          miopen_dtype, pdesc);
    if (cache_hit) {
      assert(pdesc != 0);
      workspace_ptr =
          reinterpret_cast<uint8*>(pdesc->workspace.ptr()->opaque());
      VLOG(1) << "Pooling cache hit";
    } else {
      VLOG(1) << "Pooling cache miss";
      assert(workspace_allocator);
      auto allocated =
          workspace_allocator->AllocateBytes(workspace_size_in_bytes);
      if (!allocated.ok() || (workspace = allocated.value()) == nullptr) {
        return absl::InternalError(
            "Failed to allocate backward pooling workspace");
      }
      DeviceMemory<uint8> dest2;  // duplicated dest from forward:
      int64_t dest2_size = 0;

      // miopen requires the strides and dims to be ordered as BDYX.
      std::vector<int64_t> dims64 =
          output_dimensions.full_dims(dnn::DataLayout::kBatchDepthYX);
      // miopen does not use strides and must have 4D tensor.
      // std::vector<int> dims(pooling_dimensions.ndims() + 2);

      dest2_size = (element_type == dnn::DataType::kFloat)
                       ? sizeof(float)
                       : sizeof(Eigen::half);
      for (auto& x : dims64) dest2_size *= x;

      if (dest2_size > 0) {
        assert(workspace_allocator);
        auto allocated = workspace_allocator->AllocateBytes(dest2_size);
        if (!allocated.ok() || (dest2 = allocated.value()) == nullptr) {
          return absl::InternalError(
              "Failed to allocate backward pooling workspace");
        }
      } else {
        LOG(ERROR) << "Failed to calculate tensor size to chain forward and "
                      "backward pooling";
      }

      status = wrap::miopenPoolingForward(
          miopen.handle(), pooling_desc.handle(), &alpha, src_desc.handle(),
          input_data.opaque(), &beta, dest_desc.handle(), dest2.opaque(), true,
          workspace.opaque(), workspace_size_in_bytes);

      if (status != miopenStatusSuccess) {
        return absl::InternalError(absl::StrCat(
            "Failed to enqueue forward pooling (before backward) on stream: ",
            ToString(status)));
      }
      workspace_ptr = reinterpret_cast<uint8*>(workspace.opaque());
    }
  }

  status = wrap::miopenPoolingBackward(
      miopen.handle(), pooling_desc.handle(), &alpha, dest_desc.handle(),
      output_data.opaque(), dest_desc.handle(), input_diff_data.opaque(),
      src_desc.handle(), input_data.opaque(), &beta, src_desc.handle(),
      output_diff_data.opaque(), workspace_ptr);

  if (status != miopenStatusSuccess) {
    return absl::InternalError(absl::StrCat(
        "Failed to enqueue backward pooling on stream: ", ToString(status)));
  }
  return absl::OkStatus();
}

#define ASSIGN_OR_RETURN_FALSE(lhs, rexpr) \
  ASSIGN_OR_RETURN_FALSE_IMPL(             \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define ASSIGN_OR_RETURN_FALSE_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                \
  if (TF_PREDICT_FALSE(!statusor.ok())) {                 \
    return false;                                         \
  }                                                       \
  lhs = std::move(statusor).value()

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
  ASSIGN_OR_RETURN_FALSE(auto dims, scope(dimensions, miopenFloat));
  ASSIGN_OR_RETURN_FALSE(auto normalize, scope(normalize_descriptor));

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

  ASSIGN_OR_RETURN_FALSE(auto dims, scope(dimensions, miopenFloat));
  ASSIGN_OR_RETURN_FALSE(auto normalize, scope(normalize_descriptor));

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
        workspace_allocator->AllocateBytes(workspace_size_in_bytes);
    if (!allocated.ok() || (workspace = allocated.value()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  }

  DeviceMemory<uint8> dest2;  // duplicated dest from forward:
  int dest2_size = 0;

  // miopen requires the strides and dims to be ordered as BDYX.
  std::vector<int64_t> dims64 =
      dimensions.full_dims(dnn::DataLayout::kBatchDepthYX);

  // miopen does not use strides and must have 4D tensor.
  std::vector<int> dimsint(4);

  std::transform(dims64.cbegin(), dims64.cend(), dimsint.begin(),
                 &CheckedNarrowing<int64_t, int>);

  dest2_size =
      dimsint[0] * dimsint[1] * dimsint[2] * dimsint[3] * sizeof(float);

  if (dest2_size > 0) {
    assert(workspace_allocator);
    auto allocated = workspace_allocator->AllocateBytes(dest2_size);
    if (!allocated.ok() || (dest2 = allocated.value()) == nullptr) {
      LOG(ERROR)
          << "Failed to allocate tensor to chain forward and backward LRN";
      return false;
    }
  } else {
    LOG(ERROR) << "Failed to calculate tensor size to chain forward and "
                  "backward LRN";
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

bool MIOpenSupport::DeriveOutputBatchDescriptor(
    const BatchDescriptor& batch_descriptor,
    const FilterDescriptor& filter_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::BatchDescriptor* output_batch_descriptor) {
  ASSIGN_OR_RETURN_FALSE(auto input_nd, scope(batch_descriptor, miopenFloat));
  ASSIGN_OR_RETURN_FALSE(auto filter, scope(filter_descriptor, miopenFloat));
  ASSIGN_OR_RETURN_FALSE(auto conv, scope(convolution_descriptor));

  int dn = batch_descriptor.ndims() + 2;
  std::vector<int> dims(dn);  // in BDYX
  auto status = wrap::miopenGetConvolutionNdForwardOutputDim(
      conv.handle(), input_nd.handle(), filter.handle(), &dn, dims.data());
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

class RocmFusedConvRunner : public dnn::FusedConvRunner {
 public:
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
    VLOG(2) << "RocmFusedConvRunner()";
    if (parent_ != stream->parent()) {
      return absl::InternalError(
          "RocmFusedConvRunner cached across multiple StreamExecutors.");
    }

    // We can't reliably detect whether this sequence can be fused until
    // we come here and actually try to fuse it. So, we need a fallback.
    bool do_unfused =
        (side_input_scale_ != 0.0) || !fusion_plan_.CompilationSucceeded();

    if (do_unfused)
      return execute_unfused(stream, profile_result, scratch_memory, input_data,
                             filter_data, side_input_data, bias_data,
                             output_data);
    auto algo = MakeAlgorithmDesc();
    auto miopen = miopen_->GetHandle(parent_, stream);
    fusion_plan_.SetConvolutionArgs(filter_data.opaque());
    fusion_plan_.SetBiasArgs(bias_data.opaque());
    if (activation_desc_.miopen_activation_mode_ != miopenActivationPASTHRU)
      fusion_plan_.SetActivationForwardArgs(activation_desc_);

    std::optional<GpuTimer> timer;
    if (profile_result) {
      auto timer_or_status = GpuTimer::Create(AsGpuStream(stream));
      if (!timer_or_status.ok()) {
        LOG(ERROR) << "Failed to create timer";
        return absl::InternalError("Failed to start timer");
      }
      timer.emplace(std::move(*timer_or_status));
    }

    miopenStatus_t status;
    status = wrap::miopenExecuteFusionPlan(
        miopen.handle(), fusion_plan_.fusion_plan_, input_nd_.handle(),
        input_data.opaque(), output_nd_.handle(), output_data.opaque(),
        fusion_plan_.fusion_args_);

    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "Failed to enqueue fused convolution on stream: "
                 << stream_executor::gpu::ToString(status);
      return absl::InternalError(
          "Failed to enqueue fused convolution on stream: " +
          stream_executor::gpu::ToString(status));
    }

    if (profile_result) {
      absl::StatusOr<absl::Duration> elapsed = timer->GetElapsedDuration();
      if (!elapsed.ok()) {
        LOG(ERROR) << "Failed to get elapsed duration";
        return absl::InternalError("Timer failure");
      }
      profile_result->set_elapsed_time_in_ms(
          absl::ToDoubleMilliseconds(*elapsed));
      profile_result->set_algorithm(algo);
      profile_result->set_scratch_size(scratch_memory.size());
    }

    return absl::OkStatus();
  }

 public:
  // Queries the workspace size and constructs a 'RocmFusedConvRunner'.
  static absl::StatusOr<std::unique_ptr<const dnn::FusedConvRunner>> Create(
      GpuExecutor* parent, Stream* stream, MIOpenAccess* miopen,
      const dnn::AlgorithmDesc& algo, dnn::DataType input_type,
      dnn::DataType bias_type, double conv_scale, double side_input_scale,
      double leakyrelu_alpha, BatchDescriptor input_nd,
      BatchDescriptor output_nd, FilterDescriptor filter,
      BatchDescriptor bias_nd, ConvolutionDescriptor conv,
      dnn::ActivationMode activation) {
    TF_ASSIGN_OR_RETURN(
        auto input_nd_,
        scope(input_nd, ToMIOpenDataType(input_type, input_nd.layout())));
    TF_ASSIGN_OR_RETURN(
        auto output_nd_,
        scope(output_nd, ToMIOpenDataType(input_type, input_nd.layout())));
    TF_ASSIGN_OR_RETURN(auto filter_,
                        scope(filter, ToMIOpenDataType(input_type)));
    TF_ASSIGN_OR_RETURN(auto bias_nd_,
                        scope(bias_nd, ToMIOpenDataType(bias_type)));
    TF_ASSIGN_OR_RETURN(auto conv_, scope(conv));

    TF_ASSIGN_OR_RETURN(
        auto activation_desc,
        ScopedActivationDescriptor::Create(activation, leakyrelu_alpha));

    TF_ASSIGN_OR_RETURN(
        auto fusion_plan,
        ScopedFusionPlanConvolutionBiasActivation::Create(
            miopen->GetHandle(parent, stream).handle(), input_nd_.handle(),
            filter_.handle(), conv_.handle(), bias_nd_.handle(),
            activation_desc));

    VLOG(2) << "RocmFusedConvRunner";
    auto mi = miopen->GetHandle(parent, stream);

    size_t maxSolutionCount = 0;
    auto status = wrap::miopenConvolutionForwardGetSolutionCount(
        mi.handle(), filter_.handle(), input_nd_.handle(), conv_.handle(),
        output_nd_.handle(), &maxSolutionCount);

    size_t solutionCount = 0;
    std::unique_ptr<miopenConvSolution_t[]> solutions(
        new miopenConvSolution_t[maxSolutionCount]);

    status = wrap::miopenConvolutionForwardGetSolution(
        mi.handle(), filter_.handle(), input_nd_.handle(), conv_.handle(),
        output_nd_.handle(), maxSolutionCount, &solutionCount, solutions.get());

    VLOG(2) << solutionCount << " solutions";

    if (solutionCount == 0) return absl::InternalError("No algorithms found");

    size_t workspace_size_1 = solutions[0].workspace_size;
    size_t true_workspace_size = 0;
    status = wrap::miopenConvolutionForwardGetWorkSpaceSize(
        mi.handle(), filter_.handle(), input_nd_.handle(), conv_.handle(),
        output_nd_.handle(), &true_workspace_size);

    VLOG(2) << "True workspace size " << workspace_size_1 << " "
            << true_workspace_size;

    auto obj = new RocmFusedConvRunner(
        parent, stream, miopen, static_cast<int64_t>(solutions[0].solution_id),
        true_workspace_size, input_type, bias_type, conv_scale,
        side_input_scale, leakyrelu_alpha, input_nd, output_nd, filter, bias_nd,
        conv, activation, input_nd_, output_nd_, filter_, bias_nd_, conv_,
        activation_desc, fusion_plan);

    return std::unique_ptr<const dnn::FusedConvRunner>(obj);
  }

 private:
  // Private to prevent passing in the wrong workspace_size.
  RocmFusedConvRunner(
      GpuExecutor* parent, Stream* stream, MIOpenAccess* miopen,
      int64_t algo_id, size_t workspace_size, dnn::DataType input_type,
      dnn::DataType bias_type, double conv_scale, double side_input_scale,
      double leakyrelu_alpha, BatchDescriptor dnn_input_nd,
      BatchDescriptor dnn_output_nd, FilterDescriptor dnn_filter,
      BatchDescriptor dnn_bias_nd, ConvolutionDescriptor dnn_conv,
      dnn::ActivationMode activation, ScopedTensorDescriptor& input_nd,
      ScopedTensorDescriptor& output_nd, ScopedFilterDescriptor& filter,
      ScopedTensorDescriptor& bias_nd, ScopedConvolutionDescriptor& conv,
      ScopedActivationDescriptor& activation_desc,
      ScopedFusionPlanConvolutionBiasActivation& fusion_plan)
      : parent_(parent),
        miopen_(miopen),
        algo_id_(algo_id),
        workspace_size_(workspace_size),
        input_type_(input_type),
        bias_type_(bias_type),

        conv_scale_(conv_scale),
        side_input_scale_(side_input_scale),
        leakyrelu_alpha_(leakyrelu_alpha),
        side_input_scale_f32_(static_cast<float>(side_input_scale)),

        activation_mode_(activation),
        dnn_input_nd_(dnn_input_nd),
        dnn_output_nd_(dnn_output_nd),
        dnn_filter_(dnn_filter),
        dnn_bias_nd_(dnn_bias_nd),
        dnn_conv_(dnn_conv),

        input_nd_(std::move(input_nd)),
        output_nd_(std::move(output_nd)),
        filter_(std::move(filter)),
        bias_nd_(std::move(bias_nd)),
        conv_(std::move(conv)),
        activation_desc_(std::move(activation_desc)),
        fusion_plan_(std::move(fusion_plan)) {}

  absl::Status execute_unfused(
      Stream* stream, dnn::ProfileResult* profile_result,
      DeviceMemoryBase scratch_memory, DeviceMemoryBase input_data,
      DeviceMemoryBase filter_data, DeviceMemoryBase side_input_data,
      DeviceMemoryBase bias_data, DeviceMemoryBase output_data) const {
    auto miopen = miopen_->GetHandle(parent_, stream);
    auto status = wrap::miopenConvolutionForwardImmediate(
        miopen.handle(), filter_.handle(), filter_data.opaque(),
        input_nd_.handle(), input_data.opaque(), conv_.handle(),
        output_nd_.handle(), output_data.opaque(), scratch_memory.opaque(),
        scratch_memory.size(), static_cast<uint64_t>(algo_id_));
    if (status != miopenStatusSuccess) {
      VLOG(0) << "Failed to enqueue convolution: "
              << stream_executor::gpu::ToString(status);
      return absl::InternalError("Failed to enqueue convolution: " +
                                 stream_executor::gpu::ToString(status));
    }

    int batch;
    std::vector<int64_t> dims_output =
        dnn_output_nd_.full_dims(dnn_output_nd_.layout());
    int rank = dims_output.size();
    if (rank != 4 && rank != 5)
      return absl::InternalError(
          "RocmFusedConvRunner expects 4d or 5d descriptors");
    int d1 = 1, d2 = 1;
    bool bNCHW = (dnn_output_nd_.layout() != dnn::DataLayout::kBatchYXDepth);
    batch = dims_output[0];
    if (bNCHW) {
      d1 = dims_output[1];
      for (int i = 2; i < rank; i++) d2 *= dims_output[i];
    } else {
      d2 = dims_output[rank - 1];
      for (int i = 1; i < rank - 1; i++) d1 *= dims_output[i];
    }

    float param = activation_desc_.alpha_;

    auto inplace_call = [&](auto out, auto bias) {
      return InplaceBiasActivation(stream, out, bias, side_input_data,
                                   side_input_scale_f32_, activation_mode_,
                                   batch, d1, d2, d2, param, bNCHW);
    };

    absl::Status biasActStatus;
    if (input_type_ == dnn::DataType::kFloat &&
        bias_type_ == dnn::DataType::kFloat)
      biasActStatus = inplace_call(DeviceMemory<float>(output_data),
                                   DeviceMemory<float>(bias_data));
    else if (input_type_ == dnn::DataType::kHalf &&
             bias_type_ == dnn::DataType::kFloat)
      biasActStatus = inplace_call(DeviceMemory<Eigen::half>(output_data),
                                   DeviceMemory<float>(bias_data));
    else if (input_type_ == dnn::DataType::kHalf &&
             bias_type_ == dnn::DataType::kHalf)
      biasActStatus = inplace_call(DeviceMemory<Eigen::half>(output_data),
                                   DeviceMemory<Eigen::half>(bias_data));
    else if (input_type_ == dnn::DataType::kBF16 &&
             bias_type_ == dnn::DataType::kFloat)
      biasActStatus = inplace_call(DeviceMemory<Eigen::bfloat16>(output_data),
                                   DeviceMemory<float>(bias_data));
    else if (input_type_ == dnn::DataType::kBF16 &&
             bias_type_ == dnn::DataType::kBF16)
      biasActStatus = inplace_call(DeviceMemory<Eigen::bfloat16>(output_data),
                                   DeviceMemory<Eigen::bfloat16>(bias_data));
    else
      return absl::InternalError("Unsupported data type");

    return absl::OkStatus();
  }

  // Internal form of ToAlgorithmDesc without the StatusOr.
  dnn::AlgorithmDesc MakeAlgorithmDesc() const {
    return {algo_id_, /*tensor_ops_enabled_*/ true, workspace_size_};
  }

  std::string desc_;

  GpuExecutor* parent_;
  MIOpenAccess* miopen_;
  int64_t algo_id_;
  size_t workspace_size_;
  dnn::DataType input_type_, bias_type_;
  double conv_scale_, side_input_scale_, leakyrelu_alpha_;
  float side_input_scale_f32_;
  dnn::ActivationMode activation_mode_;

  BatchDescriptor dnn_input_nd_;
  BatchDescriptor dnn_output_nd_;
  FilterDescriptor dnn_filter_;
  BatchDescriptor dnn_bias_nd_;
  ConvolutionDescriptor dnn_conv_;

  ScopedTensorDescriptor input_nd_;
  ScopedTensorDescriptor output_nd_;
  ScopedFilterDescriptor filter_;
  ScopedTensorDescriptor bias_nd_;
  ScopedConvolutionDescriptor conv_;
  mutable ScopedActivationDescriptor activation_desc_;
  mutable ScopedFusionPlanConvolutionBiasActivation fusion_plan_;
};

absl::StatusOr<std::unique_ptr<const dnn::FusedConvRunner>>
MIOpenSupport::FusedConvolveRunnerFromDesc(
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
  VLOG(2) << "MIOpenSupport::FusedConvolveRunnerFromDesc "
          << filter_descriptor.ndims() << " " << side_input_scale << " "
          << convolution_descriptor.ToString() << getTypeName(input_type) << " "
          << getTypeName(bias_type) << " " << getTypeName(output_type);

  // note: these checks need to be duplicated in XLA logic, because XLA calls
  // this function directly and it terminates the process on error

  return RocmFusedConvRunner::Create(
      parent_, stream, miopen_.get(), algorithm_desc, input_type, bias_type,
      conv_scale, side_input_scale, leakyrelu_alpha, input_descriptor,
      output_descriptor, filter_descriptor, bias_descriptor,
      convolution_descriptor, activation_mode);
}

absl::Status MIOpenSupport::GetFusedConvolveRunners(
    bool use_cudnn_frontend, dnn::ConvolutionKind kind,
    dnn::DataType input_type, dnn::DataType bias_type,
    dnn::DataType output_type, double conv_scale, double side_input_scale,
    double leakyrelu_alpha, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& bias_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    dnn::ActivationMode activation_mode, const NumericOptions& numeric_options,
    std::vector<std::unique_ptr<const dnn::FusedConvRunner>>* out_exec_plans) {
  VLOG(2) << "MIOpenSupport::GetFusedConvolveRunners";
  VLOG(2) << "filter_descriptor " << filter_descriptor.ndims();

  std::vector<dnn::AlgorithmDesc> algorithms{
      // clang-format off
      dnn::AlgorithmDesc(miopenConvolutionFwdAlgoGEMM, false, 0),
      dnn::AlgorithmDesc(miopenConvolutionFwdAlgoDirect, false, 0),
      dnn::AlgorithmDesc(miopenConvolutionFwdAlgoFFT, false, 0),
      dnn::AlgorithmDesc(miopenConvolutionFwdAlgoWinograd, false, 0),
      // clang-format on
  };

  for (const auto& algo : algorithms) {
    auto runner_or = FusedConvolveRunnerFromDesc(
        stream, algo, kind, input_type, bias_type, output_type, conv_scale,
        side_input_scale, leakyrelu_alpha, input_descriptor, filter_descriptor,
        bias_descriptor, output_descriptor, convolution_descriptor,
        activation_mode);
    if (!runner_or.ok()) continue;
    out_exec_plans->push_back(std::move(runner_or).value());
  }

  VLOG(2) << "MIOpenSupport::GetFusedConvolveRunners returns "
          << out_exec_plans->size() << " runners";
  return absl::OkStatus();
}

bool UseNhwcLayoutForRocm() {
#if TF_ROCM_VERSION >= 50100
  static bool is_enabled = [] {
    bool is_enabled = false;
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_USE_ROCM_NHWC",
                                        /*default_val=*/false, &is_enabled));
    return is_enabled;
  }();
  return is_enabled;
#else  // TF_ROCM_VERSION < 50000
  return false;
#endif
}

}  // namespace gpu

void initialize_miopen() {
  auto miopenAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      rocm::kROCmPlatformId, PluginKind::kDnn);

  if (!miopenAlreadyRegistered) {
    absl::Status status =
        PluginRegistry::Instance()->RegisterFactory<PluginRegistry::DnnFactory>(
            rocm::kROCmPlatformId, "MIOpen",
            [](StreamExecutorInterface* parent) -> dnn::DnnSupport* {
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
      LOG(ERROR) << "Unable to register MIOpen factory: " << status.message();
    }
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_miopen, {
  stream_executor::initialize_miopen();
});
