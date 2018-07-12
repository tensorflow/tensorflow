/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// This is essentially unsupported/CXX11/Eigen/Tensor.h
// TODO(petewarden) - move this to a common location in Eigen itself.

// clang-format off


#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_TENSOR_REDUCED_INSTANTIATIONS_OSS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_TENSOR_REDUCED_INSTANTIATIONS_OSS_H_


#include "Eigen/Core"

#if defined(EIGEN_USE_SYCL)
#undef min
#undef max
#undef isnan
#undef isinf
#undef isfinite
#include <CL/sycl.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#endif
#include <cmath>
#include <cstddef>
#include <cstring>





#ifdef _WIN32
typedef __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#include <windows.h>
#else
#include <stdint.h>
#include <unistd.h>
#endif

#if __cplusplus > 199711 || EIGEN_COMP_MSVC >= 1900
#include <random>
#endif

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <time.h>
#endif

// #if defined(EIGEN_USE_LIBXSMM)
// #include "libxsmm.h"
// #endif

#ifdef EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/ThreadPool"
#endif


#include "Eigen/src/Core/util/DisableStupidWarnings.h"

#include "unsupported/Eigen/SpecialFunctions"
#include "unsupported/Eigen/CXX11/src/util/CXX11Meta.h"
#include "unsupported/Eigen/CXX11/src/util/MaxSizeVector.h"


#include "unsupported/Eigen/CXX11/src/Tensor/TensorMacros.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorMeta.h"

#include "unsupported/Eigen/CXX11/src/Tensor/TensorFunctors.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorCostModel.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceThreadPool.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceGpu.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceSycl.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorIndexList.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDimensionList.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorInitializer.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorTraits.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorRandom.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorUInt128.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorIntDiv.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorGlobalFunctions.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorBase.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorEvaluator.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorExpr.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorArgMax.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorConcatenation.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorContractionMapper.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorContractionBlocking.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorContraction.h"

#undef TENSOR_CONTRACTION_DISPATCH
#define TENSOR_CONTRACTION_DISPATCH(METHOD, ALIGNMENT, ARGS)    \
  if (this->m_lhs_inner_dim_contiguous &&                       \
      this->m_rhs_inner_dim_contiguous &&                       \
      !this->m_rhs_inner_dim_reordered) {                       \
    METHOD<true, true, false, ALIGNMENT> ARGS;                  \
  } else {                                                      \
    eigen_assert(false && "Unsupported contraction formats");   \
  }


#include "unsupported/Eigen/CXX11/src/Tensor/TensorContractionThreadPool.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorContractionGpu.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorConversion.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorConvolution.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorFFT.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorPatch.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorImagePatch.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorVolumePatch.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorBroadcasting.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorChipping.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorInflation.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorLayoutSwap.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorMorphing.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorPadding.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorReverse.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorShuffling.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorStriding.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorCustomOp.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorEvalTo.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorForcedEval.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorGenerator.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorAssign.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorScan.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorTrace.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorSycl.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDevice.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorStorage.h"
#include "unsupported/Eigen/CXX11/src/Tensor/Tensor.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorFixedSize.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorMap.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorRef.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorIO.h"

#include "Eigen/src/Core/util/ReenableStupidWarnings.h"


#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_TENSOR_REDUCED_INSTANTIATIONS_OSS_H_
