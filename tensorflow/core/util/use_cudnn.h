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

// The utility to check Cudnn dependency and set Cudnn-related flags.

#ifndef TENSORFLOW_CORE_UTIL_USE_CUDNN_H_
#define TENSORFLOW_CORE_UTIL_USE_CUDNN_H_

#include <cstdint>

#include "xla/tsl/util/use_cudnn.h"

namespace tensorflow {

using tsl::CudnnDisableConv1x1Optimization;
using tsl::CudnnRnnUseAutotune;
using tsl::CudnnUseAutotune;
using tsl::CudnnUseFrontend;
using tsl::CudnnUseRuntimeFusion;
using tsl::DebugCudnnRnn;
using tsl::DebugCudnnRnnAlgo;
using tsl::DebugCudnnRnnUseTensorOps;
using tsl::ShouldCudnnGroupedConvolutionBeUsed;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_USE_CUDNN_H_
