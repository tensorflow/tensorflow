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

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// FP16ConvMode: The mode to set the internal compute type for cudnn convolution
// when the input data type is float16. Two types of modes are supported:
//   kAccurate: Always use float32 as the internal compute type.
//   kFast: Include both float32 and float16 compute type in the autotune.
enum class FP16ConvMode {
  kAccurate = 1,
  kFast = 2,
};

bool CanUseCudnn();
bool CudnnUseAutotune();
bool CudnnRnnUseAutotune();
bool CudnnDisableConv1x1Optimization();
FP16ConvMode CudnnConvComputeMode();
bool DebugCudnnRnn();
bool DebugCudnnRnnUseTensorOps();
int64 DebugCudnnRnnAlgo();

// Returns true if the CuDNN depthwise convolution can be used. See cudnn
// release note 7.6.3.
// (https://docs.nvidia.com/deeplearning/sdk/cudnn-release-notes/rel_763.html)
bool IsCudnnSupportedFilterSize(const int32 filter_rows,
                                const int32 filter_cols, const int32 in_depth,
                                const int32 out_depth);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_USE_CUDNN_H_
