/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_TRT_PARAMETERS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_TRT_PARAMETERS_H_

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tensorrt {

// The PrecisionMode controls the precision used in TRT converted parts of the
// model. Setting PrecisionMode other than FP32 enables TensorRT to select
// lower-precision implementations when searching for the fastest kernels.
//
// For regularized models whose input dynamic range is approximately one, this
// typically produces significant speedups with negligible change in accuracy.
// There is additional complexity when working with INT8, see Calibration.
//
// - FP32
// - FP16 Enable FP16 layer selection, with FP32 fallback.
// - INT8 Enable Int8 layer selection, with FP32 and FP16 fallback.
//
// Note that TensorRT will still choose a higher-precision kernel if it results
// in overall lower runtime, or if no low-precision implementation exists.
enum class TrtPrecisionMode { FP32, FP16, INT8 };

Status TrtPrecisionModeToName(const TrtPrecisionMode mode, string* name);

Status TrtPrecisionModeFromName(const string& name, TrtPrecisionMode* mode);

string DebugString(const TrtPrecisionMode mode);

// Optimization profile generation strategies.
// - `kRange`: create one profile that works for inputs with dimension values
//   in the range of [min_dims, max_dims] where min_dims and max_dims are
//   derived from the provided inputs.
// - `kOptimal`: create one profile for each input. The profile only works for
//   inputs with the same dimensions as the input it is created for. The GPU
//   engine will be run with optimal performance with such inputs.
// - `kRangeOptimal`: create the profiles for both `Range` and `Optimal`.
// - `kImplicitBatchModeCompatible`: create the profiles that will produce the
//   same GPU engines as the implicit_batch_mode would produce.
enum class ProfileStrategy {
  kRange,
  kOptimal,
  kRangeOptimal,
  kImplicitBatchModeCompatible,
};

string ProfileStrategyToName(const ProfileStrategy strategy);
Status ProfileStrategyFromName(const string& name, ProfileStrategy* strategy);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_TRT_PARAMETERS_H_
