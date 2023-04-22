/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/image/sampling_kernels.h"

#include <string>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace functor {

SamplingKernelType SamplingKernelTypeFromString(const StringPiece str) {
  const string lower_case = absl::AsciiStrToLower(str);
  if (lower_case == "lanczos1") return Lanczos1Kernel;
  if (lower_case == "lanczos3") return Lanczos3Kernel;
  if (lower_case == "lanczos5") return Lanczos5Kernel;
  if (lower_case == "gaussian") return GaussianKernel;
  if (lower_case == "box") return BoxKernel;
  if (lower_case == "triangle") return TriangleKernel;
  if (lower_case == "keyscubic") return KeysCubicKernel;
  if (lower_case == "mitchellcubic") return MitchellCubicKernel;
  return SamplingKernelTypeEnd;
}

}  // namespace functor
}  // namespace tensorflow
