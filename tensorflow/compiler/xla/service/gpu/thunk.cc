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

#include "tensorflow/compiler/xla/service/gpu/thunk.h"

namespace xla {
namespace gpu {

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind) {
  switch (kind) {
    case Thunk::kConditional:
      return os << "kConditional";
    case Thunk::kConvolution:
      return os << "kConvolution";
    case Thunk::kCopy:
      return os << "kCopy";
    case Thunk::kCudnnBatchNormBackward:
      return os << "kCudnnBatchNormBackward";
    case Thunk::kCudnnBatchNormForwardInference:
      return os << "kCudnnBatchNormForwardInference";
    case Thunk::kCudnnBatchNormForwardTraining:
      return os << "kCudnnBatchNormForwardTraining";
    case Thunk::kFft:
      return os << "kFft";
    case Thunk::kGemm:
      return os << "kGemm";
    case Thunk::kInfeed:
      return os << "kInfeed";
    case Thunk::kKernel:
      return os << "kKernel";
    case Thunk::kMemset32BitValue:
      return os << "kMemset32BitValue";
    case Thunk::kMemzero:
      return os << "kMemzero";
    case Thunk::kOutfeed:
      return os << "kOutfeed";
    case Thunk::kSequential:
      return os << "kSequential";
    case Thunk::kTuple:
      return os << "kTuple";
    case Thunk::kWhile:
      return os << "kWhile";
  }
}

}  // namespace gpu
}  // namespace xla
