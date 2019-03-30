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

absl::string_view ThunkKindToString(Thunk::Kind kind) {
  switch (kind) {
    case Thunk::kCholesky:
      return "kCholesky";
    case Thunk::kConditional:
      return "kConditional";
    case Thunk::kConvolution:
      return "kConvolution";
    case Thunk::kCopy:
      return "kCopy";
    case Thunk::kCudnnBatchNormBackward:
      return "kCudnnBatchNormBackward";
    case Thunk::kCudnnBatchNormForwardInference:
      return "kCudnnBatchNormForwardInference";
    case Thunk::kCudnnBatchNormForwardTraining:
      return "kCudnnBatchNormForwardTraining";
    case Thunk::kNcclAllReduce:
      return "kNcclAllReduce";
    case Thunk::kFft:
      return "kFft";
    case Thunk::kGemm:
      return "kGemm";
    case Thunk::kInfeed:
      return "kInfeed";
    case Thunk::kKernel:
      return "kKernel";
    case Thunk::kMemset32BitValue:
      return "kMemset32BitValue";
    case Thunk::kMemzero:
      return "kMemzero";
    case Thunk::kOutfeed:
      return "kOutfeed";
    case Thunk::kSequential:
      return "kSequential";
    case Thunk::kTriangularSolve:
      return "kTriangularSolve";
    case Thunk::kTuple:
      return "kTuple";
    case Thunk::kWhile:
      return "kWhile";
  }
}

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind) {
  return os << ThunkKindToString(kind);
}

}  // namespace gpu
}  // namespace xla
