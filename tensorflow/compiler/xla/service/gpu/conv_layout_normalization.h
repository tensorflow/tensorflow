/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CONV_LAYOUT_NORMALIZATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CONV_LAYOUT_NORMALIZATION_H_

#include <functional>
#include <optional>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace gpu {

StatusOr<std::optional<HloInstruction*>>
NormalizeLayoutForCustomCallConvolution(HloCustomCallInstruction*);

}  // end namespace gpu
}  // end namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CONV_LAYOUT_NORMALIZATION_H_
