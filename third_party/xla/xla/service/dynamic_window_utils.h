/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_DYNAMIC_WINDOW_UTILS_H_
#define XLA_SERVICE_DYNAMIC_WINDOW_UTILS_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/statusor.h"
#include "xla/types.h"
namespace xla {
struct DynamicWindowDims {
  HloInstruction* padding_before;
  HloInstruction* output_size;
};

// This mirrors the logic in GetWindowedOutputSizeVerbose but with HLOs as
// inputs and outputs.
DynamicWindowDims GetWindowedOutputSize(HloInstruction* input_size,
                                        int64_t window_size,
                                        int64_t window_dilation,
                                        int64_t window_stride,
                                        PaddingType padding_type);

DynamicWindowDims GetWindowedInputGradSize(HloInstruction* input_size,
                                           int64_t window_size,
                                           int64_t window_dilation,
                                           int64_t window_stride,
                                           PaddingType padding_type);

}  // namespace xla

#endif  // XLA_SERVICE_DYNAMIC_WINDOW_UTILS_H_
