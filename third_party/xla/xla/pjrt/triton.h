/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_TRITON_H_
#define XLA_PJRT_TRITON_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla::triton {

struct CompilationResult {
  std::string asm_text;
  int64_t smem_bytes;
  int cluster_dim_x;
  int cluster_dim_y;
  int cluster_dim_z;
};

absl::StatusOr<CompilationResult> Compile(absl::string_view module,
                                          absl::string_view arch_name,
                                          int num_warps, int num_ctas,
                                          int num_stages);

}  // namespace xla::triton

#endif  // XLA_PJRT_TRITON_H_
