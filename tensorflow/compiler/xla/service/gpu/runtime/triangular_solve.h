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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_TRIANGULAR_SOLVE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_TRIANGULAR_SOLVE_H_

#include <string_view>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace gpu {

using runtime::CustomCall;

struct TriangularSolve {
  // Adaptor from XlaCustomCall API to properly typed TriangularSolve handler.
  static absl::Status run(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          CustomCall::RemainingArgs args,
                          std::string_view backend_config);

  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          runtime::StridedMemrefView a,
                          runtime::StridedMemrefView b,
                          runtime::StridedMemrefView result,
                          runtime::FlatMemrefView temp, bool left_side,
                          bool lower, bool unit_diagonal,
                          TriangularSolveOptions::Transpose transpose_a) const;

  static TriangularSolve Handler() { return TriangularSolve(); }
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_TRIANGULAR_SOLVE_H_
