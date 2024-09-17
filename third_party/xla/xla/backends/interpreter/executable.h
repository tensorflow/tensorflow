/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_INTERPRETER_EXECUTABLE_H_
#define XLA_BACKENDS_INTERPRETER_EXECUTABLE_H_

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/interpreter/executable_base.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace interpreter {

// Responsible for running a HLO graph through the HloEvaluator and output
// buffer allocation. Refer to interpreter/README.md for more.
class InterpreterExecutable : public InterpreterExecutableBase {
 public:
  InterpreterExecutable(
      std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<HloEvaluator> evaluator,
      std::optional<DynamicDimensionInference> dynamic_dymension_inference);

  static int64_t ShapeSizeBytes(const Shape& shape);

 protected:
  absl::StatusOr<Literal> Evaluate(
      const ServiceExecutableRunOptions* run_options,
      const HloComputation& computation,
      absl::Span<const Literal> arg_literals) override
      ABSL_LOCKS_EXCLUDED(evaluator_lock_);

  // The interpreter interprets executables with an HloEvaluator.
  std::unique_ptr<HloEvaluator> evaluator_ ABSL_PT_GUARDED_BY(evaluator_lock_);
  mutable absl::Mutex evaluator_lock_;

 private:
  std::optional<DynamicDimensionInference> dynamic_dimension_inference_;
  InterpreterExecutable(const InterpreterExecutable&) = delete;
  InterpreterExecutable& operator=(const InterpreterExecutable&) = delete;
};

}  // namespace interpreter
}  // namespace xla

#endif  // XLA_BACKENDS_INTERPRETER_EXECUTABLE_H_
