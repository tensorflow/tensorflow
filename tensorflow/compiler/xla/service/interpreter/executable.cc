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

#include "tensorflow/compiler/xla/service/interpreter/executable.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/interpreter/executable_base.h"
#include "tensorflow/compiler/xla/service/interpreter/executor.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace interpreter {

InterpreterExecutable::InterpreterExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloEvaluator> evaluator,
    absl::optional<DynamicDimensionInference> dynamic_dymension_inference)
    : InterpreterExecutableBase(std::move(hlo_module)),
      evaluator_(std::move(evaluator)),
      dynamic_dimension_inference_(std::move(dynamic_dymension_inference)) {
  if (dynamic_dimension_inference_.has_value()) {
    evaluator_->set_dynamic_dimension_inference(
        &dynamic_dimension_inference_.value());
  }
}

StatusOr<Literal> InterpreterExecutable::Evaluate(
    const ServiceExecutableRunOptions* run_options,
    const HloComputation& computation, absl::Span<const Literal> arg_literals) {
  // Execute the graph using the HloEvaluator.
  tensorflow::mutex_lock lock(evaluator_lock_);
  evaluator_->ResetVisitStates();
  return evaluator_->Evaluate(computation, arg_literals);
}

/*static*/ int64 InterpreterExecutable::ShapeSizeBytes(const Shape& shape) {
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

}  // namespace interpreter
}  // namespace xla
