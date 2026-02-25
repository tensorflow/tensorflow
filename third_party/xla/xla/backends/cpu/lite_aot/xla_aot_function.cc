/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/lite_aot/xla_aot_function.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<XlaAotFunction>> XlaAotFunction::Create(
    const CompilationResultProto& compilation_result) {
  TF_ASSIGN_OR_RETURN(
      ProgramShape program_shape,
      ProgramShape::FromProto(
          compilation_result.hlo_module().hlo_module().host_program_shape()));

  TF_ASSIGN_OR_RETURN(
      auto nanort_executable,
      NanoRtExecutable::Create(compilation_result, program_shape));

  std::vector<Literal> results_literals;
  if (nanort_executable->program_shape()->result().IsTuple()) {
    results_literals.reserve(
        nanort_executable->program_shape()->result().tuple_shapes().size());
    for (const Shape& shape :
         nanort_executable->program_shape()->result().tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(results_literals.emplace_back(),
                          Literal::Make(shape, /*allocate_arrays=*/true));
    }

  } else {
    TF_ASSIGN_OR_RETURN(
        results_literals.emplace_back(),
        Literal::Make(nanort_executable->program_shape()->result(),
                      /*allocate_arrays=*/true));
  }

  TF_ASSIGN_OR_RETURN(
      Literal temp_literal,
      Literal::Make(
          ShapeUtil::MakeShape(U8, {static_cast<int64_t>(
                                       nanort_executable->temp_buffer_size())}),
          /*allocate_arrays=*/true));

  return absl::WrapUnique(new XlaAotFunction(std::move(nanort_executable),
                                             std::move(results_literals),
                                             std::move(temp_literal)));
}

XlaAotFunction::XlaAotFunction(std::unique_ptr<NanoRtExecutable> executable,
                               std::vector<Literal> results_literals,
                               Literal temp_literal)
    : executable_(std::move(executable)),
      results_literals_(std::move(results_literals)),
      temp_literal_(std::move(temp_literal)) {
  // We don't back this with literals as users should set it themselves.
  auto program_shape = executable_->program_shape().value();
  arguments_.reserve(program_shape.parameters_size());
  for (size_t i = 0; i < program_shape.parameters_size(); ++i) {
    argument_sizes_.push_back(
        ShapeUtil::ByteSizeOfElements(program_shape.parameters(i)));
    arguments_.emplace_back(nullptr, 0);
  }

  results_.reserve(results_literals_.size());
  for (auto& result_literal : results_literals_) {
    results_.emplace_back(result_literal.untyped_data(),
                          result_literal.size_bytes());
  }

  temp_ = NanoRtExecutable::PreallocatedTemp(
      static_cast<std::byte*>(temp_literal_.untyped_data()),
      temp_literal_.size_bytes());
}

absl::Status XlaAotFunction::Execute() {
  auto event = executable_->Execute(arguments_, results_, temp_);
  tsl::BlockUntilReady(event);
  if (event.IsError()) {
    return event.GetError();
  }
  return absl::OkStatus();
}

}  // namespace xla::cpu
