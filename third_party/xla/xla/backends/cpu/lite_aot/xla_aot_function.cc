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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/literal.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

namespace {

struct ExecutableAndSupportingLiterals {
  std::unique_ptr<NanoRtExecutable> nanort_executable;
  std::vector<Literal> results_literals;
  Literal temp_literal;
};

absl::StatusOr<ProgramShape> GetProgramShape(
    const NanoRtExecutable& nanort_executable) {
  auto maybe_nanort_program_shape = nanort_executable.program_shape();
  if (!maybe_nanort_program_shape.has_value()) {
    return absl::InternalError(
        "Program shape was not set in the NanoRtExecutable.");
  }
  return maybe_nanort_program_shape.value();
}

absl::StatusOr<ExecutableAndSupportingLiterals>
CreateExecutableAndSupportingLiterals(
    const CompilationResultProto& compilation_result) {
  TF_ASSIGN_OR_RETURN(
      ProgramShape program_shape,
      ProgramShape::FromProto(
          compilation_result.hlo_module().hlo_module().host_program_shape()));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<NanoRtExecutable> nanort_executable,
      NanoRtExecutable::Create(compilation_result, program_shape));

  std::vector<Literal> results_literals;

  TF_ASSIGN_OR_RETURN(auto nanort_program_shape,
                      GetProgramShape(*nanort_executable));
  if (nanort_program_shape.result().IsTuple()) {
    auto tuple_shapes = nanort_program_shape.result().tuple_shapes();
    results_literals.reserve(tuple_shapes.size());
    for (const Shape& shape : tuple_shapes) {
      TF_ASSIGN_OR_RETURN(results_literals.emplace_back(),
                          Literal::Make(shape, /*allocate_arrays=*/true));
    }

  } else {
    TF_ASSIGN_OR_RETURN(results_literals.emplace_back(),
                        Literal::Make(nanort_program_shape.result(),
                                      /*allocate_arrays=*/true));
  }

  TF_ASSIGN_OR_RETURN(
      Literal temp_literal,
      Literal::Make(
          ShapeUtil::MakeShape(U8, {static_cast<int64_t>(
                                       nanort_executable->temp_buffer_size())}),
          /*allocate_arrays=*/true));

  return ExecutableAndSupportingLiterals{std::move(nanort_executable),
                                         std::move(results_literals),
                                         std::move(temp_literal)};
}

bool AreStringsInVectorUnique(const std::vector<std::string>& strings) {
  absl::flat_hash_set<absl::string_view> unique_strings(strings.begin(),
                                                        strings.end());
  return unique_strings.size() == strings.size();
}
}  // namespace

absl::StatusOr<std::unique_ptr<XlaAotFunction>> XlaAotFunction::Create(
    const CompilationResultProto& compilation_result,
    std::vector<std::string> arg_names, std::vector<std::string> result_names) {
  if (!AreStringsInVectorUnique(arg_names)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Argument names must be unique. Got ", absl::StrJoin(arg_names, ",")));
  }
  if (!AreStringsInVectorUnique(result_names)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Result names must be unique. Got ", absl::StrJoin(result_names, ",")));
  }

  TF_ASSIGN_OR_RETURN(
      auto executable_and_supporting_literals,
      CreateExecutableAndSupportingLiterals(compilation_result));

  TF_ASSIGN_OR_RETURN(
      auto program_shape,
      GetProgramShape(*executable_and_supporting_literals.nanort_executable));

  if (program_shape.parameters_size() != arg_names.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Argument names size does not match the number "
                     "of arguments in the program shape. Got ",
                     arg_names.size(), " argument names but program shape has ",
                     program_shape.parameters_size(),
                     " arguments. Program shape: ", program_shape.ToString()));
  }

  if (executable_and_supporting_literals.results_literals.size() !=
      result_names.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Result names size does not match the number "
        "of results in the program shape. Got ",
        result_names.size(), " result names but program shape has ",
        executable_and_supporting_literals.results_literals.size(),
        " results. Program shape: ", program_shape.ToString()));
  }

  return absl::WrapUnique(new XlaAotFunction(
      std::move(executable_and_supporting_literals.nanort_executable),
      std::move(executable_and_supporting_literals.results_literals),
      std::move(executable_and_supporting_literals.temp_literal),
      std::move(arg_names), std::move(result_names)));
}

absl::StatusOr<std::unique_ptr<XlaAotFunction>> XlaAotFunction::Create(
    const CompilationResultProto& compilation_result) {
  TF_ASSIGN_OR_RETURN(
      auto executable_and_supporting_literals,
      CreateExecutableAndSupportingLiterals(compilation_result));

  auto& nanort_executable =
      executable_and_supporting_literals.nanort_executable;
  auto& results_literals = executable_and_supporting_literals.results_literals;
  auto& temp_literal = executable_and_supporting_literals.temp_literal;

  auto hlo_module = nanort_executable->executable()->shared_module();

  if (!hlo_module) {
    return absl::InternalError(
        "Cannot infer argument and result names because HLO module is null.");
  }

  std::vector<std::string> arg_names;
  arg_names.reserve(
      hlo_module->entry_computation()->parameter_instructions().size());
  for (const auto& instr :
       hlo_module->entry_computation()->parameter_instructions()) {
    arg_names.push_back(std::string(instr->name()));
  }
  std::vector<std::string> result_names;
  TF_ASSIGN_OR_RETURN(auto program_shape, GetProgramShape(*nanort_executable));
  if (program_shape.result().IsTuple()) {
    auto tuple_shapes = program_shape.result().tuple_shapes();
    absl::string_view root_name =
        hlo_module->entry_computation()->root_instruction()->name();

    result_names.reserve(tuple_shapes.size());
    for (int index = 0; index < tuple_shapes.size(); ++index) {
      result_names.push_back(absl::StrCat(root_name, "_", index));
    }

  } else {
    result_names.push_back(std::string(
        hlo_module->entry_computation()->root_instruction()->name()));
  }

  CHECK(AreStringsInVectorUnique(arg_names))
      << "Argument names must be unique. Got " << absl::StrJoin(arg_names, ",");
  CHECK(AreStringsInVectorUnique(result_names))
      << "Result names must be unique. Got "
      << absl::StrJoin(result_names, ",");

  return absl::WrapUnique(new XlaAotFunction(
      std::move(nanort_executable), std::move(results_literals),
      std::move(temp_literal), std::move(arg_names), std::move(result_names)));
}

XlaAotFunction::XlaAotFunction(std::unique_ptr<NanoRtExecutable> executable,
                               std::vector<Literal> results_literals,
                               Literal temp_literal,
                               std::vector<std::string> argument_names,
                               std::vector<std::string> result_names)
    : executable_(std::move(executable)),
      results_literals_(std::move(results_literals)),
      temp_literal_(std::move(temp_literal)) {
  VLOG(2) << "Creating XlaAotFunction with " << argument_names.size()
          << " arguments and " << result_names.size() << " results.";
  VLOG(5) << "Argument names: " << absl::StrJoin(argument_names, ",");
  VLOG(5) << "Result names: " << absl::StrJoin(result_names, ",");
  // We don't back this with literals as users should set it themselves.
  auto program_shape = executable_->program_shape().value();
  arguments_.reserve(program_shape.parameters_size());
  for (size_t i = 0; i < program_shape.parameters_size(); ++i) {
    argument_sizes_.push_back(
        ShapeUtil::ByteSizeOfElements(program_shape.parameters(i)));
    arguments_.emplace_back(nullptr, 0);
    name_to_argument_index_[argument_names[i]] = i;
  }

  results_.reserve(results_literals_.size());
  for (size_t i = 0; i < results_literals_.size(); ++i) {
    auto& result_literal = results_literals_[i];
    results_.emplace_back(result_literal.untyped_data(),
                          result_literal.size_bytes());
    name_to_result_index_[result_names[i]] = i;
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
