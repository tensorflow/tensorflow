/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_input_output_alias_config.h"

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {

bool HloInputOutputAliasConfig::OutputHasAlias(
    const ShapeIndex& output_index) const {
  return alias_.element(output_index).has_value();
}

absl::Status HloInputOutputAliasConfig::SetUpAlias(
    const ShapeIndex& output_index, int64_t param_number,
    const ShapeIndex& param_index,
    HloInputOutputAliasConfig::AliasKind must_alias) {
  TF_RET_CHECK(ShapeUtil::IndexIsValid(alias_.shape(), output_index))
      << "Trying to set up alias at " << output_index.ToString()
      << " which is an invalid index for shape "
      << ShapeUtil::HumanString(alias_.shape());
  TF_RET_CHECK(param_number >= 0) << param_number;
  // Output can't be aliased with multiple parameters.
  TF_RET_CHECK(!alias_.element(output_index)) << absl::StrFormat(
      "Trying to set up output alias for param %lld at %s but failed: output "
      "index %s is already aliased with param %lld at %s",
      param_number, param_index.ToString(), output_index.ToString(),
      alias_.element(output_index)->parameter_number,
      alias_.element(output_index)->parameter_index.ToString());
  (*alias_.mutable_element(output_index)) =
      Alias(param_number, param_index, must_alias);
  VLOG(4) << "Set up alias between output index " << output_index.ToString()
          << " and parameter " << param_number << " at index "
          << param_index.ToString();
  return absl::OkStatus();
}

HloInputOutputAliasProto HloInputOutputAliasConfig::ToProto() const {
  HloInputOutputAliasProto result;
  alias_.ForEachElement(
      [&](const ShapeIndex& index, const std::optional<Alias>& data) {
        if (data) {
          HloInputOutputAliasProto::AliasEntryProto entry;
          for (int64_t i : index) {
            entry.add_output_shape_index(i);
          }
          entry.set_parameter_number(data->parameter_number);
          for (int64_t i : data->parameter_index) {
            entry.add_parameter_shape_index(i);
          }
          if (data->must_alias()) {
            entry.set_kind(Kind::MUST_ALIAS);
          } else {
            entry.set_kind(Kind::MAY_ALIAS);
          }
          result.add_entries()->Swap(&entry);
        }
      });
  return result;
}

absl::StatusOr<HloInputOutputAliasConfig>
HloInputOutputAliasConfig::CreateFromProto(
    Shape output_shape, const HloInputOutputAliasProto& proto) {
  HloInputOutputAliasConfig result(std::move(output_shape));
  for (const HloInputOutputAliasProto::AliasEntryProto& entry :
       proto.entries()) {
    ShapeIndex output_index(entry.output_shape_index().begin(),
                            entry.output_shape_index().end());
    int64_t param_number = entry.parameter_number();
    ShapeIndex param_index(entry.parameter_shape_index().begin(),
                           entry.parameter_shape_index().end());
    AliasKind kind = entry.kind() == Kind::MAY_ALIAS ? kMayAlias : kMustAlias;
    TF_RETURN_IF_ERROR(
        result.SetUpAlias(output_index, param_number, param_index, kind));
  }
  return result;
}

const Shape& HloInputOutputAliasConfig::shape() const { return alias_.shape(); }

std::string HloInputOutputAliasConfig::ToString() const {
  std::vector<std::string> pieces;
  pieces.push_back("HloInputOutputAliasConfig");
  pieces.push_back(
      absl::StrFormat("  Output shape: %s", alias_.shape().ToString()));

  ForEachAlias([&](const ShapeIndex& output_index, const Alias& alias) {
    pieces.push_back(absl::StrFormat(
        "  OutputIndex %s is %saliased with parameter %lld at %s:",
        output_index.ToString(), alias.kind == kMustAlias ? "must-" : "may-",
        alias.parameter_number, alias.parameter_index.ToString()));
  });
  return absl::StrJoin(pieces, "\n");
}

std::string HloInputOutputAliasConfig::ToShortString() const {
  std::vector<std::string> pieces;
  for (const auto& p : alias_) {
    const ShapeIndex& index = p.first;
    if (std::optional<Alias> alias = p.second) {
      pieces.push_back(
          absl::StrFormat("%s: %s", index.ToString(), alias->ToString()));
    }
  }
  return absl::StrJoin(pieces, ", ");
}

bool HloInputOutputAliasConfig::ParameterMustAlias(
    int64_t param_number, const ShapeIndex& param_index) const {
  bool result = false;
  alias_.ForEachElement(
      [&](const xla::ShapeIndex&, std::optional<Alias> alias) {
        if (alias && alias->parameter_number == param_number &&
            alias->parameter_index == param_index && alias->must_alias()) {
          result = true;
        }
      });
  return result;
}

std::optional<ShapeIndex> HloInputOutputAliasConfig::GetAliasedOutput(
    int64_t param_number, const ShapeIndex& param_index) const {
  // We use reverse iterator to preserve the semantics of
  // alias_.ForEachElement() which was used before.
  for (auto it = alias_.rbegin(); it != alias_.rend(); ++it) {
    if (it->second.has_value() &&
        it->second->parameter_number == param_number &&
        it->second->parameter_index == param_index) {
      return it->first;
    }
  }
  return std::nullopt;
}

std::optional<HloInputOutputAliasConfig::Alias>
HloInputOutputAliasConfig::GetAliasedParameter(
    const ShapeIndex& output_index) const {
  CHECK(ShapeUtil::IndexIsValid(alias_.shape(), output_index))
      << ToString() << " " << alias_.shape().ToString() << " " << output_index;
  return alias_.element(output_index);
}

void HloInputOutputAliasConfig::ForEachAlias(AliasFn fn) const {
  alias_.ForEachElement(
      [&](const ShapeIndex& output_index, std::optional<Alias> aliased) {
        if (aliased) {
          fn(output_index, *aliased);
        }
      });
}

absl::Status HloInputOutputAliasConfig::ForEachAliasWithStatus(
    AliasFnWithStatus fn) const {
  return alias_.ForEachElementWithStatus(
      [&](const ShapeIndex& output_index, std::optional<Alias> aliased) {
        if (aliased) {
          TF_RETURN_IF_ERROR(fn(output_index, *aliased));
        }
        return absl::OkStatus();
      });
}

absl::Status HloInputOutputAliasConfig::Verify(
    const HloModule& module,
    absl::FunctionRef<int64_t(const Shape&)> size_func) const {
  std::vector<ShapeTree<bool>> param_has_seen;
  const HloComputation* entry = module.entry_computation();
  for (int64_t i = 0; i < entry->num_parameters(); ++i) {
    HloInstruction* param = entry->parameter_instruction(i);
    param_has_seen.emplace_back(param->shape());
  }
  return ForEachAliasWithStatus([&](const ShapeIndex& output_index,
                                    const Alias& alias) -> absl::Status {
    TF_RET_CHECK(0 <= alias.parameter_number);
    TF_RET_CHECK(entry->num_parameters() > alias.parameter_number);
    const Shape& param_shape =
        module.entry_computation_layout().parameter_shape(
            alias.parameter_number);
    const Shape& output_shape =
        module.entry_computation_layout().result_shape();
    TF_RET_CHECK(ShapeUtil::IndexIsValid(param_shape, alias.parameter_index));
    TF_RET_CHECK(ShapeUtil::IndexIsValid(output_shape, output_index));

    const Shape& param_subshape =
        ShapeUtil::GetSubshape(param_shape, alias.parameter_index);
    const Shape& output_subshape =
        ShapeUtil::GetSubshape(output_shape, output_index);
    TF_RET_CHECK(LayoutUtil::IsDenseArray(param_subshape));
    TF_RET_CHECK(LayoutUtil::IsDenseArray(output_subshape));

    if (size_func(param_subshape) != size_func(output_subshape)) {
      return Internal(
          "Expected aliased input %lld at index %s and output at index %s to "
          "have the same size. Input sub-shape is %s with size %lld, output "
          "sub-shape is %s with size %lld",
          alias.parameter_number, alias.parameter_index.ToString(),
          output_index.ToString(),
          ShapeUtil::HumanStringWithLayout(param_subshape),
          size_func(param_subshape),
          ShapeUtil::HumanStringWithLayout(output_subshape),
          size_func(output_subshape));
    }

    // Check each alias.parameter_number and alias.parameter_index pair only
    // show up once. No input can be aliased with output buffers.
    TF_RET_CHECK(param_has_seen[alias.parameter_number].element(
                     alias.parameter_index) == false);
    *(param_has_seen[alias.parameter_number].mutable_element(
        alias.parameter_index)) = true;
    return absl::OkStatus();
  });
}

std::ostream& operator<<(std::ostream& out,
                         const HloInputOutputAliasConfig& config) {
  out << config.ToString();
  return out;
}

absl::Status HloBufferDonorConfig::AddBufferDonor(
    int64_t param_number, const ShapeIndex& param_index) {
  TF_RET_CHECK(param_number >= 0) << param_number;
  VLOG(4) << "Register the parameter " << param_number << " at index "
          << param_index.ToString() << " as a buffer donor.";
  buffer_donor_.emplace(BufferDonor(param_number, param_index));
  return absl::OkStatus();
}

absl::Status HloBufferDonorConfig::RemoveBufferDonor(
    int64_t param_number, const ShapeIndex& param_index) {
  TF_RET_CHECK(param_number >= 0) << param_number;
  buffer_donor_.erase(BufferDonor(param_number, param_index));
  return absl::OkStatus();
}

HloBufferDonorProto HloBufferDonorConfig::ToProto() const {
  HloBufferDonorProto result;
  for (const auto& donor : buffer_donor_) {
    HloBufferDonorProto::BufferDonorEntryProto entry;
    entry.set_parameter_number(donor.param_number);
    for (int64_t i : donor.param_index) {
      entry.add_parameter_shape_index(i);
    }
    result.add_entries()->Swap(&entry);
  }
  return result;
}

absl::StatusOr<HloBufferDonorConfig> HloBufferDonorConfig::CreateFromProto(
    const HloBufferDonorProto& proto) {
  HloBufferDonorConfig result;
  for (const HloBufferDonorProto::BufferDonorEntryProto& entry :
       proto.entries()) {
    int64_t param_number = entry.parameter_number();
    ShapeIndex param_index(entry.parameter_shape_index().begin(),
                           entry.parameter_shape_index().end());
    TF_RETURN_IF_ERROR(result.AddBufferDonor(param_number, param_index));
  }
  return result;
}

std::string HloBufferDonorConfig::ToString() const {
  std::vector<std::string> pieces;
  pieces.push_back("HloBufferDonorConfig");
  for (const auto& donor : buffer_donor_) {
    pieces.push_back(absl::StrFormat(
        "  Parameter %lld at %s is registered as a buffer donor.",
        donor.param_number, donor.param_index.ToString()));
  }
  return absl::StrJoin(pieces, "\n");
}

std::string HloBufferDonorConfig::ToShortString() const {
  std::vector<std::string> pieces;
  pieces.reserve(buffer_donor_.size());
  for (const auto& donor : buffer_donor_) {
    pieces.push_back(absl::StrFormat("(%lld, %s)", donor.param_number,
                                     donor.param_index.ToString()));
  }
  return absl::StrJoin(pieces, ", ");
}

bool HloBufferDonorConfig::ParameterIsBufferDonor(
    int64_t param_number, const ShapeIndex& param_index) const {
  auto it = buffer_donor_.find(BufferDonor(param_number, param_index));
  return it != buffer_donor_.end();
}

absl::Status HloBufferDonorConfig::Verify(const HloModule& module) const {
  const HloComputation* entry = module.entry_computation();
  const auto& alias_config = module.input_output_alias_config();
  for (const auto& donor : buffer_donor_) {
    TF_RET_CHECK(donor.param_number >= 0);
    TF_RET_CHECK(donor.param_number < entry->num_parameters());

    const Shape& param_shape =
        module.entry_computation_layout().parameter_shape(donor.param_number);
    TF_RET_CHECK(ShapeUtil::IndexIsValid(param_shape, donor.param_index));

    const Shape& param_subshape =
        ShapeUtil::GetSubshape(param_shape, donor.param_index);
    TF_RET_CHECK(LayoutUtil::IsDenseArray(param_subshape));

    if (alias_config.ParameterHasAlias(donor.param_number, donor.param_index)) {
      return Internal(
          "Input %lld at index %s is registered as a buffer donor. However, it "
          "is also in the input output alias config.",
          donor.param_number, donor.param_index.ToString());
    }
  }

  // Since buffer_donor_ is a set, we do not need to check if one input has
  // registered as a buffer donor many times.
  return absl::OkStatus();
}

std::ostream& operator<<(std::ostream& out,
                         const HloBufferDonorConfig& config) {
  out << config.ToString();
  return out;
}

}  // namespace xla
