/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
Status HloInputOutputAliasConfig::SetUpAlias(const ShapeIndex& output_index,
                                             int64 param_number,
                                             const ShapeIndex& param_index) {
  TF_RET_CHECK(ShapeUtil::IndexIsValid(alias_.shape(), output_index))
      << absl::StrCat("Tring to set up alias at ", output_index.ToString(),
                      " which is an invalid index for shape ",
                      ShapeUtil::HumanString(alias_.shape()));
  // Output can't be aliased with multiple parameters.
  TF_RET_CHECK(!alias_.element(output_index)) << absl::StrFormat(
      "Trying to set up output alias for param %lld at %s but failed: output "
      "index %s is already aliased with param %lld at %s",
      param_number, param_index.ToString(), output_index.ToString(),
      alias_.element(output_index)->first,
      alias_.element(output_index)->second.ToString());
  (*alias_.mutable_element(output_index)) =
      std::make_pair(param_number, param_index);
  VLOG(4) << "Set up alias between output index " << output_index.ToString()
          << " and parameter " << param_index << " at index "
          << param_index.ToString();
  return Status::OK();
}

HloInputOutputAliasProto HloInputOutputAliasConfig::ToProto() const {
  HloInputOutputAliasProto result;
  alias_.ForEachElement(
      [&](const ShapeIndex& index,
          const absl::optional<std::pair<int64, ShapeIndex>>& data) {
        if (data) {
          HloInputOutputAliasProto::AliasEntryProto entry;
          for (int64 i : index) {
            entry.add_output_shape_index(i);
          }
          entry.set_parameter_number(data->first);
          for (int64 i : data->second) {
            entry.add_parameter_shape_index(i);
          }
          result.add_entries()->Swap(&entry);
        }
      });
  return result;
}

StatusOr<HloInputOutputAliasConfig> HloInputOutputAliasConfig::CreateFromProto(
    const Shape& output_shape, const HloInputOutputAliasProto& proto) {
  HloInputOutputAliasConfig result(output_shape);
  for (const HloInputOutputAliasProto::AliasEntryProto& entry :
       proto.entries()) {
    ShapeIndex output_index(entry.output_shape_index().begin(),
                            entry.output_shape_index().end());

    int64 param_number = entry.parameter_number();
    ShapeIndex param_index(entry.parameter_shape_index().begin(),
                           entry.parameter_shape_index().end());
    TF_RETURN_IF_ERROR(
        result.SetUpAlias(output_index, param_number, param_index));
  }

  return result;
}

string HloInputOutputAliasConfig::ToString() const {
  std::vector<string> pieces;
  pieces.push_back("HloInputOutputAliasConfig");

  ForEachAlias([&](const ShapeIndex& output_index, int64 param_number,
                   const ShapeIndex& param_index) {
    pieces.push_back(absl::StrFormat(
        "  OutputIndex %s is aliased with parameter %lld at %s:",
        output_index.ToString(), param_number, param_index.ToString()));
  });

  return absl::StrJoin(pieces, "\n");
}

bool HloInputOutputAliasConfig::ParameterHasAlias(
    int64 param_number, const ShapeIndex& param_index) const {
  bool output = false;
  alias_.ForEachElement(
      [&](const xla::ShapeIndex&,
          absl::optional<std::pair<int64, ShapeIndex>> alias) {
        if (alias && alias->first == param_number &&
            alias->second == param_index) {
          output = true;
        }
      });
  return output;
}

absl::optional<ShapeIndex> HloInputOutputAliasConfig::GetAliasedOutput(
    int64 param_number, const ShapeIndex& param_index) const {
  absl::optional<ShapeIndex> output;
  alias_.ForEachElement(
      [&](const xla::ShapeIndex& output_index,
          absl::optional<std::pair<int64, ShapeIndex>> alias) {
        if (alias && alias->first == param_number &&
            alias->second == param_index) {
          output = output_index;
        }
      });
  return output;
}

absl::optional<std::pair<int64, ShapeIndex>>
HloInputOutputAliasConfig::GetAliasedParameter(
    const ShapeIndex& output_index) const {
  CHECK(ShapeUtil::IndexIsValid(alias_.shape(), output_index));
  return alias_.element(output_index);
}

void HloInputOutputAliasConfig::ForEachAlias(AliasFn fn) const {
  alias_.ForEachElement(
      [&](const ShapeIndex& output_index,
          absl::optional<std::pair<int64, ShapeIndex>> aliased) {
        if (aliased) {
          fn(output_index, aliased->first, aliased->second);
        }
      });
}

Status HloInputOutputAliasConfig::ForEachAliasWithStatus(
    AliasFnWithStatus fn) const {
  return alias_.ForEachElementWithStatus(
      [&](const ShapeIndex& output_index,
          absl::optional<std::pair<int64, ShapeIndex>> aliased) {
        if (aliased) {
          TF_RETURN_IF_ERROR(fn(output_index, aliased->first, aliased->second));
        }
        return Status::OK();
      });
}

Status HloInputOutputAliasConfig::Verify(
    const HloModule& module,
    std::function<int64(const Shape&)> size_func) const {
  std::vector<ShapeTree<bool>> param_has_seen;
  const HloComputation* entry = module.entry_computation();
  for (int64 i = 0; i < entry->num_parameters(); ++i) {
    HloInstruction* param = entry->parameter_instruction(i);
    param_has_seen.emplace_back(param->shape());
  }
  return ForEachAliasWithStatus([&](const ShapeIndex& output_index,
                                    int64 param_number,
                                    const ShapeIndex& param_index) -> Status {
    const HloInstruction* root = entry->root_instruction();

    TF_RET_CHECK(0 <= param_number);
    TF_RET_CHECK(entry->num_parameters() > param_number);
    const Shape& param_shape =
        entry->parameter_instruction(param_number)->shape();
    const Shape& output_shape = root->shape();
    TF_RET_CHECK(ShapeUtil::IndexIsValid(param_shape, param_index));
    TF_RET_CHECK(ShapeUtil::IndexIsValid(output_shape, output_index));

    const Shape& param_subshape =
        ShapeUtil::GetSubshape(param_shape, param_index);
    const Shape& output_subshape =
        ShapeUtil::GetSubshape(output_shape, output_index);
    TF_RET_CHECK(LayoutUtil::IsDenseArray(param_subshape));
    TF_RET_CHECK(LayoutUtil::IsDenseArray(output_subshape));

    if (size_func(param_subshape) != size_func(output_subshape)) {
      return InternalError(
          "Expected aliased input %lld at index %s and output at index %s to "
          "have the same size. Input sub-shape is %s with size %lld, output "
          "sub-shape is %s with size %lld",
          param_number, param_index.ToString(), output_index.ToString(),
          ShapeUtil::HumanStringWithLayout(param_subshape),
          size_func(param_subshape),
          ShapeUtil::HumanStringWithLayout(output_subshape),
          size_func(output_subshape));
    }

    // Check each param_number and param_index pair only show up once. No
    // input can be aliased with output buffers.
    TF_RET_CHECK(param_has_seen[param_number].element(param_index) == false);

    *(param_has_seen[param_number].mutable_element(param_index)) = true;

    return Status::OK();
  });
}

std::ostream& operator<<(std::ostream& out,
                         const HloInputOutputAliasConfig& config) {
  out << config.ToString();
  return out;
}
}  // namespace xla
