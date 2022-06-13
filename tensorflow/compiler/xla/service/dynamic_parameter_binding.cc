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

#include "tensorflow/compiler/xla/service/dynamic_parameter_binding.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

Status DynamicParameterBinding::Bind(
    const DynamicParameter& dynamic_parameter,
    const DynamicDimension& dynamic_dimension) {
  auto result = bindings_.emplace(dynamic_dimension, dynamic_parameter);
  TF_RET_CHECK(result.second);
  return OkStatus();
}

std::optional<DynamicParameterBinding::DynamicParameter>
DynamicParameterBinding::GetBinding(
    const DynamicDimension& dynamic_dimension) const {
  auto param_iter = bindings_.find(dynamic_dimension);
  if (param_iter == bindings_.end()) {
    return std::nullopt;
  }
  return param_iter->second;
}

DynamicParameterBindingProto DynamicParameterBinding::ToProto() const {
  DynamicParameterBindingProto result;
  for (const auto& binding : bindings_) {
    const DynamicDimension& dynamic_dimension = binding.first;
    const DynamicParameter& dynamic_param = binding.second;
    DynamicParameterBindingProto::Binding binding_proto;
    binding_proto.set_dynamic_param_num(dynamic_param.parameter_num);
    for (int64_t i : dynamic_param.parameter_index) {
      binding_proto.add_dynamic_param_index(i);
    }

    binding_proto.set_target_param_num(dynamic_dimension.parameter_num);

    for (int64_t i : dynamic_dimension.parameter_index) {
      binding_proto.add_target_param_index(i);
    }

    binding_proto.set_target_param_dim_num(dynamic_dimension.dimension);
    result.add_entries()->Swap(&binding_proto);
  }
  return result;
}

StatusOr<DynamicParameterBinding> DynamicParameterBinding::CreateFromProto(
    const DynamicParameterBindingProto& proto) {
  DynamicParameterBinding result;
  for (const DynamicParameterBindingProto::Binding& binding : proto.entries()) {
    int64_t dynamic_param_num = binding.dynamic_param_num();
    ShapeIndex dynamic_param_index(binding.dynamic_param_index().begin(),
                                   binding.dynamic_param_index().end());
    int64_t target_param_num = binding.target_param_num();
    ShapeIndex target_param_index(binding.target_param_index().begin(),
                                  binding.target_param_index().end());
    int64_t target_dim_num = binding.target_param_dim_num();

    TF_RETURN_IF_ERROR(
        result.Bind(DynamicParameter{dynamic_param_num, dynamic_param_index},
                    DynamicDimension{target_param_num, target_param_index,
                                     target_dim_num}));
  }

  return result;
}

std::string DynamicParameterBinding::ToString() const {
  std::vector<std::string> pieces;
  pieces.push_back("DynamicParameterBinding: ");
  for (const auto& binding : bindings_) {
    const DynamicDimension& dynamic_dimension = binding.first;
    const DynamicParameter& dynamic_param = binding.second;
    pieces.push_back(absl::StrFormat(
        " -- Input param number %lld at %s has dim %lld as dynamic"
        " dimension, which is represented by param number %lld at "
        "%s",
        dynamic_dimension.parameter_num,
        dynamic_dimension.parameter_index.ToString(),
        dynamic_dimension.dimension, dynamic_param.parameter_num,
        dynamic_param.parameter_index.ToString()));
  }
  return absl::StrJoin(pieces, "\n");
}

Status DynamicParameterBinding::ForEachBinding(BindingFn fn) const {
  for (const auto& binding : bindings_) {
    TF_RETURN_IF_ERROR(fn(binding.second, binding.first));
  }
  return OkStatus();
}

Status DynamicParameterBinding::Verify(const HloModule& module) const {
  const HloComputation* entry = module.entry_computation();
  return ForEachBinding([&](const DynamicParameter& dynamic_parameter,
                            const DynamicDimension& dynamic_dimension)
                            -> Status {
    TF_RET_CHECK(dynamic_parameter.parameter_num >= 0 &&
                 dynamic_parameter.parameter_num < entry->num_parameters());
    TF_RET_CHECK(dynamic_dimension.parameter_num < entry->num_parameters());
    TF_RET_CHECK(ShapeUtil::IndexIsValid(
        entry->parameter_instruction(dynamic_parameter.parameter_num)->shape(),
        dynamic_parameter.parameter_index));
    TF_RET_CHECK(ShapeUtil::IndexIsValid(
        entry->parameter_instruction(dynamic_dimension.parameter_num)->shape(),
        dynamic_dimension.parameter_index));
    TF_RET_CHECK(
        dynamic_dimension.dimension <
        ShapeUtil::GetSubshape(
            entry->parameter_instruction(dynamic_dimension.parameter_num)
                ->shape(),
            dynamic_dimension.parameter_index)
            .rank());
    return OkStatus();
  });
}

std::ostream& operator<<(std::ostream& out,
                         const DynamicParameterBinding& binding) {
  out << binding.ToString();
  return out;
}

}  // namespace xla
