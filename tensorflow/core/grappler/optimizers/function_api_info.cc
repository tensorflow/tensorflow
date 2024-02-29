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

#include "tensorflow/core/grappler/optimizers/function_api_info.h"

#include <string>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
FunctionApiInfo::FunctionApiInfo() {}
FunctionApiInfo::~FunctionApiInfo() {}

Status FunctionApiInfo::Init(const FunctionDef& function_def) {
  function_type_ = FunctionApiInfo::FunctionType::INFERENCE;
  for (const auto& attr : function_def.attr()) {
    if (attr.first == "api_preferred_device") {
      preferred_device_ = attr.second.s();
    }
    if (attr.first == "api_implements") {
      interface_name_ = attr.second.s();
    }
    if (attr.first == "forward_function_name") {
      function_type_ = FunctionApiInfo::FunctionType::BACKWARD;
      pairing_function_name_ = attr.second.s();
    }
    if (attr.first == "backward_function_name") {
      function_type_ = FunctionApiInfo::FunctionType::FORWARD;
      pairing_function_name_ = attr.second.s();
    }
  }

  input_arg_dtypes_.reserve(function_def.signature().input_arg_size());
  for (const auto& input_arg : function_def.signature().input_arg()) {
    input_arg_dtypes_.emplace_back(input_arg.type());
  }
  output_arg_dtypes_.reserve(function_def.signature().output_arg_size());
  for (const auto& output_arg : function_def.signature().output_arg()) {
    output_arg_dtypes_.emplace_back(output_arg.type());
  }

  if (interface_name_.empty() && !preferred_device_.empty()) {
    return errors::InvalidArgument(
        "Function '", function_def.signature().name(),
        "' has a preferred device, but does not implement an interface");
  }
  return absl::OkStatus();
}

const string& FunctionApiInfo::preferred_device() const {
  return preferred_device_;
}

const string& FunctionApiInfo::interface_name() const {
  return interface_name_;
}

const FunctionApiInfo::FunctionType FunctionApiInfo::function_type() const {
  return function_type_;
}

const string& FunctionApiInfo::pairing_function_name() const {
  return pairing_function_name_;
}

const DataTypeVector& FunctionApiInfo::input_arg_dtypes() const {
  return input_arg_dtypes_;
}

const DataTypeVector& FunctionApiInfo::output_arg_dtypes() const {
  return output_arg_dtypes_;
}

FunctionLibraryApiInfo::FunctionLibraryApiInfo() {}
FunctionLibraryApiInfo::~FunctionLibraryApiInfo() {}

namespace {
bool IsSameArgDef(const OpDef::ArgDef& arg1, const OpDef::ArgDef& arg2) {
  if (arg1.type() != arg2.type()) return false;
  if (arg1.type_attr() != arg2.type_attr()) return false;
  if (arg1.number_attr() != arg2.number_attr()) return false;
  if (arg1.type_list_attr() != arg2.type_list_attr()) return false;
  if (arg1.is_ref() != arg2.is_ref()) return false;
  return true;
}

bool IsSameSignature(const FunctionDef& f1, const FunctionDef& f2,
                     const bool check_inputs, const bool check_outputs) {
  const auto& sig1 = f1.signature();
  const auto& sig2 = f2.signature();
  // Functions have positional semantics, so we don't check for names.
  if (check_inputs) {
    if (sig1.input_arg_size() != sig2.input_arg_size()) return false;
    for (int k = 0; k < sig1.input_arg_size(); ++k) {
      if (!IsSameArgDef(sig1.input_arg(k), sig2.input_arg(k))) return false;
    }
  }
  if (check_outputs) {
    if (f1.ret().size() != f2.ret().size()) return false;
    if (sig1.output_arg_size() != sig2.output_arg_size()) return false;
    for (int k = 0; k < sig1.output_arg_size(); ++k) {
      if (!IsSameArgDef(sig1.output_arg(k), sig2.output_arg(k))) return false;
    }
  }
  return true;
}

Status ValidateSignature(const string& interface_name,
                         const std::vector<const FunctionDef*>& equiv_funcs,
                         const FunctionApiInfo::FunctionType function_type) {
  if (equiv_funcs.size() < 2) return absl::OkStatus();
  for (size_t k = 1; k < equiv_funcs.size(); ++k) {
    const bool check_input =
        (function_type == FunctionApiInfo::FunctionType::INFERENCE ||
         function_type == FunctionApiInfo::FunctionType::FORWARD);
    const bool check_output =
        (function_type == FunctionApiInfo::FunctionType::INFERENCE ||
         function_type == FunctionApiInfo::FunctionType::BACKWARD);
    if (!IsSameSignature(*equiv_funcs[0], *equiv_funcs[k], check_input,
                         check_output)) {
      return errors::InvalidArgument(
          "Functions '", equiv_funcs[0]->signature().name(), "' and '",
          equiv_funcs[k]->signature().name(), "' both implement '",
          interface_name, "' but their signatures do not match.");
    }
  }
  return absl::OkStatus();
}

Status ValidateSignatures(
    const std::unordered_map<string, std::vector<const FunctionDef*>>&
        intf_to_func,
    const FunctionApiInfo::FunctionType function_type) {
  for (const auto& item : intf_to_func)
    TF_RETURN_IF_ERROR(
        ValidateSignature(item.first, item.second, function_type));
  return absl::OkStatus();
}
}  // namespace

Status FunctionLibraryApiInfo::Init(
    const FunctionDefLibrary& function_library) {
  std::unordered_map<string, std::vector<const FunctionDef*>> infer_funcs;
  std::unordered_map<string, std::vector<const FunctionDef*>> fwd_funcs;
  std::unordered_map<string, std::vector<const FunctionDef*>> bwd_funcs;
  for (const auto& function : function_library.function()) {
    std::unique_ptr<FunctionApiInfo> func_info(new FunctionApiInfo);
    TF_RETURN_IF_ERROR(func_info->Init(function));
    // Ignore the function if it does not implement any interface.
    if (func_info->interface_name().empty()) continue;

    const string& function_name = function.signature().name();
    const string& interface_name = func_info->interface_name();
    VLOG(3) << "Got " << func_info->function_type()
            << " function: " << function_name
            << " with interface: " << interface_name;
    switch (func_info->function_type()) {
      case FunctionApiInfo::FunctionType::INFERENCE:
        intf_to_inference_funcs_[interface_name].emplace_back(function_name);
        infer_funcs[interface_name].emplace_back(&function);
        break;
      case FunctionApiInfo::FunctionType::FORWARD:
        intf_to_forward_funcs_[interface_name].emplace_back(function_name);
        fwd_funcs[interface_name].emplace_back(&function);
        break;
      case FunctionApiInfo::FunctionType::BACKWARD:
        intf_to_backward_funcs_[interface_name].emplace_back(function_name);
        bwd_funcs[interface_name].emplace_back(&function);
        break;
      default:
        return errors::InvalidArgument("Unrecognized function type: ",
                                       func_info->function_type());
    }
    func_info_[function_name] = std::move(func_info);
  }
  TF_RETURN_IF_ERROR(ValidateSignatures(
      infer_funcs, FunctionApiInfo::FunctionType::INFERENCE));
  TF_RETURN_IF_ERROR(
      ValidateSignatures(fwd_funcs, FunctionApiInfo::FunctionType::FORWARD));
  TF_RETURN_IF_ERROR(
      ValidateSignatures(bwd_funcs, FunctionApiInfo::FunctionType::BACKWARD));
  return absl::OkStatus();
}

Status FunctionLibraryApiInfo::GetEquivalentImplementations(
    const string& function_name, std::vector<string>* other_functions) const {
  const auto func_it = func_info_.find(function_name);
  if (func_it == func_info_.end()) return absl::OkStatus();
  const FunctionApiInfo* func_info = func_it->second.get();

  absl::flat_hash_map<string, std::vector<string>>::const_iterator it;
  switch (func_info->function_type()) {
    case FunctionApiInfo::FunctionType::INFERENCE:
      it = intf_to_inference_funcs_.find(func_info->interface_name());
      break;
    case FunctionApiInfo::FunctionType::FORWARD:
      it = intf_to_forward_funcs_.find(func_info->interface_name());
      break;
    case FunctionApiInfo::FunctionType::BACKWARD:
      it = intf_to_backward_funcs_.find(func_info->interface_name());
      break;
    default:
      return errors::InvalidArgument("Unrecognized function type: ",
                                     func_info->function_type());
  }

  for (const auto& func_name : it->second) {
    if (func_name == function_name) continue;
    other_functions->emplace_back(func_name);
  }
  return absl::OkStatus();
}

const FunctionApiInfo* FunctionLibraryApiInfo::GetApiInfo(
    const string& function_name) const {
  const auto it = func_info_.find(function_name);
  if (it == func_info_.end()) return nullptr;
  return it->second.get();
}

}  // end namespace grappler
}  // end namespace tensorflow
