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
  for (const auto& attr : function_def.attr()) {
    if (attr.first == "experimental_api_preferred_device") {
      preferred_device_ = attr.second.s();
    }
    if (attr.first == "experimental_api_implements") {
      interface_name_ = attr.second.s();
    }
  }
  if (interface_name_.empty() && !preferred_device_.empty()) {
    return errors::InvalidArgument(
        "Function '", function_def.signature().name(),
        "' has a preferred device, but does not implement an interface");
  }
  return Status::OK();
}

const string& FunctionApiInfo::preferred_device() const {
  return preferred_device_;
}

const string& FunctionApiInfo::interface_name() const {
  return interface_name_;
}

FunctionLibraryApiInfo::FunctionLibraryApiInfo() {}
FunctionLibraryApiInfo::~FunctionLibraryApiInfo() {}

namespace {
bool IsSameSignature(const FunctionDef& f1, const FunctionDef& f2) {
  if (f1.ret().size() != f2.ret().size()) return false;
  const auto& sig1 = f1.signature();
  const auto& sig2 = f2.signature();
  // Functions have positional semantics, so we don't check for names.
  if (sig1.input_arg_size() != sig2.input_arg_size()) return false;
  for (int k = 0; k < sig1.input_arg_size(); ++k) {
    const OpDef::ArgDef& arg1 = sig1.input_arg(k);
    const OpDef::ArgDef& arg2 = sig2.input_arg(k);
    if (arg1.type() != arg2.type()) return false;
    if (arg1.type_attr() != arg2.type_attr()) return false;
    if (arg1.number_attr() != arg2.number_attr()) return false;
    if (arg1.type_list_attr() != arg2.type_list_attr()) return false;
    if (arg1.is_ref() != arg2.is_ref()) return false;
  }
  return true;
}

Status ValidateSignature(const string& interface_name,
                         const std::vector<const FunctionDef*>& equiv_funcs) {
  if (equiv_funcs.size() < 2) return Status::OK();
  for (size_t k = 1; k < equiv_funcs.size(); ++k) {
    if (!IsSameSignature(*equiv_funcs[0], *equiv_funcs[k]))
      return errors::InvalidArgument(
          "Functions '", equiv_funcs[0]->signature().name(), "' and '",
          equiv_funcs[k]->signature().name(), "' both implement '",
          interface_name, "' but their signatures do not match.");
  }
  return Status::OK();
}

Status ValidateSignatures(
    const std::unordered_map<string, std::vector<const FunctionDef*>>&
        intf_to_func) {
  for (const auto& item : intf_to_func)
    TF_RETURN_IF_ERROR(ValidateSignature(item.first, item.second));
  return Status::OK();
}
}  // namespace

Status FunctionLibraryApiInfo::Init(
    const FunctionDefLibrary& function_library) {
  std::unordered_map<string, std::vector<const FunctionDef*>> intf_to_func;
  for (const auto& function : function_library.function()) {
    std::unique_ptr<FunctionApiInfo> func_info(new FunctionApiInfo);
    TF_RETURN_IF_ERROR(func_info->Init(function));
    // Ignore the function if it does not implement any interface.
    if (func_info->interface_name().empty()) continue;

    const string& function_name = function.signature().name();
    const string& interface_name = func_info->interface_name();
    func_to_intf_[function_name] = interface_name;
    intf_to_funcs_[interface_name].emplace_back(function_name);
    intf_to_func[interface_name].emplace_back(&function);
    func_info_[function_name] = std::move(func_info);
  }
  TF_RETURN_IF_ERROR(ValidateSignatures(intf_to_func));
  return Status::OK();
}

void FunctionLibraryApiInfo::GetEquivalentImplementations(
    const string& function_name, std::vector<string>* other_names) const {
  const auto intf_it = func_to_intf_.find(function_name);
  // The function does not implement any interface.
  if (intf_it == func_to_intf_.end()) return;
  CHECK(!intf_it->second.empty()) << "Function " << function_name
                                  << "should at least implement 1 interface.";
  const auto it = intf_to_funcs_.find(intf_it->second);
  CHECK(it != intf_to_funcs_.end())
      << "Function " << function_name << " maps to " << intf_it->second
      << " but no reverse mapping was found";
  CHECK_GE(it->second.size(), 1) << "Class " << it->first << " is empty";
  other_names->reserve(it->second.size() - 1);
  for (const auto& other_name : it->second) {
    if (other_name == function_name) continue;
    other_names->emplace_back(other_name);
  }
}

void FunctionLibraryApiInfo::GetBestImplementation(
    const string& function_name, const string& device,
    string* best_func_name) const {
  CHECK(best_func_name != nullptr);
  const auto func_it = func_to_intf_.find(function_name);
  if (func_it == func_to_intf_.end()) return;

  const auto it = intf_to_funcs_.find(func_it->second);
  // No function found for the given interface.
  if (it == intf_to_funcs_.end()) return;
  for (const auto& func_name : it->second) {
    const auto func_api_info = func_info_.find(func_name)->second.get();
    if (func_api_info->preferred_device() == device) {
      best_func_name->assign(func_name);
      return;
    }
  }
  // Didn't find a function with the match device name, choose the first one
  // among all the available functions.
  best_func_name->assign(it->second.front());
}

const FunctionApiInfo* FunctionLibraryApiInfo::GetApiInfo(
    const string& function_name) const {
  const auto it = func_info_.find(function_name);
  if (it == func_info_.end()) return nullptr;
  return it->second.get();
}

}  // end namespace grappler
}  // end namespace tensorflow
