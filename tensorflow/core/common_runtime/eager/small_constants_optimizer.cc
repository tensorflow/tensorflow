/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/small_constants_optimizer.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow::small_constants_optimizer {
namespace {

constexpr char kRuntimeConstantOptimization[] = "runtime_constant_optimization";

FunctionDef FoldBoolInputTensor(const FunctionDef& fdef,
                                absl::string_view input_name,
                                const bool input_value) {
  // TODO(b/272805674):
  // -  Inline the evaluated conditional expressions by promoting the
  //    appropriate true/false branch into the function body.
  // -  Repeat process recursively for each promoted
  //    Partitioned/StatefulPartitionedCall.
  // -  Compare with trace time constant folding.
  // -  Compare with grappler constant folding.
  FunctionDef result = fdef;

  // Rename the new fdef.
  result.mutable_signature()->set_name(
      FoldedFunctionName(fdef.signature().name(), input_name, input_value));

  // Remove boolean tensor from input arg signature.
  result.mutable_signature()->clear_input_arg();
  for (const auto& input_arg : fdef.signature().input_arg()) {
    if (input_arg.name() == input_name) continue;
    *result.mutable_signature()->add_input_arg() = input_arg;
  }

  // Point all references of the boolean tensor to the constant tensor.
  for (auto& node_def : *result.mutable_node_def()) {
    for (auto& node_name : *node_def.mutable_input()) {
      if (node_name == input_name) {
        node_name.append(":output:0");
      }
    }
  }

  // Insert top level const tensor node.
  auto* const_tensor = result.add_node_def();
  const_tensor->set_name(std::string(input_name));
  const_tensor->set_op("Const");
  AttrValue dtype_value;
  dtype_value.set_type(DT_BOOL);
  const_tensor->mutable_attr()->emplace("dtype", dtype_value);
  AttrValue tensor_value;
  auto* tensor = tensor_value.mutable_tensor();
  tensor->set_dtype(DT_BOOL);
  tensor->mutable_tensor_shape();
  tensor->add_bool_val(true);
  const_tensor->mutable_attr()->emplace("value", tensor_value);

  return result;
}

void DisableBoolInputFolding(FunctionDef& fdef) {
  auto it = fdef.mutable_attr()->find(kRuntimeConstantOptimization);
  if (it == fdef.mutable_attr()->end()) return;
  it->second.set_b(false);
}

// Returns a list of input arguments that have dtype tf.bool in a FunctionDef.
// NOTE: This function requires that the FunctionDef outlive the returned
// result.
std::vector<absl::string_view> GetBoolInputNames(const FunctionDef& fdef) {
  std::vector<absl::string_view> result;
  for (const auto& input_arg : fdef.signature().input_arg()) {
    if (input_arg.type() == DT_BOOL) {
      result.emplace_back(input_arg.name());
    }
  }
  return result;
}

void GenerateTrueAndFalseFunctions(const FunctionDef& fdef,
                                   std::vector<FunctionDef>& result) {
  // Limit the folding to functions that have a single boolean tensor input to
  // avoid exponential FunctionDef creation. We might consider easing this
  // restriction later.
  std::vector<absl::string_view> bool_input_names = GetBoolInputNames(fdef);
  if (bool_input_names.size() != 1) return;
  absl::string_view input_name_to_fold = bool_input_names[0];

  // Add f_true.
  auto true_fdef = FoldBoolInputTensor(fdef, input_name_to_fold, true);
  DisableBoolInputFolding(true_fdef);
  result.push_back(std::move(true_fdef));
  // Add f_false.
  auto false_fdef = FoldBoolInputTensor(fdef, input_name_to_fold, false);
  DisableBoolInputFolding(false_fdef);
  result.push_back(std::move(false_fdef));
}

}  // namespace

std::string FoldedFunctionName(absl::string_view fname,
                               absl::string_view input_name, bool input_value) {
  return absl::StrCat(fname, "_", input_name, "_", input_value);
}

bool IsSmallConstantOptimizationEnabled(const FunctionDef& fdef) {
  auto it = fdef.attr().find(kRuntimeConstantOptimization);
  if (it == fdef.attr().end()) return false;
  return it->second.b();
}

std::vector<FunctionDef> FoldInputTensors(
    const FunctionDef& fdef, const FunctionLibraryDefinition& flib) {
  std::vector<FunctionDef> result;
  if (!IsSmallConstantOptimizationEnabled(fdef)) return result;

  // Add f_true and f_false to the result.
  GenerateTrueAndFalseFunctions(fdef, result);

  return result;
}

}  // namespace tensorflow::small_constants_optimizer
