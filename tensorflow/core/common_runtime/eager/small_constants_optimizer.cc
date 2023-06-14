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

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
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
constexpr char kIfOp[] = "If";
constexpr char kStatelessIfOp[] = "StatelessIf";
constexpr char kPartitionedCallOp[] = "PartitionedCall";
constexpr char kStatefulPartitionedCallOp[] = "StatefulPartitionedCall";

// Limit the folding to functions that have a single boolean tensor input to
// avoid exponential FunctionDef creation. We might consider easing this
// restriction later.
constexpr int32_t kMaxBoolArguments = 1;

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

// Recursively folds the boolean input tensor as a constant tensor into the
// FunctionDef and its dependent functions.
std::vector<FunctionDef> FoldBoolInputTensor(
    const FunctionDef& fdef, absl::string_view input_name, bool input_value,
    bool delete_input, const FunctionLibraryDefinition& flib,
    absl::flat_hash_set<std::string>& folded_functions);

// Updates the name of the nested function referred to by `nested_func` and
// extends `results` with the folded versions of the underlying FunctionDefs.
void FoldBoolInputTensorInNestedFunction(
    const int32_t input_idx, const bool input_value, const NodeDef& outer_ndef,
    const FunctionLibraryDefinition& flib, NameAttrList& nested_func,
    std::vector<FunctionDef>& results,
    absl::flat_hash_set<std::string>& folded_functions) {
  // Extract the FunctionDef and the appropriate input argument of the nested
  // function to fold.
  const std::string& fname = nested_func.name();
  const FunctionDef* internal_fdef = flib.Find(fname);
  if (internal_fdef == nullptr) return;

  // Validate the input arguments of the inner function.
  if (outer_ndef.input_size() != internal_fdef->signature().input_arg_size()) {
    return;
  }
  if (internal_fdef->signature().input_arg(input_idx).type() != DT_BOOL) {
    return;
  }

  // Update the nested function's name.
  const std::string& internal_name_to_fold =
      internal_fdef->signature().input_arg(input_idx).name();
  nested_func.set_name(
      FoldedFunctionName(fname, internal_name_to_fold, input_value));

  // Fold the nested function's boolean input into its FunctionDef.
  auto folded_funcs =
      FoldBoolInputTensor(*internal_fdef, internal_name_to_fold, input_value,
                          /*delete_input=*/false, flib, folded_functions);
  results.insert(results.end(), std::make_move_iterator(folded_funcs.begin()),
                 std::make_move_iterator(folded_funcs.end()));
}

std::vector<FunctionDef> FoldBoolInputTensor(
    const FunctionDef& fdef, absl::string_view input_name,
    const bool input_value, const bool delete_input,
    const FunctionLibraryDefinition& flib,
    absl::flat_hash_set<std::string>& folded_functions) {
  std::vector<FunctionDef> results;
  // TODO(b/272805674):
  // -  Inline the evaluated conditional expressions by promoting the
  //    appropriate true/false branch into the function body.
  // -  Compare with trace time constant folding.
  const std::string folded_function_name =
      FoldedFunctionName(fdef.signature().name(), input_name, input_value);
  if (folded_functions.contains(folded_function_name)) return results;
  FunctionDef result = fdef;

  // Rename the new fdef.
  result.mutable_signature()->set_name(folded_function_name);

  // Remove boolean tensor from input arg signature.
  // TODO(b/272805674): Investigate if deleting input at top level makes any
  //                    difference in performance.
  if (delete_input) {
    result.mutable_signature()->clear_input_arg();
    for (const auto& input_arg : fdef.signature().input_arg()) {
      if (input_arg.name() == input_name) continue;
      *result.mutable_signature()->add_input_arg() = input_arg;
    }
  }

  // Update nodes in the FunctionDef to use the correct inputs and
  // PartitionedCalls.
  for (auto& node_def : *result.mutable_node_def()) {
    // Note: `inputs` will be invalidated after the handling for `If` and
    // `StatelessIf` ops.
    auto& inputs = *node_def.mutable_input();
    auto it = std::find(inputs.begin(), inputs.end(), input_name);

    // Only process nodes that have the boolean tensor input defined by
    // `input_name`.
    if (it == inputs.end()) continue;
    int32_t input_idx = it - inputs.begin();

    // Point all references of the boolean tensor to the constant tensor.
    it->append("_rt_const_opt:output:0");

    // Inline the true/false PartitionedCall for cond ops relying on the const
    // bool tensor.
    if (node_def.op() == kIfOp || node_def.op() == kStatelessIfOp) {
      // Update the Node's op type.
      if (node_def.op() == kIfOp) node_def.set_op(kStatefulPartitionedCallOp);
      if (node_def.op() == kStatelessIfOp) node_def.set_op(kPartitionedCallOp);

      // Update the node's inputs. The cond op does not pass its first argument
      // to the true/false PartitionedCalls.
      const auto node_inputs = node_def.input();
      node_def.clear_input();
      for (int32_t i = 1; i < node_inputs.size(); ++i) {
        node_def.add_input(node_inputs[i]);
      }

      // Update the node's attributes.
      const auto node_attrs = node_def.attr();
      node_def.clear_attr();
      for (const auto& [attr_key, attr_value] : node_attrs) {
        // Skip redundant attributes.
        if (attr_key == "Tcond") continue;
        if (attr_key == "_lower_using_switch_merge") continue;

        // Promote the true branch when input_value is `true`.
        if (attr_key == "then_branch") {
          if (input_value) node_def.mutable_attr()->insert({"f", attr_value});
          continue;
        }
        // Promote the false branch when input_value is `false`.
        if (attr_key == "else_branch") {
          if (!input_value) node_def.mutable_attr()->insert({"f", attr_value});
          continue;
        }

        // All other attributes should be copied over.
        node_def.mutable_attr()->insert({attr_key, attr_value});
      }
    }

    // Recursively point the nested functions to the folded equivalent.
    for (auto& attr : *node_def.mutable_attr()) {
      AttrValue& attr_value = attr.second;
      // Nested functions can be stored in a node's attr in two ways.
      // 1. A single function is stored in the attr_value as AttrValue.func.
      if (attr_value.has_func()) {
        FoldBoolInputTensorInNestedFunction(input_idx, input_value, node_def,
                                            flib, *attr_value.mutable_func(),
                                            results, folded_functions);
      }

      // 2. A list of functions is stored as AttrValue.ListValue.func.
      if (attr_value.has_list()) {
        for (auto& func : *attr_value.mutable_list()->mutable_func()) {
          FoldBoolInputTensorInNestedFunction(input_idx, input_value, node_def,
                                              flib, func, results,
                                              folded_functions);
        }
      }
    }
  }

  // Insert top level const tensor node.
  auto* const_tensor = result.add_node_def();
  const_tensor->set_name(absl::StrCat(input_name, "_rt_const_opt"));
  const_tensor->set_op("Const");
  AttrValue dtype_value;
  dtype_value.set_type(DT_BOOL);
  const_tensor->mutable_attr()->insert({"dtype", dtype_value});
  AttrValue tensor_value;
  auto* tensor = tensor_value.mutable_tensor();
  tensor->set_dtype(DT_BOOL);
  tensor->mutable_tensor_shape();
  tensor->add_bool_val(true);
  const_tensor->mutable_attr()->insert({"value", tensor_value});

  // Mark the current `FunctionDef` as folded and return the results.
  results.push_back(std::move(result));
  folded_functions.insert(folded_function_name);
  return results;
}

void DisableBoolInputFolding(FunctionDef& fdef) {
  auto it = fdef.mutable_attr()->find(kRuntimeConstantOptimization);
  if (it == fdef.mutable_attr()->end()) return;
  it->second.set_b(false);
}

void GenerateTrueAndFalseFunctions(const FunctionDef& fdef,
                                   const FunctionLibraryDefinition& flib,
                                   std::vector<FunctionDef>& result) {
  std::vector<absl::string_view> bool_input_names = GetBoolInputNames(fdef);
  if (bool_input_names.size() != kMaxBoolArguments) return;
  absl::string_view input_name_to_fold = bool_input_names[0];
  absl::flat_hash_set<std::string> folded_functions;

  // Add f_true(s).
  auto true_fdefs =
      FoldBoolInputTensor(fdef, input_name_to_fold, /*input_value=*/true,
                          /*delete_input=*/true, flib, folded_functions);
  for (FunctionDef& fdef : true_fdefs) DisableBoolInputFolding(fdef);
  result.insert(result.end(), std::make_move_iterator(true_fdefs.begin()),
                std::make_move_iterator(true_fdefs.end()));
  // Add f_false(s).
  auto false_fdefs =
      FoldBoolInputTensor(fdef, input_name_to_fold, /*input_value=*/false,
                          /*delete_input=*/true, flib, folded_functions);
  for (FunctionDef& fdef : false_fdefs) DisableBoolInputFolding(fdef);
  result.insert(result.end(), std::make_move_iterator(false_fdefs.begin()),
                std::make_move_iterator(false_fdefs.end()));
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
  GenerateTrueAndFalseFunctions(fdef, flib, result);

  return result;
}

}  // namespace tensorflow::small_constants_optimizer
