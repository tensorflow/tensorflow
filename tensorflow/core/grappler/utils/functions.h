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

#ifndef TENSORFLOW_GRAPPLER_UTILS_FUNCTIONS_H_
#define TENSORFLOW_GRAPPLER_UTILS_FUNCTIONS_H_

#include <memory>
#include <string>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

// Depending on the function instantiation attributes, input argument to the
// function might be a single tensor, list of tensors of the same type, or a
// list of tensors of different types.
//
// InputArgExpansion keeps track of the placeholders that were added to the
// function body in place of function inputs.
struct InputArgExpansion {
  string input_name;                 // name of the function input argument
  std::vector<string> placeholders;  // names of placeholder nodes in the
                                     // function body
};

// Depending on the function instantiation attributes, output argument is mapped
// to one or more outputs of one of the function body nodes.
//
// OutputArgExpansion keeps mapping from a function output arg to the output
// tensors of a function body nodes, that compute function outputs.
struct OutputArgExpansion {
  string output_name;                  // name of the function output argument
  std::vector<string> output_tensors;  // names of output tensors from the
                                       // function body graph nodes
};

// FunctionDef uses different connectivity encoding for the function body nodes,
// then a GraphDef (see function.proto for details). Input name in FunctionDef
// can potentially represent a sequence of tensors (instead just one tensor in
// GraphDef), we need to expand it when converting from FunctionDef to GraphDef,
// and fold it back when doing backward conversion.
class GrapplerFunctionConnectivity {
 public:
  void RegisterInputArgExpansion(const InputArgExpansion& input_arg_expansion);
  void RegisterFunctionBodyOutputs(const string& node_name,
                                   const tensorflow::NameRangeMap& outputs);

  // Expand input encoded in FunctionDef format (name[:output][:position]) into
  // multiple inputs in GraphDef format (name[:position]).
  Status ExpandFunctionDefInput(const string& func_def_input,
                                std::vector<string>* graph_def_inputs) const;

  // Update Node inputs from FunctionDef to GraphDef format
  Status ExpandNodeInputs(NodeDef* function_body_node) const;

  // TODO(ezhulenev): fold GraphDef inputs back to FunctionDef format
  // Status FoldGraphDefInputs(const std::vector<sting> graph_def_inputs,
  //                          std::vector<string>* function_def_inputs) const;

 private:
  std::unordered_map<string, InputArgExpansion> input_arg_expansions_;
  std::unordered_map<string, tensorflow::NameRangeMap> function_body_outputs_;
};

// Helper methods to build GrapplerFunctionItem from a function def and function
// attributes.
class GrapplerFunctionItemBuilder {
 public:
  using FunctionAttr = std::unordered_map<string, AttrValue>;

  explicit GrapplerFunctionItemBuilder(const FunctionAttr* func_attr)
      : func_attr_(func_attr) {}

  // Get DataType from attributes by name. Return error if attribute is missing,
  // or it doesn't define a valid data type.
  Status GetTypeAttr(const string& type_attr_name, DataType* data_type) const;

  // Get argument data type. If data type is not explicitly defined, uses
  // provided attribute name to look it up in function attributes.
  Status GetArgType(const OpDef::ArgDef& arg, DataType* data_type) const;

 private:
  const FunctionAttr* func_attr_;  // do not own
};

// A special case of GrapplerItem, constructed from a TensorFlow Function.
class GrapplerFunctionItem : public GrapplerItem {
 public:
  GrapplerFunctionItem() {}
  GrapplerFunctionItem(
      const string& function_name,
      const std::vector<InputArgExpansion>& input_arg_expansions,
      const std::vector<OutputArgExpansion>& output_arg_expansions,
      GraphDef&& function_body);

  const string& function_name() const;

  const std::vector<InputArgExpansion>& inputs() const;
  const InputArgExpansion& input(int i) const;
  const std::size_t input_size() const;

  const std::vector<OutputArgExpansion>& outputs() const;
  const OutputArgExpansion& output(int i) const;
  const std::size_t output_size() const;

  const GraphDef& function_body() const;
  GraphDef& mutable_function_body();

 private:
  string function_name_;
  std::vector<InputArgExpansion> input_arg_expansions_;
  std::vector<OutputArgExpansion> output_arg_expansions_;
};

// Return all output tensors referenced by item output args.
std::vector<string> OutputTensors(const GrapplerFunctionItem& item);

// Make a GrapplerFunctionItem from the function definition and attributes.
// Return error if the given function def cannot be converted.
Status MakeGrapplerFunctionItem(
    const FunctionDef& func,
    const std::unordered_map<string, AttrValue>& func_attr,
    const FunctionLibraryDefinition& func_library, GrapplerFunctionItem* item);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_UTILS_FUNCTIONS_H_
