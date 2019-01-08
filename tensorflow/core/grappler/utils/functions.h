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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_FUNCTIONS_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_FUNCTIONS_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace tensorflow {
namespace grappler {

// Depending on the function instantiation attributes, input argument to the
// function might be a single tensor, list of tensors of the same type, or a
// list of tensors of different types.
//
// InputArgExpansion keeps track of the placeholders that were added to the
// function body in place of function inputs and a resolved input data type.
struct InputArgExpansion {
  // TODO(ezhulenev): Add support for functions with tensor sequence inputs of
  // different data types.
  // TODO(ezhulenev): Support type parametrized inputs?
  string input_name;                 // name of the function input argument
  DataType data_type;                // input data type
  bool is_ref;                       // if true, inputs are required to be refs
  std::vector<string> placeholders;  // names of placeholder nodes in the
                                     // function body
};

// Depending on the function instantiation attributes, output argument is mapped
// to one or more outputs of one of the function body nodes.
//
// OutputArgExpansion keeps mapping from a function output arg to the output
// tensors of a function body nodes and a resolved output data type
struct OutputArgExpansion {
  // TODO(ezhulenev): Add support for functions with tensor sequence outputs of
  // different data types.
  // TODO(ezhulenev): Support type parametrized outputs?
  string output_name;                  // name of the function output argument
  DataType data_type;                  // output data type
  bool is_ref;                         // if true, outputs are refs
  std::vector<string> output_tensors;  // names of output tensor from the
                                       // function body nodes
};

// FunctionDef uses different connectivity encoding for the function body nodes,
// then a GraphDef (see function.proto for details). Input name in FunctionDef
// can potentially represent a sequence of tensors (instead just one tensor in
// GraphDef), we need to expand it when converting from FunctionDef to GraphDef,
// and fold it back when doing backward conversion.
class GrapplerFunctionConnectivity {
 public:
  void RegisterInputArgExpansion(InputArgExpansion input_arg_expansion);
  void RegisterFunctionBodyOutputs(const string& node_name,
                                   tensorflow::NameRangeMap&& outputs);

  // Expands input encoded in FunctionDef format (name[:output][:position]) into
  // multiple inputs in GraphDef format (name[:position]).
  Status ExpandFunctionDefInput(const string& func_def_input,
                                std::vector<string>* graph_def_inputs) const;

  // Updates Node inputs from FunctionDef to GraphDef format.
  Status ExpandNodeInputs(NodeDef* function_body_node) const;

  // When expanding inputs in function def format, single input might be
  // expanded into multiple tensors. When converting back to the function def
  // format from graph def format, it's always a 1-to-1 relationship.
  // FunctionDef built from GrapplerFunctionItem is always specialized to it's
  // instantiation attributes and length of input args (and node def outputs) is
  // known.

  // Converts input name from GraphDef format (name[:position]) to the
  // FunctionDef input format (name[:output][:position]) using registered input
  // arg expansion and function body outputs.
  Status AsFunctionDefInput(const string& graph_def_input,
                            string* func_def_input) const;

  // Updates Node inputs from GraphDef to FunctionDef format.
  Status AsFunctionDefNode(NodeDef* function_body_node) const;

 private:
  // Mapping from input name to input arg expansion.
  absl::flat_hash_map<string, InputArgExpansion> input_arg_expansions_;
  // Mapping from function body node name to output names range map.
  absl::flat_hash_map<string, tensorflow::NameRangeMap> function_body_outputs_;

  // For each placeholder added to the function instantiation graph, we keep a
  // mapping back to the function input argument name and index.
  struct InputArgPlaceholder {
    string input_name;  // Name of the function input argument.
    int input_index;    // Index of a tensor in the function input argument
                        // expansion, it can be greater than `0` if input
                        // argument is a list of tensors (aka list(type)).
  };
  // Mapping from input arg placeholder to the function input tensor.
  absl::flat_hash_map<string, InputArgPlaceholder> input_arg_placeholders_;
};

// Get Function type attributes using attributes of a node that instantiated
// a function.
class GrapplerFunctionItemInstantiation {
 public:
  explicit GrapplerFunctionItemInstantiation(AttrSlice func_instantiation_attr)
      : func_instantiation_attr_(func_instantiation_attr) {}

  // Get DataType from attributes by name. Return error if attribute is missing,
  // or it doesn't define a valid data type.
  Status GetTypeAttr(const string& type_attr_name, DataType* data_type) const;

  // Get argument data type. If data type is not explicitly defined, uses
  // provided attribute name to look it up in function attributes.
  Status GetArgType(const OpDef::ArgDef& arg, DataType* data_type) const;

 private:
  const AttrSlice func_instantiation_attr_;  // do not own
};

// A special case of GrapplerItem, constructed from a TensorFlow Function.
class GrapplerFunctionItem : public GrapplerItem {
 public:
  GrapplerFunctionItem() = default;

  const string& description() const;

  bool IsInputPlaceholder(const string& node_name) const;

  const std::vector<InputArgExpansion>& inputs() const;
  const InputArgExpansion& input(int i) const;
  const std::size_t input_size() const;

  const std::vector<OutputArgExpansion>& outputs() const;
  const OutputArgExpansion& output(int i) const;
  const std::size_t output_size() const;

  const AttrSlice& func_attr() const;
  const GraphDef& function_body() const;
  GraphDef& mutable_function_body();

  bool is_stateful() const;

  GrapplerFunctionItem& SwapFunctionBody(GraphDef&& other);

 private:
  friend Status MakeGrapplerFunctionItem(const FunctionDef&, const AttrSlice&,
                                         const FunctionLibraryDefinition&, int,
                                         GrapplerFunctionItem*);
  friend Status ReplaceInputWithConst(const NodeDef&, int,
                                      GrapplerFunctionItem*);
  friend Status RemoveUnusedOutputs(
      const absl::flat_hash_set<int>& active_outputs,
      GrapplerFunctionItem* item,
      std::vector<std::pair<int, int>>* output_mapping);

  GrapplerFunctionItem(string func_name, string description,
                       AttrSlice func_attr,
                       std::vector<InputArgExpansion> input_arg_expansions,
                       std::vector<OutputArgExpansion> output_arg_expansions,
                       std::vector<string> keep_nodes, int graph_def_version,
                       bool is_stateful, GraphDef&& function_body);

  string description_;
  AttrSlice func_attr_;  // Attributes specific to function definition that
                         // produced this item (FuncDef.attr field).

  std::vector<InputArgExpansion> input_arg_expansions_;
  std::vector<OutputArgExpansion> output_arg_expansions_;

  std::set<string> input_arg_placeholders_;

  bool is_stateful_ = false;
};

// Check if function input/output types are fully defined only at instantiation
// time (parametrized by it's instantiation node).
bool HasParametrizedType(const FunctionDef& func);

// Check if a function body is parametrized by it's instantiation node. Function
// body is parametrized, if it has at least one node with a 'placeholder'
// attribute.
bool HasParametrizedBody(const FunctionDef& func);

// Check if function has parametrized type or body.
bool IsParametrized(const FunctionDef& func);

// Resolve function instantiation type parameters from the attributes of the
// caller node. Return error if type can't be resolved.
Status InstantiationTypeParameters(
    const FunctionDef& func, const AttrSlice& func_instantiation_attr,
    absl::flat_hash_map<string, DataType>* type_parameters);

// Resolve function instantiation body parameters (values for the function body
// attr placeholders) from the attributes of the caller node. Return error if
// type can't be resolved.
Status InstantiationBodyParameters(
    const FunctionDef& func, const AttrSlice& func_instantiation_attr,
    absl::flat_hash_map<string, AttrValue>* body_parameters);

// Register GrapplerFunctionItem input arg expansion and function body outputs
// in the GrapplerFunctionConnectivity. Use function library definition to
// lookup function body nodes output names and ranges.
Status RegisterGrapplerFunctionConnectivity(
    const GrapplerFunctionItem& item, const FunctionLibraryDefinition& flib,
    GrapplerFunctionConnectivity* connectivity);

// Replace one of the function inputs with a constant.
Status ReplaceInputWithConst(const NodeDef& input_const, int input_index,
                             GrapplerFunctionItem* item);

// Remove function output arguments that do not have any active outputs (output
// tensor connected to other node inputs or in a fetch set). Active outputs uses
// GraphDef output position encoding, and multiple active outputs could
// potentially be connected to the same output argument (in case of tensor list
// outputs). Add output mapping for all active outputs that changed it's output
// position (std::pair<old position, new position>).
Status RemoveUnusedOutputs(const absl::flat_hash_set<int>& active_outputs,
                           GrapplerFunctionItem* item,
                           std::vector<std::pair<int, int>>* output_mapping);

// Make a GrapplerFunctionItem from the function definition and function
// instantiation attributes (caller node attributes). Returns error if the given
// function def cannot be converted (e.g. not all attributes are defined).
Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const AttrSlice& func_instantiation_attr,
                                const FunctionLibraryDefinition& flib,
                                int graph_def_version,
                                GrapplerFunctionItem* item);

// Make a GrapplerFunction item from the function definition. Function must be
// fully defined (no type or body parametrization).
// TODO(ezhulenev): Support parametrized functions without fully defined
// instantiation attributes? Do we ever want to optimize parametrized function
// without specializing it to it's instantiation attributes (at least types)?
Status MakeGrapplerFunctionItem(const FunctionDef& func,
                                const FunctionLibraryDefinition& flib,
                                int graph_def_version,
                                GrapplerFunctionItem* item);

// Make a FunctionDef from the GrapplerFunctionItem. Use function library
// definition to lookup function body nodes output names and ranges.
Status MakeFunctionDef(const GrapplerFunctionItem& item,
                       const FunctionLibraryDefinition& flib,
                       FunctionDef* func);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_FUNCTIONS_H_
