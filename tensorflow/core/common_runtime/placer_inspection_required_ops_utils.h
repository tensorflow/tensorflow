/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_INSPECTION_REQUIRED_OPS_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_INSPECTION_REQUIRED_OPS_UTILS_H_

// Operations calling functions are becoming ubiquitous in TF 2.0.
// Examples include PartitionedCallOp, functional If/While, and Dataset ops.
// Such operations might require deep inspection - looking at the body of the
// called function - to place them and surrounding ops correctly.

// This file contains some utilities for placer to correctly place such ops
// including:
// - PlacerInspectionRequiredOpChecker: A simple class with a single
// IsPlacerInspectionRequired method.
// - IsolatePlacerInspectionRequiredOps: This function adds Identity ops for
// each input/output of ops requiring placer inspection. It greatly simplifies
// the implementation of placing such ops.

#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// PlacerInspectionRequiredOpChecker allows one to check if Placer needs to
// look deeply into the op to place ops consuming the outputs correctly.
//
// It is a class instead of a standalone method because checking whether
// a function returns a resource takes non-trivial time and we cache the
// results.
class PlacerInspectionRequiredOpChecker {
 public:
  // Constructs a PlacerInspectionRequiredOpChecker for nodes of `graph`.
  // The functions referenced by nodes in `graph` will be looked up in
  // `flib_def`
  PlacerInspectionRequiredOpChecker(const Graph* graph,
                                    const FunctionLibraryDefinition* flib_def);

  // If `node` is considered a deep op, sets `*is_deep` to true and returns
  // OkStatus(). If an error occurs, returns that error, and the value of
  // `*is_deep` is undefined.
  // Currently, an op is considered deep, if it is a calling a function
  // returning a resource. This definition is driven by Placer's need to
  // look inside the op.
  // REQUIRES: `node` is part of `graph` passed into constructor.
  Status IsPlacerInspectionRequired(const Node& node, bool* is_deep);

 private:
  const Graph& graph_;
  const FunctionLibraryDefinition& flib_def_;
  // Indexed by the node id.
  // If cache_[node_id] is empty, the deepness of the node with id `node_id` has
  // not been computed yet. Else, it contains the value already computed.
  std::vector<absl::optional<bool>> cache_;
};

// Extracts `fdef` and `func` from `flib_def` for the function identified
// in "f" attribute of `node`.
Status GetFunctionDefAndAttrs(const FunctionLibraryDefinition& flib_def,
                              const Node& node,
                              core::RefCountPtr<FunctionRecord>* fdef,
                              NameAttrList* func);

// The "call" stack of functions.
// Useful for better error messages as well as for detecting recursion.
// Stores references to graph nodes. These references must outlive this.
class FunctionStack {
 public:
  explicit FunctionStack(const string& function_name);

  // `node_in_current_function` must outlive this.
  FunctionStack Push(const Node* node_in_current_function,
                     const string& new_current_function) const;

  // Returns true iff this stack already includes `function_name`.
  bool HasFunction(const string& function_name) const;

  const string& current_function_name() const { return current_function_name_; }

  // Format's this suitable for error interpolation that retrieves
  // Python files and line numbers.
  string FormatForError() const;

 private:
  struct Frame {
    Frame(const string& function, const Node* node)
        : function_name(function), node(node) {}

    string function_name;
    const Node* node;
  };

  // The function at the top of the stack. In other words, the function
  // that is currently being inspected for placement.
  string current_function_name_;

  // The stack of frames that got the placement to the current_function_name_.
  // frames_[0].function_name is the top function that Placer was constructed
  // with. frames_[0].function_name can be empty if placer was constructed with
  // a nameless graph, not a function.  frames_[0].node_name is a name of a node
  // in frames_[0].function_name that required deep inspection (e.g. a
  // PartitionedCallOp). The function that this node invoked is
  // frames_[1].function_name, if frames_.size() > 1.  Else, the function that
  // this node invoked is current_function_name_.
  std::vector<Frame> frames_;
};

// Adds Identities for each input and output of function-calling ops in `graph`
//
// For example, the following graph calling a function on inputs `a` and `b`
// and producing output `y` will be rewritten to include identities on all
// edges:
//
//      a             b
//      |             |
//      v             v
//    f (PartitionedCallOp)
//         |
//         v
//         y
//
// is transformed to
//
//      a             b
//      |             |
//  a_f (Identity)   b_f (Identity)
//      |             |
//      v             v
//    f (PartitionedCallOp)
//         |
//      f_y (Identity)
//         |
//         v
//         y
//
Status IsolatePlacerInspectionRequiredOps(
    const FunctionLibraryDefinition& flib_def, Graph* graph);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_INSPECTION_REQUIRED_OPS_UTILS_H_
