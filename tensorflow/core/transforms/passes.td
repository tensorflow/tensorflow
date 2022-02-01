/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/Pass/PassBase.td"

def TopoSort : Pass<"tfg-toposort"> {
  let summary = "Topologically sort graph and function regions";
  let description = [{
    This pass performs a topological sort of the body of a graph or function.
    The main purpose is readability, as well as potentially faster processing
    of the body by analyses/transformation as it'll ensure that all uses appear
    after the producer (but note that cycles are allowed in TensorFlow graphs).
    This is supposed to not change the semantics, and could be graduated to a
    canonicalization in the future.
  }];

  // A constructor must be provided to specify how to create a default instance.
  let constructor = "CreateTopoSortPass()";
}

def DedupeAndHoistConstant: Pass<"tfg-dedupe-hoist-constant"> {
  let summary = "Dedupe and hoist constants";
  let description = [{
    Deduplicate small constants by utilizing single constant op per constant
    value and removing/forwarding control dependencies to users.

    This pass is restricted to operate on "small" constants currently as there
    is no cost model nor attempt to limit the live range of values.

    Currently it only performs this deduplication if there are no possible
    function calls in the graph. This is to avoid changing strictness of the
    execution.

    Note: This pass does not aim to retain the names of nodes dedupe'd.
  }];

  let options = [
    Option<"max_size_", "max-size", "int64_t",
           /*default=*/"10",
           "The maximum number of elements when considering whether a constant is small">
  ];
  let constructor = "CreateDedupeAndHoistConstantPass()";
}

def DropOutputShapesAttr : Pass<"tfg-drop-unregistered-output-shapes"> {
  let summary = "Drop _output_shapes attribute";
  let description = [{
    Drop the unregistered `_output_shapes` attribute from all ops unless
    member of skipped set.
  }];

  let options = [
   ListOption<"skip_", "skip", "std::string",
              "Comma separated list of ops that will be skipped.",
              "llvm::cl::MiscFlags::CommaSeparated">,
  ];
  let constructor = "CreateDropOutputShapesAttrPass()";
}

def GraphToFunc : Pass<"tfg-lift-graph-to-func", "ModuleOp"> {
  let summary = "Turns a graph into a function.";
  let description = [{
    This pass takes a list of feeds and fetches and turns the graph into a
    function.
  }];
  // A constructor must be provided to specify how to create a default instance.
  let constructor = "CreateGraphToFuncPass()";
  let options = [
   ListOption<"feeds_", "feeds", "std::string",
              "Comma separated list of ops that will be turned into arguments.",
              "llvm::cl::MiscFlags::CommaSeparated">,
   ListOption<"fetches_", "fetches", "std::string",
              "Comma separated list of ops that will be turned into results.",
              "llvm::cl::MiscFlags::CommaSeparated">,
   ListOption<"control_rets_", "control_rets", "std::string",
              "Comma separated list of ops that will be turned into control "
              "returned.",
              "llvm::cl::MiscFlags::CommaSeparated">
  ];
}
