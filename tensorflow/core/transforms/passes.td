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

def DedupeAndHoistConstant: Pass<"tfg-dedupe-hoist-constant", "ModuleOp"> {
  let summary = "Dedupe and hoist constants";
  let description = [{
    Deduplicate small constants by utilizing single constant op per constant
    value and removing/forwarding control dependencies to users.

    This pass is restricted to operate on "small" constants currently as there
    is no cost model nor attempt to limit the live range of values.

    Currently it only performs this deduplication if there are no possible
    function calls in the graph or if all calls may be assumed to be strict.
    This is to avoid changing strictness of the execution.

    Note: This pass does not aim to retain the names of nodes dedupe'd.
  }];

  let options = [
    Option<"max_size_", "max-size", "int64_t",
           /*default=*/"10",
           "The maximum number of elements when considering whether a constant "
           "is small">,
    Option<"assume_strict_calls_", "assume-strict-calls", "bool",
           /*default=*/"false",
           "Assume all function calls are stricts, that is operands are "
           "evaluated prior to the call">,
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

def ControlFlowSink : Pass<"tfg-cf-sink", "ModuleOp"> {
  let summary = "Perform control-flow sink on region-based control-flow ops.";
  let description = [{
    This pass implements control-flow sinking on TFG region-based control-flow
    ops. Singly-executed regions (i.e. regions known to be executed at most
    once) are analyzed for values whose only uses are within the region. If the
    value is produced by a stateless/side-effect free op, the op is moved inside
    the region so that it is not computed along control paths where its result
    is not needed.

    This pass affects all regions of `tfg.IfRegion` and `tfg.CaseRegion` only,
    as their regions are known to be executed at most once.

    Example:

    ```mlir
    %Add, %ctl = Add(%arg0, %arg1)
    %IfRegion, %ctl_0 = If %cond then {
      yield(%Add) // only use of %Add
    } else {
      yield(%arg0)
    }
    ```

    Will become:

    ```mlir
    %IfRegion, %ctl_0 = If %cond then {
      %Add, %ctl = Add(%arg0, %arg1)
      yield(%Add)
    } else {
      yield(%arg0)
    }
    ```
  }];

  let constructor = "CreateControlFlowSinkPass()";
}

def Remapper : Pass<"remapper"> {
  let summary = "Remap operations to decrease amount of operations";
  let description = [{
    This is TF-level instruction selection. It remaps operations onto other
    set of operations to decrease the amount operations to perform computation.
    For example, we may convert a MulOp to a _MklSwishOp if it has the following
    form,

        MulOp               _MklSwishOp
        /   \                    \
       x  SigmoidOp   ->          x
             \
              x
  }];
  let options = [
    Option<"enable_mkl_patterns_", "enable-mkl-patterns", "bool",
    /*default=*/"false",
    "Enable the the MKL related patterns.">,
  ];
  let constructor = "CreateRemapperPass()";
}

def ConsolidateAttributes : Pass<"tfg-consolidate-attrs", "ModuleOp"> {
  let summary = "Reify type data from attributes to types.";
  let description = [{
    The TensorFlow type system, all tensors are treated as unranked and all
    resources are treated as opaque. However, shape and handle information can
    be specified for certain tensors and resources through registered and
    unregistered attributes, e.g. `_output_shapes` and `_handle_data`. TFG's
    type system has a richer type representation; it is more consistent to store
    information about the types in there instead of in attributes. This pass
    uses attributes to refine function argument, function result, and operation
    result types using these attributes. The attributes are dropped, reducing
    the amount of book-keeping that passes need to do.

    Certain ops also have registered attributes that indicate shapes. For
    example, `If`, `Case`, and `While` have an `output_shapes` attribute. These
    attributes are optional in TFG and are dropped by this pass, but they must
    be present on the operation before export.

    For operations, the following attributes are used to refine result shapes
    and then are dropped:
      * `_output_shapes`
      * `output_shapes` (If, Case, While)

    For operations, the following attributes are dropped because they contain
    redundant data.
      * `Tcond` (If)
      * `Tin` (If, Case)
      * `Tout` (If, Case)
      * `T` (While, For)

    For functions:
      * `_input_shapes` is used to refine function argument types

    For function arguments:
      * `_output_shapes` is used to refine function argument types
      * `tfg.handle_data` is used to refine function result types

    For function results:
      * `_output_shapes` is used to refine function result types
      * `tfg.handle_data` is used to refine function result types

    The following function argument attributes are dropped because they contain
    redundant data:
      * `tfg.dtype`
      * `tfg.is_ref`

    The following function result attributes are dropped because they contain
    redundant data:
      * `tfg.dtype`
  }];

  let constructor = "CreateConsolidateAttributesPass()";
}

def PrepareAttributesForExport : Pass<"tfg-prepare-attrs-export", "ModuleOp"> {
  let summary = "Legalize ops' attributes for export.";
  let description = [{
    The canonical form of TFG for performing transformations is after running
    the consolidate attributes pass, which minimizes the attributes in the IR,
    including reifying type information in attributes into the types themselves.
    The pass also removes registered attributes from operations that are
    optional in TFG. These attributes are required for export, and this pass
    will materialize those attributes.

    The TFG exporter also expects the presence of certain `tfg.` attributes.
    In addition, reified type information (e.g. shapes and handle data) are
    stored as unregistered attributes. The types are not converted back to
    unranked/opaque.
  }];

  let constructor = "CreatePrepareAttributesForExportPass()";
}
