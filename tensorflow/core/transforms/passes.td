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

    Note that some Grappler optimizers are sensitive to the textual order of the
    operations in the graph. This pass is meant to be approximately stable, in
    that the order of operations in the input graph is preserved as much as
    possible so as to not alter the results of these optimizers.
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
              "Comma separated list of ops that will be skipped.">,
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
              "Comma separated list of ops that will be turned into arguments.">,
   ListOption<"fetches_", "fetches", "std::string",
              "Comma separated list of ops that will be turned into results.">,
   ListOption<"control_rets_", "control_rets", "std::string",
              "Comma separated list of ops that will be turned into control "
              "returned.">
  ];
}

def FuncToGraph : Pass<"tfg-lower-func-to-graph", "ModuleOp"> {
  let summary = "Turns a function back to a graph";
  let description = [{
    This pass converts a function which was lifted from a graph back to the
    graph. Note that only functions lifted by `tfg-lift-graph-to-func` pass are
    valid for this conversion.

    For example,

    ```mlir
    tfg.func @_mlir_lifted_graph(%FooA0: tensor<*xf32> {tfg.name = "Foo:0"})
         -> (tensor<*xf32> {tfg.name = "Bar:0"})
          attributes {tfg.lifted_graph_version =
                      #tf_type.version<producer = 34, min_consumer = 5>} {
      %Foo, %ctl = Foo name("Foo") {dtype = i32} : () -> (tensor<*xf32>)
      %Bar, %ctl_1 = Bar(%Foo) name("Bar") : (tensor<*xf32>) -> (tensor<*xf32>)
      return (%Bar) [%ctl] : tensor<*xf32>
    }
    ```

    will be turned into

    ```mlir
    tfg.graph #tf_type.version<producer = 34, min_consumer = 5> {
      %Foo, %ctl = Foo name("Foo") {dtype = i32} : () -> (tensor<*xf32>)
      %Bar, %ctl_1 = Bar(%Foo) name("Bar") : (tensor<*xf32>) -> (tensor<*xf32>)
    }
    ```

  }];
  let constructor = "CreateFuncToGraphPass()";
}

def FunctionalToRegion : Pass<"tfg-functional-to-region", "ModuleOp"> {
  let summary = "Convert functional control-flow ops to region-based.";
  let description = [{
    This pass converts functional TFG control-flow operations to region-based
    forms. For example, it converts `If` and `While` to `IfRegion` and
    `WhileRegion`.

    Region-based control-flow is a more natural form for performing control-flow
    optimizations, and it intrinsically enables inter-procedural effects in
    those optimizations. Region-based form can be used to perform loop-invariant
    code motion, control-flow sinking, dead-code analysis, etc.

    For example, consider the following conversion of an `If`.

    ```mlir
    func @true(%a: tensor<*xi32>, %b: tensor<*xi32>) -> (tensor<*xi32>) {
      return %a : tensor<*xi32>
    }

    func @false(%a: tensor<*xi32>, %b: tensor<*xi32>) -> (tensor<*xi32>) {
      return %b : tensor<*xi32>
    }

    func @main(%cond: tensor<i32>, %a: tensor<*xi32>, %b: tensor<*xi32>) {
      %If, %ctl = If(%cond, %a, %b) {
        then_branch = #tf_type.func<@true, {}>,
        else_branch = #tf_type.func<@false, {}>
      } : (tensor<i32>, tensor<*ix32>, tensor<*xi32>) -> (tensor<*xi32>)
      return
    }
    ```

    Will become:

    ```mlir
    func @true(%a: tensor<*xi32>, %b: tensor<*xi32>) -> (tensor<*xi32>) {
      return %a : tensor<*xi32>
    }

    func @false(%a: tensor<*xi32>, %b: tensor<*xi32>) -> (tensor<*xi32>) {
      return %b : tensor<*xi32>
    }

    func @main(%cond: tensor<i32>, %a: tensor<*xi32>, %b: tensor<*xi32>) {
      %If, %ctl = IfRegion %cond then {
        yield(%a) : tensor<*xi32>
      } else {
        yield(%b) : tensor<*xi32>
      } : (tensor<i32>) -> (tensor<*xi32>)
      return
    }
    ```
  }];

  let constructor = "CreateFunctionalToRegionPass()";
  let dependentDialects = ["tfg::TFGraphDialect"];
}


def RegionToFunctional : Pass<"tfg-region-to-functional", "ModuleOp"> {
  let summary = "Convert region-based control-flow ops to functional.";
  let description = [{
    This pass converts region-based TFG control-flow operations to functional
    forms. For example, it converts `IfRegion` and `WhileRegion` to `If` and
    `While`.

    If a control token is implicitly captured in the region but whose defining op
    does not produce a data value, it cannot be converted to an explicit capture.
    Example:

    ```mlir
    %ctl = NoOp [...]
    %Case, %ctl_0 = CaseRegion %branch_index {
      %Thing, %ctl_1 = Thing [%ctl]
    }
    ```

    Because `%ctl` has no associated data value, the control edge cannot be
    propagated into the region body using a block argument.

    However, if `force_control_capture` is set, all region ops are guaranteed to
    be converted to functional ops by inserting a chain `Const` op:

    ```mlir
    %ctl = NoOp [...]
    %unused, %ctl_captured = Const [%ctl]
    %Case, %ctl_0 = CaseRegion %branch_index {
      %Thing, %ctl_1 = Thing [%ctl_captured]
    }
    ```

    When converted to functional form:

    ```mlir
    %ctl = NoOp [...]
    %unused, %ctl_captured = Const [%ctl]
    %Case, %ctl_0 = Case(%branch_index, %unused)
    ```

    And in the function body:

    ```mlir
    %Thing, %ctl = Thing [%unused.ctl]
    ```
  }];

  let constructor = "CreateRegionToFunctionalPass()";
  let dependentDialects = ["tfg::TFGraphDialect"];

  let options = [
    Option<"force_control_capture", "force-control-capture", "bool",
    /*default=*/"false", "Force the capture of control tokens by inserting "
                         "chain `Const` ops">
  ];
}

def ControlFlowSink : Pass<"tfg-cf-sink", "GraphFuncOp"> {
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

  let statistics = [
    Statistic<"num_sunk", "num-sunk", "Number of operations sunk">,
  ];
}

def Remapper : Pass<"tfg-remapper"> {
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
    "Enable the MKL related patterns.">,
    Option<"verify_pdll_patterns_only_", "verify-pdll-patterns-only", "bool",
    /*default=*/"false",
    "Only enable PDLL patterns.">,
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

def EliminatePassthroughIterArgs : Pass<"tfg-eliminate-passthrough-iter-args",
                                        "ModuleOp"> {
  let summary = "Eliminate passthrough loop iteration arguments.";

  let description = [{
    This pass will analyze loops and eliminate loop iteration arguments that are
    passthrough, i.e. they are the same for every loop iteration. For example:

    ```mlir
    %For:2, %ctl = ForRegion(%a, %b) from %start to %end by %step {
    %bb0(%index: tensor<i32>, %arg0: tensor<*xi32>, %arg1: tensor<*xi32>):
      %ctl_0 = Use(%arg0) : tensor<*xi32>
      %Foo, %ctl_1 = Foo(%arg1) : (tensor<*xi32>) -> (tensor<*xi32>)
      yield(%arg0, %Foo) : tensor<*xi32>, tensor<*xi32>
    }
    %ctl_2 = Use(%For#0, For#1) : tensor<*xi32>, tensor<*xi32>
    ```

    Will become:

    ```mlir
    %For, %ctl = ForRegion(%b) from %start to %end by %step {
    %bb0(%index: tensor<i32>, %arg0: tensor<*xi32>):
      %ctl_0 = Use(%a) : tensor<*xi32>
      %Foo, %ctl_1 = Foo(%arg0) : (tensor<*xi32>) -> (tensor<*xi32>)
      yield(%Foo) : tensor<*xi32>
    }
    %ctl_2 = Use(%a, For#1) : tensor<*xi32>, tensor<*xi32>
    ```
  }];

  let constructor = "CreateEliminatePassthroughIterArgsPass()";
  let dependentDialects = ["tfg::TFGraphDialect"];
}

def ShapeInference : Pass<"tfg-shape-inference"> {
  let summary = "Infer the output shape of operations";
  let description = [{
    Infer the result shape of an operation if possible. Besides inferring the op
    result tensor shape, it'll also try to evalute the tensor value which can be
    used to do more inferences.
  }];
  let options = [
    Option<"graph_version_", "graph-version", "int", /*default=*/"",
           "The graph producer version">
  ];
  let constructor = "CreateShapeInferencePass()";
}

def ConstantFoldingPass : Pass<"tfg-constant-folding"> {
  let summary = "constant-folding on tfg";
  let description = [{
    This pass will fold operations into constants and do full/partial propagte
    the constants iteratively.
  }];

  let constructor = "CreateConstantFoldingPass()";
  let options = [
   ListOption<"nodes_to_preserve_", "nodes-to-preserve", "std::string",
              "Comma separated list of ops that won't be optimized.",
              "llvm::cl::ZeroOrMore">,
   ListOption<"feeds_", "feeds", "std::string",
              "Comma separated list of feed ops.", "llvm::cl::ZeroOrMore">,
   ListOption<"fetches_", "fetches", "std::string",
              "Comma separated list of fetch ops.", "llvm::cl::ZeroOrMore">,
   Option<"disable_compressed_tensor_optimization_",
          "disable-compressed-tensor-optimization", "bool",
          /*default=*/"false",
          "Determine if we should disable compressed tensor optimization">,
   Option<"fold_quantization_emulation_",
          "fold-quantization-emulation", "bool",
          /*default=*/"true",
          "Determine if we should fold quantization emulation ops">
  ];
}

def LiftLegacyCall : Pass<"tfg-lift-legacy-call", "ModuleOp"> {
  let summary = "Tag legacy calls with symbol references to add symbol uses";
  let description = [{
    A legacy function call is when an operation's name refers to a function in 
    the function library. For example,

    ```
    %ctl = tfg.Foo

    tfg.func @Foo() -> ()
    ```

    This is problematic because the symbol table framework will not recognize
    `tfg.Foo` as a symbol use of `@Foo`, which can break passes like symbol DCE.
    The symbol table walks operations' nested attributes for symbol reference
    attributes to find symbol uses.

    This pass will tag legacy function call operations with a symbol attribute
    `tfg.legacy_call` so that the symbol table framework can find the symbol use
    in the operation:

    ```
    %ctl = tfg.Foo {tfg.legacy_call = @Foo}
    ```

    Now, symbol table will return this operation when querying the uses of the 
    function `@Foo`.
  }];

  let constructor = "CreateLiftLegacyCallPass()";
}

def CSEPass : Pass<"tfg-cse", "GraphFuncOp"> {
  let summary = "Common sub-expression elimination, ignoring op names";
  let constructor = "CreateCSEPass()";
}

def NameCompress : Pass<"tfg-name-compress", "GraphFuncOp"> {
  let summary = "Compress the graph by shortening names";
  let description = [{
    This pass will rename operations, function arguments, results, and control
    results to have "compressed" names, reducing the size of the graph. Only
    operations in funtions can be safely renamed, and names need only be unique
    in the scope of the function.
    
    This pass also assigns unique names to arguments, results, control results,
    and operations without names.
  }];
  let constructor = "CreateNameCompressPass()";
}

def StripDefaultAttrs : Pass<"tfg-strip-default-attrs"> {
  let summary = "Removes default-valued attributes from the graph";
  let description = [{
    This pass reduces the size of graphs by removing operation attributes equal
    to their default values. This pass requires the TensorFlow registry
    interface to inject an instance of the operation registry.

    Note that removing default-valued attributes may render some pass inoperable
    if they expect default-valued attributes to be present.

    This pass is the opposite of `tfg-add-default-attrs`, which adds missing
    default-valued attributes.
  }];
  let constructor = "CreateStripDefaultAttrsPass()";
}

def AddDefaultAttrs : Pass<"tfg-add-default-attrs"> {
  let summary = "Add default-valued attributes to the graph";
  let description = [{
    This pass adds missing attributes with default values to the graph. This
    pass requires the TensorFlow registry interface to inject an instance of the
    operation registry.

    This pass is the opposite of `tfg-strip-default-attrs`, which removes
    default-valued attributes equal to their default values.
  }];
  let constructor = "CreateAddDefaultAttrsPass()";
}