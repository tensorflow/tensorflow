/* Copyright 2023 The StableHLO Authors.

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

def StablehloCanonicalizeDynamismPass : Pass<"stablehlo-ext-canonicalize-dynamism", "func::FuncOp"> {
  let summary = "Canonicalizes dynamic StableHLO ops into static ops, with some XLA specialization.";
  let description = [{
    Experimental version of the --stablehlo-canonicalize-dynamism pass.
  }];
  let dependentDialects = ["mlir::chlo::ChloDialect"];
}

def StablehloRefineShapesPass : Pass<"stablehlo-ext-refine-shapes", "ModuleOp"> {
  let summary = "Refines shapes across a StableHLO program, with some XLA specialization.";
  let description = [{
    Experimental version of the --stablehlo-refine-shapes pass.
  }];
}

def ChloRecomposeOpsPass : Pass<"stablehlo-ext-chlo-recompose-ops", "ModuleOp"> {
  let summary = "(Experimental) Recompose CHLO ops serialized as custom calls.";
  let description = [{
    Experimental version of CHLO serialization support.
  }];
  let dependentDialects = ["chlo::ChloDialect"];
}

def StablehloPrepareForHloExportPass : Pass<"stablehlo-ext-prepare-for-hlo-export", "mlir::func::FuncOp"> {
  let summary = "Prepare StableHLO for HLO export";

  let description = [{
    This pass transforms functions in preparation for exporting to HLO. This

    * converts splat constants to constants and broadcasts to reduce size of
      and speedup the creation of the generated proto during export.

    Note: The result of this pass need not be a module in canonical form and
    canonicalization may undo transformations.
  }];
}

def StablehloFlattenTuplePass : Pass<"stablehlo-ext-flatten-tuple", "func::FuncOp"> {
  let summary = "Flatten tuples in operands and results of operators that "
    "support both tuple and variadic type.";
}

def StablehloFlattenEntryFunctionTuplesPass : Pass<"stablehlo-ext-expand-flatten-entry-function-tuples", "ModuleOp"> {
  let summary = "Flatten HLO tuple for the entry function of the module.";
  let options = [
    Option<"entryFunctionNameOption", "entry-function", "std::string",
           /*default=*/"", "the name of entry function of the module">,
  ];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def StablehloLegalizeQuantCompositePass : Pass<"stablehlo-ext-legalize-quant-composite", "ModuleOp"> {
  let summary = "Lowers the quantization related composites op to native quantized ops.";
  let description = [{
    Legalizes the following composite ops to native quantized ops:
    * quant.composite "quant.quantize" to `quant.quantize`
    * quant.composite "quant.dequantize" to `quant.dequantize`
    * quant.composite "quant.fake_quant" to `quant.quantize` and `quant.dequantize`

    Only the following patterns are supported for legalization:
    1.
    ```
    // before legalize
    int_output = quant.composite "quant.quantize" float_input
    float_output = quant.composite "quant.dequantize" int_output
    // after legalize
    qint_output = quant.quantize float_input
    float_output = quant.dequantize qint_output
    ```

    2.
    ```
    // before legalize
    func.func @main(...) -> int_type
      int_output = quant.composite "quant.quantize" float_input
      return int_output
    // after legalize
    func.func @main(...) -> qint_type
      qint_output = quant.quantize float_input
      return qint_output
    ```

    3.
    ```
    // before legalize
    func.func @main(int_argument)
      float_output = quant.composite "quant.dequantize" int_argument
    // after legalize
    func.func @main(qint_argument)
      float_output = quant.dequantize qint_argument
    ```
  }];
  let dependentDialects = [
    "mlir::quant::QuantDialect",
    "stablehlo::StablehloDialect"
  ];
}

def StablehloAddQDQAfterConvPass : Pass<"stablehlo-ext-add-qdq-after-conv", "ModuleOp"> {
  let summary = "Add quant and dequant ops after convolution op.";
 let dependentDialects = [
    "mlir::quant::QuantDialect",
    "stablehlo::StablehloDialect"
  ];
} 

