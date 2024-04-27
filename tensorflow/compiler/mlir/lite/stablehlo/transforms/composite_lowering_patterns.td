/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

// Pattern definition file for direct lowering of mhlo composites to tflite ops.
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mhlo/IR/hlo_ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_utils.td"
include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_avg_pool_patterns.td"

def LegalizeHardSwishComposite: Pat<
                    (MHLO_CompositeOp:$old_val
                    (variadic $input),
                    ConstantStrAttr<StrAttr, "aten.hardswish.default">, $_, $_, $_),
                    (TFL_HardSwishOp $input)>;

// Checks if the given op is an InterpolateBilinear op with NCHW layout.
// Supplied arguments are the input, output op values and the output shape.
def IsSupportedNchwUpsampleBlinear: Constraint<CPred<
  "IsSupportedNchwUpsampleBlinear($0, $1[0], $2.get(\"output\").dyn_cast<DenseIntElementsAttr>())">>;

def LegalizeTorchUpsampleBlinear2dComposite: Pat<
                    (MHLO_CompositeOp:$old_val
                    (variadic $input),
                    ConstantStrAttr<StrAttr, "odml.upsample_bilinear2d">, $attrs, $_, $_),
                    (TFL_TransposeOp
                        (TFL_ResizeBilinearOp
                            (TFL_TransposeOp $input,
                                (Arith_ConstantOp
                                    ConstantAttr<RankedI32ElementsAttr<[4]>,"{0, 2, 3, 1}">)),
                            (Arith_ConstantOp:$output_size (GetI32DenseAttr (GetAsVectorAttr<"output"> $attrs))),
                            (GetCompositeAttributeAs<"align_corners", "BoolAttr"> $attrs),
                            ConstBoolAttrTrue,
                            (returnType (GetNhwcReturnTypeFromNchw $old_val))),
                        (Arith_ConstantOp
                            ConstantAttr<RankedI32ElementsAttr<[4]>,"{0, 3, 1, 2}">)),
                    [(IsSupportedNchwUpsampleBlinear $input, $old_val, $attrs)]>;
