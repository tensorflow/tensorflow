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

include "mlir/IR/PatternBase.td"
include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_utils.td"


// See the function doc in the header file.
def GetPaddedType : NativeCodeCall<
  "GetPaddedType((*$0.begin()).getDefiningOp())">;

// See the function doc in the header file.
def GetPadding: 
  NativeCodeCall<"GetPaddingStringAttr($_builder, (*$0.begin()).getDefiningOp())">;

// Returns true if the provided padding in the composite op can *not* be 
// satisfied by SAME or VALID tensorflow padding.
def HasCustomPadding:
  Constraint<CPred<"GetPaddingStringAttr($_builder, (*$0.begin()).getDefiningOp()) == $_builder.getStringAttr(\"CUSTOM\")">>;

// Returns true if the provided padding in the composite op can be satisfied 
// by SAME or VALID tensorflow padding.
def HasSameOrValidPadding: Constraint<Neg<HasCustomPadding.predicate>>;

// See the function doc in the header file.
def GetPaddingArrayAttr: NativeCodeCall<"GetPaddingArrayAttr($_builder, (*$0.begin()).getDefiningOp())">;

// Replaces an ate.avg_pool2d with a (T -> tfl.average_pool_2d  -> T).
// Constraints are added on the attributes of the aten.avg_pool2d to ensure only
// ops that match the behaviour of tfl.average_pool_2d are directly lowered.
def LegalizeAvgPool2dComposite: Pat<
                    (MHLO_CompositeOp:$old_val
                    (variadic $a_input), 
                    ConstantStrAttr<StrAttr, "aten.avg_pool2d.default">, $attrs, $_, $_),
                    (TFL_TransposeOp 
                        (TFL_AveragePool2DOp 
                            /*input*/ (TFL_TransposeOp $a_input,
                                (Arith_ConstantOp
                                    ConstantAttr<RankedI32ElementsAttr<[4]>,"{0, 2, 3, 1}">)),
                            /*filter_height*/(GetI32At<0> (GetAsVectorAttr<"kernel_size"> $attrs)),
                            /*filter_width*/(GetI32At<1> (GetAsVectorAttr<"kernel_size"> $attrs)),
                            /*padding*/(GetPadding $old_val),
                            /*stride_h*/(GetI32At<0> (GetAsVectorAttr<"stride"> $attrs)),
                            /*stride_w*/(GetI32At<1> (GetAsVectorAttr<"stride"> $attrs)),
                            /*fused_activation_function*/TFL_AF_None,
                            (returnType (GetNhwcReturnTypeFromNchw $old_val))), 
                        (Arith_ConstantOp
                            ConstantAttr<RankedI32ElementsAttr<[4]>,"{0, 3, 1, 2}">)), 
                    [(IsBoolCompositeAttribute<"ceil_mode", "false"> $attrs),
                    (IsBoolCompositeAttribute<"count_include_pad", "false"> $attrs),
                    (IsStrCompositeAttribute<"divisor_override", "py_None"> $attrs),
                    (HasSameOrValidPadding $old_val)]>;

// Replaces an ate.avg_pool2d with (T -> tfl.pad -> tfl.average_pool_2d  -> T).
def LegalizeAvgPool2dWithPadComposite: Pat<
                    (MHLO_CompositeOp:$old_val
                    (variadic $a_input), 
                    ConstantStrAttr<StrAttr, "aten.avg_pool2d.default">, $attrs, $_, $_),
                    (TFL_TransposeOp 
                        (TFL_AveragePool2DOp:$padded_value 
                            /*input*/ (TFL_PadOp 
                                (TFL_TransposeOp $a_input,
                                    (Arith_ConstantOp
                                        ConstantAttr<RankedI32ElementsAttr<[4]>,"{0, 2, 3, 1}">)),
                                (Arith_ConstantOp
                                    (GetPaddingArrayAttr $old_val)),
                                (returnType (GetPaddedType $old_val))),
                            /*filter_height*/(GetI32At<0> (GetAsVectorAttr<"kernel_size"> $attrs)),
                            /*filter_width*/(GetI32At<1> (GetAsVectorAttr<"kernel_size"> $attrs)),
                            /*padding*/TFL_PAD_Valid,
                            /*stride_h*/(GetI32At<0> (GetAsVectorAttr<"stride"> $attrs)),
                            /*stride_w*/(GetI32At<1> (GetAsVectorAttr<"stride"> $attrs)),
                            /*fused_activation_function*/TFL_AF_None,
                            (returnType (GetNhwcReturnTypeFromNchw $old_val))), 
                        (Arith_ConstantOp
                            ConstantAttr<RankedI32ElementsAttr<[4]>,"{0, 3, 1, 2}">)), 
                    [(IsBoolCompositeAttribute<"ceil_mode", "false"> $attrs),
                    (IsStrCompositeAttribute<"divisor_override", "py_None"> $attrs),
                    (IsBoolCompositeAttribute<"count_include_pad", "true"> $attrs)]>;
