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

def IsNchwLayoutOp: Constraint<CPred<
  "$0.get(\"is_nchw_op\") && $0.get(\"is_nchw_op\").dyn_cast<BoolAttr>() "
      "== mlir::BoolAttr::get($_builder.getContext(), true)">>;

def IsNhwcLayoutOp: Constraint<Neg<IsNchwLayoutOp.predicate>>;

class HasRank<int n> : Constraint<
    CPred<"$0.getType().cast<ShapedType>().hasRank() && "
          "$0.getType().cast<ShapedType>().getRank() == " # n>>;

class HasRankAtLeast<int n> : Constraint<
    CPred<"$0.getType().cast<ShapedType>().hasRank() && "
          "$0.getType().cast<ShapedType>().getRank() >= " # n>>;

def I32ElementsVal : Constraint<CPred<
  "$0.getType().cast<TensorType>().getElementType().isInteger(32)">,
  "32 bit integer tensor">;

// TODO(b/343278954): Move the creation of transposes to a separate prepare pass
// to avoid creating multiple pattern-rewrite rules for the same composite op.
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
        ConstantAttr<RankedI32ElementsAttr<[4]>,"{0, 3, 1, 2}">))>;

// pattern to lower a stablehlo.composite of `jax` and `pytorch` image resize fuctions
// in `nearest` mode and with NHWC inputs to a tflite.resize_nearest_neighbor op.
def LegalizeNhwcResizeNearestNeighbor2dComposite: Pat<
    (MHLO_CompositeOp:$old_val
    (variadic $input),
    ConstantStrAttr<StrAttr, "tfl.resize_nearest_neighbor">, $attrs, $_, $_),
    (TFL_ResizeNearestNeighborOp
      $input,
      (Arith_ConstantOp:$output_size (GetI32DenseAttr (GetAsVectorAttr<"size"> $attrs))),
      ConstBoolAttrFalse,
      ConstBoolAttrTrue),
    [(IsNhwcLayoutOp $attrs)]>;

// pattern to lower a stablehlo.composite of `jax` and `pytorch` image resize fuctions
// in `nearest`mode and with NCHW inputs to a tflite.resize_nearest_neighbor op.
// TODO(b/343278954): Move the creation of transposes to a separate prepare pass
// to avoid creating multiple pattern-rewrite rules for the same composite op.
def LegalizeNchwResizeNearestNeighbor2dComposite: Pat<
    (MHLO_CompositeOp:$old_val
    (variadic $input),
    ConstantStrAttr<StrAttr, "tfl.resize_nearest_neighbor">, $attrs, $_, $_),
    (TFL_TransposeOp
      (TFL_ResizeNearestNeighborOp
        (TFL_TransposeOp $input,
          (Arith_ConstantOp
            ConstantAttr<RankedI32ElementsAttr<[4]>,"{0, 2, 3, 1}">)),
        (Arith_ConstantOp:$output_size (GetI32DenseAttr (GetAsVectorAttr<"size"> $attrs))),
        ConstBoolAttrFalse,
        ConstBoolAttrTrue,
        (returnType (GetNhwcReturnTypeFromNchw $old_val))),
      (Arith_ConstantOp
        ConstantAttr<RankedI32ElementsAttr<[4]>,"{0, 3, 1, 2}">)),
    [(IsNchwLayoutOp $attrs)]>;

def LegalizeCompositeExactAtenGELU : Pat<
        (MHLO_CompositeOp:$composite
          (variadic $inputs),
          ConstantStrAttr<StrAttr, "aten.gelu.default">, $attrs, $_, $_),
        (TFL_GeluOp $inputs, ConstBoolAttrFalse),
        [(IsStrCompositeAttribute<"approximate", "none"> $attrs)]>;

def LegalizeCompositeApproximateAtenGELU : Pat<
        (MHLO_CompositeOp:$composite
          (variadic $inputs),
          ConstantStrAttr<StrAttr, "aten.gelu.default">, $attrs, $_, $_),
        (TFL_GeluOp $inputs, ConstBoolAttrTrue),
        [(IsStrCompositeAttribute<"approximate", "tanh"> $attrs)]>;

def LegalizeCompositeOdmlEmbeddingLookup : Pat<
        (MHLO_CompositeOp:$composite
          (variadic $indices, $table),
          ConstantStrAttr<StrAttr, "odml.embedding_lookup">, $attrs, $_, $_),
        (TFL_EmbeddingLookupOp $indices, $table),
        [(HasRank<1> $indices),
        (I32ElementsVal $indices),
        (HasRankAtLeast<2> $table)]>;

def LegalizeCompositeOdmlEmbeddingLookupDynamicShaped : Pat<
        (MHLO_CompositeOp:$composite
          (variadic $_, $_, $_, $indices, $table),
          ConstantStrAttr<StrAttr, "odml.embedding_lookup">, $attrs, $_, $_),
        (TFL_EmbeddingLookupOp $indices, $table),
        [(HasRank<1> $indices),
        (I32ElementsVal $indices),
        (HasRankAtLeast<2> $table)]>;
