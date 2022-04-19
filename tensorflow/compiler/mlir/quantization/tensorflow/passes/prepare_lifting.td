/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.td"

// Converts tf.FusedBatchNormV3 into a sequence of more primitive arithmetic
// operations. Specifically, performs the following calculation:
//
//   (x - mean) * scale / sqrt(variance + epsilon) + offset
//
// Let multiplier = scale / sqrt(variance + epsilon),
// to compute
//   (x - mean) * scale / sqrt(variance + epsilon) + offset,
// is then to compute
//   (x * multiplier) + (offset - mean * multiplier).
//
// TODO(b/228916181): There is a known issue with this DDR rule that it doesn't
// take into account broadcasting conditions. If the issue needs to be handled,
// see tensorflow/compiler/mlir/lite/transforms/prepare_tf.cc
def FoldFusedBatchNormV3: Pattern<
    (TF_FusedBatchNormV3Op:$root
        $x, $scale, $offset, $mean, $variance,
        F32Attr:$epsilon, $exponential_avg_factor,
        $data_format, FalseBoolAttr:$is_training),
    [(TF_AddOp
        (TF_MulOp
            $x,
            (TF_MulOp:$multiplier
                $scale,
                (TF_RsqrtOp
                    (TF_AddOp $variance,
                              (TF_ConstOp $epsilon))))),
        (TF_SubOp $offset, (TF_MulOp $mean, $multiplier))),
     // We already guaranteed that the last five results have no use so it does
     // not matter what value we provide here for replacement.
     /*batch_mean=*/(replaceWithValue $x),
     /*batch_variance=*/(replaceWithValue $x),
     /*reserve_space_1=*/(replaceWithValue $x),
     /*reserve_space_2=*/(replaceWithValue $x),
     /*reserve_space_3=*/(replaceWithValue $x)],
    [(HasNoUseOf:$root__1), (HasNoUseOf:$root__2),
     (HasNoUseOf:$root__3), (HasNoUseOf:$root__4),
     (HasNoUseOf:$root__5)]>;
