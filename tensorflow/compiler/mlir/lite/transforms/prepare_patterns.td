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

include "tensorflow/compiler/mlir/tensorflow/transforms/optimize.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

def FalseBoolAttr : AttrConstraint<CPred<"!$_self.getValue()">>;

def HasNoUse: Constraint<
    CPred<"$0->use_begin() == $0->use_end()">, "has no use">;

// Converts tf.FusedBatchNorm into a sequence of more primitive arithmetic
// operations. Specifically, performs the following calculation:
//
//   (x - mean) * scale / sqrt(variance + epsilon) + offset
//
// Let multiplier = scale / sqrt(variance + epsilon),
// to compute
//   (x - mean) * scale / sqrt(variance + epsilon) + offset,
// is then to compute
//   (x * multiplier) + (offset - mean * multiplier).
def : Pattern<
    (TF_FusedBatchNormOp:$root
        $x, $scale, $offset, $mean, $variance,
        F32Attr:$epsilon, $data_format, FalseBoolAttr:$is_training),
    [(TF_AddOp
        (TF_MulOp
            $x,
            (TF_MulOp:$multiplier
                $scale,
                (TF_RsqrtOp
                    (TF_AddOp $variance,
                              (TF_ConstOp $epsilon))))),
        (TF_SubOp $offset, (TF_MulOp $mean, $multiplier))),
     // We already guaranteed that the last four results has no use so it does
     // not matter what value we provide here for replacement.
     /*batch_mean=*/(replaceWithValue $x),
     /*batch_variance=*/(replaceWithValue $x),
     /*reserve_space_1=*/(replaceWithValue $x),
     /*reserve_space_2=*/(replaceWithValue $x)],
    [(HasNoUse $root__1), (HasNoUse $root__2),
     (HasNoUse $root__3), (HasNoUse $root__4)]>;

// TODO(jpienaar): Move to opbase something more general.
def TFi32ElementsAttr : Attr<CPred<"$_self.isa<DenseIntElementsAttr>">,
                                   "scalar int attribute"> {
  let storageType = [{ DenseIntElementAttr }];
  let constBuilderCall = "$_builder.getDenseElementsAttr("
    "$_builder.getTensorType({}, $_builder.getIntegerType(32)), "
      "{$_builder.getI32IntegerAttr($0)})";
}
class TFi32<int v> : ConstantAttr<TFi32ElementsAttr, !cast<string>(v)>;

// Matmul without transpose on b to matmul with explicit transpose op and
// transposed b.
def : Pat<(TF_MatMulOp $a, $b, ConstBoolAttrFalse:$at, ConstBoolAttrFalse),
          (TF_MatMulOp $a, (TF_TransposeOp $b, (TF_SubOp (TF_RangeOp
             /*start=*/(TF_RankOp $b),
             /*limit=*/(ConstantOp TFi32<0>),
             /*delta=*/(ConstantOp TFi32<-1>)), (ConstantOp TFi32<1>))),
           $at, ConstBoolAttrTrue)>;

// Matmul with transpose on a to matmul with explicit transpose op and a not
// transposed.
def : Pat<(TF_MatMulOp $a, $b, ConstBoolAttrTrue, $bt),
          (TF_MatMulOp (TF_TransposeOp $a, (TF_SubOp (TF_RangeOp
             /*start=*/(TF_RankOp $a),
             /*limit=*/(ConstantOp TFi32<0>),
             /*delta=*/(ConstantOp TFi32<-1>)), (ConstantOp TFi32<1>))), $b,
           ConstBoolAttrFalse, $bt)>;

def : Pat<(TF_StopGradientOp $arg), (TF_IdentityOp $arg)>;

//===----------------------------------------------------------------------===//
// Op removal patterns.
//===----------------------------------------------------------------------===//
def : Pat<(TF_IdentityOp $arg), (replaceWithValue $arg)>;

//===----------------------------------------------------------------------===//
// Op quantization pass-through patterns.
//===----------------------------------------------------------------------===//
// TODO(fengliuai): Implement similar rule in the QuantizePass if the constant
// folding hook of tfl.transpose and tfl.reshape are implemented.
def : Pat<(TF_TransposeOp
              (TF_FakeQuantWithMinMaxVarsOp
                  $input, $min, $max, $num_bits, $narrow_range),
              $perm),
          (TF_FakeQuantWithMinMaxVarsOp (TF_TransposeOp $input, $perm),
              $min, $max, $num_bits, $narrow_range)>;

def : Pat<(TF_ReshapeOp
              (TF_FakeQuantWithMinMaxVarsOp
                  $input, $min, $max, $num_bits, $narrow_range),
              $shape),
          (TF_FakeQuantWithMinMaxVarsOp (TF_ReshapeOp $input, $shape),
              $min, $max, $num_bits, $narrow_range)>;

// Casts result type of $1 to a quantized type by using the quantization
// parameters from the type in $0.
def UpdateShape : NativeCodeCall<
  "CastQuantizedTypeAttrFromExpressedType($_builder, $0, GetFirstResultType($1))">;

// When the op is passing-through, the output types of the quantized ops need
// to be updated as well. Since the quantize op manages its own type by the
// "qtype" attribute, we should update the type shape in this attribute.
def : Pat<(TF_TransposeOp:$op
              (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype)), $perm),
          (TFL_DequantizeOp (TFL_QuantizeOp (TF_TransposeOp $input, $perm),
                                            (UpdateShape $qtype, $op)))>;

def : Pat<(TF_ReshapeOp:$op
              (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype)), $shape),
          (TFL_DequantizeOp
              (TFL_QuantizeOp (TF_ReshapeOp $input, $shape),
              (UpdateShape $qtype, $op)))>;
