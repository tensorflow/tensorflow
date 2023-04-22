/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

namespace {
Status RiscBinaryNonBroadcastOpShapeFn(shape_inference::InferenceContext* c) {
  const auto rank = c->Rank(c->input(0));
  if (rank != c->Rank(c->input(1))) {
    return errors::InvalidArgument("Mismatch rank for input.");
  }
  for (int i = 0; i < rank; ++i) {
    if (!c->ValueKnown(c->Dim(c->input(0), i)) ||
        !c->ValueKnown(c->Dim(c->input(1), i))) {
      continue;
    }
    if (c->Value(c->Dim(c->input(0), i)) != c->Value(c->Dim(c->input(1), i))) {
      return errors::InvalidArgument("Mismatch shapes for input.");
    }
  }
  c->set_output(0, c->input(0));
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data != nullptr) {
    c->set_output_handle_shapes_and_types(0, *handle_data);
  }
  return Status::OK();
}
}  // namespace

REGISTER_OP("RiscAbs")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscAdd")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn)
    .SetIsAggregate()
    .SetIsCommutative();

// TODO(b/178234771): retire this.
REGISTER_OP("RiscBinaryArithmetic")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("op_type: {'ADD', 'SUB', 'MUL', 'DIV', 'REM', 'MIN', 'POW'}")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscBinaryComparison")
    .Input("x: T")
    .Input("y: T")
    .Output("z: bool")
    .Attr("op_type: {'EQ', 'NE', 'GE', 'GT', 'LE', 'LT'}")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscBitcast")
    .Input("x: SrcT")
    .Output("y: DstT")
    .Attr("SrcT: type")
    .Attr("DstT: type")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscBroadcast")
    .Input("input: T")
    .Input("shape: Tidx")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscCast")
    .Input("x: SrcT")
    .Output("y: DstT")
    .Attr("SrcT: type")
    .Attr("DstT: type")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscCeil")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscCholesky")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscConcat")
    .Input("values: N * T")
    .Input("axis: Tidx")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ConcatV2Shape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscCondition")
    .Input("pred: bool")
    .Input("input_true: SrcT")
    .Input("input_false: SrcT")
    .Output("output: DstT")
    .Attr("func_true: func")
    .Attr("func_false: func")
    .Attr("SrcT: {bfloat16, half, float, double}")
    .Attr("DstT: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscConv")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::UnknownShape)
    .Attr("dilations: list(int) = [1, 1, 1, 1]");

REGISTER_OP("RiscCos")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscDiv")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

REGISTER_OP("RiscDot")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::MatMulShape);

REGISTER_OP("RiscExp")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscFft")
    .Input("input: Tcomplex")
    .Output("output: Tcomplex")
    .Attr("Tcomplex: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscFloor")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscGather")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Input("axis: Taxis")
    .Attr("batch_dims: int = 0")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {int32,int64}")
    .Attr("Taxis: {int32,int64}")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscImag")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscIsFinite")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscLog")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscLogicalAnd")
    .Input("x: bool")
    .Input("y: bool")
    .Output("z: bool")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscLogicalNot")
    .Input("x: bool")
    .Output("z: bool")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscLogicalOr")
    .Input("x: bool")
    .Input("y: bool")
    .Output("z: bool")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscMax")
    .Input("x: T")
    .Input("y: T")
    .Output("max: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

REGISTER_OP("RiscMin")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

REGISTER_OP("RiscMul")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

REGISTER_OP("RiscNeg")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscPad")
    .Input("input: T")
    .Input("paddings: Tpaddings")
    .Input("constant_values: T")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tpaddings: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscPool")
    .Input("value: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("pooling_type: {'AVG', 'MAX'}")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscPow")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

REGISTER_OP("RiscRandomUniform")
    .Input("shape: T")
    .Output("output: float")
    .Attr("seed: int = 0")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape);

REGISTER_OP("RiscReal")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscReduce")
    .Input("tensor: T")
    .Input("axis: Index")
    .Output("output: T")
    .Attr("reduce_type: {'MEAN', 'SUM'}")
    .Attr("Index: {int32,int64} = DT_INT32")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscRem")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscReshape")
    .Input("tensor: T")
    .Input("shape: Tshape")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscReverse")
    .Input("tensor: T")
    .Input("axis: Tidx")
    .Output("output: T")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscScatter")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Input("shape: Tindices")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscShape")
    .Input("input: T")
    .Output("output: out_type")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("out_type: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscSign")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RiscSlice")
    .Input("input: T")
    .Input("begin: Index")
    .Input("size: Index")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Index: {int32,int64}")
    .SetShapeFn(shape_inference::SliceShape);

REGISTER_OP("RiscSort")
    .Input("input: T")
    .Input("axis: Index")
    .Output("output: T")
    .Attr("Index: {int32,int64} = DT_INT32")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("direction: {'ASCENDING', 'DESCENDING'}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscSqueeze")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("squeeze_dims: list(int) >= 0 = []")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscSub")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(RiscBinaryNonBroadcastOpShapeFn);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscTranspose")
    .Input("x: T")
    .Input("perm: Tperm")
    .Output("y: T")
    .Attr("T: type")
    .Attr("Tperm: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("adjoint: bool = False")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/178234771): retire this.
REGISTER_OP("RiscUnary")
    .Input("x: T")
    .Output("y: T")
    .Attr(
        "op_type: {'ABL', 'CEIL', 'COS', 'EXP', 'FLOOR', 'IMAG', 'LOG', 'NEG', "
        "'REAL', 'SIGN'}")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/178234771): change shape function.
REGISTER_OP("RiscWhile")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("cond: func")
    .Attr("body: func")
    .Attr("output_shapes: list(shape) = []")
    .Attr("parallel_iterations: int = 10")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow
