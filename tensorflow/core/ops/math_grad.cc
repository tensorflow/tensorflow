/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <vector>
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef FunctionDefHelper FDH;

// Cwise binary ops
Status GradForUnaryCwise(FunctionDef* g, std::vector<FDH::Node> nodes) {
  for (auto& n : nodes) {
    if (n.attr.empty()) {
      n.attr = {{"T", "$T"}};
    }
  }
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {{"T: {half, float, double}"}},
      // Nodes
      nodes);
  return Status::OK();
}

Status AbsGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"sign"}, "Sign", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "sign"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Abs", AbsGrad);

Status NegGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"dx"}, "Neg", {"dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Neg", NegGrad);

Status InvGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Inv", {"x"}},
      {{"y2"}, "Square", {"y"}, {}, {"dy"}},
      {{"y2_neg"}, "Neg", {"y2"}},
      {{"dx"}, "Mul", {"dy", "y2_neg"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Inv", InvGrad);

Status SquareGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("c", 2LL),
      {{"two"}, "Cast", {"c"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
      {{"x2"}, "Mul", {"x", "two"}, {}, {"dy"}},  // x * 2
      {{"dx"}, "Mul", {"dy", "x2"}},              // dy * (x * 2)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Square", SquareGrad);

Status SqrtGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Sqrt", {"x"}},
      {{"y_inv"}, "Inv", {"y"}, {}, {"dy"}},
      FDH::Const("const", 0.5f),
      {{"half"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Mul", {"half", "y_inv"}},  // .5 * 1/y
      {{"dx"}, "Mul", {"dy", "a"}},  // dy * (.5 * 1/y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sqrt", SqrtGrad);

Status RsqrtGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"x_inv"}, "Inv", {"x"}, {}, {"dy"}},
      {{"y"}, "Rsqrt", {"x"}},
      FDH::Const("const", -.5f),
      {{"neghalf"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Mul", {"neghalf", "x_inv"}},   // -0.5 * 1/x
      {{"b"}, "Mul", {"a", "y"}},             // -0.5 * 1/x * y
      {{"dx"}, "Mul", {"dy", "b"}},           // dy * (1/y * .5)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Rsqrt", RsqrtGrad);

Status ExpGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Exp", {"x"}},
      {{"dx"}, "Mul", {"dy", "y"}},           // dy * y
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Exp", ExpGrad);

Status LogGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"x_inv"}, "Inv", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "x_inv"}},           // dy * 1/x
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Log", LogGrad);

Status TanhGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Tanh", {"x"}},
      {{"y2"}, "Square", {"y"}, {}, {"dy"}},
      FDH::Const("const", 1.0f),
      {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Sub", {"one", "y2"}},
      {{"dx"}, "Mul", {"dy", "a"}},           // dy * (1 - y*y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Tanh", TanhGrad);

Status SigmoidGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Sigmoid", {"x"}},
      FDH::Const("const", 1.0f),
      {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Sub", {"one", "y"}, {}, {"dy"}},
      {{"b"}, "Mul", {"y", "a"}},             // y * (1 - y)
      {{"dx"}, "Mul", {"dy", "b"}},           // dy * y * (1 - y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sigmoid", SigmoidGrad);

Status SignGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"s"}, "Shape", {"x"}},
      FDH::Const("zero", 0.f),
      {{"val"}, "Cast", {"zero"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"dx"}, "Fill", {"s", "val"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sign", SignGrad);

Status SinGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"cos"}, "Cos", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "cos"}},  // dy * cos(x)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sin", SinGrad);

Status CosGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"sin"}, "Sin", {"x"}, {}, {"dy"}},
      {{"neg"}, "Neg", {"sin"}},
      {{"dx"}, "Mul", {"dy", "neg"}},  // dy * (-sin(x))
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Cos", CosGrad);

Status RealGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("zero", 0.f),
      {{"dx"}, "Complex", {"dy", "zero"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Real", RealGrad);

Status ImagGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("zero", 0.f),
      {{"dx"}, "Complex", {"zero", "dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Imag", ImagGrad);

Status ConjGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"dx"}, "Conj", {"dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Conj", ConjGrad);

// Cwise binary ops
//
// TODO(zhifengc): This can be arrange as a function in the standard
// library.
Status GradForBinaryCwise(FunctionDef* g, std::vector<FDH::Node> body) {
  // clang-format off
  std::vector<FDH::Node> nodes = {
    {{"sx"}, "Shape", {"x"}},
    {{"sy"}, "Shape", {"y"}},
  };
  nodes.insert(nodes.end(), body.begin(), body.end());
  std::vector<FDH::Node> reshapes = {
    {{"sum_gx"}, "Sum", {"gx", "rx"}},
    {{"dx"}, "Reshape", {"sum_gx", "sx"}},
    {{"sum_gy"}, "Sum", {"gy", "ry"}},
    {{"dy"}, "Reshape", {"sum_gy", "sy"}},
  };
  nodes.insert(nodes.end(), reshapes.begin(), reshapes.end());

  // clang-format on
  for (auto& n : nodes) {
    if (n.attr.empty()) {
      n.attr = {{"T", "$T"}};
    }
  }
  // "BroadcastGradientArgs" doesn't need any attrs.
  nodes.push_back({{"rx", "ry"}, "BroadcastGradientArgs", {"sx", "sy"}});
  *g = FDH::Define(
      // Arg defs
      {"x: T", "y: T", "dz: T"},
      // Ret val defs
      {"dx: T", "dy: T"},
      // Attr defs
      {{"T: {half, float, double}"}},
      // Nodes
      nodes);
  return Status::OK();
}

Status AddGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Identity", {"dz"}},
      {{"gy"}, "Identity", {"dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Add", AddGrad);

Status SubGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Identity", {"dz"}},
      {{"gy"}, "Neg", {"dz"}},          // -dz
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sub", SubGrad);

Status MulGrad(const AttrSlice& attrs, FunctionDef* g) {
  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));
  if (T == DT_COMPLEX64) {
    return GradForBinaryCwise(
        g, {
               {{"cy"}, "Conj", {"y"}, {}, {"dz"}},
               {{"gx"}, "Mul", {"dz", "cy"}},  // dz * Conj(y)
               {{"cx"}, "Conj", {"x"}, {}, {"dz"}},
               {{"gy"}, "Mul", {"cx", "dz"}},  // Conj(x) * dz
           });
  } else {
    // clang-format off
    return GradForBinaryCwise(g, {
        {{"gx"}, "Mul", {"dz", "y"}},  // dz * y
        {{"gy"}, "Mul", {"x", "dz"}},  // x * dz
    });
    // clang-format on
  }
}
REGISTER_OP_GRADIENT("Mul", MulGrad);

Status DivGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Div", {"dz", "y"}},
      {{"nx"}, "Neg", {"x"}, {}, {"dz"}},
      {{"y2"}, "Square", {"y"}, {}, {"dz"}},
      {{"nx_y2"}, "Div", {"nx", "y2"}},
      {{"gy"}, "Mul", {"dz", "nx_y2"}},  // dz * (- x / y^2)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Div", DivGrad);

Status PowGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"z"}, "Pow", {"x", "y"}},
      // dz * y * Pow(x, y - 1)
      FDH::Const("const", 1.0f),
      {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"t0"}, "Sub", {"y", "one"}, {}, {"dz"}},
      {{"t1"}, "Pow", {"x", "t0"}},
      {{"t2"}, "Mul", {"dz", "y"}},
      {{"gx"}, "Mul", {"t1", "t2"}},
      // dz * z * Log(x)
      {{"t3"}, "Log", {"x"}, {}, {"dz"}},
      {{"t4"}, "Mul", {"dz", "z"}},
      {{"gy"}, "Mul", {"t3", "t4"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Pow", PowGrad);

Status MaximumMinimumGradHelper(const string& comparator,
                                const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"c"}, comparator, {"x", "y"}, {}, {"dz"}},
      {{"mask"}, "Cast", {"c"}, {{"SrcT", DT_BOOL}, {"DstT", "$T"}}},
      {{"gx"}, "Mul", {"dz", "mask"}},
      {{"gy"}, "Sub", {"dz", "gx"}},
  });
  // clang-format on
}

Status MaximumGrad(const AttrSlice& attrs, FunctionDef* g) {
  return MaximumMinimumGradHelper("GreaterEqual", attrs, g);
}
REGISTER_OP_GRADIENT("Maximum", MaximumGrad);

Status MinimumGrad(const AttrSlice& attrs, FunctionDef* g) {
  return MaximumMinimumGradHelper("LessEqual", attrs, g);
}
REGISTER_OP_GRADIENT("Minimum", MinimumGrad);

Status ComplexGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Real", {"dz"}},
      {{"gy"}, "Imag", {"dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Complex", ComplexGrad);

// Cwise ternary ops.
Status SelectGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      {"c:bool", "x:T", "y:T", "dz:T"},
      {"dc:bool", "dx:T", "dy:T"},
      {{"T: {half, float, double}"}},
      {
        {{"dc"}, "ZerosLike", {"c"}, {{"T", DT_BOOL}}, {"dz"}},
        {{"zeros"}, "ZerosLike", {"x"}, {{"T", "$T"}}, {"dz"}},
        {{"dx"}, "Select", {"c", "dz", "zeros"}, {{"T", "$T"}}},
        {{"dy"}, "Select", {"c", "zeros", "dz"}, {{"T", "$T"}}},
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Select", SelectGrad);

// N-ry ops
// REGISTER_OP_GRADIENT("AddN", AddNGrad);

// Reduction ops
//
// TODO(zhifengc): This helper is pretty ugly. Do something better.
// TODO(zhifengc): This can be arrange as a function in the standard library.
Status GradForReductionOp(FunctionDef* g, std::vector<FDH::Node> body) {
  // Shape manipulation nodes.

  // clang-format off
  std::vector<FDH::Node> nodes = {
   {{"x_shape"}, "Shape", {"x"}},
   {{"x_rank"}, "Rank", {"x"}},
   {{"i_shape"}, "Shape", {"i"}, {{"T", DT_INT32}}},
   FDH::Const("zero", 0),
   FDH::Const("one", 1),
   // stitch_idx0 = Range(0, x_rank, 1)
   {{"stitch_idx1"}, "Identity", {"i"}, {{"T", DT_INT32}}},
   {{"stitch_idx"}, "_ListToArray", {"stitch_idx0", "stitch_idx1"},
    {{"Tin", DataTypeSlice{DT_INT32, DT_INT32}},
     {"T", DT_INT32}, {"N", 2}}},
   {{"stitch_val0"}, "Identity", {"x_shape"}, {{"T", DT_INT32}}},
   {{"stitch_val1"}, "Fill", {"i_shape", "one"}, {{"T", DT_INT32}}},
   {{"stitch_val"}, "_ListToArray", {"stitch_val0", "stitch_val1"},
    {{"Tin", DataTypeSlice{DT_INT32, DT_INT32}},
     {"T", DT_INT32}, {"N", 2}}},
   {{"y_shape"}, "DynamicStitch", {"stitch_idx", "stitch_val"},
                 {{"N", 2}, {"T", DT_INT32}}},
   {{"tile_scaling"}, "Div", {"x_shape", "y_shape"}, {{"T", DT_INT32}}},
   {{"di"}, "ZerosLike", {"i"}, {{"T", DT_INT32}}}
  };
  // clang-format on
  nodes.insert(nodes.end(), body.begin(), body.end());
  for (auto& n : nodes) {
    if (n.attr.empty()) {
      n.attr = {{"T", "$T"}};
    }
  }
  // "Range" doesn't need any attr.
  nodes.push_back({{"stitch_idx0"}, "Range", {"zero", "x_rank", "one"}, {}});
  *g = FDH::Define(
      // Arg defs
      {"x:T", "i:int32", "dy:T"},
      // Ret val defs
      {"dx:T", "di:int32"},
      // Attr defs
      {{"T: {half, float, double}"}},
      // Nodes
      nodes);
  return Status::OK();
}

Status SumGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForReductionOp(g, {
    {{"dy_reshaped"}, "Reshape", {"dy", "y_shape"}},
    {{"dx"}, "Tile", {"dy_reshaped", "tile_scaling"}},
  });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Sum", SumGrad);

Status MeanGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForReductionOp(g, {
    {{"factor"}, "Prod", {"tile_scaling", "zero"}, {{"T", DT_INT32}}},
    {{"factor_T"}, "Cast", {"factor"}, {{"SrcT", DT_INT32}, {"DstT", "$T"}}},
    {{"dy_scaled"}, "Div", {"dy", "factor_T"}},
    {{"dy_reshaped"}, "Reshape", {"dy_scaled", "y_shape"}},
    {{"dx"}, "Tile", {"dy_reshaped", "tile_scaling"}},
  });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Mean", MeanGrad);

// REGISTER_OP_GRADIENT("Prod", ProdGrad);
// REGISTER_OP_GRADIENT("SegmentSum", SegmentSumGrad);
// REGISTER_OP_GRADIENT("SegmentMean", SegmentMeanGrad);
// REGISTER_OP_GRADIENT("SparseSegmentSum", SparseSegmentSumGrad);
// REGISTER_OP_GRADIENT("SparseSegmentMean", SparseSegmentMeanGrad);
// REGISTER_OP_GRADIENT("SparseSegmentSqrtN", SparseSegmentSqrtNGrad);
// REGISTER_OP_GRADIENT("SegmentMin", SegmentMinGrad);
// REGISTER_OP_GRADIENT("SegmentMax", SegmentMaxGrad);
// REGISTER_OP_GRADIENT("UnsortedSegmentSum", UnsortedSegmentSumGrad);

Status MinMaxGradHelper(const string& op, const AttrSlice& attrs,
                        FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x:T", "i:int32", "dy:T"},
      // Ret val defs
      {"dx:T", "di:int32"},
      // Attr defs
      {{"T: {half, float, double}"}},
      {
        // keep_dims because we need to do x == y, which requires x
        // and y are broadcastable.
        {{"y"}, op, {"x", "i"}, {{"T", "$T"}, {"keep_dims", true}}},
        {{"mask"}, "Equal", {"x", "y"}, {{"T", "$T"}}},
        {{"mask_cast"}, "Cast", {"mask"}, {{"SrcT", DT_BOOL}, {"DstT", "$T"}}},
        {{"mask_sum"}, "Sum", {"mask_cast", "i"}, {{"T", "$T"}}},
        {{"norm_dy"}, "Div", {"dy", "mask_sum"}, {{"T", "$T"}}},
        {{"sy"}, "Shape", {"y"}, {{"T", "$T"}}},
        {{"norm_dy_reshaped"}, "Reshape", {"norm_dy", "sy"}, {{"T", "$T"}}},
        {{"dx"}, "Mul", {"mask_cast", "norm_dy_reshaped"}, {{"T", "$T"}}},
        {{"di"}, "ZerosLike", {"i"}, {{"T", DT_INT32}}}
      });
  // clang-format on
  return Status::OK();
}

Status MaxGrad(const AttrSlice& attrs, FunctionDef* g) {
  return MinMaxGradHelper("Max", attrs, g);
}
REGISTER_OP_GRADIENT("Max", MaxGrad);

Status MinGrad(const AttrSlice& attrs, FunctionDef* g) {
  return MinMaxGradHelper("Min", attrs, g);
}
REGISTER_OP_GRADIENT("Min", MinGrad);

static Status MatMulGradHelper(FunctionDef* g, const string& x0, bool tx0,
                               const string& x1, bool tx1, const string& y0,
                               bool ty0, const string& y1, bool ty1) {
  *g = FDH::Define(
      // Arg defs
      {"x: T", "y: T", "dz: T"},
      // Ret val defs
      {"dx: T", "dy: T"},
      // Attr defs
      {{"T: {float, double}"}},
      // Nodes
      {
          {{"dx"},
           "MatMul",
           {x0, x1},
           {{"T", "$T"}, {"transpose_a", tx0}, {"transpose_b", tx1}}},
          {{"dy"},
           "MatMul",
           {y0, y1},
           {{"T", "$T"}, {"transpose_a", ty0}, {"transpose_b", ty1}}},
      });
  return Status::OK();
}

Status MatMulGrad(const AttrSlice& attrs, FunctionDef* g) {
  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));
  if (T == DT_COMPLEX64) {
    return errors::Unimplemented(
        "MatMul gradient for complex is not supported yet.");
  }
  bool ta;
  bool tb;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "transpose_a", &ta));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "transpose_b", &tb));
  if (!ta && !tb) {
    return MatMulGradHelper(g, "dz", false, "y", true, "x", true, "dz", false);
  }
  if (!ta && tb) {
    return MatMulGradHelper(g, "dz", false, "y", false, "dz", true, "x", false);
  }
  if (ta && !tb) {
    return MatMulGradHelper(g, "y", false, "dz", true, "x", false, "dz", false);
  }
  CHECK(ta && tb);
  return MatMulGradHelper(g, "y", true, "dz", true, "dz", true, "x", true);
}
REGISTER_OP_GRADIENT("MatMul", MatMulGrad);

// REGISTER_OP_GRADIENT("SparseMatMul", SparseMatMulGrad);
// REGISTER_OP_GRADIENT("BatchMatMul", BatchMatMulGrad);

// Comparison ops.
REGISTER_OP_NO_GRADIENT("Less");
REGISTER_OP_NO_GRADIENT("LessEqual");
REGISTER_OP_NO_GRADIENT("Greater");
REGISTER_OP_NO_GRADIENT("GreaterEqual");
REGISTER_OP_NO_GRADIENT("Equal");
REGISTER_OP_NO_GRADIENT("NotEqual");

// Logical ops.
REGISTER_OP_NO_GRADIENT("LogicalAnd");
REGISTER_OP_NO_GRADIENT("LogicalOr");
REGISTER_OP_NO_GRADIENT("LogicalNot");

// Sequence generation ops.
REGISTER_OP_NO_GRADIENT("Range");
REGISTER_OP_NO_GRADIENT("LinSpace");

}  // end namespace tensorflow
