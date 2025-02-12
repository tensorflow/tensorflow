/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

typedef FunctionDefHelper FDH;

// Cwise binary ops
absl::Status GradForUnaryCwise(FunctionDef* g, std::vector<FDH::Node> nodes) {
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
  return absl::OkStatus();
}

absl::Status AbsGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"sign"}, "Sign", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "sign"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Abs", AbsGrad);

absl::Status NegGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"dx"}, "Neg", {"dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Neg", NegGrad);

absl::Status InvGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Reciprocal", {"x"}},
      {{"y2"}, "Square", {"y"}, {}, {"dy"}},
      {{"y2_neg"}, "Neg", {"y2"}},
      {{"dx"}, "Mul", {"dy", "y2_neg"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Inv", InvGrad);
REGISTER_OP_GRADIENT("Reciprocal", InvGrad);

absl::Status SquareGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("c", int64_t{2}),
      {{"two"}, "Cast", {"c"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
      {{"x2"}, "Mul", {"x", "two"}, {}, {"dy"}},  // x * 2
      {{"dx"}, "Mul", {"dy", "x2"}},              // dy * (x * 2)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Square", SquareGrad);

absl::Status SqrtGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Sqrt", {"x"}},
      {{"y_inv"}, "Reciprocal", {"y"}, {}, {"dy"}},
      FDH::Const("const", 0.5f),
      {{"half"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Mul", {"half", "y_inv"}},  // .5 * 1/y
      {{"dx"}, "Mul", {"dy", "a"}},  // dy * (.5 * 1/y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sqrt", SqrtGrad);

absl::Status RsqrtGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"x_inv"}, "Reciprocal", {"x"}, {}, {"dy"}},
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

absl::Status ExpGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Exp", {"x"}},
      {{"dx"}, "Mul", {"dy", "y"}},           // dy * y
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Exp", ExpGrad);

absl::Status Expm1Grad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Exp", {"x"}},
      {{"dx"}, "Mul", {"dy", "y"}},           // dy * y
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Expm1", Expm1Grad);

absl::Status LogGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"x_inv"}, "Reciprocal", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "x_inv"}},           // dy * 1/x
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Log", LogGrad);

absl::Status Log1pGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("const", 1.0f),
      {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"a"}, "Add", {"one", "x"}},
      {{"dx"}, "Div", {"dy", "a"}},           // dy / (1 + x)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Log1p", Log1pGrad);

absl::Status SinhGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"cosh"}, "Cosh", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "cosh"}},  // dy * cosh(x)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sinh", SinhGrad);

absl::Status CoshGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"sinh"}, "Sinh", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "sinh"}},  // dy * sinh(x)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Cosh", CoshGrad);

absl::Status TanhGrad(const AttrSlice& attrs, FunctionDef* g) {
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

absl::Status AsinhGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Asinh", {"x"}},
      {{"cosh"}, "Cosh", {"y"}},
      {{"dx"}, "Mul", {"dy", "cosh"}},  // dy * cosh(y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Asinh", AsinhGrad);

absl::Status AcoshGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"y"}, "Acosh", {"x"}},
      {{"sinh"}, "Sinh", {"y"}},
      {{"dx"}, "Mul", {"dy", "sinh"}},  // dy * sinh(y)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Acosh", AcoshGrad);

absl::Status AtanhGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
    {{"x2"}, "Square", {"x"}},
    FDH::Const("const", 1.0f),
    {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"a"}, "Sub", {"one", "x2"}}, // 1 - x^2
    {{"inv"}, "Reciprocal", {"a"}},
    {{"dx"}, "Mul", {"dy", "inv"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Atanh", AtanhGrad);

absl::Status SigmoidGrad(const AttrSlice& attrs, FunctionDef* g) {
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

absl::Status SignGrad(const AttrSlice& attrs, FunctionDef* g) {
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

absl::Status SinGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"cos"}, "Cos", {"x"}, {}, {"dy"}},
      {{"dx"}, "Mul", {"dy", "cos"}},  // dy * cos(x)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sin", SinGrad);

absl::Status CosGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"sin"}, "Sin", {"x"}, {}, {"dy"}},
      {{"neg"}, "Neg", {"sin"}},
      {{"dx"}, "Mul", {"dy", "neg"}},  // dy * (-sin(x))
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Cos", CosGrad);

absl::Status AcosGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
    {{"x2"}, "Square", {"x"}},
    FDH::Const("const", 1.0f),
    {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"a"}, "Sub", {"one", "x2"}}, // 1 - x^2
    {{"b"}, "Sqrt", {"a"}},
    {{"inv"}, "Reciprocal", {"b"}},
    {{"neg"}, "Neg", {"inv"}},
    {{"dx"}, "Mul", {"dy", "neg"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Acos", AcosGrad);

absl::Status AsinGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
    {{"x2"}, "Square", {"x"}},
    FDH::Const("const", 1.0f),
    {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"a"}, "Sub", {"one", "x2"}}, // 1 - x^2
    {{"b"}, "Sqrt", {"a"}},
    {{"inv"}, "Reciprocal", {"b"}},
    {{"dx"}, "Mul", {"dy", "inv"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Asin", AsinGrad);

absl::Status AtanGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
    {{"x2"}, "Square", {"x"}},
    FDH::Const("const", 1.0f),
    {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"a"}, "Add", {"one", "x2"}}, // 1 + x^2
    {{"inv"}, "Reciprocal", {"a"}},
    {{"dx"}, "Mul", {"dy", "inv"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Atan", AtanGrad);

absl::Status TanGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
    {{"cosx"}, "Cos", {"x"}},
    {{"secx"}, "Reciprocal", {"cosx"}},
    {{"secx2"}, "Square", {"secx"}},
    {{"dx"}, "Mul", {"dy", "secx2"}}
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Tan", TanGrad);

absl::Status RealGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("zero", 0.f),
      {{"dx"}, "Complex", {"dy", "zero"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Real", RealGrad);

absl::Status ImagGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      FDH::Const("zero", 0.f),
      {{"dx"}, "Complex", {"zero", "dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Imag", ImagGrad);

absl::Status AngleGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"re"}, "Real", {"x"}},
      {{"im"}, "Imag", {"x"}},
      {{"z"}, "Complex", {"im", "re"}},
      {{"z_inv"}, "Reciprocal", {"z"}},
      {{"neg"}, "Neg", {"z_inv"}},
      {{"dx"}, "Mul", {"neg", "dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Angle", AngleGrad);

absl::Status ConjGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForUnaryCwise(g, {
      {{"dx"}, "Conj", {"dy"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Conj", ConjGrad);

absl::Status CastGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: SrcT", "dy: DstT"},
      // Ret val defs
      {"dx: SrcT"},
      // Attr defs
      {{"SrcT: type"}, {"DstT: type"}},
      // Nodes
      {{{"dx"}, "Cast", {"dy"}, {{"SrcT", "$DstT"}, {"DstT", "$SrcT"}}}});
  return absl::OkStatus();
  // clang-format on
}
REGISTER_OP_GRADIENT("Cast", CastGrad);

// Cwise binary ops
//
// TODO(zhifengc): This can be arrange as a function in the standard
// library.
absl::Status GradForBinaryCwise(FunctionDef* g, std::vector<FDH::Node> body) {
  // clang-format off
  std::vector<FDH::Node> nodes = {
    {{"sx"}, "Shape", {"x"}},
    {{"sy"}, "Shape", {"y"}},
  };
  nodes.insert(nodes.end(), body.begin(), body.end());
  std::vector<FDH::Node> reshapes = {
    {{"rx", "ry"}, "BroadcastGradientArgs", {"sx", "sy"}},
    {{"sum_gx"}, "Sum", {"gx", "rx"}},
    {{"dx"}, "Reshape", {"sum_gx", "sx"}},
    {{"sum_gy"}, "Sum", {"gy", "ry"}},
    {{"dy"}, "Reshape", {"sum_gy", "sy"}},
  };
  nodes.insert(nodes.end(), reshapes.begin(), reshapes.end());

  // clang-format on
  for (auto& n : nodes) {
    // "BroadcastGradientArgs" doesn't need any attrs.
    if (n.attr.empty() && n.op != "BroadcastGradientArgs") {
      n.attr = {{"T", "$T"}};
    }
  }
  *g = FDH::Define(
      // Arg defs
      {"x: T", "y: T", "dz: T"},
      // Ret val defs
      {"dx: T", "dy: T"},
      // Attr defs
      {{"T: {half, float, double}"}},
      // Nodes
      nodes);
  return absl::OkStatus();
}

absl::Status AddGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Identity", {"dz"}},
      {{"gy"}, "Identity", {"dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Add", AddGrad);
REGISTER_OP_GRADIENT("AddV2", AddGrad);

absl::Status SubGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Identity", {"dz"}},
      {{"gy"}, "Neg", {"dz"}},          // -dz
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sub", SubGrad);

absl::Status MulGrad(const AttrSlice& attrs, FunctionDef* g) {
  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));
  if (T == DT_COMPLEX64 || T == DT_COMPLEX128) {
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

absl::Status MulNoNanGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "MulNoNan", {"y", "dz"}},  // y * dz
      {{"gy"}, "MulNoNan", {"x", "dz"}},  // x * dz
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("MulNoNan", MulGrad);

absl::Status DivGrad(const AttrSlice& attrs, FunctionDef* g) {
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

absl::Status RealDivGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "RealDiv", {"dz", "y"}},
      {{"nx"}, "Neg", {"x"}, {}, {"dz"}},
      {{"y2"}, "Square", {"y"}, {}, {"dz"}},
      {{"nx_y2"}, "RealDiv", {"nx", "y2"}},
      {{"gy"}, "Mul", {"dz", "nx_y2"}},  // dz * (- x / y^2)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("RealDiv", RealDivGrad);

absl::Status DivNoNanGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "DivNoNan", {"dz", "y"}},
      {{"nx"}, "Neg", {"x"}, {}, {"dz"}},
      {{"y2"}, "Square", {"y"}, {}, {"dz"}},
      {{"nx_y2"}, "DivNoNan", {"nx", "y2"}},
      {{"gy"}, "Mul", {"dz", "nx_y2"}},  // dz * (- x / y^2)
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("DivNoNan", DivNoNanGrad);

absl::Status PowGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  std::vector<FDH::Node> nodes = {
    {{"z"}, "Pow", {"x", "y"}},
    // dz * y * Pow(x, y - 1)
    FDH::Const("const_zero", 0.0f),
    FDH::Const("const_one", 1.0f),
    {{"zero"}, "Cast", {"const_zero"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"one"}, "Cast", {"const_one"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
    {{"t0"}, "Sub", {"y", "one"}, {}, {"dz"}},
    {{"t1"}, "Pow", {"x", "t0"}},
    {{"t2"}, "Mul", {"dz", "y"}},
    {{"gx"}, "Mul", {"t1", "t2"}},
    {{"unsafe_log"}, "Log", {"x"}, {}, {"dz"}},
    {{"zeros"}, "ZerosLike", {"x"}}};
  // clang-format on
  std::vector<FDH::Node> log_x_handling;
  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));
  if (T == DT_COMPLEX64 || T == DT_COMPLEX128) {
    // dz * z * (x != 0 ? Log(x) : 0)
    // clang-format off
    log_x_handling = {
      {{"nz_x"}, "NotEqual", {"x", "zero"}},
      {{"safe_log"}, "Select", {"nz_x", "unsafe_log", "zeros"}}};
    // clang-format on
  } else {
    // dz * z * (x > 0 ? Log(x) : 0)
    // clang-format off
    log_x_handling = {
      {{"pos_x"}, "Greater", {"x", "zero"}},
      {{"safe_log"}, "Select", {"pos_x", "unsafe_log", "zeros"}}};
    // clang-format on
  }
  nodes.insert(nodes.end(), log_x_handling.begin(), log_x_handling.end());
  nodes.push_back({{"t4"}, "Mul", {"dz", "z"}});
  nodes.push_back({{"gy"}, "Mul", {"safe_log", "t4"}});
  return GradForBinaryCwise(g, nodes);
}
REGISTER_OP_GRADIENT("Pow", PowGrad);

absl::Status XlogyGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"zeros"}, "ZerosLike", {"x"}},
      {{"is_x_zero"}, "NotEqual", {"x", "zeros"}},
      {{"is_zero_cast"}, "Cast", {"is_x_zero"},
        {{"SrcT", DT_BOOL}, {"DstT", "$T"}}},
      {{"safe_logy"}, "Xlogy", {"is_zero_cast", "y"}},
      {{"xlogygrad"}, "Xdivy", {"x", "y"}},
      {{"gx"}, "Mul", {"safe_logy", "dz"}},
      {{"gy"}, "Mul", {"xlogygrad", "dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Xlogy", XlogyGrad);

absl::Status Xlog1pyGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      FDH::Const("const", 1.0f),
      {{"one"}, "Cast", {"const"}, {{"SrcT", DT_FLOAT}, {"DstT", "$T"}}},
      {{"zeros"}, "ZerosLike", {"x"}},
      {{"yp1"}, "Add", {"y", "one"}},
      {{"is_x_zero"}, "NotEqual", {"x", "zeros"}},
      {{"is_zero_cast"}, "Cast", {"is_x_zero"},
        {{"SrcT", DT_BOOL}, {"DstT", "$T"}}},
      {{"safe_log1py"}, "Xlog1py", {"is_zero_cast", "y"}},
      {{"xlog1pygrad"}, "Xdivy", {"x", "yp1"}},
      {{"gx"}, "Mul", {"safe_log1py", "dz"}},
      {{"gy"}, "Mul", {"xlog1pygrad", "dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Xlog1py", Xlog1pyGrad);

absl::Status XdivyGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"zeros"}, "ZerosLike", {"x"}},
      {{"is_x_zero"}, "NotEqual", {"x", "zeros"}},
      {{"is_zero_cast"}, "Cast", {"is_x_zero"},
        {{"SrcT", DT_BOOL}, {"DstT", "$T"}}},
      {{"safe_divy"}, "Xdivy", {"is_zero_cast", "y"}},
      {{"y2"}, "Square", {"y"}},
      {{"negy2"}, "Neg", {"y2"}},
      {{"xdivygrad"}, "Xdivy", {"x", "negy2"}},
      {{"gx"}, "Mul", {"safe_divy", "dz"}},
      {{"gy"}, "Mul", {"xdivygrad", "dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Xdivy", XdivyGrad);

absl::Status SquaredDifferenceGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      FDH::Const("c", int64_t{2}),
      {{"two"}, "Cast", {"c"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
      {{"x_sub_y"}, "Sub", {"x", "y"}},
      {{"two_x_sub_y"}, "Mul", {"two", "x_sub_y"}},  // 2 * (x - y)
      {{"gx"}, "Mul", {"two_x_sub_y", "dz"}},
      {{"gy"}, "Neg", {"gx"}}
    });
  // clang-format on
}
REGISTER_OP_GRADIENT("SquaredDifference", SquaredDifferenceGrad);

absl::Status MaximumMinimumGradHelper(const string& comparator,
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

absl::Status MaximumGrad(const AttrSlice& attrs, FunctionDef* g) {
  return MaximumMinimumGradHelper("GreaterEqual", attrs, g);
}
REGISTER_OP_GRADIENT("Maximum", MaximumGrad);

absl::Status MinimumGrad(const AttrSlice& attrs, FunctionDef* g) {
  return MaximumMinimumGradHelper("LessEqual", attrs, g);
}
REGISTER_OP_GRADIENT("Minimum", MinimumGrad);

absl::Status ComplexGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForBinaryCwise(g, {
      {{"gx"}, "Real", {"dz"}},
      {{"gy"}, "Imag", {"dz"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Complex", ComplexGrad);

// Cwise ternary ops.
absl::Status SelectGrad(const AttrSlice& attrs, FunctionDef* g) {
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
  return absl::OkStatus();
}
REGISTER_OP_GRADIENT("Select", SelectGrad);

// N-ry ops
// REGISTER_OP_GRADIENT("AddN", AddNGrad);

// Reduction ops
//
// TODO(zhifengc): This helper is pretty ugly. Do something better.
// TODO(zhifengc): This can be arrange as a function in the standard library.
absl::Status GradForReductionOp(FunctionDef* g, std::vector<FDH::Node> body) {
  // Shape manipulation nodes.

  // clang-format off
  std::vector<FDH::Node> nodes = {
   {{"x_shape"}, "Shape", {"x"}},
   {{"x_rank"}, "Rank", {"x"}},
   {{"i_shape"}, "Shape", {"i"}, {{"T", DT_INT32}}},
   FDH::Const("zero", 0),
   FDH::Const("one", 1),
   // stitch_idx0 = Range(0, x_rank, 1)
   {{"stitch_val1"}, "Fill", {"i_shape:output:0", "one:output:0"},
    {{"T", DT_INT32}}},
   {{"y_shape"}, "DynamicStitch",
    {"stitch_idx0:output:0", "i",
     "x_shape:output:0", "stitch_val1:output:0"},
    {{"N", 2}, {"T", DT_INT32}}},
   {{"tile_scaling"}, "Div", {"x_shape:output:0", "y_shape:merged:0"},
    {{"T", DT_INT32}}},
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
  nodes.push_back({{"stitch_idx0"},
                   "Range",
                   {"zero:output:0", "x_rank:output:0", "one:output:0"},
                   {}});
  *g = FDH::Create("_",
                   // Input defs
                   {"x:T", "i:int32", "dy:T"},
                   // Ret val defs
                   {"dx:T", "di:int32"},
                   // Attr defs
                   {{"T: {half, float, double}"}},
                   // Nodes
                   nodes,
                   // Return values
                   {{"dx", "dx:output:0"}, {"di", "di:y:0"}});
  return absl::OkStatus();
}

absl::Status SumGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForReductionOp(g, {
    {{"dy_reshaped"}, "Reshape", {"dy", "y_shape:merged:0"}},
    {{"dx"}, "Tile", {"dy_reshaped:output:0", "tile_scaling:z:0"}},
  });
  // clang-format on
}
REGISTER_OP_GRADIENT("Sum", SumGrad);

absl::Status MeanGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  return GradForReductionOp(g, {
    {{"factor"}, "Prod", {"tile_scaling:z:0", "zero:output:0"},
                   {{"T", DT_INT32}}},
    {{"factor_T"}, "Cast", {"factor:output:0"},
                   {{"SrcT", DT_INT32}, {"DstT", "$T"}}},
    {{"dy_scaled"}, "Div", {"dy", "factor_T:y:0"}},
    {{"dy_reshaped"}, "Reshape", {"dy_scaled:z:0", "y_shape:merged:0"}},
    {{"dx"}, "Tile", {"dy_reshaped:output:0", "tile_scaling:z:0"}},
  });
  // clang-format on
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
// REGISTER_OP_GRADIENT("UnsortedSegmentMax", UnsortedSegmentMaxGrad);

absl::Status MinMaxGradHelper(const string& op, const AttrSlice& attrs,
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
  return absl::OkStatus();
}

absl::Status MaxGrad(const AttrSlice& attrs, FunctionDef* g) {
  return MinMaxGradHelper("Max", attrs, g);
}
REGISTER_OP_GRADIENT("Max", MaxGrad);

absl::Status MinGrad(const AttrSlice& attrs, FunctionDef* g) {
  return MinMaxGradHelper("Min", attrs, g);
}
REGISTER_OP_GRADIENT("Min", MinGrad);

static absl::Status MatMulGradHelper(FunctionDef* g, const string& opname,
                                     const string& attr_adj_x,
                                     const string& attr_adj_y, const string& x0,
                                     bool ax0, const string& x1, bool ax1,
                                     const string& y0, bool ay0,
                                     const string& y1, bool ay1,
                                     bool enable_broadcasting) {
  // The final outputs are "dx" and "dy". If we're broadcasting compute
  // intermediate nodes for now.
  std::vector<FDH::Node> nodes = {
      {{(enable_broadcasting ? "gx" : "dx")},
       opname,
       {x0, x1},
       {{"T", "$T"}, {attr_adj_x, ax0}, {attr_adj_y, ax1}}},
      {{(enable_broadcasting ? "gy" : "dy")},
       opname,
       {y0, y1},
       {{"T", "$T"}, {attr_adj_x, ay0}, {attr_adj_y, ay1}}},
  };
  // TODO(anudhyan): Figure out a way to inspect the static shapes of "x" and
  // "y". If they have the same batch dimensions, then we can omit adding the
  // broadcasting-specific ops.
  if (enable_broadcasting) {
    std::vector<FDH::Node> unbroadcast_gradients = {
        FDH::Const<int32>("zero", absl::Span<const int32>{0}),
        FDH::Const<int32>("one", absl::Span<const int32>{1}),
        FDH::Const<int32>("minustwo", absl::Span<const int32>{-2}),
        // Compute the batch shapes of the inputs (all but last two dims).
        {{"sx"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"sy"}, "Shape", {"y"}, {{"T", "$T"}}},
        {{"batch_sx"},
         "StridedSlice",
         {"sx", "zero", "minustwo", "one"},
         {{"T", DT_INT32}, {"Index", DT_INT32}}},
        {{"batch_sy"},
         "StridedSlice",
         {"sy", "zero", "minustwo", "one"},
         {{"T", DT_INT32}, {"Index", DT_INT32}}},
        // Sum along dimensions that the inputs were broadcasted across.
        {{"rx", "ry"}, "BroadcastGradientArgs", {"batch_sx", "batch_sy"}},
        {{"sum_gx"}, "Sum", {"gx", "rx"}, {{"T", "$T"}}},
        {{"sum_gy"}, "Sum", {"gy", "ry"}, {{"T", "$T"}}},
        {{"dx"}, "Reshape", {"sum_gx", "sx"}, {{"T", "$T"}}},
        {{"dy"}, "Reshape", {"sum_gy", "sy"}, {{"T", "$T"}}}};
    nodes.insert(nodes.end(), unbroadcast_gradients.begin(),
                 unbroadcast_gradients.end());
  }
  *g = FDH::Define(
      // Arg defs
      {"x: T", "y: T", "dz: T"},
      // Ret val defs
      {"dx: T", "dy: T"},
      // Attr defs
      {{"T: {half, float, double}"}},
      // Nodes
      nodes);
  return absl::OkStatus();
}

absl::Status MatMulGradCommon(const string& opname, const string& attr_adj_x,
                              const string& attr_adj_y, const AttrSlice& attrs,
                              FunctionDef* g, bool enable_broadcasting) {
  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));
  if (T == DT_COMPLEX64 || T == DT_COMPLEX128) {
    return errors::Unimplemented(
        "MatMul gradient for complex is not supported yet.");
  }
  bool ta;
  bool tb;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, attr_adj_x, &ta));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, attr_adj_y, &tb));
  if (!ta && !tb) {
    return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "dz", false, "y",
                            true, "x", true, "dz", false, enable_broadcasting);
  }
  if (!ta && tb) {
    return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "dz", false, "y",
                            false, "dz", true, "x", false, enable_broadcasting);
  }
  if (ta && !tb) {
    return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "y", false, "dz",
                            true, "x", false, "dz", false, enable_broadcasting);
  }
  CHECK(ta && tb);
  return MatMulGradHelper(g, opname, attr_adj_x, attr_adj_y, "y", true, "dz",
                          true, "dz", true, "x", true, enable_broadcasting);
}

absl::Status MatMulGrad(const AttrSlice& attrs, FunctionDef* g) {
  return MatMulGradCommon("MatMul", "transpose_a", "transpose_b", attrs, g,
                          false /* enable_broadcasting */);
}
REGISTER_OP_GRADIENT("MatMul", MatMulGrad);

absl::Status BatchMatMulGrad(const AttrSlice& attrs, FunctionDef* g) {
  return MatMulGradCommon("BatchMatMul", "adj_x", "adj_y", attrs, g,
                          false /* enable_broadcasting */);
}
REGISTER_OP_GRADIENT("BatchMatMul", BatchMatMulGrad);

absl::Status BatchMatMulV2Grad(const AttrSlice& attrs, FunctionDef* g) {
  return MatMulGradCommon("BatchMatMulV2", "adj_x", "adj_y", attrs, g,
                          true /* enable_broadcasting */);
}
REGISTER_OP_GRADIENT("BatchMatMulV2", BatchMatMulV2Grad);

// REGISTER_OP_GRADIENT("SparseMatMul", SparseMatMulGrad);

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

REGISTER_OP_NO_GRADIENT("Floor");
REGISTER_OP_NO_GRADIENT("FloorDiv");
REGISTER_OP_NO_GRADIENT("TruncateDiv");

}  // end namespace tensorflow
