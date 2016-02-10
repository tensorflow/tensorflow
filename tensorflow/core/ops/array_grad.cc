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

REGISTER_OP_NO_GRADIENT("Shape");
REGISTER_OP_NO_GRADIENT("Rank");
REGISTER_OP_NO_GRADIENT("Size");
REGISTER_OP_NO_GRADIENT("ZerosLike");
REGISTER_OP_NO_GRADIENT("Const");
REGISTER_OP_NO_GRADIENT("EditDistance");
REGISTER_OP_NO_GRADIENT("StopGradient");

Status ReshapeGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "shape: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dshape: int32"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
        {{"x_shape"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"dx"}, "Reshape", {"dy", "x_shape"}, {{"T", "$T"}}},
        {{"dshape"}, "ZerosLike", {"shape"}, {{"T", DT_INT32}}},
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Reshape", ReshapeGrad);
REGISTER_OP_GRADIENT("ExpandDims", ReshapeGrad);

Status SqueezeGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
        {{"x_shape"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"dx"}, "Reshape", {"dy", "x_shape"}, {{"T", "$T"}}},
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Squeeze", SqueezeGrad);

Status IdentityGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
        {{"dx"}, "Identity", {"dy"}, {{"T", "$T"}}},
      });
  // clang-format on
  VLOG(1) << "IdentityGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Identity", IdentityGrad);

Status PackGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: N*T", "dy: T"},
      // Ret val defs
      {"dx: N*T"},
      // Attr defs
      {"T: type", "N: int"},
      // Nodes
      {
        {{"dx"}, "Unpack", {"dy"}, {{"T", "$T"}, {"num", "$N"}}},
      });
  // clang-format on
  VLOG(1) << "PackGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Pack", PackGrad);

Status UnpackGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: num*T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {"T: type", "num: int"},
      // Nodes
      {
        {{"dx"}, "Pack", {"dy"}, {{"T", "$T"}, {"N", "$num"}}},
      });
  // clang-format on
  VLOG(1) << "UnpackGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Unpack", UnpackGrad);

Status ConcatGrad(const AttrSlice& attrs, FunctionDef* g) {
  int N;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "N", &N));
  DataType T;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &T));

  std::vector<string> shape_i;
  std::vector<string> offset_i;
  std::vector<string> dx_i;
  for (int i = 0; i < N; ++i) {
    shape_i.push_back(strings::StrCat("shape_lst:", i));
    offset_i.push_back(strings::StrCat("offset_lst:", i));
    dx_i.push_back(strings::StrCat("dx_", i));
  }
  DataTypeVector dtype_list(N, T);

  // ConcatGrad(dim, x, dy):
  //   for i in range(N):
  //     dx[i] = Slice(dy, offset[i], shape[x[i]]),
  // where offset[i] is the offset of x[i] in the output y,
  // which is the same as dx[i]'s offset within dy.
  std::vector<FDH::Node> nodes{
      {{"shapes"}, "ShapeN", {"x"}, {{"T", "$T"}, {"N", "$N"}}},
      {{"shape_lst"},
       "_ArrayToList",
       {"shapes"},
       {{"T", DT_INT32},
        {"N", "$N"},
        {"out_types", DataTypeVector(N, DT_INT32)}}},
      {{"offset"}, "ConcatOffset", {"dim", "shapes"}, {{"N", "$N"}}},
      {{"offset_lst"},
       "_ArrayToList",
       {"offset"},
       {{"T", DT_INT32},
        {"N", "$N"},
        {"out_types", DataTypeVector(N, DT_INT32)}}},
      {{"d_dim"}, "ZerosLike", {"dim"}, {{"T", DT_INT32}}},
      {{"dx"},
       "_ListToArray",
       dx_i,
       {{"T", "$T"}, {"N", "$N"}, {"Tin", DataTypeVector(N, T)}}}};

  // For each dx[i], we take a slice of dy. The offset and size of the
  // slice is given by offset[i] and shape[i].
  for (int i = 0; i < N; ++i) {
    nodes.push_back({{dx_i[i]},
                     "Slice",
                     {"dy", offset_i[i], shape_i[i]},
                     {{"T", "$T"}, {"Index", DT_INT32}}});
  }
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"dim: int32", "x: N*T", "dy: T"},
      // Ret val defs
      {"d_dim: int32", "dx: N*T"},
      // Attr defs
      {"T: type", "N: int"},
      // Nodes
      nodes);
  // clang-format on
  VLOG(1) << "ConcatGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Concat", ConcatGrad);

Status SplitGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"dim: int32", "x: T", "dy: num_split*T"},
      // Ret val defs
      {"d_dim: int32", "dx: T"},
      // Attr defs
      {"T: type", "num_split: int"},
      // Nodes
      {
        {{"d_dim"}, "ZerosLike", {"dim"}, {{"T", DT_INT32}}},
        {{"dx"}, "Concat", {"dim", "dy"}, {{"T", "$T"}, {"N", "$num_split"}}}
      });
  // clang-format on
  VLOG(1) << "SplitGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Split", SplitGrad);

Status ArrayToListGrad(const AttrSlice& attrs, FunctionDef* g) {
  int N;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "N", &N));
  std::vector<string> dys;
  for (int i = 0; i < N; ++i) {
    dys.push_back(strings::StrCat("dy:", i));
  }
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: N*T", "dy: out_types"},
      // Ret val defs
      {"dx: N*T"},
      // Attr defs
      {"T: type", "N: int", "out_types: list(type)"},
      // Nodes
      {
        {{"dx"}, "_ListToArray", dys,
         {{"T", "$T"}, {"N", "$N"}, {"Tin", "$out_types"}}}
      });
  // clang-format on
  VLOG(1) << "ArrayToListGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("_ArrayToList", ArrayToListGrad);

Status ListToArrayGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: Tin", "dy: N*T"},
      // Ret val defs
      {"dx: Tin"},
      // Attr defs
      {"T: type", "N: int", "Tin: list(type)"},
      // Nodes
      {
        {{"dx"}, "_ArrayToList", {"dy"},
         {{"T", "$T"}, {"N", "$N"}, {"out_types", "$Tin"}}}
      });
  // clang-format on
  VLOG(1) << "ListToArrayGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("_ListToArray", ListToArrayGrad);

Status FillGrad(const AttrSlice& attrs, FunctionDef* g) {
  *g = FDH::Define(
      // Arg defs
      {"dims: int32", "x: T", "dy: T"},
      // Ret val defs
      {"d_dims: int32", "dx: T"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
          {{"d_dims"}, "ZerosLike", {"dims"}, {{"T", DT_INT32}}},
          FDH::Const("zero", 0),
          {{"rank"}, "Rank", {"dy"}, {{"T", "$T"}}},
          FDH::Const("one", 1),
          {{"r"}, "Range", {"zero", "rank", "one"}, {}},
          // dx = sum(dy)
          {{"dx"}, "Sum", {"dy", "r"}, {{"T", "$T"}}},
      });
  VLOG(1) << "FillGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Fill", FillGrad);

Status TransposeGrad(const AttrSlice& attrs, FunctionDef* g) {
  *g = FDH::Define(
      // Arg defs
      {"x: T", "p: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dp: int32"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
          {{"q"}, "InvertPermutation", {"p"}, {}},
          {{"dx"}, "Transpose", {"dy", "q"}, {{"T", "$T"}}},
          {{"dp"}, "ZerosLike", {"p"}, {{"T", DT_INT32}}},
      });
  VLOG(1) << "TransposeGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Transpose", TransposeGrad);

Status ReverseGrad(const AttrSlice& attrs, FunctionDef* g) {
  *g = FDH::Define(
      // Arg defs
      {"x: T", "d: bool", "dy: T"},
      // Ret val defs
      {"dx: T", "dd: bool"},
      // Attr defs
      {"T: type"},
      // Nodes
      {
          {{"dx"}, "Reverse", {"dy", "d"}, {{"T", "$T"}}},
          {{"dd"}, "ZerosLike", {"d"}, {{"T", DT_BOOL}}},
      });
  VLOG(1) << "ReverseGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Reverse", ReverseGrad);

Status SliceGrad(const AttrSlice& attrs, FunctionDef* g) {
  DataType itype;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "Index", &itype));
  if (itype != DT_INT32) {
    return errors::Unimplemented(
        "SliceGrad for int64 index are not supported.");
  }
  *g = FDH::Define(
      // Arg defs
      {"x: T", "b: int32", "s: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "db: int32", "ds: int32"},
      // Attr defs
      {"T: type"},
      // Nodes
      {// paddings = concat(1, [b, shape(x) - b - s])
       FDH::Const("one", 1),
       {{"b1"}, "ExpandDims", {"b", "one"}, {{"T", DT_INT32}}},
       {{"xs"}, "Shape", {"x"}, {{"T", "$T"}}},
       {{"xs_b"}, "Sub", {"xs", "b"}, {{"T", DT_INT32}}},
       {{"xs_b_s"}, "Sub", {"xs_b", "s"}, {{"T", DT_INT32}}},
       {{"a1"}, "ExpandDims", {"xs_b_s", "one"}, {{"T", DT_INT32}}},
       {{"b_and_a"},
        "_ListToArray",
        {"b1", "a1"},
        {{"T", DT_INT32},
         {"N", 2},
         {"Tin", DataTypeVector{DT_INT32, DT_INT32}}}},
       {{"paddings"},
        "Concat",
        {"one", "b_and_a"},
        {{"N", 2}, {"T", DT_INT32}}},
       // dx = Pad(dy, paddings)
       {{"dx"}, "Pad", {"dy", "paddings"}, {{"T", "$T"}}},
       {{"db"}, "ZerosLike", {"b"}, {{"T", DT_INT32}}},
       {{"ds"}, "ZerosLike", {"s"}, {{"T", DT_INT32}}}});
  VLOG(1) << "SliceGrad " << DebugString(*g);
  return Status::OK();
}
REGISTER_OP_GRADIENT("Slice", SliceGrad);

}  // end namespace tensorflow
