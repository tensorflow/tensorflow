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

#include "tensorflow/core/framework/function.h"
#include <vector>
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef FunctionDefHelper FDH;

Status GetOpSig(const string& op, const OpDef** sig) {
  return OpRegistry::Global()->LookUpOpDef(op, sig);
}

REGISTER_OP("One")
    .Output("y: T")
    .Attr("T: {float, double, int32, int64}")
    .Doc(R"doc(
Returns a tensor with a single element (1) of type T.

y: A scalar in type T.

)doc");

static InstantiateAttrValueMap kNoAttrs;

TEST(TFunc, SquarePlusOne) {
  auto fdef = FDH::Create(
      // Name
      "SquarePlusOne",
      // Inputs
      {"x: T"},
      // Outputs
      {"y: T"},
      // Attrs
      {"T: {float, double, int32, int64}"},
      // Nodes
      {// a = Square<T>(x)
       {{"a"}, "Square", {"x"}, {{"T", "$T"}}},
       // o = One<T>()
       // NOTE: We can also have a Cast<Tin, Tout>(x) instead.
       {{"o"}, "One", {}, {{"T", "$T"}}},
       // y = Add<T>(a, o)
       {{"y"}, "Add", {"a:y", "o:y"}, {{"T", "$T"}}}},
      // Returns
      {{"y", "y:z:0"}});

  const char* e = R"P(
SquarePlusOne[T:{float, double, int32, int64}](x:T) -> (y:T) {
  a = Square[T=$T](x)
  o = One[T=$T]()
  y = Add[T=$T](a:y, o:y)
  return y = y:z:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  // Instantiate one with T=float
  InstantiationResult result;
  TF_ASSERT_OK(InstantiateFunction(fdef, {{"T", DT_FLOAT}}, GetOpSig, &result));
  const char* e2 = R"P(
(n0:float) -> (n3:float) {
  n1 = Square[T=float](n0)
  n2 = One[T=float]()
  n3 = Add[T=float](n1, n2)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.gdef), e2);
}

TEST(TFunc, ControlDep) {
  auto fdef = FDH::Create(
      // Name
      "ControlDep",
      // Inputs
      {"x: int32"},
      // Outputs
      {"y: int32"},
      // Attrs
      {},
      // Nodes
      {// a = Identity<int32>(x)
       {{"a"}, "Identity", {"x"}, {{"T", DT_INT32}}},
       // o = NoOp(^a)
       {{"o"}, "NoOp", {"^a"}, {}},
       // y = Identity<int32>(a, ^o)
       {{"y"}, "Identity", {"a:output:0", "^o"}, {{"T", DT_INT32}}}},
      // Returns
      {{"y", "y:output:0"}});

  const char* e = R"P(
ControlDep(x:int32) -> (y:int32) {
  a = Identity[T=int32](x)
  o = NoOp() @ a
  y = Identity[T=int32](a:output:0) @ o
  return y = y:output:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  // Instantiate one with T=float
  InstantiationResult result;
  TF_ASSERT_OK(InstantiateFunction(fdef, {{"T", DT_FLOAT}}, GetOpSig, &result));
  const char* e2 = R"P(
(n0:int32) -> (n3:int32) {
  n1 = Identity[T=int32](n0)
  n2 = NoOp() @ n1
  n3 = Identity[T=int32](n1) @ n2
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_INT32}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_INT32}));
  EXPECT_EQ(DebugString(result.gdef), e2);
}

REGISTER_OP("HasDefaultType")
    .Output("out: T")
    .Attr("T: {float, double, int32, int64} = DT_FLOAT");

// This verifies that a function using an op before a type attr (with
// a default) is added, still works.  This is important for backwards
// compatibilty.
TEST(TFunc, MissingTypeAttr) {
  auto fdef = FDH::Create(
      // Name
      "BackCompat",
      // Args
      {},
      // Return values
      {"y: float"},
      // Attrs
      {},
      // Nodes
      {// y = HasDefaultType(x), T missing, defaults to float
       {{"a"}, "HasDefaultType", {}, {}}},
      // Returns
      {{"y", "a:out:0"}});

  const char* e = R"P(
BackCompat() -> (y:float) {
  a = HasDefaultType()
  return y = a:out:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  TF_ASSERT_OK(
      InstantiateFunction(fdef, InstantiateAttrValueMap{}, GetOpSig, &result));
  // Should get T=float from Op's default.
  const char* e2 = R"P(
() -> (n0:float) {
  n0 = HasDefaultType[T=float]()
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector());
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.gdef), e2);
}

TEST(TFunc, NTimesT) {
  auto fdef = FDH::Create(
      // Name
      "NTimesT",
      // Inputs
      {"x: float", "y: float"},
      // Outputs
      {"z: float"},
      // Attrs
      {},
      // Nodes
      {// a = AddN<N=2>(x, y)
       {{"a"}, "AddN", {"x", "y"}, {{"T", DT_FLOAT}, {"N", 2}}}},
      // Returns
      {{"z", "a:sum:0"}});

  const char* e = R"P(
NTimesT(x:float, y:float) -> (z:float) {
  a = AddN[N=2, T=float](x, y)
  return z = a:sum:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  TF_ASSERT_OK(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result));
  const char* e2 = R"P(
(n0:float, n1:float) -> (n2:float) {
  n2 = AddN[N=2, T=float](n0, n1)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT, DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.gdef), e2);
}

// NOTE: This is the simplest Map op. It takes a f:T->U.
REGISTER_OP("Map")
    .Input("x: N * T")
    .Output("y: N * U")
    .Attr("T: type")
    .Attr("U: type")
    .Attr("N: int >= 1")
    // .Attr("func: func_name_with_attr")
    .Doc(R"doc(
Applies the 'func' on every input. I.e.,

y[i] = func<...>(x[i])

x: N tensors, each of type T;
y: N tensors, each of type U;

)doc");

TEST(TFunc, AddSquared) {
  auto fdef = FDH::Create(
      // Name
      "AddSquared",
      // Args
      {"x: N*T"},
      // Return values
      {"y: T"},
      // Attrs
      {"N:int", "T:{float, double, int32, int64}"},
      // Nodes
      {// a = Map<func=Square<$T>,T=$T,U=$T,N=$N>(x)
       {{"a"},
        "Map",
        {"x"},
        {{"func", FDH::FunctionRef("Square", {{"T", "$T"}})},
         {"T", "$T"},
         {"U", "$T"},
         {"N", "$N"}}},
       // y = AddN<N=$N,T=$T>(a)
       {{"y"}, "AddN", {"a:y"}, {{"N", "$N"}, {"T", "$T"}}}},
      {{"y", "y:sum"}});

  const char* e = R"P(
AddSquared[N:int, T:{float, double, int32, int64}](x:N*T) -> (y:T) {
  a = Map[N=$N, T=$T, U=$T, func=Square[T=$T]](x)
  y = AddN[N=$N, T=$T](a:y)
  return y = y:sum
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  // Instantiate one with T=float
  InstantiationResult result;
  TF_ASSERT_OK(InstantiateFunction(fdef, {{"N", 3}, {"T", DT_FLOAT}}, GetOpSig,
                                   &result));
  const char* e2 = R"P(
(n0:float, n1:float, n2:float) -> (n4:float) {
  n3 = Map[N=3, T=float, U=float, func=Square[T=float]](n0, n1, n2)
  n4 = AddN[N=3, T=float](n3, n3:1, n3:2)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT, DT_FLOAT, DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.gdef), e2);
}

TEST(TFunc, ControlDeps) {
  auto fdef = FDH::Define(
      // Name
      "ControlDeps",
      // Args
      {"x: float"},
      // Return values
      {},
      // Attrs
      {},
      // Nodes
      {
          {{"a"}, "One", {}, {{"T", DT_FLOAT}}, {"x"}},
          {{"u"}, "NoOp", {}, {}, {"a"}},
          {{"b"}, "One", {}, {{"T", DT_FLOAT}}, {"u"}},
          {{"v"}, "NoOp", {}, {}, {"b"}},
          {{"c"}, "One", {}, {{"T", DT_FLOAT}}, {"a", "v"}},
      });
  const char* e = R"P(
ControlDeps(x:float) -> () {
  a = One[T=float]() @ x
  u = NoOp() @ a
  b = One[T=float]() @ u
  v = NoOp() @ b
  c = One[T=float]() @ a, v
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  TF_ASSERT_OK(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result));
  const char* e2 = R"P(
(n0:float) -> () {
  n1 = One[T=float]() @ n0
  n2 = NoOp() @ n1
  n3 = One[T=float]() @ n2
  n4 = NoOp() @ n3
  n5 = One[T=float]() @ n1, n4
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({}));
  EXPECT_EQ(DebugString(result.gdef), e2);
}

TEST(TFunc, XTimesTwo) {
  auto expect = R"P(
XTimesTwo[T:{float, double, int32, int64}](x:T) -> (y:T) {
  two = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  scale = Cast[DstT=$T, SrcT=int64](two:output:0)
  y = Mul[T=$T](x, scale:y:0)
  return y = y:z:0
}
)P";
  EXPECT_EQ(expect, DebugString(test::function::XTimesTwo()));
}

TEST(TFunc, WXPlusB) {
  auto expect = R"P(
WXPlusB[T:{float, double}](w:T, x:T, b:T) -> (y:T) {
  mm = MatMul[T=$T, _kernel="eigen", transpose_a=false, transpose_b=false](w, x)
  y = Add[T=$T](mm:product:0, b)
  return y = y:z:0
}
)P";
  EXPECT_EQ(expect, DebugString(test::function::WXPlusB()));
}

TEST(TFunc, Body_TypeList) {
  const Tensor kZero = test::AsScalar<int32>(0);
  auto fdef = FDH::Create(
      // Name
      "Test",
      // Args
      {"i:float"},
      // Return values
      {"o:float"},
      // Attrs
      {},
      // Nodes
      {{{"zero"}, "Const", {}, {{"value", kZero}, {"dtype", DT_INT32}}},
       {{"s"},
        "Split",
        {"zero:output:0", "i"},
        {{"num_split", 4}, {"T", DT_FLOAT}}},
       {{"l"}, "Mul", {"s:output:0", "s:output:1"}, {{"T", DT_FLOAT}}},
       {{"r"}, "Mul", {"s:output:2", "s:output:3"}, {{"T", DT_FLOAT}}},
       {{"x"},
        "_ListToArray",
        {"l:z", "r:z"},
        {{"N", 2},
         {"T", DT_FLOAT},
         {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}}}},
       {{"o"}, "AddN", {"x:output"}, {{"N", 2}, {"T", DT_FLOAT}}}},
      {{"o", "o:sum:0"}});

  const char* e = R"P(
Test(i:float) -> (o:float) {
  zero = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  s = Split[T=float, num_split=4](zero:output:0, i)
  l = Mul[T=float](s:output:0, s:output:1)
  r = Mul[T=float](s:output:2, s:output:3)
  x = _ListToArray[N=2, T=float, Tin={float, float}](l:z, r:z)
  o = AddN[N=2, T=float](x:output)
  return o = o:sum:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  TF_ASSERT_OK(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result));
  const char* e2 = R"P(
(n0:float) -> (n6:float) {
  n1 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  n2 = Split[T=float, num_split=4](n1, n0)
  n3 = Mul[T=float](n2, n2:1)
  n4 = Mul[T=float](n2:2, n2:3)
  n5 = _ListToArray[N=2, T=float, Tin={float, float}](n3, n4)
  n6 = AddN[N=2, T=float](n5, n5:1)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.gdef), e2);
}

REGISTER_OP("Cond")
    .Input("input: Tin")
    .Output("output: out_types")
    .Attr("Tin: list(type)")
    .Attr("out_types: list(type)")
    .Attr("cond: func")
    .Attr("then_branch: func")
    .Attr("else_branch: func")
    .Doc(R"doc(
output = Cond(input) ? then_branch(input) : else_branch(input)

cond: A function takes 'input' and returns a scalar.
then_branch: A function takes 'input' and returns 'output'.
else_branch: A function takes 'input' and returns 'output'.
)doc");

TEST(TFunc, Body_Array_List_Converter) {
  auto fdef = FDH::Define(
      // Name
      "MySelect",
      // Args
      {"x:float"},
      // Return values
      {"z:float"},
      // Attrs
      {},
      // Nodes
      {
          {{"y"},
           "Cond",
           {"x"},
           {{"Tin", DataTypeSlice{DT_FLOAT}},
            {"out_types", DataTypeSlice{DT_FLOAT}},
            {"cond", FDH::FunctionRef("MyCond")},
            {"then_branch", FDH::FunctionRef("MyThen")},
            {"else_branch", FDH::FunctionRef("MyElse")}}},
          {{"z"},
           "Cond",
           {"y", "y"},
           {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
            {"out_types", DataTypeSlice{DT_FLOAT}},
            {"cond", FDH::FunctionRef("MyCond2")},
            {"then_branch", FDH::FunctionRef("MyThen2")},
            {"else_branch", FDH::FunctionRef("MyElse2")}}},
      });

  const char* e = R"P(
MySelect(x:float) -> (z:float) {
  y = Cond[Tin={float}, cond=MyCond, else_branch=MyElse, out_types={float}, then_branch=MyThen](x)
  z = Cond[Tin={float, float}, cond=MyCond2, else_branch=MyElse2, out_types={float}, then_branch=MyThen2](y:output:0, y:output:0)
  return z = z:output:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  TF_ASSERT_OK(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result));
  const char* e2 = R"P(
(n0:float) -> (n2:float) {
  n1 = Cond[Tin={float}, cond=MyCond, else_branch=MyElse, out_types={float}, then_branch=MyThen](n0)
  n2 = Cond[Tin={float, float}, cond=MyCond2, else_branch=MyElse2, out_types={float}, then_branch=MyThen2](n1, n1)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.gdef), e2);
}

static void HasError(const Status& s, const string& substr) {
  EXPECT_TRUE(StringPiece(s.ToString()).contains(substr))
      << ">>" << s << "<<, expected substring >>" << substr << "<<";
}

TEST(InstantiateErrors, Not_Sufficient_Attrs) {
  auto fdef =
      FDH::Define("nop", {}, {}, {"T:{float, double, int32, int64}"}, {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, {{"U", DT_FLOAT}}, GetOpSig, &result),
           "Attr T is not found from ");
}

#if 0  // TODO(josh11b): Enable this test once having an extra attr is an error.
TEST(InstantiateErrors, Too_Many_Attrs) {
  auto fdef =
      FDH::Define("nop", {}, {}, {"T:{float, double, int32, int64}"}, {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, {{"T", DT_INT32}, {"U", DT_FLOAT}},
                               GetOpSig, &result),
           "Attr U is not found in ");
}
#endif

TEST(InstantiateErrors, AttrValue_Value_Placeholder) {
  auto fdef =
      FDH::Define("nop", {}, {}, {"T:{float, double, int32, int64}"}, {});
  InstantiationResult result;
  HasError(
      InstantiateFunction(fdef, {{"T", "$bad"}}, GetOpSig, &result),
      "AttrValue had value with unexpected type 'placeholder'\n\tfor attr 'T'");
}

TEST(InstantiateErrors, Unbounded_Attr) {
  auto fdef = FDH::Define("test", {}, {}, {"T:{float, double, int32, int64}"},
                          {
                              {{"a"}, "One", {}, {{"T", "$unknown"}}, {"x"}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, {{"T", DT_FLOAT}}, GetOpSig, &result),
           "Failed to bind all placeholders");
}

TEST(InstantiateErrors, DupArgs) {
  auto fdef = FDH::Define("test", {"x:float", "x:float"}, {}, {}, {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Duplicated arg name");
}

TEST(InstantiateErrors, Dup_Node_Names) {
  auto fdef = FDH::Define("test", {"x:float"}, {}, {},
                          {
                              {{"y"}, "One", {}, {{"T", DT_FLOAT}}},
                              {{"y"}, "One", {}, {{"T", DT_FLOAT}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Duplicated ret name");
}

TEST(InstantiateErrors, Node_Arg_Notfound) {
  auto fdef = FDH::Create("test", {"x:float"}, {}, {},
                          {
                              {{"y"}, "Add", {"x", "z"}, {{"T", DT_FLOAT}}},
                          },
                          {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "input z is not found");
}

TEST(InstantiateErrors, Node_Arg_TypeMismatch) {
  auto fdef = FDH::Define("test", {"x:float"}, {}, {},
                          {
                              {{"y"}, "Add", {"x", "x"}, {{"T", DT_INT32}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "input x[0] expected type int32 != float, the type of x[0]");
}

TEST(InstantiateErrors, Node_Arg_ControlMissing) {
  auto fdef =
      FDH::Define("test", {"x:float"}, {}, {},
                  {
                      {{"y"}, "Add", {"x", "x"}, {{"T", DT_FLOAT}}, {"z"}},
                  });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "input[2] == '^z', is not found.");
}

TEST(InstantiateErrors, FuncRet_Missing) {
  auto fdef = FDH::Create("test", {}, {"y: float"}, {},
                          {
                              {{"x"}, "One", {}, {{"T", DT_FLOAT}}},
                          },
                          {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Return y missing");
}

TEST(InstantiateErrors, FuncRet_NotFound) {
  auto fdef = FDH::Create("test", {}, {"y: float"}, {},
                          {
                              {{"x"}, "One", {}, {{"T", DT_FLOAT}}},
                          },
                          {{"y", "z"}});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Return y -> z is not found");
}

TEST(InstantiateErrors, FuncRet_NameMismatch) {
  auto fdef = FDH::Create("test", {}, {"y: float"}, {},
                          {
                              {{"x"}, "One", {}, {{"T", DT_FLOAT}}},
                          },
                          {{"z", "x:y:0"}});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Return y missing");
}

// TODO(josh11b): Make this an error.
// TEST(InstantiateErrors, FuncRet_Extra) {
//   auto fdef = FDH::Create("test", {}, {"y: float"}, {},
//                           {
//                               {{"x"}, "One", {}, {{"T", DT_FLOAT}}},
//                           },
//                           {{"y", "x:y:0"}, {"z", "x:y:0"}});
//   InstantiationResult result;
//   HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
//            "ret is not found");
// }

TEST(InstantiateErrors, FuncRet_TypeMismatch) {
  auto fdef = FDH::Define("test", {}, {"y: float"}, {},
                          {
                              {{"y"}, "One", {}, {{"T", DT_DOUBLE}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Invalid ret types y : float vs. double\n\tIn function output y");
}

TEST(InstantiateErrors, TypeList_Missing_Retval_Attr) {
  auto fdef = FDH::Create(
      // Name
      "MySelect",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attrs
      {},
      // Nodes
      {
          {{"y"},
           "Cond",
           {"x", "x"},
           {{"tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
            {"cond", FDH::FunctionRef("MyCond2")},
            {"then_branch", FDH::FunctionRef("MyThen2")},
            {"else_branch", FDH::FunctionRef("MyElse2")}}},
      },
      {{"y", "y:output"}});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "type attr not found: out_types");
}

TEST(InstantiateErrors, TypeList_Num_Retval_Mismatch) {
  auto fdef = FDH::Create(
      // Name
      "MySelect",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attrs
      {},
      // Nodes
      {
          {{"y"},
           "Cond",
           {"x", "x"},
           {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
            {"out_types", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
            {"cond", FDH::FunctionRef("MyCond2")},
            {"then_branch", FDH::FunctionRef("MyThen2")},
            {"else_branch", FDH::FunctionRef("MyElse2")}}},
      },
      {{"y", "y:output"}});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Invalid ret types");
}

TEST(InstantiateErrors, TypeList_Missing_Arg) {
  auto fdef = FDH::Create(
      // Name
      "MySelect",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attrs
      {},
      // Nodes
      {
          {{"y"},
           "Cond",
           {"x", "unknown"},
           {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
            {"out_types", DataTypeSlice{DT_FLOAT}},
            {"cond", FDH::FunctionRef("MyCond2")},
            {"then_branch", FDH::FunctionRef("MyThen2")},
            {"else_branch", FDH::FunctionRef("MyElse2")}}},
      },
      {{"y", "y:output"}});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "input unknown is not found");
}

TEST(InstantiateErrors, TooManyInputs) {
  auto fdef = FDH::Create(
      // Name
      "TooManyInputs",
      // Inputs
      {"x: float", "y: float"},
      // Outputs
      {"z: float"},
      // Attrs
      {},
      // Nodes
      {// a = AddN<N=2>(x, y, x)
       {{"a"}, "AddN", {"x", "y", "x"}, {{"T", DT_FLOAT}, {"N", 2}}}},
      // Returns
      {{"z", "a:sum:0"}});

  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Expected input[2] == 'x' to be a control input.");
}

TEST(InstantiateErrors, TooFewInputs) {
  auto fdef = FDH::Create(
      // Name
      "TooFewInputs",
      // Inputs
      {"x: float", "y: float"},
      // Outputs
      {"z: float"},
      // Attrs
      {},
      // Nodes
      {// a = AddN<N=3>(x, y)
       {{"a"}, "AddN", {"x", "y"}, {{"T", DT_FLOAT}, {"N", 3}}}},
      // Returns
      {{"z", "a:sum:0"}});

  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Attempt to access beyond input size: 2 >= 2");
}

TEST(InstantiateErrors, TooManyInputsFromArray1) {
  auto fdef = FDH::Create(
      // Name
      "TooManyInputsFromArray",
      // Inputs
      {"x: float", "y: float"},
      // Outputs
      {"z: float"},
      // Attrs
      {},
      // Nodes
      {// a = _ListToArray(x,y)
       {{"a"},
        "_ListToArray",
        {"x", "y"},
        {{"N", 2},
         {"T", DT_FLOAT},
         {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}}}},
       // b = AddN<N=2>(a, y)
       {{"b"}, "AddN", {"a:output", "y"}, {{"T", DT_FLOAT}, {"N", 2}}}},
      // Returns
      {{"z", "a:sum:0"}});

  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Expected input[1] == 'y' to be a control input.");
}

TEST(InstantiateErrors, TooManyInputsFromArray2) {
  auto fdef = FDH::Create(
      // Name
      "TooManyInputsFromArray",
      // Inputs
      {"x: float", "y: float"},
      // Outputs
      {"z: float"},
      // Attrs
      {},
      // Nodes
      {// a = _ListToArray(x,y)
       {{"a"},
        "_ListToArray",
        {"x", "y"},
        {{"N", 2},
         {"T", DT_FLOAT},
         {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}}}},
       // b = AddN<N=2>(x, a)
       {{"b"}, "AddN", {"x", "a:output"}, {{"T", DT_FLOAT}, {"N", 2}}}},
      // Returns
      {{"z", "a:sum:0"}});

  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Input a:output too long for inputs");
}

TEST(InstantiateErrors, TypeMismatch) {
  auto fdef = FDH::Create(
      // Name
      "TypeMismatch",
      // Inputs
      {"x: float", "y: int32"},
      // Outputs
      {"z: float"},
      // Attrs
      {},
      // Nodes
      {// a = AddN<N=2>(x, y)
       {{"a"}, "AddN", {"x", "y"}, {{"T", DT_FLOAT}, {"N", 3}}}},
      // Returns
      {{"z", "a:sum:0"}});

  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "input inputs[1] expected type float != int32, the type of y[0]");
}

TEST(FunctionCallFrame, Void_Void) {
  FunctionCallFrame frame({}, {});
  TF_EXPECT_OK(frame.SetArgs({}));
  auto a = test::AsTensor<float>({100});
  HasError(frame.SetArgs({a}), "Invalid argument");
  Tensor v;
  HasError(frame.GetArg(0, &v), "Invalid argument");
  HasError(frame.SetRetval(0, v), "Invalid argument");
  std::vector<Tensor> rets;
  TF_EXPECT_OK(frame.GetRetvals(&rets));
  EXPECT_EQ(rets.size(), 0);
}

TEST(FunctionCallFrame, Float_Float_Float) {
  FunctionCallFrame frame({DT_FLOAT, DT_FLOAT}, {DT_FLOAT});
  HasError(frame.SetArgs({}), "Invalid argument: Expects 2 arguments");
  auto a = test::AsTensor<float>({100});
  auto b = test::AsTensor<float>({200});
  auto c = test::AsTensor<int64>({300});
  HasError(frame.SetArgs({a, c}),
           "Invalid argument: Expects arg[1] to be float");
  TF_EXPECT_OK(frame.SetArgs({a, b}));

  Tensor v;
  HasError(frame.GetArg(-1, &v), "Invalid argument");
  HasError(frame.GetArg(2, &v), "Invalid argument");
  TF_EXPECT_OK(frame.GetArg(0, &v));
  test::ExpectTensorEqual<float>(a, v);
  TF_EXPECT_OK(frame.GetArg(1, &v));
  test::ExpectTensorEqual<float>(b, v);

  v = test::AsTensor<float>({-100});
  HasError(frame.SetRetval(-1, v), "Invalid argument");
  HasError(frame.SetRetval(1, v), "Invalid argument");
  HasError(frame.SetRetval(0, test::AsTensor<int64>({-100})),
           "Invalid argument: Expects ret[0] to be float");

  std::vector<Tensor> rets;
  HasError(frame.GetRetvals(&rets), "does not have value");
  TF_EXPECT_OK(frame.SetRetval(0, v));
  HasError(frame.SetRetval(0, v), "has already been set");

  TF_EXPECT_OK(frame.GetRetvals(&rets));
  EXPECT_EQ(rets.size(), 1);
  test::ExpectTensorEqual<float>(rets[0], v);
}

TEST(Canonicalize, Basic) {
  EXPECT_EQ(Canonicalize("MatMul", {{"T", DT_FLOAT},
                                    {"transpose_a", false},
                                    {"transpose_b", false}}),
            "MatMul[T=float,transpose_a=false,transpose_b=false]");
  EXPECT_EQ(Canonicalize("MatMul", {{"T", DT_FLOAT},
                                    {"transpose_b", false},
                                    {"transpose_a", false}}),
            "MatMul[T=float,transpose_a=false,transpose_b=false]");
  EXPECT_EQ(Canonicalize("MatMul", {{"T", DT_DOUBLE},
                                    {"transpose_b", true},
                                    {"transpose_a", false}}),
            "MatMul[T=double,transpose_a=false,transpose_b=true]");
}

TEST(FunctionLibraryDefinitionTest, Find) {
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);

  EXPECT_EQ(lib_def.Find("XTimes16"), nullptr);

  auto expect = R"P(
XTimesTwo[T:{float, double, int32, int64}](x:T) -> (y:T) {
  two = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  scale = Cast[DstT=$T, SrcT=int64](two:output:0)
  y = Mul[T=$T](x, scale:y:0)
  return y = y:z:0
}
)P";
  auto found = lib_def.Find("XTimesTwo");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(expect, DebugString(*found));
}

TEST(FunctionLibraryDefinitionTest, LookUp) {
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);

  const OpDef* op_def;
  EXPECT_TRUE(!lib_def.LookUpOpDef("XTimes16", &op_def).ok());

  TF_EXPECT_OK(lib_def.LookUpOpDef("XTimesTwo", &op_def));
  ASSERT_NE(op_def, nullptr);
  EXPECT_EQ(op_def->DebugString(),
            test::function::XTimesTwo().signature().DebugString());
}

TEST(FunctionLibraryDefinitionTest, AddFunctionDef) {
  // Add one function to the proto lib before constructing 'lib_def'.
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);

  // Add a new function def to the library.
  TF_EXPECT_OK(lib_def.AddFunctionDef(test::function::WXPlusB()));

  // Test lookup of first function.
  const OpDef* first;
  TF_EXPECT_OK(lib_def.LookUpOpDef("XTimesTwo", &first));
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->DebugString(),
            test::function::XTimesTwo().signature().DebugString());

  // Test lookup of second function.
  const OpDef* second;
  TF_EXPECT_OK(lib_def.LookUpOpDef("WXPlusB", &second));
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second->DebugString(),
            test::function::WXPlusB().signature().DebugString());
}

TEST(FunctionLibraryDefinitionTest, ToProto) {
  FunctionDefLibrary proto1;
  *proto1.add_function() = test::function::XTimesTwo();
  *proto1.add_function() = test::function::WXPlusB();
  FunctionLibraryDefinition lib_def1(OpRegistry::Global(), proto1);

  // Call 'ToProto' and make sure both protos have the same function lib size.
  FunctionDefLibrary proto2 = lib_def1.ToProto();
  EXPECT_EQ(proto1.function_size(), proto2.function_size());

  // Initialize 'lib_def2' with proto returned by 'ToProto' call.
  FunctionLibraryDefinition lib_def2(OpRegistry::Global(), proto2);

  // Test that the first function exists in both libraries.
  const OpDef *f1, *f2, *f3, *f4;
  TF_EXPECT_OK(lib_def1.LookUpOpDef("XTimesTwo", &f1));
  TF_EXPECT_OK(lib_def2.LookUpOpDef("XTimesTwo", &f2));
  EXPECT_EQ(f1->DebugString(), f2->DebugString());

  // Test that the second function exists in both libraries.
  TF_EXPECT_OK(lib_def1.LookUpOpDef("WXPlusB", &f3));
  TF_EXPECT_OK(lib_def2.LookUpOpDef("WXPlusB", &f4));
  EXPECT_EQ(f3->DebugString(), f4->DebugString());
}

TEST(FunctionLibraryDefinitionTest, GetAttr_FuncNoAttr) {
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  FunctionLibraryDefinition lib(OpRegistry::Global(), proto);

  NodeDef ndef;
  bool annotation;

  // Not a function.
  ndef.set_op("Matmul");
  EXPECT_FALSE(lib.GetAttr(ndef, "annotation", &annotation).ok());

  // A function. No attr defined.
  ndef.set_op("XTimesTwo");
  EXPECT_FALSE(lib.GetAttr(ndef, "annotation", &annotation).ok());

  // ndef defines the attr. But we don't care.
  AddNodeAttr("annotation", true, &ndef);
  EXPECT_FALSE(lib.GetAttr(ndef, "annotation", &annotation).ok());
}

template <typename T>
void SetAttrValue(FunctionDef* fdef, const string& attr, const T& value) {
  AttrValue attr_value;
  SetAttrValue(value, &attr_value);
  fdef->mutable_attr()->insert({attr, attr_value});
}

TEST(FunctionLibraryDefinitionTest, GetAttr_FuncWithAttr) {
  FunctionDefLibrary proto;
  auto fdef = proto.add_function();
  *fdef = test::function::XTimesTwo();
  SetAttrValue(fdef, "annotation", true);
  SetAttrValue(fdef, "options", "some string data");
  FunctionLibraryDefinition lib(OpRegistry::Global(), proto);

  NodeDef ndef;
  bool annotation;

  // A function. No attr defined in ndef.
  ndef.set_op("XTimesTwo");
  TF_EXPECT_OK(lib.GetAttr(ndef, "annotation", &annotation));
  EXPECT_EQ(annotation, true);

  string str;
  TF_EXPECT_OK(lib.GetAttr(ndef, "options", &str));
  EXPECT_EQ(str, "some string data");
}

TEST(FunctionLibraryDefinitionTest, GetAttr_Gradient) {
  FunctionDefLibrary proto;
  auto fdef = proto.add_function();
  *fdef = test::function::XTimesTwo();
  SetAttrValue(fdef, "annotation", true);
  *fdef = test::function::WXPlusB();
  SetAttrValue(fdef, "annotation", false);
  auto func_grad = proto.add_gradient();
  func_grad->set_function_name("XTimesTwo");
  func_grad->set_gradient_func("WXPlusB");
  FunctionLibraryDefinition lib(OpRegistry::Global(), proto);

  NodeDef ndef;
  ndef.set_op(FunctionLibraryDefinition::kGradientOp);

  bool annotation;
  EXPECT_FALSE(lib.GetAttr(ndef, "annotation", &annotation).ok());

  NameAttrList nal;
  nal.set_name("XTimesTwo");
  AddNodeAttr(FunctionLibraryDefinition::kFuncAttr, nal, &ndef);
  TF_EXPECT_OK(lib.GetAttr(ndef, "annotation", &annotation));
  EXPECT_EQ(annotation, false);  // XTimesTwo's gradient is WXPlusB.

  nal.set_name("WXPlusB");
  ndef.clear_attr();
  AddNodeAttr(FunctionLibraryDefinition::kFuncAttr, nal, &ndef);
  TF_EXPECT_OK(lib.GetAttr(ndef, "annotation", &annotation));
  EXPECT_EQ(annotation, false);  // WXPlusB has no custom gradient.
}

}  // end namespace tensorflow
