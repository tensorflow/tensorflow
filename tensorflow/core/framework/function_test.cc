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
  Status s;
  *sig = OpRegistry::Global()->LookUp(op, &s);
  return s;
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
  auto fdef = FDH::Define(
      // Name
      "SquarePlusOne",
      // Args
      {"x: T"},
      // Return values
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
       {{"y"}, "Add", {"a", "o"}, {{"T", "$T"}}}});

  const char* e = R"P(
SquarePlusOne[T:{float, double, int32, int64}](x:T) -> (y:T) {
  a = Square[T=$T](x)
  o = One[T=$T]()
  y = Add[T=$T](a, o)
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  // Instantiate one with T=float
  InstantiationResult result;
  TF_CHECK_OK(InstantiateFunction(fdef, {{"T", DT_FLOAT}}, GetOpSig, &result));
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
  auto fdef = FDH::Define(
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
       {{"y"}, "AddN", {"a"}, {{"N", "$N"}, {"T", "$T"}}}});

  const char* e = R"P(
AddSquared[N:int, T:{float, double, int32, int64}](x:N*T) -> (y:T) {
  a = Map[N=$N, T=$T, U=$T, func=Square[T=$T]](x)
  y = AddN[N=$N, T=$T](a)
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  // Instantiate one with T=float
  InstantiationResult result;
  TF_CHECK_OK(InstantiateFunction(fdef, {{"N", 3}, {"T", DT_FLOAT}}, GetOpSig,
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
  TF_CHECK_OK(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result));
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
  scale = Cast[DstT=$T, SrcT=int64](two)
  y = Mul[T=$T](x, scale)
}
)P";
  EXPECT_EQ(expect, DebugString(test::function::XTimesTwo()));
}

TEST(TFunc, WXPlusB) {
  auto expect = R"P(
WXPlusB[T:{float, double}](w:T, x:T, b:T) -> (y:T) {
  mm = MatMul[T=$T, _kernel="eigen", transpose_a=false, transpose_b=false](w, x)
  y = Add[T=$T](mm, b)
}
)P";
  EXPECT_EQ(expect, DebugString(test::function::WXPlusB()));
}

TEST(TFunc, Body_TypeList) {
  const Tensor kZero = test::AsScalar<int32>(0);
  auto fdef = FDH::Define(
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
       {{"s"}, "Split", {"zero", "i"}, {{"num_split", 4}, {"T", DT_FLOAT}}},
       {{"lst"},
        "_ArrayToList",
        {"s"},
        {{"N", 4},
         {"T", DT_FLOAT},
         {"out_types", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}}}},
       {{"l"}, "Mul", {"lst:0", "lst:1"}, {{"T", DT_FLOAT}}},
       {{"r"}, "Mul", {"lst:2", "lst:3"}, {{"T", DT_FLOAT}}},
       {{"x"},
        "_ListToArray",
        {"l", "r"},
        {{"N", 2},
         {"T", DT_FLOAT},
         {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}}}},
       {{"o"}, "AddN", {"x"}, {{"N", 2}, {"T", DT_FLOAT}}}});

  const char* e = R"P(
Test(i:float) -> (o:float) {
  zero = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  s = Split[T=float, num_split=4](zero, i)
  lst = _ArrayToList[N=4, T=float, out_types={float, float, float, float}](s)
  l = Mul[T=float](lst:0, lst:1)
  r = Mul[T=float](lst:2, lst:3)
  x = _ListToArray[N=2, T=float, Tin={float, float}](l, r)
  o = AddN[N=2, T=float](x)
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  TF_CHECK_OK(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result));
  const char* e2 = R"P(
(n0:float) -> (n7:float) {
  n1 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  n2 = Split[T=float, num_split=4](n1, n0)
  n3 = _ArrayToList[N=4, T=float, out_types={float, float, float, float}](n2, n2:1, n2:2, n2:3)
  n4 = Mul[T=float](n3, n3:1)
  n5 = Mul[T=float](n3:2, n3:3)
  n6 = _ListToArray[N=2, T=float, Tin={float, float}](n4, n5)
  n7 = AddN[N=2, T=float](n6, n6:1)
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
  z = Cond[Tin={float, float}, cond=MyCond2, else_branch=MyElse2, out_types={float}, then_branch=MyThen2](y, y)
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  InstantiationResult result;
  TF_CHECK_OK(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result));
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
      << s << ", expected substring " << substr;
}

TEST(InstantiateErrors, Not_Sufficient_Attrs) {
  auto fdef =
      FDH::Define("nop", {}, {}, {"T:{float, double, int32, int64}"}, {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, {{"U", DT_FLOAT}}, GetOpSig, &result),
           "T is not found");
}

TEST(InstantiateErrors, AttrValue_Value_Placeholder) {
  auto fdef =
      FDH::Define("nop", {}, {}, {"T:{float, double, int32, int64}"}, {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, {{"T", "$bad"}}, GetOpSig, &result),
           "T in attr_values is still a placeholder");
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

TEST(InstantiateErrors, Dup_Arg_Node_Name) {
  auto fdef = FDH::Define("test", {"x:float"}, {}, {},
                          {
                              {{"x"}, "One", {}, {{"T", DT_FLOAT}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Duplicated ret name");
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

TEST(InstantiateErrors, Node_Signature_Mismatch_NoOp) {
  auto fdef = FDH::Define("test", {"x:float"}, {}, {},
                          {
                              {{"y", "z"}, "NoOp", {}, {{"T", DT_FLOAT}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Expect one ret name");
}

TEST(InstantiateErrors, Node_Signature_Mismatch) {
  auto fdef = FDH::Define("test", {"x:float"}, {}, {},
                          {
                              {{"y", "z"}, "One", {}, {{"T", DT_FLOAT}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Malformed function node (#ret)");
}

TEST(InstantiateErrors, Node_Arg_Notfound) {
  auto fdef = FDH::Define("test", {"x:float"}, {}, {},
                          {
                              {{"y"}, "Add", {"x", "z"}, {{"T", DT_FLOAT}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "arg[1] is not found");
}

TEST(InstantiateErrors, Node_Arg_Mismatch) {
  auto fdef = FDH::Define("test", {"x:float"}, {}, {},
                          {
                              {{"y"}, "Add", {"x", "x"}, {{"T", DT_INT32}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Invalid arg(0) for function arg");
}

TEST(InstantiateErrors, Node_Arg_ControlMissing) {
  auto fdef =
      FDH::Define("test", {"x:float"}, {}, {},
                  {
                      {{"y"}, "Add", {"x", "x"}, {{"T", DT_FLOAT}}, {"z"}},
                  });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "dep[0] is not found");
}

TEST(InstantiateErrors, FuncRet_Missing) {
  auto fdef = FDH::Define("test", {}, {"y: float"}, {},
                          {
                              {{"x"}, "One", {}, {{"T", DT_FLOAT}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "ret is not found");
}

TEST(InstantiateErrors, FuncRet_Mismatch) {
  auto fdef = FDH::Define("test", {}, {"y: float"}, {},
                          {
                              {{"y"}, "One", {}, {{"T", DT_DOUBLE}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Invalid ret types y : float vs. double\n\t In y");
}

TEST(InstantiateErrors, TypeList_Missing_Retval_Attr) {
  auto fdef = FDH::Define(
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
      });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "type attr not found: out_types");
}

TEST(InstantiateErrors, TypeList_Num_Retval_Mismatch) {
  auto fdef = FDH::Define(
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
      });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "Invalid ret types");
}

TEST(InstantiateErrors, TypeList_Missing_Arg) {
  auto fdef = FDH::Define(
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
      });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, kNoAttrs, GetOpSig, &result),
           "arg[1] is not found");
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
  FunctionLibraryDefinition lib_def(proto);

  EXPECT_EQ(lib_def.Find("XTimes16"), nullptr);

  auto expect = R"P(
XTimesTwo[T:{float, double, int32, int64}](x:T) -> (y:T) {
  two = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  scale = Cast[DstT=$T, SrcT=int64](two)
  y = Mul[T=$T](x, scale)
}
)P";
  auto found = lib_def.Find("XTimesTwo");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(expect, DebugString(*found));
}

TEST(FunctionLibraryDefinitionTest, LookUp) {
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  FunctionLibraryDefinition lib_def(proto);

  Status s;
  EXPECT_EQ(lib_def.LookUp("XTimes16", &s), nullptr);

  auto found = lib_def.LookUp("XTimesTwo", &s);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->DebugString(),
            test::function::XTimesTwo().signature().DebugString());
}

TEST(FunctionLibraryDefinitionTest, AddFunctionDef) {
  // Add one function to the proto lib before constructing 'lib_def'.
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  FunctionLibraryDefinition lib_def(proto);

  // Add a new function def to the library.
  TF_EXPECT_OK(lib_def.AddFunctionDef(test::function::WXPlusB()));

  // Test lookup of first function.
  Status s;
  auto first = lib_def.LookUp("XTimesTwo", &s);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->DebugString(),
            test::function::XTimesTwo().signature().DebugString());

  // Test lookup of second function.
  auto second = lib_def.LookUp("WXPlusB", &s);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second->DebugString(),
            test::function::WXPlusB().signature().DebugString());
}

TEST(FunctionLibraryDefinitionTest, ToProto) {
  FunctionDefLibrary proto1;
  *proto1.add_function() = test::function::XTimesTwo();
  *proto1.add_function() = test::function::WXPlusB();
  FunctionLibraryDefinition lib_def1(proto1);

  // Call 'ToProto' and make sure both protos have the same function lib size.
  FunctionDefLibrary proto2 = lib_def1.ToProto();
  EXPECT_EQ(proto1.function_size(), proto2.function_size());

  // Initialize 'lib_def2' with proto returned by 'ToProto' call.
  FunctionLibraryDefinition lib_def2(proto2);

  // Test that the first function exists in both libraries.
  Status s;
  auto f1 = lib_def1.LookUp("XTimesTwo", &s);
  TF_EXPECT_OK(s);
  auto f2 = lib_def1.LookUp("XTimesTwo", &s);
  TF_EXPECT_OK(s);
  EXPECT_EQ(f1->DebugString(), f2->DebugString());

  // Test that the second function exists in both libraries.
  auto f3 = lib_def1.LookUp("WXPlusB", &s);
  TF_EXPECT_OK(s);
  auto f4 = lib_def1.LookUp("WXPlusB", &s);
  TF_EXPECT_OK(s);
  EXPECT_EQ(f3->DebugString(), f4->DebugString());
}

}  // end namespace tensorflow
