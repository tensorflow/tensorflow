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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/optimized_function_graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

// A helper class to make AttrSlice from initializer lists
class Attrs {
 public:
  Attrs(const std::initializer_list<  // NOLINT(runtime/explicit)
        std::pair<string, FunctionDefHelper::AttrValueWrapper>>
            attrs) {
    for (const auto& aval : attrs) {
      map_.insert({aval.first, aval.second.proto});
    }
  }

  operator AttrSlice() { return AttrSlice(&map_); }  // NOLINT(runtime/explicit)

 private:
  AttrValueMap map_;
};

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
  TF_ASSERT_OK(
      InstantiateFunction(fdef, Attrs({{"T", DT_FLOAT}}), GetOpSig, &result));
  const char* e2 = R"P(
(x:float) -> (y:float) {
  a = Square[T=float](x)
  o = One[T=float]()
  y = Add[T=float](a, o)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.nodes), e2);
}

TEST(TFunc, CopyDebugInfo) {
  auto fdef = FDH::Create(
      // Name
      "Square",
      // Inputs
      {"x: T"},
      // Outputs
      {"y: T"},
      // Attrs
      {"T: {float, double, int32, int64}"},
      // Nodes
      {// a = Sqaure<T>(x)
       {{"a"},
        {"Square"},
        {"x"},
        {{"T", "$T"}},
        {},
        "",
        "",
        {"node_name"},
        {"func_name"}}},
      // Returns
      {{"y", "a:y:0"}});
  const char* e = R"P(
Square[T:{float, double, int32, int64}](x:T) -> (y:T) {
  a = Square[T=$T](x)
  return y = a:y:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);
  InstantiationResult result;
  TF_ASSERT_OK(
      InstantiateFunction(fdef, Attrs({{"T", DT_FLOAT}}), GetOpSig, &result));
  const char* e2 = R"P(
(x:float) -> (a:float) {
  a = Square[T=float](x)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.nodes), e2);
  EXPECT_EQ(result.nodes.size(), 3);
  NodeDef node;
  for (auto n : result.nodes) {
    if (n.name() == "a") {
      node = n;
      break;
    }
  }
  EXPECT_TRUE(node.has_experimental_debug_info());
  EXPECT_EQ(node.experimental_debug_info().original_node_names().size(), 1);
  EXPECT_EQ(node.experimental_debug_info().original_func_names().size(), 1);
  EXPECT_EQ(node.experimental_debug_info().original_node_names()[0],
            "node_name");
  EXPECT_EQ(node.experimental_debug_info().original_func_names()[0],
            "func_name");
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
  TF_ASSERT_OK(
      InstantiateFunction(fdef, Attrs({{"T", DT_FLOAT}}), GetOpSig, &result));
  const char* e2 = R"P(
(x:int32) -> (y:int32) {
  a = Identity[T=int32](x)
  o = NoOp() @ a
  y = Identity[T=int32](a) @ o
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_INT32}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_INT32}));
  EXPECT_EQ(DebugString(result.nodes), e2);
}

TEST(TFunc, ControlRet) {
  auto fdef = FDH::Create(
      // Name
      "ControlRet",
      // Inputs
      {"x: int32"},
      // Outputs
      {"y: int32"},
      // Attrs
      {},
      // Nodes
      {
          {{"a"}, "Identity", {"x"}, {{"T", DT_INT32}}},
      },
      // Returns
      {{"y", "a:output:0"}},
      // Control returns
      {{"must_execute", "a"}});

  const char* e = R"P(
ControlRet(x:int32) -> (y:int32) {
  a = Identity[T=int32](x)
  @return must_execute = a
  return y = a:output:0
}
)P";
  EXPECT_EQ(DebugString(fdef), e);

  // Instantiate one with T=float
  InstantiationResult result;
  TF_ASSERT_OK(
      InstantiateFunction(fdef, Attrs({{"T", DT_FLOAT}}), GetOpSig, &result));
  const char* e2 = R"P(
(x:int32) -> (a:int32) {
  a = Identity[T=int32](x)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_INT32}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_INT32}));
  EXPECT_EQ(DebugString(result.nodes), e2);
}

REGISTER_OP("HasDefaultType")
    .Output("out: T")
    .Attr("T: {float, double, int32, int64} = DT_FLOAT");

// This verifies that a function using an op before a type attr (with
// a default) is added, still works.  This is important for backwards
// compatibility.
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
  TF_ASSERT_OK(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result));
  // Should get T=float from Op's default.
  const char* e2 = R"P(
() -> (a:float) {
  a = HasDefaultType[T=float]()
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector());
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.nodes), e2);
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
  TF_ASSERT_OK(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result));
  const char* e2 = R"P(
(x:float, y:float) -> (a:float) {
  a = AddN[N=2, T=float](x, y)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT, DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.nodes), e2);
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
  TF_ASSERT_OK(InstantiateFunction(fdef, Attrs({{"N", 3}, {"T", DT_FLOAT}}),
                                   GetOpSig, &result));
  const char* e2 = R"P(
(x_0:float, x_1:float, x_2:float) -> (y:float) {
  a = Map[N=3, T=float, U=float, func=Square[T=float]](x_0, x_1, x_2)
  y = AddN[N=3, T=float](a, a:1, a:2)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT, DT_FLOAT, DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.nodes), e2);
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
  TF_ASSERT_OK(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result));
  const char* e2 = R"P(
(x:float) -> () {
  a = One[T=float]() @ x
  u = NoOp() @ a
  b = One[T=float]() @ u
  v = NoOp() @ b
  c = One[T=float]() @ a, v
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({}));
  EXPECT_EQ(DebugString(result.nodes), e2);
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
  mm = MatMul[T=$T, transpose_a=false, transpose_b=false](w, x)
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
  TF_ASSERT_OK(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result));
  const char* e2 = R"P(
(i:float) -> (o:float) {
  zero = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  s = Split[T=float, num_split=4](zero, i)
  l = Mul[T=float](s, s:1)
  r = Mul[T=float](s:2, s:3)
  x = _ListToArray[N=2, T=float, Tin={float, float}](l, r)
  o = AddN[N=2, T=float](x, x:1)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.nodes), e2);
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
  TF_ASSERT_OK(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result));
  const char* e2 = R"P(
(x:float) -> (z:float) {
  y = Cond[Tin={float}, cond=MyCond, else_branch=MyElse, out_types={float}, then_branch=MyThen](x)
  z = Cond[Tin={float, float}, cond=MyCond2, else_branch=MyElse2, out_types={float}, then_branch=MyThen2](y, y)
}
)P";
  EXPECT_EQ(result.arg_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(result.ret_types, DataTypeVector({DT_FLOAT}));
  EXPECT_EQ(DebugString(result.nodes), e2);
}

TEST(TFunc, IntsOnDeviceArgNotSet) {
  auto fdef = test::function::XTimesTwoInt32();
  InstantiationResult result;
  TF_ASSERT_OK(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result));
  EXPECT_EQ(5, result.nodes.size());
  EXPECT_EQ("_Retval", result.nodes[4].op());
}

TEST(TFunc, IntsOnDeviceArgSet) {
  auto fdef = test::function::XTimesTwoInt32();
  (*fdef.mutable_attr())[FunctionLibraryDefinition::kIntsOnDeviceAttr].set_b(
      true);
  InstantiationResult result;
  TF_ASSERT_OK(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result));
  EXPECT_EQ(5, result.nodes.size());
  EXPECT_EQ("_DeviceRetval", result.nodes[4].op());
}

static void HasError(const Status& s, const string& substr) {
  EXPECT_TRUE(absl::StrContains(s.ToString(), substr))
      << ">>" << s << "<<, expected substring >>" << substr << "<<";
}

TEST(InstantiateErrors, Not_Sufficient_Attrs) {
  auto fdef =
      FDH::Define("nop", {}, {}, {"T:{float, double, int32, int64}"}, {});
  InstantiationResult result;
  HasError(
      InstantiateFunction(fdef, Attrs({{"U", DT_FLOAT}}), GetOpSig, &result),
      "Attr T is not found from ");
}

#if 0  // TODO(josh11b): Enable this test once having an extra attr is an error.
TEST(InstantiateErrors, Too_Many_Attrs) {
  auto fdef =
      FDH::Define("nop", {}, {}, {"T:{float, double, int32, int64}"}, {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, Attrs({{"T", DT_INT32}, {"U", DT_FLOAT}}),
                               GetOpSig, &result),
           "Attr U is not found in ");
}
#endif

TEST(InstantiateErrors, AttrValue_Value_Placeholder) {
  auto fdef =
      FDH::Define("nop", {}, {}, {"T:{float, double, int32, int64}"}, {});
  InstantiationResult result;
  HasError(
      InstantiateFunction(fdef, Attrs({{"T", "$bad"}}), GetOpSig, &result),
      "AttrValue had value with unexpected type 'placeholder'\n\tfor attr 'T'");
}

TEST(InstantiateErrors, Unbounded_Attr) {
  auto fdef = FDH::Define("test", {}, {}, {"T:{float, double, int32, int64}"},
                          {
                              {{"a"}, "One", {}, {{"T", "$unknown"}}, {"x"}},
                          });
  InstantiationResult result;
  HasError(
      InstantiateFunction(fdef, Attrs({{"T", DT_FLOAT}}), GetOpSig, &result),
      "Failed to bind all placeholders");
}

TEST(InstantiateErrors, DupArgs) {
  auto fdef = FDH::Define("test", {"x:float", "x:float"}, {}, {}, {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
           "Duplicated arg name");
}

TEST(InstantiateErrors, Dup_Node_Names) {
  auto fdef = FDH::Define("test", {"x:float"}, {}, {},
                          {
                              {{"y"}, "One", {}, {{"T", DT_FLOAT}}},
                              {{"y"}, "One", {}, {{"T", DT_FLOAT}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
           "Duplicated ret name");
}

TEST(InstantiateErrors, Node_Arg_Notfound) {
  auto fdef = FDH::Create("test", {"x:float"}, {}, {},
                          {
                              {{"y"}, "Add", {"x", "z"}, {{"T", DT_FLOAT}}},
                          },
                          {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
           "input z is not found");
}

TEST(InstantiateErrors, Node_Arg_TypeMismatch) {
  auto fdef = FDH::Define("test", {"x:float"}, {}, {},
                          {
                              {{"y"}, "Add", {"x", "x"}, {{"T", DT_INT32}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
           "input x[0] expected type int32 != float, the type of x[0]");
}

TEST(InstantiateErrors, Node_Arg_ControlMissing) {
  auto fdef =
      FDH::Define("test", {"x:float"}, {}, {},
                  {
                      {{"y"}, "Add", {"x", "x"}, {{"T", DT_FLOAT}}, {"z"}},
                  });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
           "input[2] == '^z', is not found.");
}

TEST(InstantiateErrors, FuncRet_Missing) {
  auto fdef = FDH::Create("test", {}, {"y: float"}, {},
                          {
                              {{"x"}, "One", {}, {{"T", DT_FLOAT}}},
                          },
                          {});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
           "Return y missing");
}

TEST(InstantiateErrors, FuncRet_NotFound) {
  auto fdef = FDH::Create("test", {}, {"y: float"}, {},
                          {
                              {{"x"}, "One", {}, {{"T", DT_FLOAT}}},
                          },
                          {{"y", "z"}});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
           "Return y -> z is not found");
}

TEST(InstantiateErrors, FuncRet_NameMismatch) {
  auto fdef = FDH::Create("test", {}, {"y: float"}, {},
                          {
                              {{"x"}, "One", {}, {{"T", DT_FLOAT}}},
                          },
                          {{"z", "x:y:0"}});
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
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
//   HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
//            "ret is not found");
// }

TEST(InstantiateErrors, FuncRet_TypeMismatch) {
  auto fdef = FDH::Define("test", {}, {"y: float"}, {},
                          {
                              {{"y"}, "One", {}, {{"T", DT_DOUBLE}}},
                          });
  InstantiationResult result;
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
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
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
           "type list attr not found: out_types");
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
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
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
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
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
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
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
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
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
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
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
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
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
  HasError(InstantiateFunction(fdef, AttrSlice(), GetOpSig, &result),
           "input inputs[1] expected type float != int32, the type of y[0]");
}

TEST(FunctionCallFrame, Void_Void) {
  FunctionCallFrame frame({}, {});
  TF_EXPECT_OK(frame.SetArgs({}));
  auto a = test::AsTensor<float>({100});
  EXPECT_EQ(frame.SetArgs({a}).code(), error::INVALID_ARGUMENT);
  const Tensor* v = nullptr;
  EXPECT_EQ(frame.GetArg(0, &v).code(), error::INVALID_ARGUMENT);
  if (v != nullptr) {
    // v is null in certain environments.
    EXPECT_EQ(frame.SetRetval(0, *v).code(), error::INVALID_ARGUMENT);
  }
  std::vector<Tensor> rets;
  TF_EXPECT_OK(frame.GetRetvals(&rets));
  EXPECT_EQ(rets.size(), 0);
}

TEST(FunctionCallFrame, Float_Float_Float) {
  FunctionCallFrame frame({DT_FLOAT, DT_FLOAT}, {DT_FLOAT});
  EXPECT_EQ(frame.SetArgs({}).code(), error::INVALID_ARGUMENT);
  auto a = test::AsTensor<float>({100});
  auto b = test::AsTensor<float>({200});
  auto c = test::AsTensor<int64_t>({300});
  EXPECT_EQ(frame.SetArgs({a, c}).code(), error::INVALID_ARGUMENT);
  TF_EXPECT_OK(frame.SetArgs({a, b}));

  const Tensor* v;

  EXPECT_EQ(frame.GetArg(-1, &v).code(), error::INVALID_ARGUMENT);
  EXPECT_EQ(frame.GetArg(2, &v).code(), error::INVALID_ARGUMENT);
  TF_EXPECT_OK(frame.GetArg(0, &v));
  test::ExpectTensorEqual<float>(a, *v);
  TF_EXPECT_OK(frame.GetArg(1, &v));
  test::ExpectTensorEqual<float>(b, *v);

  Tensor w = test::AsTensor<float>({-100});
  EXPECT_EQ(frame.SetRetval(-1, w).code(), error::INVALID_ARGUMENT);
  EXPECT_EQ(frame.SetRetval(1, w).code(), error::INVALID_ARGUMENT);
  EXPECT_EQ(frame.SetRetval(0, test::AsTensor<int64_t>({-100})).code(),
            error::INVALID_ARGUMENT);

  std::vector<Tensor> rets;
  HasError(frame.GetRetvals(&rets), "does not have value");
  TF_EXPECT_OK(frame.SetRetval(0, *v));
  HasError(frame.SetRetval(0, *v), "has already been set");

  TF_EXPECT_OK(frame.GetRetvals(&rets));
  EXPECT_EQ(rets.size(), 1);
  test::ExpectTensorEqual<float>(rets[0], *v);
}

TEST(Canonicalize, Basic) {
  EXPECT_EQ(Canonicalize("MatMul", Attrs({{"T", DT_FLOAT},
                                          {"transpose_a", false},
                                          {"transpose_b", false}})),
            "MatMul[T=float,transpose_a=false,transpose_b=false]");
  EXPECT_EQ(Canonicalize("MatMul", Attrs({{"T", DT_FLOAT},
                                          {"transpose_b", false},
                                          {"transpose_a", false}})),
            "MatMul[T=float,transpose_a=false,transpose_b=false]");
  EXPECT_EQ(Canonicalize("MatMul", Attrs({{"T", DT_DOUBLE},
                                          {"transpose_b", true},
                                          {"transpose_a", false}})),
            "MatMul[T=double,transpose_a=false,transpose_b=true]");
  EXPECT_EQ(Canonicalize("CheckNumericsV2",
                         Attrs({{"T", DT_HALF},
                                {"message", "Message should get hashed"}}),
                         FunctionLibraryRuntime::InstantiateOptions()),
            "CheckNumericsV2[T=half,message=811750450553548470]");
}

TEST(FunctionLibraryDefinitionTest, Contains) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), {});
  TF_CHECK_OK(lib_def.AddFunctionDef(test::function::XTimesTwo()));

  EXPECT_FALSE(lib_def.Contains("XTimes16"));
  EXPECT_TRUE(lib_def.Contains("XTimesTwo"));
}

TEST(FunctionLibraryDefinitionTest, Find) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), {});
  TF_CHECK_OK(lib_def.AddFunctionDef(test::function::XTimesTwo()));

  EXPECT_EQ(lib_def.Find("XTimes16"), nullptr);

  auto found = lib_def.Find("XTimesTwo");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(test::function::XTimesTwo().DebugString(), found->DebugString());
}

TEST(FunctionLibraryDefinitionTest, LookUp) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), {});
  TF_CHECK_OK(lib_def.AddFunctionDef(test::function::XTimesTwo()));

  const OpDef* op_def;
  EXPECT_FALSE(lib_def.LookUpOpDef("XTimes16", &op_def).ok());

  TF_EXPECT_OK(lib_def.LookUpOpDef("XTimesTwo", &op_def));
  ASSERT_NE(op_def, nullptr);
  EXPECT_EQ(op_def->DebugString(),
            test::function::XTimesTwo().signature().DebugString());

  const OpRegistrationData* op_reg_data;
  TF_EXPECT_OK(lib_def.LookUp("XTimesTwo", &op_reg_data));
  ASSERT_NE(op_reg_data, nullptr);
  // Shape inference function is initialized to UnknownShape.
  ASSERT_NE(op_reg_data->shape_inference_fn, nullptr);
}

TEST(FunctionLibraryDefinitionTest, AddFunctionDef) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), {});
  TF_CHECK_OK(lib_def.AddFunctionDef(test::function::XTimesTwo()));

  // Test lookup of existing function.
  const OpDef* op_def;
  TF_EXPECT_OK(lib_def.LookUpOpDef("XTimesTwo", &op_def));
  ASSERT_NE(op_def, nullptr);
  EXPECT_EQ(op_def->DebugString(),
            test::function::XTimesTwo().signature().DebugString());

  // Test that adding a function with same name as existing op fails.
  FunctionDef fdef = test::function::XTimesTwo();
  fdef.mutable_signature()->set_name("Add");
  Status s = lib_def.AddFunctionDef(fdef);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "Cannot add function 'Add' because an op with the same name "
            "already exists.");

  // Test that adding the same functions again does not produce an error.
  TF_EXPECT_OK(lib_def.AddFunctionDef(test::function::XTimesTwo()));
}

TEST(FunctionLibraryDefinitionTest, AddGradientDef) {
  // AddGradientDef() doesn't check that functions referenced exist (yet?)
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), FunctionDefLibrary());

  // Test adding a gradient (XTimesFour isn't a valid grad function for
  // XTimesTwo but that's ok for now)
  GradientDef grad;
  grad.set_function_name(test::function::XTimesTwo().signature().name());
  grad.set_gradient_func(test::function::XTimesFour().signature().name());
  TF_EXPECT_OK(lib_def.AddGradientDef(grad));

  // Already-added gradients don't produce error
  TF_EXPECT_OK(lib_def.AddGradientDef(grad));

  // Test that adding a duplicate gradient fails
  grad.set_gradient_func(test::function::XTimes16().signature().name());
  Status s = lib_def.AddGradientDef(grad);
  EXPECT_EQ(s.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_EQ(s.error_message(),
            "Cannot assign gradient function 'XTimes16' to 'XTimesTwo' because "
            "it already has gradient function 'XTimesFour'");
}

TEST(FunctionLibraryDefinitionTest, RemoveFunction) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), {});
  TF_CHECK_OK(lib_def.AddFunctionDef(test::function::XTimesTwo()));

  Status s = lib_def.RemoveFunction("XTimes16");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(),
            "Tried to remove non-existent function 'XTimes16'.");

  EXPECT_TRUE(lib_def.Contains("XTimesTwo"));
  TF_EXPECT_OK(lib_def.RemoveFunction("XTimesTwo"));
  EXPECT_FALSE(lib_def.Contains("XTimesTwo"));
}

TEST(FunctionLibraryDefinitionTest, Clear) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), {});
  TF_CHECK_OK(lib_def.AddFunctionDef(test::function::XTimesTwo()));
  TF_CHECK_OK(lib_def.AddFunctionDef(test::function::XAddX()));

  lib_def.Clear();
  EXPECT_FALSE(lib_def.Contains("XTimesTwo"));
  EXPECT_FALSE(lib_def.Contains("XAddX"));
}

TEST(FunctionLibraryDefinitionTest, AddLibrary) {
  // Create lib def with single function
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);

  // Add gradient
  GradientDef grad;
  grad.set_function_name(test::function::XTimesTwo().signature().name());
  grad.set_gradient_func(test::function::XTimesFour().signature().name());
  TF_EXPECT_OK(lib_def.AddGradientDef(grad));

  // Error if you try to add conflicting function
  proto.Clear();
  FunctionDef fdef = test::function::XTimesFour();
  fdef.mutable_signature()->set_name(
      test::function::XTimesTwo().signature().name());
  *proto.add_function() = fdef;
  FunctionLibraryDefinition lib_def2(OpRegistry::Global(), proto);
  Status s = lib_def.AddLibrary(lib_def2);
  EXPECT_EQ(s.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_EQ(s.error_message(),
            "Cannot add function 'XTimesTwo' because a different function with "
            "the same name already exists.");

  // Error if you try to add conflicting gradient
  proto.Clear();
  grad.set_gradient_func(test::function::XTimes16().signature().name());
  *proto.add_gradient() = grad;
  FunctionLibraryDefinition lib_def3(OpRegistry::Global(), proto);
  s = lib_def.AddLibrary(lib_def3);
  EXPECT_EQ(s.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_EQ(s.error_message(),
            "Cannot assign gradient function 'XTimes16' to 'XTimesTwo' because "
            "it already has gradient function 'XTimesFour'");

  // No conflicting functions or gradients OK
  proto.Clear();
  *proto.add_function() = test::function::XTimesFour();
  grad.set_function_name(test::function::XTimes16().signature().name());
  *proto.add_gradient() = grad;
  FunctionLibraryDefinition lib_def4(OpRegistry::Global(), proto);
  TF_EXPECT_OK(lib_def.AddLibrary(lib_def4));

  // OK to add the same functions and gradients twice
  TF_EXPECT_OK(lib_def.AddLibrary(lib_def));
}

GradientDef MakeGradDef(const string& f, const string& g) {
  GradientDef grad;
  grad.set_function_name(f);
  grad.set_gradient_func(g);
  return grad;
}

TEST(FunctionLibraryDefinitionTest, AddLibrary_Atomic) {
  // Create lib def containing two functions with equal names
  FunctionDefLibrary proto;
  const string x2_name = test::function::XTimesTwo().signature().name();
  const string x4_name = test::function::XTimesFour().signature().name();
  *proto.add_function() = test::function::XTimesTwo();
  FunctionDef fdef = test::function::XTimesFour();
  fdef.mutable_signature()->set_name(x2_name);
  *proto.add_function() = fdef;
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), FunctionDefLibrary());

  // Try adding the two functions to lib_def
  Status s = lib_def.AddLibrary(proto);
  EXPECT_EQ(error::Code::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(
      "Cannot add function 'XTimesTwo' because a different function with "
      "the same name already exists.",
      s.error_message());

  // Verify that none of the functions are added
  EXPECT_TRUE(lib_def.Find(x2_name) == nullptr);

  // Fix the name in proto but add two gradient names for it
  proto.mutable_function(1)->mutable_signature()->set_name(x4_name);
  *proto.add_gradient() = MakeGradDef(x2_name, x4_name);
  *proto.add_gradient() = MakeGradDef(x2_name, "SecondGradName");

  // Try adding the library and check that nothing was added
  s = lib_def.AddLibrary(proto);
  EXPECT_EQ(error::Code::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(s.error_message(),
            "Cannot assign gradient function 'SecondGradName' to 'XTimesTwo' "
            "because it already has gradient function 'XTimesFour'");
  EXPECT_TRUE(lib_def.Find(x2_name) == nullptr);
  EXPECT_EQ(0, lib_def.ToProto().function_size());
  EXPECT_EQ(0, lib_def.ToProto().gradient_size());
}

TEST(FunctionLibraryDefinitionTest, AddLibraryDefinition_Atomic_FuncConflict) {
  const string x2_name = test::function::XTimesTwo().signature().name();
  const string x4_name = test::function::XTimesFour().signature().name();
  const string wx_name = test::function::WXPlusB().signature().name();

  // Create FunctionLibraryDefinition with
  // (func = XTimesTwo, grad = XTimesFour)
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  *proto.add_gradient() = MakeGradDef(x2_name, x4_name);
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);
  EXPECT_EQ(1, lib_def.ToProto().function_size());
  EXPECT_EQ(1, lib_def.ToProto().gradient_size());

  // Create FunctionLibraryDefinition with (func = WXPlusB, grad = XTimesTwo)
  // and function (name = XTimesTwo, body = XTimeFour)
  FunctionDefLibrary proto2;
  *proto2.add_function() = test::function::WXPlusB();
  *proto2.add_gradient() = MakeGradDef(wx_name, x2_name);
  *proto2.add_function() = test::function::XTimesFour();
  proto2.mutable_function(1)->mutable_signature()->set_name(x2_name);
  FunctionLibraryDefinition lib_def2(OpRegistry::Global(), proto2);

  // Verify that adding lib_def2 will fail because of function conflict
  // and WXPlusB is not added.
  Status s = lib_def.AddLibrary(lib_def2);
  EXPECT_EQ(error::Code::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(
      "Cannot add function 'XTimesTwo' because a different function "
      "with the same name already exists.",
      s.error_message());
  EXPECT_TRUE(lib_def.Find(wx_name) == nullptr);
  EXPECT_EQ(1, lib_def.ToProto().function_size());
  EXPECT_EQ(1, lib_def.ToProto().gradient_size());
}

TEST(FunctionLibraryDefinitionTest, AddLibraryDefinition_Atomic_GradConflict) {
  const string x2_name = test::function::XTimesTwo().signature().name();
  const string x4_name = test::function::XTimesFour().signature().name();
  const string wx_name = test::function::WXPlusB().signature().name();

  // Create FunctionLibraryDefinition with
  // (func = XTimesTwo, grad = XTimesFour)
  FunctionDefLibrary proto;
  *proto.add_function() = test::function::XTimesTwo();
  *proto.add_gradient() = MakeGradDef(x2_name, x4_name);
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);
  EXPECT_EQ(1, lib_def.ToProto().function_size());
  EXPECT_EQ(1, lib_def.ToProto().gradient_size());

  // Create FunctionLibraryDefinition with (func = WXPlusB, grad = XTimesTwo)
  // and (func = XTimesTwo, grad = WXPlusB)
  FunctionDefLibrary proto2;
  *proto2.add_function() = test::function::WXPlusB();
  *proto2.add_gradient() = MakeGradDef(wx_name, x2_name);
  *proto2.add_function() = test::function::XTimesTwo();
  *proto2.add_gradient() = MakeGradDef(x2_name, wx_name);
  FunctionLibraryDefinition lib_def2(OpRegistry::Global(), proto2);

  // Verify that adding lib_def2 will fail because of gradient conflict
  // and WXPlusB is not added.
  Status s = lib_def.AddLibrary(lib_def2);
  EXPECT_EQ(error::Code::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(
      "Cannot assign gradient function 'WXPlusB' to 'XTimesTwo'"
      " because it already has gradient function 'XTimesFour'",
      s.error_message());
  EXPECT_TRUE(lib_def.Find(wx_name) == nullptr);
  EXPECT_EQ(1, lib_def.ToProto().function_size());
  EXPECT_EQ(1, lib_def.ToProto().gradient_size());
}

TEST(FunctionLibraryDefinitionTest, ToProto) {
  FunctionLibraryDefinition lib_def1(OpRegistry::Global(), {});
  TF_CHECK_OK(lib_def1.AddFunctionDef(test::function::XTimesTwo()));
  TF_CHECK_OK(lib_def1.AddFunctionDef(test::function::WXPlusB()));

  FunctionDefLibrary proto = lib_def1.ToProto();
  EXPECT_EQ(proto.function_size(), 2);

  // Initialize 'lib_def2' with proto returned by 'ToProto' call.
  FunctionLibraryDefinition lib_def2(OpRegistry::Global(), proto);

  // Test that the functions exists in both libraries.
  for (auto name : {"XTimesTwo", "WXPlusB"}) {
    const OpDef *f1, *f2;
    TF_EXPECT_OK(lib_def1.LookUpOpDef(name, &f1));
    TF_EXPECT_OK(lib_def2.LookUpOpDef(name, &f2));
    EXPECT_EQ(f1->DebugString(), f2->DebugString());
  }
}

TEST(FunctionLibraryDefinitionTest, ListFunctionNames) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), {});
  TF_CHECK_OK(lib_def.AddFunctionDef(test::function::XTimesTwo()));
  TF_CHECK_OK(lib_def.AddFunctionDef(test::function::WXPlusB()));

  const std::vector<string> function_names = lib_def.ListFunctionNames();
  const std::vector<string> expected = {"XTimesTwo", "WXPlusB"};
  EXPECT_EQ(function_names, expected);
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

TEST(FunctionLibraryDefinitionTest, ReachableDefinitions) {
  using ::tensorflow::test::function::GDef;
  using ::tensorflow::test::function::NDef;
  using FDH = ::tensorflow::FunctionDefHelper;

  const auto make_simple_fdef = [](const string& name,
                                   const string& interface_name) {
    auto func_def = FDH::Create(
        name, {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
        {{{"output"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
        /* Mapping between function returns and function node outputs. */
        {{"z", "output:z:0"}});

    if (!interface_name.empty()) {
      auto* attr = func_def.mutable_attr();
      (*attr)["api_implements"].set_s(interface_name);
    }
    return func_def;
  };

  FunctionDef func_1 = make_simple_fdef("Func1", "");
  FunctionDef func_2 = make_simple_fdef("Func2", "");
  FunctionDef func_3 = make_simple_fdef("Func3", "");
  FunctionDef func_4 = make_simple_fdef("Func4", "api_1");
  FunctionDef func_5 = make_simple_fdef("Func5", "api_1");
  FunctionDef func_6 = make_simple_fdef("Func6", "api_2");

  FunctionDef func_2_grad = make_simple_fdef("Func2_grad", "");

  constexpr char kDevice[] = "/device:CPU:0";

  GraphDef graph = GDef(
      {
          NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
          NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
          NDef("x", "Func1", {"a", "b"}, {{"T", DT_FLOAT}}, kDevice),
          NDef("y", "PartitionedCall", {"a", "b"},
               {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
                {"Tout", DataTypeSlice{DT_FLOAT}},
                {"f", FDH::FunctionRef("Func2", {{"T", DT_FLOAT}})}},
               kDevice),
          NDef("z", "Func4", {"a", "b"}, {{"T", DT_FLOAT}}, kDevice),
      },
      // FunctionLib
      {func_1, func_2, func_3, func_2_grad, func_4, func_5, func_6});

  // Register custom function gradient after the graph was constructed.
  GradientDef* func3_grad_def = graph.mutable_library()->add_gradient();
  func3_grad_def->set_function_name("Func2");
  func3_grad_def->set_gradient_func("Func2_grad");

  FunctionLibraryDefinition flib(OpRegistry::Global(), graph.library());

  // - 'Func1' is called directly from the graph.
  // - 'Func2' is called indirectly via a PartitionedCall attribute, and it also
  //   has a custom gradient ('Func2_grad') that must remain in the library.
  // - 'Func3' is unreachable and has to be removed from the library
  // - 'Func4' is called directly from the graph
  // - 'Func5' is not called directly, but it implements same interface as Func4
  //   which is directly called.
  // - 'Func6' is not called directly, and the interface it implements has not
  //   not been called by another nodes in the graph.
  FunctionLibraryDefinition reachable_flib = flib.ReachableDefinitions(graph);
  EXPECT_EQ(reachable_flib.num_functions(), 5);
  EXPECT_TRUE(reachable_flib.Contains("Func1"));
  EXPECT_TRUE(reachable_flib.Contains("Func2"));
  EXPECT_TRUE(reachable_flib.Contains("Func2_grad"));
  EXPECT_FALSE(reachable_flib.Contains("Func3"));
  EXPECT_TRUE(reachable_flib.Contains("Func4"));
  EXPECT_TRUE(reachable_flib.Contains("Func5"));
  EXPECT_FALSE(reachable_flib.Contains("Func6"));
}

TEST(FunctionLibraryDefinitionTest, AddAndFindOptimizedFunctionGraph) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), {});
  EXPECT_EQ(lib_def.FindOptimizedFunctionGraph("test"), nullptr);
  OptimizedFunctionGraph proto;
  lib_def.AddOptimizedFunctionGraph("test", proto);
  EXPECT_NE(lib_def.FindOptimizedFunctionGraph("test"), nullptr);
}

// TODO(skyewm): this could be more thorough
TEST(FunctionDefsEqualTest, TestFunctionDefsEqual) {
  // Equal functions
  const FunctionDef fdef1 = test::function::XTimesTwo();
  FunctionDef fdef2 = test::function::XTimesTwo();
  uint64 hash1 = FunctionDefHash(fdef1);
  EXPECT_TRUE(FunctionDefsEqual(fdef1, fdef2));
  EXPECT_EQ(hash1, FunctionDefHash(fdef2));

  // Different functions
  fdef2 = test::function::XTimesFour();
  EXPECT_FALSE(FunctionDefsEqual(fdef1, fdef2));
  EXPECT_NE(hash1, FunctionDefHash(fdef2));

  // Different signatures
  fdef2 = test::function::XTimesTwo();
  fdef2.mutable_signature()->mutable_input_arg(0)->set_name("foo");
  EXPECT_FALSE(FunctionDefsEqual(fdef1, fdef2));
  EXPECT_NE(hash1, FunctionDefHash(fdef2));

  // Descriptions must be equal
  fdef2 = test::function::XTimesTwo();
  fdef2.mutable_signature()->mutable_input_arg(0)->set_description("foo");
  EXPECT_FALSE(FunctionDefsEqual(fdef1, fdef2));
  EXPECT_NE(hash1, FunctionDefHash(fdef2));

  // Different NodeDefs
  fdef2 = test::function::XTimesTwo();
  NodeDef* ndef = fdef2.add_node_def();
  *ndef = fdef2.node_def(0);
  ndef->set_name("new_name");
  EXPECT_FALSE(FunctionDefsEqual(fdef1, fdef2));
  EXPECT_NE(hash1, FunctionDefHash(fdef2));

  // Different return values
  fdef2 = test::function::XTimesTwo();
  (*fdef2.mutable_ret())["y"] = "y:z:1";  // originally is "y:z:0"
  EXPECT_FALSE(FunctionDefsEqual(fdef1, fdef2));
  EXPECT_NE(hash1, FunctionDefHash(fdef2));

  // Different attributes
  fdef2 = test::function::XTimesTwo();
  SetAttrValue(&fdef2, "ExtraAttr", true);
  EXPECT_FALSE(FunctionDefsEqual(fdef1, fdef2));
  EXPECT_NE(hash1, FunctionDefHash(fdef2));

  // Multiple equivalent attributes; the two functions should be equal.
  fdef2 = test::function::XTimesTwo();
  FunctionDef fdef3 = test::function::XTimesTwo();
  SetAttrValue(&fdef2, "Foo", true);
  SetAttrValue(&fdef3, "Foo", true);
  SetAttrValue(&fdef2, "Bar", 123);
  SetAttrValue(&fdef3, "Bar", 123);
  SetAttrValue(&fdef2, "Baz", "abc");
  SetAttrValue(&fdef3, "Baz", "abc");
  EXPECT_TRUE(FunctionDefsEqual(fdef2, fdef3));
  EXPECT_EQ(FunctionDefHash(fdef2), FunctionDefHash(fdef3));
}

TEST(InstantiateFunctionTest, ArgAttrs) {
  auto fdef = FDH::Create(
      // Name
      "Func",
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
  AttrValue shape_attr;
  TensorShapeProto* shape_proto = shape_attr.mutable_list()->add_shape();
  shape_proto->add_dim()->set_size(2);
  shape_proto->add_dim()->set_size(4);
  shape_proto->add_dim()->set_size(6);
  shape_proto->add_dim()->set_size(8);
  FunctionDef::ArgAttrs arg_attrs;
  (*arg_attrs.mutable_attr())["_output_shapes"] = std::move(shape_attr);
  (*fdef.mutable_arg_attr())[0] = std::move(arg_attrs);

  // Instantiate one with T=float
  InstantiationResult result;
  TF_ASSERT_OK(
      InstantiateFunction(fdef, Attrs({{"T", DT_FLOAT}}), GetOpSig, &result));
  bool found = false;
  for (const auto& node : result.nodes) {
    if (node.name() != "x") {
      continue;
    }
    found = true;
    auto it = node.attr().find("_output_shapes");
    ASSERT_TRUE(it != node.attr().end());
    const auto& attr = it->second;
    ASSERT_EQ(attr.list().shape_size(), 1);
    const auto& shape_attr = attr.list().shape(0);
    ASSERT_FALSE(shape_attr.unknown_rank());
    ASSERT_EQ(shape_attr.dim_size(), 4);
    EXPECT_EQ(shape_attr.dim(0).size(), 2);
    EXPECT_EQ(shape_attr.dim(1).size(), 4);
    EXPECT_EQ(shape_attr.dim(2).size(), 6);
    EXPECT_EQ(shape_attr.dim(3).size(), 8);
  }
  EXPECT_TRUE(found);
}

TEST(InstantiateFunctionTest, ResourceInputDevice) {
  FunctionDef fdef = FDH::Create(
      // Name
      "Func",
      // Args
      {{"x0: resource"}, {"x1: resource"}},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"read0"},
           "ReadVariableOp",
           {"x0"},
           {{"dtype", DT_FLOAT}},
           {},
           "/device:CPU:1"},
          {{"read1"},
           "ReadVariableOp",
           {"x1"},
           {{"dtype", DT_FLOAT}},
           {},
           "/device:CPU:0"},
          {{"add"},
           "Add",
           {"read0:value:0", "read1:value:0"},
           {{"T", DT_FLOAT}},
           {},
           "/device:CPU:0"},
      },
      {{"y", "add:z:0"}});
  FunctionDef::ArgAttrs arg_attrs;
  *(*arg_attrs.mutable_attr())["_composite_device"].mutable_s() =
      "/device:COMPOSITE:0";
  (*fdef.mutable_arg_attr())[0] = arg_attrs;
  absl::flat_hash_map<string, std::vector<string>> composite_devices;

  Tensor arg0(DT_RESOURCE, TensorShape({2}));
  ResourceHandle resource_handle0;
  resource_handle0.set_device("/device:CPU:0");
  ResourceHandle resource_handle1;
  resource_handle1.set_device("/device:CPU:1");
  arg0.flat<ResourceHandle>()(0) = resource_handle0;
  arg0.flat<ResourceHandle>()(1) = resource_handle1;

  Tensor arg1(DT_RESOURCE, TensorShape({}));
  arg1.scalar<ResourceHandle>()() = resource_handle0;

  const string device0 = GetFunctionResourceInputDevice(
      arg0, /*arg_index=*/0, fdef, &composite_devices);
  const string device1 = GetFunctionResourceInputDevice(
      arg1, /*arg_index=*/1, fdef, &composite_devices);

  EXPECT_EQ(device0, "/device:COMPOSITE:0");
  EXPECT_EQ(device1, "/device:CPU:0");
  EXPECT_EQ(composite_devices.size(), 1);
  EXPECT_EQ(composite_devices.at("/device:COMPOSITE:0").size(), 2);
}

}  // end namespace
}  // end namespace tensorflow
