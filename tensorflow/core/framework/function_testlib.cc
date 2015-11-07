#include "tensorflow/core/framework/function_testlib.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {
namespace test {
namespace function {

typedef FunctionDefHelper FDH;

GraphDef GDef(gtl::ArraySlice<NodeDef> nodes,
              gtl::ArraySlice<FunctionDef> funcs) {
  GraphDef g;
  for (auto n : nodes) {
    *(g.add_node()) = n;
  }
  auto lib = g.mutable_library();
  for (auto f : funcs) {
    *(lib->add_function()) = f;
  }
  return g;
}

// Helper to construct a NodeDef.
NodeDef NDef(const string& name, const string& op,
             gtl::ArraySlice<string> inputs,
             gtl::ArraySlice<std::pair<string, FDH::AttrValueWrapper>> attrs,
             const string& device) {
  NodeDef n;
  n.set_name(name);
  n.set_op(op);
  for (auto in : inputs) n.add_input(in);
  n.set_device(device);
  for (auto na : attrs) n.mutable_attr()->insert({na.first, na.second.proto});
  return n;
}

FunctionDef NonZero() {
  return FDH::Define(
      // Name
      "NonZero",
      // Args
      {"x:T"},
      // Return values
      {"y:T"},
      // Attr def
      {"T:{float, double, int32, int64, string}"},
      // Nodes
      {
          {{"y"}, "Identity", {"x"}, {{"T", "$T"}}},
      });
}

FunctionDef XTimesTwo() {
  const Tensor kTwo = test::AsScalar<int64>(2);
  return FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
}

FunctionDef XTimesFour() {
  return FDH::Define(
      // Name
      "XTimesFour",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"x2"}, "XTimesTwo", {"x"}, {{"T", "$T"}}},
          {{"y"}, "XTimesTwo", {"x2"}, {{"T", "$T"}}},
      });
}

FunctionDef XTimes16() {
  return FDH::Define(
      // Name
      "XTimes16",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"x4"}, "XTimesFour", {"x"}, {{"T", "$T"}}},
          {{"y"}, "XTimesFour", {"x4"}, {{"T", "$T"}}},
      });
}

FunctionDef WXPlusB() {
  return FDH::Define(
      // Name
      "WXPlusB",
      // Args
      {"w: T", "x: T", "b: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double}"},
      // Nodes
      {{{"mm"},
        "MatMul",
        {"w", "x"},
        {{"T", "$T"},
         {"transpose_a", false},
         {"transpose_b", false},
         {"_kernel", "eigen"}}},
       {{"y"}, "Add", {"mm", "b"}, {{"T", "$T"}}}});
}

FunctionDef Swap() {
  return FDH::Define(
      // Name
      "Swap",
      // Args
      {"i0: T", "i1: T"},
      // Return values
      {"o0: T", "o1: T"},
      // Attr def
      {"T: {float, double}"},
      // Nodes
      {{{"o0"}, "Identity", {"i1"}, {{"T", "$T"}}},
       {{"o1"}, "Identity", {"i0"}, {{"T", "$T"}}}});
}

}  // end namespace function
}  // end namespace test
}  // end namespace tensorflow
