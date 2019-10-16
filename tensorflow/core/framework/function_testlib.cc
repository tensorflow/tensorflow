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

#include "tensorflow/core/framework/function_testlib.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace test {
namespace function {

typedef FunctionDefHelper FDH;

GraphDef GDef(gtl::ArraySlice<NodeDef> nodes,
              gtl::ArraySlice<FunctionDef> funcs) {
  GraphDef g;
  VersionDef* versions = g.mutable_versions();
  versions->set_producer(TF_GRAPH_DEF_VERSION);
  versions->set_min_consumer(TF_GRAPH_DEF_VERSION_MIN_CONSUMER);
  for (const auto& n : nodes) {
    *(g.add_node()) = n;
  }
  auto lib = g.mutable_library();
  for (const auto& f : funcs) {
    *(lib->add_function()) = f;
  }
  return g;
}

// Helper to construct a NodeDef.
NodeDef NDef(StringPiece name, StringPiece op, gtl::ArraySlice<string> inputs,
             gtl::ArraySlice<std::pair<string, FDH::AttrValueWrapper>> attrs,
             const string& device) {
  NodeDef n;
  n.set_name(string(name));
  n.set_op(string(op));
  for (const auto& in : inputs) n.add_input(in);
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

FunctionDef IsZero() {
  const Tensor kZero = test::AsScalar<int64>(0);
  return FDH::Define(
      // Name
      "IsZero",
      // Args
      {"x: T"},
      // Return values
      {"equal: bool"},
      // Attr def
      {"T:{float, double, int32, int64, string}"},
      {
          {{"zero"}, "Const", {}, {{"value", kZero}, {"dtype", DT_INT64}}},
          {{"cast"}, "Cast", {"zero"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"equal"}, "Equal", {"x", "cast"}, {{"T", "$T"}}},
      });
}

FunctionDef RandomUniform() {
  const Tensor kZero = test::AsScalar<int64>(0);

  return FDH::Define(
      // Name
      "RandomUniform",
      // Args
      {"x: T"},
      // Return values
      {"random_uniform: int64"},
      // Attr def
      {"T:{float, double, int32, int64, string}"},
      {{{"random_uniform/shape"},
        "Const",
        {},
        {{"value", kZero}, {"dtype", DT_INT64}}},
       {{"random_uniform"},
        "RandomUniform",
        {"random_uniform/shape"},
        {{"T", DT_INT32},
         {"Tout", DT_FLOAT},
         {"seed", 87654321},
         {"seed2", 42}}}});
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

FunctionDef TwoDeviceMult() {
  const Tensor kTwo = test::AsScalar<int64>(2);
  const Tensor kThree = test::AsScalar<int64>(3);
  return FDH::Create(
      // Name
      "TwoDeviceMult",
      // Args
      {"x: T"},
      // Return values
      {"y_cpu: T", "y_gpu: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"num_2"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"num_3"}, "Const", {}, {{"value", kThree}, {"dtype", DT_INT64}}},
          {{"factor_2"},
           "Cast",
           {"num_2:output:0"},
           {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"factor_3"},
           "Cast",
           {"num_3:output:0"},
           {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y_cpu"},
           "Mul",
           {"x", "factor_2:y:0"},
           {{"T", "$T"}},
           {},
           "/device:CPU:0"},
          {{"y_gpu"},
           "Mul",
           {"x", "factor_3:y:0"},
           {{"T", "$T"}},
           {},
           "/device:GPU:0"},
      },
      {{"y_cpu", "y_cpu:z:0"}, {"y_gpu", "y_gpu:z:0"}});
}

FunctionDef TwoDeviceInputOutput() {
  const Tensor kTwo = test::AsScalar<float>(2);
  const Tensor kThree = test::AsScalar<float>(3);
  return FDH::Create(
      // Name
      "TwoDeviceInputOutput",
      // Args
      {"x1: T", "x2: T"},
      // Return values
      {"y_cpu: T", "y_gpu: T"},
      // Attr def
      {"T: {float}"},
      // Nodes
      {
          {{"num_2"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_FLOAT}}},
          {{"num_3"}, "Const", {}, {{"value", kThree}, {"dtype", DT_FLOAT}}},
          {{"y_cpu"},
           "Mul",
           {"x1", "num_2:output:0"},
           {{"T", "$T"}},
           {},
           "/device:CPU:0"},
          {{"y_gpu"},
           "Mul",
           {"x2", "num_3:output:0"},
           {{"T", "$T"}},
           {},
           "/device:GPU:0"},
      },
      {{"y_cpu", "y_cpu:z:0"}, {"y_gpu", "y_gpu:z:0"}});
}

FunctionDef FuncWithListInput() {
  const Tensor kTwo = test::AsScalar<float>(2);
  return FDH::Create(
      // Name
      "FuncWithListInput",
      // Args
      {"x1: N * T"},
      // Return values
      {},
      // Attr def
      {"T: {float}", "N: int >= 1"},
      // Nodes
      {
          {{"num_2"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_FLOAT}}},
      },
      {});
}

FunctionDef FuncWithListOutput() {
  const Tensor kTwo = test::AsScalar<float>(2);
  return FDH::Create(
      // Name
      "FuncWithListOutput",
      // Args
      {},
      // Return values
      {"y: N * T"},
      // Attr def
      {"T: {float}", "N: int >= 1"},
      // Nodes
      {
          {{"num_2"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_FLOAT}}},
      },
      {{"y", "num_2:output:0"}});
}

FunctionDef XAddX() {
  return FDH::Define(
      // Name
      "XAddX",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"y"}, "Add", {"x", "x"}, {{"T", "$T"}}},
      });
}

FunctionDef XAddY() {
  return FDH::Define(
      // Name
      "XAddY",
      // Args
      {"x: T", "y: T"},
      // Return values
      {"z: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"z"}, "Add", {"x", "y"}, {{"T", "$T"}}},
      });
}

FunctionDef XTimesTwoInt32() {
  const Tensor kTwo = test::AsScalar<int64>(2);
  return FDH::Define(
      // Name
      "XTimesTwoInt32",
      // Args
      {"x: int32"},
      // Return values
      {"y: int32"}, {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"},
           "Cast",
           {"two"},
           {{"SrcT", DT_INT64}, {"DstT", DT_INT32}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", DT_INT32}}},
      });
}

FunctionDef XTimesFour() {
  return FDH::Create(
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
          {{"y"}, "XTimesTwo", {"x2:y:0"}, {{"T", "$T"}}},
      },
      {{"y", "y:y:0"}});
}

FunctionDef XTimes16() {
  return FDH::Create(
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
          {{"y"}, "XTimesFour", {"x4:y:0"}, {{"T", "$T"}}},
      },
      {{"y", "y:y:0"}});
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
      {"T: {float, double, resource}"},
      // Nodes
      {{{"o0"}, "Identity", {"i1"}, {{"T", "$T"}}},
       {{"o1"}, "Identity", {"i0"}, {{"T", "$T"}}}});
}

FunctionDef EmptyBodySwap() {
  return FDH::Create(
      // Name
      "EmptyBodySwap",
      // Args
      {"i0: T", "i1: T"},
      // Return values
      {"o0: T", "o1: T"},
      // Attr def
      {"T: {float, double, resource}"},
      // Nodes
      {},
      // Output mapping
      {{"o0", "i1"}, {"o1", "i0"}});
}

FunctionDef ResourceOutput() {
  const Tensor kTwo = test::AsScalar<float>(2);
  return FDH::Create(
      // Name
      "ResourceOutput",
      // Args
      {"x: float", "y: resource"},
      // Return values
      {"y_out: resource", "two_x: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_FLOAT}}},
          {{"mul"}, "Mul", {"x", "two:output:0"}, {{"T", DT_FLOAT}}, {}},
      },
      {{"y_out", "y"}, {"two_x", "mul:z:0"}});
}

FunctionDef ResourceIdentity() {
  return FDH::Create(
      // Name
      "ResourceIdentity",
      // Args
      {"x: resource"},
      // Return values
      {"y: resource"},
      // Attr def
      {},
      // Nodes
      {},
      // Output mapping
      {{"y", "x"}});
}

FunctionDef ReadResourceVariable() {
  return FDH::Create(
      // Name
      "ReadResourceVariable",
      // Args
      {"x: resource"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"read"}, "ReadVariableOp", {"x"}, {{"dtype", DT_FLOAT}}, {}},
      },
      {{"y", "read:value:0"}});
}

FunctionDef InvalidControlFlow() {
  return FDH::Create(
      // Name
      "InvalidControlFlow",
      // Args
      {"i: int32"},
      // Return values
      {"o: int32"},
      // Attr def
      {},
      // Nodes
      {{{"enter"}, "Enter", {"i"}, {{"T", DT_INT32}, {"frame_name", "while"}}},
       {{"add"}, "Add", {"enter:output", "i"}, {{"T", DT_INT32}}}},
      // Output mapping
      {{"o", "add:z"}});
}

FunctionDef LessThanOrEqualToN(int64 N) {
  const Tensor kN = test::AsScalar<int64>(N);
  return FDH::Define(
      // Name
      "LessThanOrEqualToN",
      // Args
      {"x: T"},
      // Return values
      {"z: bool"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"N"}, "Const", {}, {{"value", kN}, {"dtype", DT_INT64}}},
          {{"y"}, "Cast", {"N"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"z"}, "LessEqual", {"x", "y"}, {{"T", "$T"}}},
      });
}

FunctionDef XPlusOneXTimesY() {
  const Tensor kOne = test::AsScalar<int64>(1);
  return FDH::Define(
      // Name
      "XPlusOneXTimesY",
      // Args
      {"x: T", "y: T"},
      // Return values
      {"s: T", "t: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {{{"one"}, "Const", {}, {{"value", kOne}, {"dtype", DT_INT64}}},
       {{"increment"}, "Cast", {"one"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
       {{"s"}, "Add", {"x", "increment"}, {{"T", "$T"}}},
       {{"t"}, "Mul", {"x", "y"}, {{"T", "$T"}}}});
}

FunctionDef XYXLessThanOrEqualToN(int64 N) {
  const Tensor kN = test::AsScalar<int64>(N);
  return FDH::Define(
      // Name
      "XYXLessThanOrEqualToN",
      // Args
      {"x: T", "y: T"},
      // Return values
      {"z: bool"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"N"}, "Const", {}, {{"value", kN}, {"dtype", DT_INT64}}},
          {{"N1"}, "Cast", {"N"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"z"}, "LessEqual", {"x", "N1"}, {{"T", "$T"}}},
      });
}

FunctionDef RandomUniformLess() {
  const Tensor kZero = test::AsScalar<int32>(0);
  const Tensor kOne = test::AsScalar<int32>(1);
  const Tensor k005 = test::AsScalar<float>(0.05);

  return FDH::Define(
      // Name
      "RandomUniformLess",
      // Args
      {"arg0: int64"},
      // Return values
      {"strided_slice: bool"},
      // Attr def
      {"T:{float, double, int32, int64, string}"},
      {{{"random_uniform/shape"},
        "Const",
        {},
        {{"value", kZero}, {"dtype", DT_INT32}}},

       {{"random_uniform/RandomUniform"},
        "RandomUniform",
        {"random_uniform/shape"},
        {{"T", DT_INT32}, {"Tout", DT_FLOAT}, {"seed", 0}, {"seed2", 0}}},

       {{"Less/y"}, "Const", {}, {{"value", k005}, {"dtype", DT_FLOAT}}},

       {{"Less"},
        "Less",
        {"random_uniform/RandomUniform", "Less/y"},
        {{"T", DT_FLOAT}}},

       {{"strided_slice/stack"},
        "Const",
        {},
        {{"value", kZero}, {"dtype", DT_INT32}}},

       {{"strided_slice/stack_1"},
        "Const",
        {},
        {{"value", kOne}, {"dtype", DT_INT32}}},

       {{"strided_slice/stack_2"},
        "Const",
        {},
        {{"value", kOne}, {"dtype", DT_INT32}}},

       {{"strided_slice"},
        "StridedSlice",
        {"Less", "strided_slice/stack", "strided_slice/stack_1",
         "strided_slice/stack_2"},
        {{"Index", DT_INT32},
         {"T", DT_BOOL},
         {"begin_mask", 0},
         {"ellipsis_mask", 0},
         {"end_mask", 0},
         {"new_axis_mask", 0},
         {"shrink_axis_mask", 0}}}});
}

FunctionDef MakeRangeDataset() {
  return FDH::Define(
      /*name=*/"MakeRangeDataset",
      /*arg_def=*/{"start: int64", "stop: int64", "step: int64"},
      /*ret_def=*/{"y:variant"},
      /*attr_def=*/
      {"output_types: list(type) >= 1", "output_shapes: list(shape) >= 1"},
      /*node_def=*/
      {{/*ret=*/{"y"},
        /*op=*/"RangeDataset",
        /*arg=*/{"start", "stop", "step"},
        /*attr=*/
        {{"output_types", "$output_types"},
         {"output_shapes", "$output_shapes"}}}});
}

FunctionDef MakeBatchDataset() {
  return FDH::Define(
      /*name=*/"MakeBatchDataset",
      /*arg_def=*/
      {"input_dataset: variant", "batch_size: int64", "drop_remainder: bool"},
      /*ret_def=*/{"y: variant"},
      /*attr_def=*/
      {"parallel_copy: bool = false", "output_types: list(type) >= 1",
       "output_shapes: list(shape) >= 1"},
      /*node_def=*/
      {{/*ret=*/{"y"},
        /*op=*/"BatchDatasetV2",
        /*arg=*/{"input_dataset", "batch_size", "drop_remainder"},
        /*attr=*/
        {{"parallel_copy", "$parallel_copy"},
         {"output_types", "$output_types"},
         {"output_shapes", "$output_shapes"}}}});
}

FunctionDef MakeMapDataset(bool has_other_args) {
  std::vector<string> args = {"input_dataset: variant"};
  std::vector<string> inputs = {"input_dataset"};
  if (has_other_args) {
    args.emplace_back("other_arguments: Targuments");
    inputs.emplace_back("other_arguments");
  }

  return FDH::Define(
      /*name=*/"MakeMapDataset",
      /*arg_def=*/args,
      /*ret_def=*/
      {"y: variant"},
      /*attr_def=*/
      {"f: func", "Targuments: list(type) >= 0",
       "output_types: list(type) >= 1", "output_shapes: list(shape) >= 1",
       "use_inter_op_parallelism: bool = true",
       "preserve_cardinality: bool = false"},
      /*node_def=*/
      {{/*ret=*/{"y"},
        /*op=*/"MapDataset",
        /*arg=*/inputs,
        /*attr=*/
        {{"f", "$f"},
         {"Targuments", "$Targuments"},
         {"output_types", "$output_types"},
         {"output_shapes", "$output_shapes"},
         {"use_inter_op_parallelism", "$use_inter_op_parallelism"},
         {"preserve_cardinality", "$preserve_cardinality"}}}});
}

FunctionDef MakeTakeDataset() {
  return FDH::Define(
      // Name
      "TakeDataset",
      // Args
      {"input_dataset: variant", "count: int64"},
      // Return values
      {"y:variant"},
      // Attr def
      {"output_types: list(type) >= 1", "output_shapes: list(shape) >= 1"},
      // Nodes
      {{{"y"},
        "TakeDataset",
        {"input_dataset", "count"},
        {{"output_types", "$output_types"},
         {"output_shapes", "$output_shapes"}}}});
}

FunctionDef MakeTensorSliceDataset() {
  return FDH::Define(
      // Name
      "MakeTensorSliceDataset",
      // Args
      {"x: Toutput_types"},
      // Return values
      {"y: variant"},
      // Attr def
      {"Toutput_types: list(type) >= 1", "output_shapes: list(shape) >= 1"},
      // Nodes
      {{{"y"},
        "TensorSliceDataset",
        {"x"},
        {{"Toutput_types", "$Toutput_types"},
         {"output_shapes", "$output_shapes"}}}});
}

FunctionDef Unique() {
  return FDH::Create(
      // Name
      "GetUnique",
      // Args
      {"x:T"},
      // Return values
      {"y:T", "idx: out_idx"},
      // Attr def
      {"T: type", "out_idx: {int32, int64} = DT_INT32"},
      // Nodes
      {
          {{"result"}, "Unique", {"x"}, {{"T", "$T"}, {"out_idx", "$out_idx"}}},
      },
      {{"y", "result:y:0"}, {"idx", "result:idx:0"}});
}

void FunctionTestSchedClosure(std::function<void()> fn) {
  static thread::ThreadPool* w =
      new thread::ThreadPool(Env::Default(), "Test", 8);
  w->Schedule(std::move(fn));
}

}  // end namespace function
}  // end namespace test
}  // end namespace tensorflow
