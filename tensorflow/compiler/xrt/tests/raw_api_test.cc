/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_compile_ops.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_execute_op.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_state_ops.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {

string* xla_test_device_ptr;  // initial value set in main()

string DeviceFromFlag() {
  string xla_test_device = *xla_test_device_ptr;
  return absl::StrCat("/device:", xla_test_device, ":0");
}

xla::LiteralProto TwoElementTuple() {
  auto array = xla::LiteralUtil::CreateR1<float>({1.0f, 3.0f});
  auto matrix = xla::LiteralUtil::CreateR2({{4, 5}, {6, 7}});
  auto tuple = xla::LiteralUtil::MakeTuple({&array, &matrix});
  return tuple.ToProto();
}

xla::LiteralProto ScalarLiteral() {
  auto scalar = xla::LiteralUtil::CreateR0<float>(12.0f);
  return scalar.ToProto();
}

xla::LiteralProto NestedTuple() {
  auto array = xla::LiteralUtil::CreateR1<float>({1.0f, 3.0f});
  auto matrix = xla::LiteralUtil::CreateR2({{4, 5}, {6, 7}});
  auto tuple = xla::LiteralUtil::MakeTuple({&array, &matrix});
  auto scalar = xla::LiteralUtil::CreateR0<float>(12.0f);
  auto nested = xla::LiteralUtil::MakeTuple({&tuple, &scalar});
  return nested.ToProto();
}

xla::LiteralProto MakeTuple0() {
  auto scalar = xla::LiteralUtil::CreateR0<float>(12.0f);
  auto array = xla::LiteralUtil::CreateR1<float>({1.0f, 3.0f});
  auto matrix = xla::LiteralUtil::CreateR2({{4, 5}, {6, 7}});
  auto tuple = xla::LiteralUtil::MakeTuple({&array, &matrix});
  auto nested0 = xla::LiteralUtil::MakeTuple({&scalar, &tuple});
  auto nested1 = xla::LiteralUtil::MakeTuple({&scalar, &nested0});
  return nested1.ToProto();
}

xla::LiteralProto FloatVector(absl::Span<const float> v) {
  auto array = xla::LiteralUtil::CreateR1<float>(v);
  return array.ToProto();
}

bool CompareLiteralProtos(const xla::LiteralProto& a,
                          const xla::LiteralProto& b) {
  auto l_a = xla::Literal::CreateFromProto(a).ValueOrDie();
  auto l_b = xla::Literal::CreateFromProto(b).ValueOrDie();
  bool equal = l_a == l_b;
  if (!equal) {
    LOG(INFO) << "LiteralProtos don't match " << a.DebugString()
              << " != " << b.DebugString();
  }
  return equal;
}

bool CompareLiteralToLiteralProto(const xla::Literal& a,
                                  const xla::LiteralProto& b) {
  auto l_b = xla::Literal::CreateFromProto(b).ValueOrDie();
  bool equal = a == l_b;
  if (!equal) {
    LOG(INFO) << "Literal and LiteralProto don't match "
              << a.ToProto().DebugString() << " != " << b.DebugString();
  }
  return equal;
}

xla::XlaComputation AddAndScale() {
  xla::XlaBuilder builder("AddAndScale");
  auto p0 = xla::Parameter(&builder, 0,
                           xla::ShapeUtil::MakeShape(xla::F32, {2}), "P0");
  auto p1 = xla::Parameter(&builder, 1,
                           xla::ShapeUtil::MakeShape(xla::F32, {2}), "P1");
  auto sum = xla::Add(p0, p1);
  auto c = xla::ConstantR0<float>(&builder, 3.0f);
  xla::Mul(sum, c);
  return builder.Build().ValueOrDie();
}

xla::XlaComputation AddAndTuple() {
  xla::XlaBuilder builder("AddAndTuple");
  auto p0 = xla::Parameter(&builder, 0,
                           xla::ShapeUtil::MakeShape(xla::F32, {2}), "P0");
  auto p1 = xla::Parameter(&builder, 1,
                           xla::ShapeUtil::MakeShape(xla::F32, {2}), "P1");
  auto sum = xla::Add(p0, p1);
  xla::Tuple(&builder, {sum});
  return builder.Build().ValueOrDie();
}

void StoreComputationSnapshot(const xla::XlaComputation& computation,
                              xla::HloSnapshot* dst) {
  auto snapshot = computation.Snapshot().ValueOrDie();
  *dst = *snapshot;
}

TEST(RawApiTest, ReadAndWriteState) {
  xrt::XLAAllocation alloc;
  alloc.set_device_ordinal(0);
  *alloc.mutable_value() = TwoElementTuple();

  Scope root = Scope::NewRootScope().WithDevice(DeviceFromFlag());
  auto value =
      ops::Const(root.WithDevice("/device:CPU:0"), alloc.SerializeAsString());
  auto handle = ops::XRTAllocate(root, value);
  auto read_back = ops::XRTReadLiteral(root, handle);
  auto release = ops::XRTReleaseAllocationHandle(
      root.WithControlDependencies(read_back), handle);
  TF_ASSERT_OK(root.status());

  tensorflow::ClientSession session(root);
  std::vector<tensorflow::Tensor> outputs;
  TF_EXPECT_OK(session.Run(tensorflow::ClientSession::FeedType(), {read_back},
                           {release}, &outputs));

  xla::LiteralProto response;
  EXPECT_TRUE(response.ParseFromString(outputs[0].scalar<string>()()));

  EXPECT_TRUE(CompareLiteralProtos(alloc.value(), response));
}

TEST(RawApiTest, ReadAndWriteStateAutoFree) {
  xrt::XLAAllocation alloc;
  alloc.set_device_ordinal(0);
  *alloc.mutable_value() = TwoElementTuple();

  Scope root = Scope::NewRootScope().WithDevice(DeviceFromFlag());
  auto value =
      ops::Const(root.WithDevice("/device:CPU:0"), alloc.SerializeAsString());
  auto handle = ops::XRTAllocate(root, value);
  auto read_back = ops::XRTReadLiteralAndRelease(root, handle);
  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run({read_back}, &outputs));

  xla::LiteralProto response;
  EXPECT_TRUE(response.ParseFromString(outputs[0].scalar<string>()()));
  EXPECT_TRUE(CompareLiteralProtos(alloc.value(), response));
}

TEST(RawApiTest, SubBuffer) {
  xrt::XLAAllocation alloc;
  alloc.set_device_ordinal(0);
  *alloc.mutable_value() = NestedTuple();

  Scope root = Scope::NewRootScope().WithDevice(DeviceFromFlag());
  auto value =
      ops::Const(root.WithDevice("/device:CPU:0"), alloc.SerializeAsString());
  auto base_handle = ops::XRTAllocate(root, value);
  auto index_0 = ops::Const(root.WithDevice("/device:CPU:0"), {0});
  auto index_1 = ops::Const(root.WithDevice("/device:CPU:0"), {1});
  auto index_00 = ops::Const(root.WithDevice("/device:CPU:0"), {0, 0});
  auto sub_0 = ops::XRTSubTuple(root, base_handle, index_0);
  auto sub_1 = ops::XRTSubTuple(root, base_handle, index_1);
  auto sub_00 = ops::XRTSubTupleAndRelease(
      root.WithControlDependencies(
          {sub_0.output_handle.op(), sub_1.output_handle.op()}),
      base_handle, index_00);
  auto value_0 = ops::XRTReadLiteralAndRelease(root, sub_0);
  auto value_1 = ops::XRTReadLiteralAndRelease(root, sub_1);
  auto value_00 = ops::XRTReadLiteralAndRelease(root, sub_00);
  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run({value_0, value_1, value_00}, &outputs));

  auto base_literal = xla::Literal::CreateFromProto(alloc.value()).ValueOrDie();
  auto base_elements = base_literal.DecomposeTuple();
  auto nested_0_elements = base_elements[0].Clone().DecomposeTuple();
  xla::LiteralProto response_0;
  EXPECT_TRUE(response_0.ParseFromString(outputs[0].scalar<string>()()));
  EXPECT_TRUE(CompareLiteralToLiteralProto(base_elements[0], response_0));
  xla::LiteralProto response_1;
  EXPECT_TRUE(response_1.ParseFromString(outputs[1].scalar<string>()()));
  EXPECT_TRUE(CompareLiteralToLiteralProto(base_elements[1], response_1));
  xla::LiteralProto response_00;
  EXPECT_TRUE(response_00.ParseFromString(outputs[2].scalar<string>()()));
  EXPECT_TRUE(CompareLiteralToLiteralProto(nested_0_elements[0], response_00));
}

TEST(RawApiTest, MakeTuple) {
  xrt::XLAAllocation alloc_0;
  alloc_0.set_device_ordinal(0);
  *alloc_0.mutable_value() = TwoElementTuple();
  xrt::XLAAllocation alloc_1;
  alloc_1.set_device_ordinal(0);
  *alloc_1.mutable_value() = ScalarLiteral();

  // The trivial tuple that just forwards its input and releases it.
  xrt::XLATupleNode desc_0;
  desc_0.set_input_index(0);
  desc_0.set_release_input_handle(true);

  xrt::XLATupleNode desc_1;
  auto subdesc_10 = desc_1.add_tuples();
  auto subdesc_11 = desc_1.add_tuples();
  subdesc_10->set_input_index(0);
  auto subdesc_110 = subdesc_11->add_tuples();
  subdesc_110->set_input_index(0);
  auto subdesc_111 = subdesc_11->add_tuples();
  subdesc_111->set_input_index(1);

  xrt::XLATupleNode desc_2;
  auto subdesc_20 = desc_2.add_tuples();
  auto subdesc_21 = desc_2.add_tuples();
  subdesc_20->set_input_index(1);
  subdesc_20->set_release_input_handle(true);
  subdesc_21->set_input_index(0);
  subdesc_21->set_release_input_handle(true);

  Scope root = Scope::NewRootScope().WithDevice(DeviceFromFlag());
  auto value_0 =
      ops::Const(root.WithDevice("/device:CPU:0"), alloc_0.SerializeAsString());
  auto handle_0 = ops::XRTAllocate(root, value_0);
  auto value_1 =
      ops::Const(root.WithDevice("/device:CPU:0"), alloc_1.SerializeAsString());
  auto handle_1 = ops::XRTAllocate(root, value_1);
  auto tuple_0 =
      ops::Const(root.WithDevice("/device:CPU:0"), desc_0.SerializeAsString());
  auto handle_2 =
      ops::XRTMakeTuple(root, tuple_0, {static_cast<Output>(handle_0)});
  // handle_0 has now been released.
  auto tuple_1 =
      ops::Const(root.WithDevice("/device:CPU:0"), desc_1.SerializeAsString());
  auto handle_3 = ops::XRTMakeTuple(
      root, tuple_1,
      {static_cast<Output>(handle_1), static_cast<Output>(handle_2)});
  auto tuple_2 =
      ops::Const(root.WithDevice("/device:CPU:0"), desc_2.SerializeAsString());
  // Make sure this runs after handle_3 has completed, since it will free
  // handle_1 and handle_2.
  auto handle_4 = ops::XRTMakeTuple(
      root.WithControlDependencies(handle_3), tuple_2,
      {static_cast<Output>(handle_1), static_cast<Output>(handle_2)});
  // handle_1 and handle_2 have now been released.

  auto res_0 = ops::XRTReadLiteralAndRelease(root, handle_3);
  auto res_1 = ops::XRTReadLiteralAndRelease(root, handle_4);
  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run({res_0, res_1}, &outputs));
  xla::LiteralProto response_0;
  EXPECT_TRUE(response_0.ParseFromString(outputs[0].scalar<string>()()));
  xla::LiteralProto response_1;
  EXPECT_TRUE(response_1.ParseFromString(outputs[1].scalar<string>()()));

  auto expected_0 = MakeTuple0();
  EXPECT_TRUE(CompareLiteralProtos(response_0, expected_0));
  auto expected_1 = NestedTuple();
  EXPECT_TRUE(CompareLiteralProtos(response_1, expected_1));
}

TEST(RawApiTest, CompileAndExecute) {
  xrt::XLAAllocation p0;
  p0.set_device_ordinal(0);
  *p0.mutable_value() = FloatVector({1.0f, 2.0f});
  xrt::XLAAllocation p1;
  p1.set_device_ordinal(0);
  *p1.mutable_value() = FloatVector({8.0f, 5.0f});

  xrt::XLAComputation c;
  auto config = c.mutable_config();
  auto shapes = config->mutable_program_shape();
  *shapes->add_parameters() = xla::ShapeUtil::MakeShape(xla::F32, {2});
  *shapes->add_parameters() = xla::ShapeUtil::MakeShape(xla::F32, {2});
  *shapes->mutable_result() = xla::ShapeUtil::MakeShape(xla::F32, {2});
  StoreComputationSnapshot(AddAndScale(), c.mutable_hlo_snapshot());

  xrt::XRTExecutionConfig e;
  e.set_release_input_handles(true);
  e.set_release_compilation_handle(true);

  Scope root = Scope::NewRootScope().WithDevice(DeviceFromFlag());
  auto e_config =
      ops::Const(root.WithDevice("/device:CPU:0"), e.SerializeAsString());
  auto computation =
      ops::Const(root.WithDevice("/device:CPU:0"), c.SerializeAsString());
  auto c_handle = ops::XRTCompile(root, computation);
  auto p0_value =
      ops::Const(root.WithDevice("/device:CPU:0"), p0.SerializeAsString());
  auto p0_handle = ops::XRTAllocate(root, p0_value);
  auto p1_value =
      ops::Const(root.WithDevice("/device:CPU:0"), p1.SerializeAsString());
  auto p1_handle = ops::XRTAllocate(root, p1_value);
  auto result = ops::XRTExecute(root, c_handle, e_config,
                                {Output(p0_handle), Output(p1_handle)});
  auto read_back = ops::XRTReadLiteralAndRelease(root, result);
  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run({read_back}, &outputs));

  xla::LiteralProto response;
  EXPECT_TRUE(response.ParseFromString(outputs[0].scalar<string>()()));

  auto expected = xla::LiteralUtil::CreateR1<float>({27.0f, 21.0f});
  EXPECT_TRUE(CompareLiteralToLiteralProto(expected, response));
}

TEST(RawApiTest, CompileAndExecuteReturnTuple) {
  xrt::XLAAllocation p0;
  p0.set_device_ordinal(0);
  *p0.mutable_value() = FloatVector({1.0f, 2.0f});
  xrt::XLAAllocation p1;
  p1.set_device_ordinal(0);
  *p1.mutable_value() = FloatVector({8.0f, 5.0f});

  xrt::XLAComputation c;
  auto config = c.mutable_config();
  auto shapes = config->mutable_program_shape();
  *shapes->add_parameters() = xla::ShapeUtil::MakeShape(xla::F32, {2});
  *shapes->add_parameters() = xla::ShapeUtil::MakeShape(xla::F32, {2});
  *shapes->mutable_result() = xla::ShapeUtil::MakeTupleShape(
      {xla::ShapeUtil::MakeShape(xla::F32, {2})});
  StoreComputationSnapshot(AddAndTuple(), c.mutable_hlo_snapshot());

  xrt::XRTExecutionConfig e;
  e.set_release_input_handles(true);
  e.set_release_compilation_handle(true);

  Scope root = Scope::NewRootScope().WithDevice(DeviceFromFlag());
  auto e_config =
      ops::Const(root.WithDevice("/device:CPU:0"), e.SerializeAsString());
  auto computation =
      ops::Const(root.WithDevice("/device:CPU:0"), c.SerializeAsString());
  auto c_handle = ops::XRTCompile(root, computation);
  auto p0_value =
      ops::Const(root.WithDevice("/device:CPU:0"), p0.SerializeAsString());
  auto p0_handle = ops::XRTAllocate(root, p0_value);
  auto p1_value =
      ops::Const(root.WithDevice("/device:CPU:0"), p1.SerializeAsString());
  auto p1_handle = ops::XRTAllocate(root, p1_value);
  auto result = ops::XRTExecute(root, c_handle, e_config,
                                {Output(p0_handle), Output(p1_handle)});
  auto read_back = ops::XRTReadLiteralAndRelease(root, result);
  TF_ASSERT_OK(root.status());

  ClientSession session(root);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run({read_back}, &outputs));

  xla::LiteralProto response;
  EXPECT_TRUE(response.ParseFromString(outputs[0].scalar<string>()()));

  auto sum = xla::LiteralUtil::CreateR1<float>({9.0f, 7.0f});
  auto expected = xla::LiteralUtil::MakeTuple({&sum});
  EXPECT_TRUE(CompareLiteralToLiteralProto(expected, response));
}

}  // namespace

}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::xla_test_device_ptr = new tensorflow::string("XLA_CPU");
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("xla_test_device", tensorflow::xla_test_device_ptr,
                       "Tensorflow device type to use for test, e.g., XLA_CPU"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
