/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class MapTest : public ClientLibraryTestBase {
 public:
  explicit MapTest(perftools::gputools::Platform* platform = nullptr)
      : ClientLibraryTestBase(platform,
                              /*disabled_pass_names=*/{"algsimp", "inline"}) {}

  // Creates a function that adds its scalar argument with the constant 1.0.
  //
  // x {R0F32} ----> (add)
  //                /
  // 1.0f ---------/
  Computation CreateAdderToOne() {
    ComputationBuilder mapped_builder(client_, TestName());
    auto x = mapped_builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    auto one = mapped_builder.ConstantR0<float>(1.0);
    auto adder_to_one = mapped_builder.Add(x, one);
    auto computation_status = mapped_builder.Build();
    TF_CHECK_OK(computation_status.status());
    return computation_status.ConsumeValueOrDie();
  }

  Computation CreateMax() {
    ComputationBuilder b(client_, TestName());
    auto lhs = b.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    auto rhs = b.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
    b.Max(lhs, rhs);
    auto computation_status = b.Build();
    TF_CHECK_OK(computation_status.status());
    return computation_status.ConsumeValueOrDie();
  }

  // Creates a computation that accepts an F32 and returns T(1) (ignoring the
  // argument).
  template <class T>
  Computation CreateScalarOne() {
    ComputationBuilder mapped_builder(client_, "scalar_one");
    (void)mapped_builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    mapped_builder.ConstantR0<T>(1);
    auto computation_status = mapped_builder.Build();
    TF_CHECK_OK(computation_status.status());
    return computation_status.ConsumeValueOrDie();
  }

  // Creates a function that multiplies its scalar argument by the constant 2.0
  //
  // x {R0F32} ----> (mul)
  //                /
  // 2.0f ---------/
  Computation CreateMulByTwo() {
    ComputationBuilder mapped_builder(client_, TestName());
    auto x = mapped_builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    auto two = mapped_builder.ConstantR0<float>(2.0);
    auto mul_by_two = mapped_builder.Mul(x, two);
    auto computation_status = mapped_builder.Build();
    TF_CHECK_OK(computation_status.status());
    return computation_status.ConsumeValueOrDie();
  }

  // Creates a function that adds its scalar argument with the constant 1.0 and
  // then multiplies by the original element.
  //
  //           /---------------\
  //          /                 \
  // x {R0F32} ----> (add) ----> (mul)
  //                /
  // 1.0f ---------/
  Computation CreateAdderToOneTimesItself() {
    ComputationBuilder mapped_builder(client_, TestName());
    auto x = mapped_builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    auto one = mapped_builder.ConstantR0<float>(1.0);
    auto adder_to_one = mapped_builder.Add(x, one);
    auto result = mapped_builder.Mul(x, adder_to_one);
    auto computation_status = mapped_builder.Build();
    TF_CHECK_OK(computation_status.status());
    return computation_status.ConsumeValueOrDie();
  }

  // Creates a function that takes a single parameter and calls map with
  // "embedded_computation" on it, and then adds "n" to the result.
  //
  // x {R0F32} -----------> (map) ----> (add)
  //                         /           /
  // embedded_computation --/       n --/
  Computation CreateMapPlusN(const Computation& embedded_computation, float n) {
    ComputationBuilder builder(client_, TestName());
    auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    auto map = builder.Map({x}, embedded_computation);
    auto constant_n = builder.ConstantR0<float>(n);
    auto add = builder.Add(map, constant_n);
    auto computation_status = builder.Build();
    TF_CHECK_OK(computation_status.status());
    return computation_status.ConsumeValueOrDie();
  }

  // Creates a binary function with signature (F32, F32) -> Pred
  // defined by (x, y) -> x > y.
  Computation CreateGt() {
    ComputationBuilder b(client_, "Gt");
    auto x = b.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    auto y = b.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
    auto gt = b.Gt(x, y);
    auto computation_status = b.Build();
    TF_CHECK_OK(computation_status.status());
    return computation_status.ConsumeValueOrDie();
  }

  // Creates a function that adds three scalar arguments
  //
  // x {R0F32} ----\
  //                \
  // y {R0F32} ----> (add) ---> (add)
  //                           /
  // z {R0F32} ---------------/
  Computation CreateTernaryAdder() {
    ComputationBuilder mapped_builder(client_, "TernaryAdder");
    auto x = mapped_builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    auto y = mapped_builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
    auto z = mapped_builder.Parameter(2, ShapeUtil::MakeShape(F32, {}), "z");
    auto xy = mapped_builder.Add(x, y);
    auto xyz = mapped_builder.Add(xy, z);
    auto computation_status = mapped_builder.Build();
    TF_CHECK_OK(computation_status.status());
    return computation_status.ConsumeValueOrDie();
  }
};

TEST_F(MapTest, MapEachElemPlusOneR0) {
  // Applies lambda (x) (+ x 1)) to an input scalar.
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal = LiteralUtil::CreateR0<float>(42.0);
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto param = builder.Parameter(0, param0_literal->shape(), "param0");
  auto map = builder.Map({param}, CreateAdderToOne());

  ComputeAndCompareR0<float>(&builder, 43.0, {param0_data.get()},
                             ErrorSpec(0.01f));
}

XLA_TEST_F(MapTest, MapEachElemPlusOneR1S0) {
  // Maps (lambda (x) (+ x 1)) onto an input R1F32 vector of length 0.
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal = LiteralUtil::CreateR1<float>({});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto param = builder.Parameter(0, param0_literal->shape(), "param0");
  auto map = builder.Map({param}, CreateAdderToOne());

  ComputeAndCompareR1<float>(&builder, {}, {param0_data.get()},
                             ErrorSpec(0.01f));
}

TEST_F(MapTest, MapEachElemPlusOneR1S4) {
  // Maps (lambda (x) (+ x 1)) onto an input R1F32 vector of length 4.
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR1<float>({2.2f, 3.3f, 4.4f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto param = builder.Parameter(0, param0_literal->shape(), "param0");
  auto map = builder.Map({param}, CreateAdderToOne());

  ComputeAndCompareR1<float>(&builder, {3.2f, 4.3f, 5.4f, 6.5f},
                             {param0_data.get()}, ErrorSpec(0.01f));
}

TEST_F(MapTest, MapEachF32ElementToS32Constant) {
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR1<float>({2.2f, 3.3f, 4.4f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto param = builder.Parameter(0, param0_literal->shape(), "param0");
  auto map = builder.Map({param}, CreateScalarOne<int32>());

  ComputeAndCompareR1<int32>(&builder, {1, 1, 1, 1}, {param0_data.get()});
}

TEST_F(MapTest, MapEachF32ElementToU32Constant) {
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR1<float>({2.2f, 3.3f, 4.4f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto param = builder.Parameter(0, param0_literal->shape(), "param0");
  auto map = builder.Map({param}, CreateScalarOne<uint32>());

  ComputeAndCompareR1<uint32>(&builder, {1, 1, 1, 1}, {param0_data.get()});
}

TEST_F(MapTest, MapEachElemLongerChainR1) {
  // Maps (lambda (x) (* (+ x 1) x)) onto an input R1F32 vector.
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR1<float>({2.6f, -5.1f, 0.1f, 0.2f, 999.0f, 255.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto param = builder.Parameter(0, param0_literal->shape(), "param0");
  auto map = builder.Map({param}, CreateAdderToOneTimesItself());

  ComputeAndCompareR1<float>(
      &builder, {9.36f, 20.91f, 0.11f, 0.24f, 999000.0f, 65535.75f},
      {param0_data.get()}, ErrorSpec(0.01f));
}

XLA_TEST_F(MapTest, MapMultipleMapsR1S0) {
  // Maps (lambda (x) (+ x 1)) onto an input R1F32 vector of length 0, and then
  // maps (lambda (x) (* x 2)) on the result.
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal = LiteralUtil::CreateR1<float>({});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto param = builder.Parameter(0, param0_literal->shape(), "param0");
  auto map1 = builder.Map({param}, CreateAdderToOne());
  auto map2 = builder.Map({map1}, CreateMulByTwo());

  ComputeAndCompareR1<float>(&builder, {}, {param0_data.get()},
                             ErrorSpec(0.01f));
}

TEST_F(MapTest, MapMultipleMapsR1S4) {
  // Maps (lambda (x) (+ x 1)) onto an input R1F32 vector of length 4, and then
  // maps (lambda (x) (* x 2)) on the result.
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR1<float>({2.2f, 3.3f, 4.4f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto param = builder.Parameter(0, param0_literal->shape(), "param0");
  auto map1 = builder.Map({param}, CreateAdderToOne());
  auto map2 = builder.Map({map1}, CreateMulByTwo());

  ComputeAndCompareR1<float>(&builder, {6.4f, 8.6f, 10.8f, 13.0f},
                             {param0_data.get()}, ErrorSpec(0.01f));
}

TEST_F(MapTest, MapEachElemPlusOneR2) {
  // Maps (lambda (x) (+ x 1)) onto an input R2F32 vector.
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal = LiteralUtil::CreateR2<float>(
      {{13.25f, 14.0f}, {-7.1f, -7.2f}, {-8.8f, 8.8f}});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto param = builder.Parameter(0, param0_literal->shape(), "param0");
  auto map = builder.Map({param}, CreateAdderToOne());

  Array2D<float> expected_array(
      {{14.25f, 15.0f}, {-6.1f, -6.2f}, {-7.8f, 9.8f}});
  ComputeAndCompareR2<float>(&builder, expected_array, {param0_data.get()},
                             ErrorSpec(0.01f));
}

XLA_TEST_F(MapTest, ComplexNestedMaps) {
  // Constructs a complex graph of embedded computations to test the computation
  // lowering order. Python equivalent:
  //
  //   embed1 = lambda x: x + 1                  #  x + 1
  //   embed2 = lambda x: embed1(x) + 2          #  x + 3
  //   embed3 = lambda x: embed1(x) + 4          #  x + 5
  //   embed4 = lambda x: embed2(x) + embed3(x)  # 2x + 8
  //   embed5 = lambda x: embed2(x) + 6          #  x + 9
  //   result = embed5(42) + embed4(7)           # (42 + 9) + (2 * 7 + 8) = 73

  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});

  auto embed1 = CreateAdderToOne();
  auto embed2 = CreateMapPlusN(embed1, 2.0);
  auto embed3 = CreateMapPlusN(embed1, 4.0);

  ComputationBuilder embed4_builder(client_, "embed4");
  auto embed4_param = embed4_builder.Parameter(0, scalar_shape, "x");
  auto embed4_map_lhs = embed4_builder.Map({embed4_param}, embed2);
  auto embed4_map_rhs = embed4_builder.Map({embed4_param}, embed3);
  auto embed4_add = embed4_builder.Add(embed4_map_lhs, embed4_map_rhs);
  auto embed4_status = embed4_builder.Build();
  ASSERT_IS_OK(embed4_status.status());
  auto embed4 = embed4_status.ConsumeValueOrDie();

  auto embed5 = CreateMapPlusN(embed2, 6.0);

  ComputationBuilder builder(client_, TestName());
  auto constant_42 = builder.ConstantR0<float>(42.0);
  auto constant_7 = builder.ConstantR0<float>(7.0);
  auto map_42 = builder.Map({constant_42}, embed5);
  auto map_7 = builder.Map({constant_7}, embed4);
  builder.Add(map_42, map_7);

  ComputeAndCompareR0<float>(&builder, 73.0, {}, ErrorSpec(0.01f));
}

TEST_F(MapTest, VersionedEmbeddedComputation) {
  // Build a computation X, use it in a map, then add an additional operation to
  // computation X and use it again in a different map. Verify that the proper
  // versions of computation X are used in each of the maps.

  // Create a (embedded) computation which adds one to its parameter argument.
  ComputationBuilder embedded_builder(client_, "EmbeddedComputation");
  auto param_0 =
      embedded_builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "param0");
  auto constant_one = embedded_builder.ConstantR0<float>(1.0);
  auto adder_to_one = embedded_builder.Add(param_0, constant_one);
  auto computation_status = embedded_builder.Build();
  ASSERT_IS_OK(computation_status.status());
  auto embedded_computation = computation_status.ConsumeValueOrDie();

  ComputationBuilder builder(client_, TestName());
  auto constant_vector = builder.ConstantR1<float>({1.0, 2.0, 3.0, 4.0});
  auto map_plus_1 = builder.Map({constant_vector}, embedded_computation);

  // Add another Add(1) operation to the existing embedded computation. This
  // requires using the stub interface because the ComputationBuilder does not
  // allow modification to the Computation objects after they have been built.
  BinaryOpRequest request;
  request.set_binop(BINOP_ADD);
  *request.mutable_lhs() = adder_to_one;
  *request.mutable_rhs() = constant_one;
  OpRequest op_request;
  *op_request.mutable_computation() = embedded_computation.handle();
  *op_request.mutable_binary_op_request() = request;
  OpResponse response;
  tensorflow::Status s = client_->stub()->Op(&op_request, &response);
  ASSERT_TRUE(s.ok());

  auto map_plus_2 = builder.Map({map_plus_1}, embedded_computation);

  // The original vector has Add(1) applied to it with a map, followed by
  // Add(1+1) resulting in a net Add(3).
  ComputeAndCompareR1<float>(&builder, {4.0, 5.0, 6.0, 7.0}, {},
                             ErrorSpec(0.01f));
}

TEST_F(MapTest, MapBinaryAdder) {
  // Maps (lambda (x y) (+ x y)) onto two R1F32 vectors.
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR1<float>({2.2f, 3.3f, 4.4f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();
  std::unique_ptr<Literal> param1_literal =
      LiteralUtil::CreateR1<float>({5.1f, 4.4f, -0.1f, -5.5f});
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();

  auto param0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto param1 = builder.Parameter(1, param1_literal->shape(), "param1");
  auto map =
      builder.Map({param0, param1}, CreateScalarAddComputation(F32, &builder));

  ComputeAndCompareR1<float>(&builder, {7.3f, 7.7, 4.3f, 0},
                             {param0_data.get(), param1_data.get()},
                             ErrorSpec(0.01f));
}

// Adds two rank-2 arrays with different layouts. This test exercises a path
// for Map that used to fail in shape inference (b/28989438).
XLA_TEST_F(MapTest, AddWithMixedLayouts) {
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      test_utils::CreateR2LiteralWithLayout({{1, 2}, {3, 4}}, {1, 0});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  std::unique_ptr<Literal> param1_literal =
      test_utils::CreateR2LiteralWithLayout({{10, 20}, {30, 40}}, {0, 1});
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();

  auto param0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto param1 = builder.Parameter(1, param1_literal->shape(), "param1");
  auto map =
      builder.Map({param0, param1}, CreateScalarAddComputation(S32, &builder));

  Array2D<int32> expected(2, 2);
  expected(0, 0) = 11;
  expected(0, 1) = 22;
  expected(1, 0) = 33;
  expected(1, 1) = 44;
  ComputeAndCompareR2<int32>(&builder, expected,
                             {param0_data.get(), param1_data.get()});
}

XLA_TEST_F(MapTest, AddR3_3x0x2) {
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR3FromArray3D<int32>(Array3D<int32>(3, 0, 2));
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  std::unique_ptr<Literal> param1_literal =
      LiteralUtil::CreateR3FromArray3D<int32>(Array3D<int32>(3, 0, 2));
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();

  auto param0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto param1 = builder.Parameter(1, param1_literal->shape(), "param1");
  auto map =
      builder.Map({param0, param1}, CreateScalarAddComputation(S32, &builder));

  ComputeAndCompareR3<int32>(&builder, Array3D<int32>(3, 0, 2),
                             {param0_data.get(), param1_data.get()});
}

TEST_F(MapTest, MapTernaryAdder) {
  // Maps (lambda (x y z) (+ x y z)) onto three R1F32 vectors.
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR1<float>({2.2f, 3.3f, 4.4f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();
  std::unique_ptr<Literal> param1_literal =
      LiteralUtil::CreateR1<float>({5.1f, 4.4f, -0.1f, -5.5f});
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();
  std::unique_ptr<Literal> param2_literal =
      LiteralUtil::CreateR1<float>({-10.0f, -100.0f, -900.0f, -400.0f});
  std::unique_ptr<GlobalData> param2_data =
      client_->TransferToServer(*param2_literal).ConsumeValueOrDie();

  auto param0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto param1 = builder.Parameter(1, param1_literal->shape(), "param1");
  auto param2 = builder.Parameter(2, param2_literal->shape(), "param2");
  auto map = builder.Map({param0, param1, param2}, CreateTernaryAdder());

  ComputeAndCompareR1<float>(
      &builder, {-2.7f, -92.3f, -895.7f, -400.0f},
      {param0_data.get(), param1_data.get(), param2_data.get()},
      ErrorSpec(0.01f));
}

TEST_F(MapTest, MapGt) {
  // Maps (x,y) -> x > y onto two R1F32 vectors.
  ComputationBuilder b(client_, TestName());
  auto gt = CreateGt();
  b.Map({b.ConstantR1<float>({1, 20}), b.ConstantR1<float>({10, 2})}, gt);
  ComputeAndCompareR1<bool>(&b, {false, true}, {});
}

TEST_F(MapTest, NestedBinaryMap) {
  Computation max_with_square;
  {
    // max_with_square(x) = do max(x, x^2) via a map.
    ComputationBuilder b(client_, "max_with_square");
    auto x = b.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
    b.Map({x, b.Mul(x, x)}, CreateMax());
    auto computation_status = b.Build();
    ASSERT_IS_OK(computation_status.status());
    max_with_square = computation_status.ConsumeValueOrDie();
  }
  ComputationBuilder b(client_, TestName());
  auto input = b.ConstantR1<float>({0.1f, 0.5f, -0.5f, 1.0f, 2.0f});
  b.Map({input}, max_with_square);
  ComputeAndCompareR1<float>(&b, {0.1f, 0.5f, 0.25f, 1.0f, 4.0f}, {});
}

TEST_F(MapTest, MapOperantionWithBuildError) {
  // Maps (lambda (x y) (+ x y)) onto two R1F32 vectors but uses an unsupported
  // type combination (F32 + U16) to test that the error is reported to the
  // outermost ComputationBuilder.
  ComputationBuilder builder(client_, TestName());

  auto sub_builder = builder.CreateSubBuilder("ErrorAdd");
  auto x = sub_builder->Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = sub_builder->Parameter(1, ShapeUtil::MakeShape(U16, {}), "y");
  auto adder = sub_builder->Add(x, y);
  auto error_add = sub_builder->BuildAndNoteError();

  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR1<float>({2.2f, 3.3f, 4.4f, 5.5f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();
  std::unique_ptr<Literal> param1_literal =
      LiteralUtil::CreateR1<float>({5.1f, 4.4f, -0.1f, -5.5f});
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();

  auto param0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto param1 = builder.Parameter(1, param1_literal->shape(), "param1");
  auto map = builder.Map({param0, param1}, error_add);

  StatusOr<Computation> computation_status = builder.Build();
  ASSERT_TRUE(!computation_status.ok());
  EXPECT_THAT(computation_status.status().ToString(),
              ::testing::HasSubstr("error from: ErrorAdd: binary op with "
                                   "different element types: f32[] and u16[]"));
}

// MapTest disables inline and algsimp. MapTestWithFullOpt runs all
// optimizations.
using MapTestWithFullOpt = ClientLibraryTestBase;

// Regression test for b/31466798. The inliner simplifies map(param0, param1,
// power) to power(param0, param1) without deleting the old subcomputation which
// is the same as the new entry computation. HloSubcomputationUnification used
// to have issues with such patterns and maybe invalidate the pointer to entry
// computation.
TEST_F(MapTestWithFullOpt, MapScalarPower) {
  ComputationBuilder builder(client_, TestName());

  auto sub_builder = builder.CreateSubBuilder("power");
  auto x = sub_builder->Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = sub_builder->Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
  sub_builder->Pow(x, y);
  auto power = sub_builder->BuildAndNoteError();

  std::unique_ptr<Literal> param0_literal = LiteralUtil::CreateR0<float>(2.0f);
  std::unique_ptr<Literal> param1_literal = LiteralUtil::CreateR0<float>(5.0f);
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();

  auto param0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto param1 = builder.Parameter(1, param1_literal->shape(), "param1");
  builder.Map({param0, param1}, power);

  ComputeAndCompareR0<float>(&builder, 32.0f,
                             {param0_data.get(), param1_data.get()},
                             ErrorSpec(0.01f));
}

// Regression test for b/35786417, where the inliner would not notice the change
// of parameter order inside the map.
TEST_F(MapTestWithFullOpt, MapSubtractOppositeOrder) {
  ComputationBuilder builder(client_, TestName());

  auto sub_builder = builder.CreateSubBuilder("power");
  auto x = sub_builder->Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = sub_builder->Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
  sub_builder->Sub(y, x);  // note that this is y - x, not x - y
  auto sub_opposite = sub_builder->BuildAndNoteError();

  std::unique_ptr<Literal> param0_literal = LiteralUtil::CreateR0<float>(2.0f);
  std::unique_ptr<Literal> param1_literal = LiteralUtil::CreateR0<float>(5.0f);
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();

  auto param0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto param1 = builder.Parameter(1, param1_literal->shape(), "param1");
  builder.Map({param0, param1}, sub_opposite);

  ComputeAndCompareR0<float>(
      &builder, 3.0f, {param0_data.get(), param1_data.get()}, ErrorSpec(0.01f));
}

// Regression test for b/35786417, where the inliner would CHECK-fail due to the
// mul inside the map having more parameters than the map does.
TEST_F(MapTestWithFullOpt, MapSquare) {
  ComputationBuilder builder(client_, TestName());

  auto sub_builder = builder.CreateSubBuilder("power");
  auto x = sub_builder->Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
  sub_builder->Mul(x, x);
  auto square = sub_builder->BuildAndNoteError();

  std::unique_ptr<Literal> param0_literal = LiteralUtil::CreateR0<float>(10.0f);
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

  auto param0 = builder.Parameter(0, param0_literal->shape(), "param0");
  builder.Map({param0}, square);

  ComputeAndCompareR0<float>(&builder, 100.0f, {param0_data.get()},
                             ErrorSpec(0.01f));
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
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
