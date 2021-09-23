/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace gpu {

XLA_TEST_F(ClientLibraryTestBase, GemmOnly) {
  Array2D<float> lhs({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Array2D<float> rhs({{10.0f, 11.0f}, {12.0f, 13.0f}});

  auto prim_type = primitive_util::NativeToPrimitiveType<float>();
  Shape lhs_shape =
      ShapeUtil::MakeShape(prim_type, {lhs.height(), lhs.width()});
  Shape rhs_shape =
      ShapeUtil::MakeShape(prim_type, {rhs.height(), rhs.width()});

  TF_ASSERT_OK_AND_ASSIGN(auto lhs_handle,
                          client_->TransferToServer(
                              LiteralUtil::CreateR2FromArray2DWithLayout<float>(
                                  lhs, LayoutUtil::MakeLayout({0, 1}))));
  TF_ASSERT_OK_AND_ASSIGN(auto rhs_handle,
                          client_->TransferToServer(
                              LiteralUtil::CreateR2FromArray2DWithLayout<float>(
                                  rhs, LayoutUtil::MakeLayout({0, 1}))));

  XlaBuilder builder(TestName());
  auto lhs_arg = Parameter(&builder, 0, lhs_shape, "lhs");
  auto rhs_arg = Parameter(&builder, 1, rhs_shape, "rhs");
  Dot(lhs_arg, rhs_arg);
  Array2D<float> expected = Array2D<float>({{34.0f, 37.0f}, {78.0f, 85.0f}});

  ComputeAndCompareR2<float>(&builder, expected,
                             {lhs_handle.get(), rhs_handle.get()},
                             ErrorSpec(1e-6));
};

XLA_TEST_F(ClientLibraryTestBase, GemmBiasOnly) {
  execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
      "layout-assignment");
  execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
      "tiling-assignment");
  // Disable algebraic simplification because the pass may replace a dot
  // instruction with a layout-changing multiplication instruction.
  execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
      "algsimp");

  const int m = 1;
  const int k = 290;
  const int n = 130;

  auto dot_lhs_data = MakeLinspaceArray2D<Eigen::half>(0.0, 1.0, m, k);
  auto dot_rhs_data = MakeLinspaceArray2D<Eigen::half>(0.0, 1.0, k, n);
  auto addend_data = MakeLinspaceArray2D<Eigen::half>(0.0, 1.0, m, n);

  auto transfer_to_server =
      [&](const Array2D<Eigen::half>& data) -> std::unique_ptr<GlobalData> {
    Literal literal = LiteralUtil::CreateR2FromArray2DWithLayout(
        data, LayoutUtil::MakeLayout({1, 0}));
    return client_->TransferToServer(literal).ConsumeValueOrDie();
  };
  auto dot_lhs_handle = transfer_to_server(*dot_lhs_data);
  auto dot_rhs_handle = transfer_to_server(*dot_rhs_data);
  auto addend_handle = transfer_to_server(*addend_data);

  XlaBuilder builder(TestName());
  auto prim_type = primitive_util::NativeToPrimitiveType<Eigen::half>();
  auto result =
      Dot(Parameter(&builder, 0,
                    ShapeUtil::MakeShapeWithLayout(prim_type, {m, k}, {1, 0}),
                    "dot_lhs"),
          Parameter(&builder, 1,
                    ShapeUtil::MakeShapeWithLayout(prim_type, {k, n}, {1, 0}),
                    "dot_rhs"));

  result =
      Add(result,
          Parameter(&builder, 2,
                    ShapeUtil::MakeShapeWithLayout(prim_type, {m, n}, {1, 0}),
                    "addend"));

  std::unique_ptr<Array2D<Eigen::half>> expected =
      ReferenceUtil::ApplyElementwise2D(
          std::plus<Eigen::half>(),
          *ReferenceUtil::MatmulArray2D(*dot_lhs_data, *dot_rhs_data),
          *addend_data);

  std::vector<GlobalData*> args = {dot_lhs_handle.get(), dot_rhs_handle.get(),
                                   addend_handle.get()};
  ComputeAndCompareR2<Eigen::half>(&builder, *expected, args,
                                   ErrorSpec(0.3, 7e-3));
}

}  // namespace gpu
}  // namespace xla
