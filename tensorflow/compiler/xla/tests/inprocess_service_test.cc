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

#include <initializer_list>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

// Tests which exercise the "InProcess" methods of xla::Client. The
// "InProcess" methods require that the client and server share the same
// process.
class InProcessServiceTest : public ClientLibraryTestBase {
 protected:
  std::unique_ptr<GlobalData> ExecuteR2F32Constant(
      std::initializer_list<std::initializer_list<float>> values,
      tensorflow::gtl::ArraySlice<int64> minor_to_major) {
    ComputationBuilder builder(client_, TestName());
    builder.ConstantR2<float>(values);
    auto computation = builder.Build().ConsumeValueOrDie();
    CHECK_EQ(2, minor_to_major.size());

    ExecutionOptions execution_options;
    *execution_options.mutable_shape_with_output_layout() =
        ShapeUtil::MakeShapeWithLayout(
            F32,
            /*dimensions=*/{static_cast<int64>(values.size()),
                            static_cast<int64>(values.begin()->size())},
            minor_to_major);
    return client_->Execute(computation, {}, &execution_options)
        .ConsumeValueOrDie();
  }

  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(InProcessServiceTest, TransferFromServer) {
  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<int32>({1, 42, 5});
  auto computation = builder.Build().ConsumeValueOrDie();

  auto handle = client_->Execute(computation, {}).ConsumeValueOrDie();

  std::vector<int32> result(3, 0);
  ASSERT_IS_OK(client_->TransferInProcess(*handle, result.data()));
  EXPECT_THAT(result, ::testing::ElementsAre(1, 42, 5));
}

XLA_TEST_F(InProcessServiceTest, TransferToServer) {
  std::vector<float> input{1.0f, 2.0f, -42.0f};
  Shape shape = ShapeUtil::MakeShape(F32, {3});
  auto data_handle = client_->TransferToServerInProcess(shape, input.data())
                         .ConsumeValueOrDie();

  ComputationBuilder builder(client_, TestName());
  auto param = builder.Parameter(0, ShapeUtil::MakeShape(F32, {3}), "param");
  builder.Add(param, param);

  ComputeAndCompareR1<float>(&builder, {2.0f, 4.0f, -84.0f},
                             {data_handle.get()}, error_spec_);
}

// TODO(b/28506710): This test case seems not to test inprocess
// methods.
TEST_F(InProcessServiceTest, GetShape) {
  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<int32>({1, 42, 5});
  auto computation = builder.Build().ConsumeValueOrDie();

  auto handle = client_->Execute(computation, {}).ConsumeValueOrDie();

  Shape shape = client_->GetShape(*handle).ConsumeValueOrDie();
  ASSERT_EQ(S32, shape.element_type());
  ASSERT_EQ(1, ShapeUtil::Rank(shape));
  ASSERT_EQ(3, shape.dimensions(0));
}

XLA_TEST_F(InProcessServiceTest, GetShapeOfClientSuppliedArrayRowMajor) {
  std::vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  shape.clear_layout();
  *shape.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  auto handle = client_->TransferToServerInProcess(shape, input.data())
                    .ConsumeValueOrDie();

  Shape shape_returned = client_->GetShape(*handle).ConsumeValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(shape, shape_returned));
}

XLA_TEST_F(InProcessServiceTest, GetShapeOfClientSuppliedArrayColMajor) {
  std::vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  shape.clear_layout();
  *shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  auto handle = client_->TransferToServerInProcess(shape, input.data())
                    .ConsumeValueOrDie();

  Shape shape_returned = client_->GetShape(*handle).ConsumeValueOrDie();
  ASSERT_TRUE(ShapeUtil::Equal(shape, shape_returned));
}

TEST_F(InProcessServiceTest, TransferToServerNoLayout) {
  std::vector<float> input{1.0f, 2.0f, -42.0f};
  Shape shape = ShapeUtil::MakeShape(F32, {3});
  shape.clear_layout();
  auto transfer_status =
      client_->TransferToServerInProcess(shape, input.data());
  ASSERT_EQ(transfer_status.status().code(),
            tensorflow::error::INVALID_ARGUMENT);
}

XLA_TEST_F(InProcessServiceTest, ExecuteRowMajor) {
  auto handle =
      ExecuteR2F32Constant({{1.0, 2.0}, {3.0, 4.0}}, /*minor_to_major=*/{1, 0});

  std::vector<float> result(4, 0.0);
  Shape shape;
  ASSERT_IS_OK(client_->TransferInProcess(*handle, result.data()));

  EXPECT_THAT(result, ::testing::ElementsAre(1.0, 2.0, 3.0, 4.0));
}

XLA_TEST_F(InProcessServiceTest, ExecuteColumnMajor) {
  auto handle =
      ExecuteR2F32Constant({{1.0, 2.0}, {3.0, 4.0}}, /*minor_to_major=*/{0, 1});

  std::vector<float> result(4, 0);
  Shape shape;
  ASSERT_IS_OK(client_->TransferInProcess(*handle, result.data()));

  EXPECT_THAT(result, ::testing::ElementsAre(1.0, 3.0, 2.0, 4.0));
}

XLA_TEST_F(InProcessServiceTest, ExecuteAndReuseDifferentLayouts) {
  // Create arrays on the server which have different layouts. Verify the
  // computation still produces the correct results.
  auto handle_rowmaj =
      ExecuteR2F32Constant({{1.0, 2.0}, {3.0, 4.0}}, /*minor_to_major=*/{1, 0});

  auto handle_colmaj = ExecuteR2F32Constant({{10.0, 20.0}, {30.0, 40.0}},
                                            /*minor_to_major=*/{0, 1});

  ComputationBuilder builder(client_, TestName());
  auto param0 =
      builder.Parameter(0, ShapeUtil::MakeShape(F32, {2, 2}), "param0");
  auto param1 =
      builder.Parameter(1, ShapeUtil::MakeShape(F32, {2, 2}), "param1");
  builder.Add(param0, param1);

  Array2D<float> expected({{11.0, 22.0}, {33.0, 44.0}});
  ComputeAndCompareR2<float>(&builder, expected,
                             {handle_rowmaj.get(), handle_colmaj.get()},
                             error_spec_);
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
