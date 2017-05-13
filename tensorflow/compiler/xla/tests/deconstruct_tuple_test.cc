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
#include <vector>

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class DeconstructTupleTest : public ClientLibraryTestBase {
 protected:
  // Build and execute the given computation then verify the results can be
  // transferred from the device successfully.
  std::unique_ptr<GlobalData> ExecuteAndCheckTransfer(
      ComputationBuilder* builder,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
    Computation computation = builder->Build().ConsumeValueOrDie();
    auto global_data =
        client_->Execute(computation, arguments).ConsumeValueOrDie();
    TF_CHECK_OK(client_->Transfer(*global_data).status());
    return global_data;
  }
};

TEST_F(DeconstructTupleTest, DeconstructTuple) {
  ComputationBuilder builder(client_, TestName());
  auto const1 = builder.ConstantR1<float>({1.0, 2.0, 3.0, 4.0});
  auto const2 = builder.ConstantR1<float>({2.0, 4.0, 6.0, 8.0});
  builder.Tuple({const1, const2});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status.ok());

  // Try copying the elements back and comparing it
  auto handles = result_status.ConsumeValueOrDie();
  std::vector<float> copy(4);
  ASSERT_IS_OK(client_->TransferInProcess(*handles[0], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(1.0, 2.0, 3.0, 4.0));
  ASSERT_IS_OK(client_->TransferInProcess(*handles[1], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(2.0, 4.0, 6.0, 8.0));
}

TEST_F(DeconstructTupleTest, DeconstructTupleTwice) {
  ComputationBuilder builder(client_, TestName());
  auto const1 = builder.ConstantR1<float>({1.0, 2.0, 3.0, 4.0});
  auto const2 = builder.ConstantR1<float>({2.0, 4.0, 6.0, 8.0});
  builder.Tuple({const1, const2});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status1 = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status1.ok());
  auto result_status2 = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status2.ok());

  auto handles1 = result_status1.ConsumeValueOrDie();
  auto handles2 = result_status2.ConsumeValueOrDie();
  std::vector<float> copy(4);

  ASSERT_IS_OK(client_->TransferInProcess(*handles1[0], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(1.0, 2.0, 3.0, 4.0));
  ASSERT_IS_OK(client_->TransferInProcess(*handles1[1], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(2.0, 4.0, 6.0, 8.0));
  handles1[0].reset();
  handles1[1].reset();

  ASSERT_IS_OK(client_->TransferInProcess(*handles2[0], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(1.0, 2.0, 3.0, 4.0));
  ASSERT_IS_OK(client_->TransferInProcess(*handles2[1], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(2.0, 4.0, 6.0, 8.0));
}

XLA_TEST_F(DeconstructTupleTest, DeconstructTupleRepeatedElement) {
  ComputationBuilder builder(client_, TestName());
  auto const1 = builder.ConstantR1<float>({1.0, 2.0, 3.0, 4.0});
  auto const2 = builder.ConstantR1<float>({2.0, 4.0, 6.0, 8.0});
  builder.Tuple({const1, const2, const2, const1});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status.ok());

  // Verify the returned GlobalDataHandle arrays have repeated elements like the
  // tuple does. That is, in the returned vector of handles, handle[0] should be
  // the same as handle[3] and handle[1] should be the same as handle[2].
  auto handles = result_status.ConsumeValueOrDie();

  std::vector<float> copy(4);
  ASSERT_IS_OK(client_->TransferInProcess(*handles[0], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(1.0, 2.0, 3.0, 4.0));
  ASSERT_IS_OK(client_->TransferInProcess(*handles[1], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(2.0, 4.0, 6.0, 8.0));
  ASSERT_IS_OK(client_->TransferInProcess(*handles[2], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(2.0, 4.0, 6.0, 8.0));
  ASSERT_IS_OK(client_->TransferInProcess(*handles[3], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(1.0, 2.0, 3.0, 4.0));
}

TEST_F(DeconstructTupleTest, DeconstructTupleThenDeallocate) {
  ComputationBuilder builder(client_, TestName());
  auto const1 = builder.ConstantR1<float>({1.0, 2.0, 3.0, 4.0});
  auto const2 = builder.ConstantR1<float>({2.0, 4.0, 6.0, 8.0});
  builder.Tuple({const1, const2, const1});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status.ok());
  auto handles = result_status.ConsumeValueOrDie();

  // Deallocate the tuple, then try copying the elements back. The elements
  // should not have been deallocated because of reference counting.
  global_data.reset();

  std::vector<float> copy(4);
  ASSERT_IS_OK(client_->TransferInProcess(*handles[0], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(1.0, 2.0, 3.0, 4.0));
  ASSERT_IS_OK(client_->TransferInProcess(*handles[1], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(2.0, 4.0, 6.0, 8.0));
  ASSERT_IS_OK(client_->TransferInProcess(*handles[2], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(1.0, 2.0, 3.0, 4.0));

  /// Try deallocating one of the repeated elements, then copy
  handles[0].reset();

  ASSERT_IS_OK(client_->TransferInProcess(*handles[2], &copy[0]));
  EXPECT_THAT(copy, ElementsAre(1.0, 2.0, 3.0, 4.0));
}

TEST_F(DeconstructTupleTest, DeconstructNonTuple) {
  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<float>({1.0, 2.0, 3.0, 4.0});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              ContainsRegex("global data handle .* is not a tuple"));
}

XLA_TEST_F(DeconstructTupleTest, DeconstructTupleFromParam) {
  ComputationBuilder builder(client_, TestName());
  std::unique_ptr<Literal> param0_literal =
      LiteralUtil::CreateR1<float>({3.14f, -100.25f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();
  auto p = builder.Parameter(0, ShapeUtil::MakeShape(F32, {2}), "param0");
  builder.Tuple({p});
  auto global_data = ExecuteAndCheckTransfer(&builder, {param0_data.get()});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status.ok());
  auto handles = result_status.ConsumeValueOrDie();
  EXPECT_NE(handles[0]->handle().handle(), param0_data->handle().handle());
}

XLA_TEST_F(DeconstructTupleTest, DeconstructNestedTuple) {
  ComputationBuilder builder(client_, TestName());
  auto const1 = builder.ConstantR1<float>({1.0, 2.0, 3.0, 4.0});
  auto const2 = builder.ConstantR1<float>({2.0, 4.0, 6.0, 8.0});
  builder.Tuple({builder.Tuple({const1, const2}), const1});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("deconstructing nested tuples not yet supported"));
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
