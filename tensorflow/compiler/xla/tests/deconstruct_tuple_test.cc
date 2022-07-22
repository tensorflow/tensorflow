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
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

using ::testing::ContainsRegex;
using ::testing::HasSubstr;

class DeconstructTupleTest : public ClientLibraryTestBase {
 protected:
  // Build and execute the given computation then verify the results can be
  // transferred from the device successfully.
  std::unique_ptr<GlobalData> ExecuteAndCheckTransfer(
      XlaBuilder* builder, absl::Span<GlobalData* const> arguments) {
    XlaComputation computation = builder->Build().value();
    auto global_data =
        client_->Execute(computation, arguments, &execution_options_).value();
    TF_CHECK_OK(client_->Transfer(*global_data).status());
    return global_data;
  }
};

TEST_F(DeconstructTupleTest, DeconstructTuple) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status.ok());

  // Try copying the elements back and comparing it
  auto handles = std::move(result_status).value();
  Literal literal;
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles[0]));
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, literal);
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles[1]));
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, literal);
}

TEST_F(DeconstructTupleTest, DeconstructTupleTwice) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status1 = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status1.ok());
  auto result_status2 = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status2.ok());

  auto handles1 = std::move(result_status1).value();
  auto handles2 = std::move(result_status2).value();

  Literal literal;
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles1[0]));
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, literal);
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles1[1]));
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, literal);

  handles1[0].reset();
  handles1[1].reset();

  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles2[0]));
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, literal);
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles2[1]));
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, literal);
}

XLA_TEST_F(DeconstructTupleTest, DeconstructTupleRepeatedElement) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2, const2, const1});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status.ok());

  // Verify the returned GlobalDataHandle arrays have repeated elements like the
  // tuple does. That is, in the returned vector of handles, handle[0] should be
  // the same as handle[3] and handle[1] should be the same as handle[2].
  auto handles = std::move(result_status).value();

  Literal literal;
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles[0]));
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, literal);
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles[1]));
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, literal);
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles[2]));
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, literal);
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles[3]));
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, literal);
}

TEST_F(DeconstructTupleTest, DeconstructTupleThenDeallocate) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {const1, const2, const1});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status.ok());
  auto handles = std::move(result_status).value();

  // Deallocate the tuple, then try copying the elements back. The elements
  // should not have been deallocated because of reference counting.
  global_data.reset();

  Literal literal;
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles[0]));
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, literal);
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles[1]));
  LiteralTestUtil::ExpectR1Equal<float>({2.0, 4.0, 6.0, 8.0}, literal);
  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles[2]));
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, literal);

  /// Try deallocating one of the repeated elements, then copy
  handles[0].reset();

  TF_ASSERT_OK_AND_ASSIGN(literal, client_->Transfer(*handles[2]));
  LiteralTestUtil::ExpectR1Equal<float>({1.0, 2.0, 3.0, 4.0}, literal);
}

TEST_F(DeconstructTupleTest, DeconstructNonTuple) {
  XlaBuilder builder(TestName());
  ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              ContainsRegex("global data handle .* is not a tuple"));
}

XLA_TEST_F(DeconstructTupleTest, DeconstructTupleFromParam) {
  XlaBuilder builder(TestName());
  Literal param0_literal = LiteralUtil::CreateR1<float>({3.14f, -100.25f});
  std::unique_ptr<GlobalData> param0_data =
      client_->TransferToServer(param0_literal).value();
  auto p = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {2}), "param0");
  Tuple(&builder, {p});
  auto global_data = ExecuteAndCheckTransfer(&builder, {param0_data.get()});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_TRUE(result_status.ok());
  auto handles = std::move(result_status).value();
  EXPECT_NE(handles[0]->handle().handle(), param0_data->handle().handle());
}

XLA_TEST_F(DeconstructTupleTest, DeconstructNestedTuple) {
  XlaBuilder builder(TestName());
  auto const1 = ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto const2 = ConstantR1<float>(&builder, {2.0, 4.0, 6.0, 8.0});
  Tuple(&builder, {Tuple(&builder, {const1, const2}), const1});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  auto result_status = client_->DeconstructTuple(*global_data);
  EXPECT_FALSE(result_status.ok());
  EXPECT_THAT(result_status.status().error_message(),
              HasSubstr("Deconstructing nested tuples is not implemented"));
}

}  // namespace
}  // namespace xla
