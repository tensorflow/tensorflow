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

#include "tensorflow/compiler/xla/tests/client_library_test_base.h"

#include <string>

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace {
// Wrapper function that creates a nicer error message (than a bare
// ValueOrDie()) if the platform we intend to test is not available.
Client* GetOrCreateLocalClientOrDie(se::Platform* platform) {
  StatusOr<Client*> result = ClientLibrary::GetOrCreateLocalClient(platform);
  TF_CHECK_OK(result.status()) << "could not create local client for testing";
  return result.ValueOrDie();
}
}  // namespace

ClientLibraryTestBase::ClientLibraryTestBase(se::Platform* platform)
    : client_(GetOrCreateLocalClientOrDie(platform)) {
  *(execution_options_.mutable_debug_options()) =
      legacy_flags::GetDebugOptionsFromFlags();

  // Disabling constant_folding so that tests (usually written using Constants)
  // will exercise the intended code paths, instead of being constant folded.
  //
  // TODO(b/38354253): Constant folding is currently disabled. Change tests to
  // use Parameters instead of Constants, and re-enable constant folding by
  // default.
  execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
      "constant_folding");
}

string ClientLibraryTestBase::TestName() const {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

StatusOr<std::unique_ptr<GlobalData>> ClientLibraryTestBase::Execute(
    ComputationBuilder* builder,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  // Build the computation, as a convenience.
  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  return client_->Execute(computation, arguments, &execution_options_);
}

StatusOr<std::unique_ptr<Literal>> ClientLibraryTestBase::ExecuteAndTransfer(
    ComputationBuilder* builder,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments,
    const Shape* shape_with_output_layout) {
  // Build the computation, as a convenience.
  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());

  ExecutionOptions execution_options = execution_options_;
  if (shape_with_output_layout != nullptr) {
    *execution_options.mutable_shape_with_output_layout() =
        *shape_with_output_layout;
  }
  return client_->ExecuteAndTransfer(computation, arguments,
                                     &execution_options);
}

std::unique_ptr<GlobalData> ClientLibraryTestBase::ExecuteOrDie(
    ComputationBuilder* builder,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  return Execute(builder, arguments).ConsumeValueOrDie();
}

std::unique_ptr<Literal> ClientLibraryTestBase::ExecuteAndTransferOrDie(
    ComputationBuilder* builder,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  return ExecuteAndTransfer(builder, arguments).ConsumeValueOrDie();
}

string ClientLibraryTestBase::ExecuteToString(
    ComputationBuilder* builder,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  StatusOr<Computation> computation_status = builder->Build();
  if (!computation_status.ok()) {
    return computation_status.status().ToString();
  }
  Computation computation = computation_status.ConsumeValueOrDie();

  auto result =
      client_->ExecuteAndTransfer(computation, arguments, &execution_options_);
  if (!result.ok()) {
    return result.status().ToString();
  } else {
    return LiteralUtil::ToString(*result.ValueOrDie());
  }
}

void ClientLibraryTestBase::ComputeAndCompareR1(
    ComputationBuilder* builder, const tensorflow::core::Bitmap& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal = LiteralUtil::CreateR1(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

void ClientLibraryTestBase::ComputeAndCompareLiteral(
    ComputationBuilder* builder, const Literal& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments,
    const Shape* shape_with_layout) {
  EXPECT_IS_OK(ComputeAndCompareLiteralWithStatus(builder, expected, arguments,
                                                  shape_with_layout));
}

void ClientLibraryTestBase::ComputeAndCompareLiteral(
    ComputationBuilder* builder, const Literal& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error,
    const Shape* shape_with_layout) {
  EXPECT_IS_OK(ComputeAndCompareLiteralWithStatus(builder, expected, arguments,
                                                  error, shape_with_layout));
}

tensorflow::Status ClientLibraryTestBase::ComputeAndCompareLiteralWithStatus(
    ComputationBuilder* builder, const Literal& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments,
    const Shape* shape_with_layout) {
  TF_ASSIGN_OR_RETURN(
      auto actual, ExecuteAndTransfer(builder, arguments, shape_with_layout));
  if (ShapeUtil::ElementIsFloating(expected.shape())) {
    LOG(WARNING) << "performing exact comparison of floating point numbers";
  } else {
    TF_RET_CHECK(ShapeUtil::ElementIsIntegral(expected.shape()) ||
                 expected.shape().element_type() == PRED);
  }
  LiteralTestUtil::ExpectEqual(expected, *actual);
  return tensorflow::Status::OK();
}

tensorflow::Status ClientLibraryTestBase::ComputeAndCompareLiteralWithStatus(
    ComputationBuilder* builder, const Literal& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error,
    const Shape* shape_with_layout) {
  TF_ASSIGN_OR_RETURN(
      auto actual, ExecuteAndTransfer(builder, arguments, shape_with_layout));
  TF_RET_CHECK(ShapeUtil::ElementIsFloating(expected.shape()));
  LiteralTestUtil::ExpectNear(expected, *actual, error);
  return tensorflow::Status::OK();
}

void ClientLibraryTestBase::ComputeAndCompareR1U8(
    ComputationBuilder* builder, tensorflow::StringPiece expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  auto actual_status = ExecuteAndTransfer(builder, arguments);
  EXPECT_IS_OK(actual_status.status());
  if (!actual_status.ok()) {
    return;
  }
  auto actual = actual_status.ConsumeValueOrDie();

  // Turn the expected value into a literal.
  std::unique_ptr<Literal> expected_literal = LiteralUtil::CreateR1U8(expected);

  VLOG(1) << "expected: " << LiteralUtil::ToString(*expected_literal);
  VLOG(1) << "actual:   " << LiteralUtil::ToString(*actual);

  EXPECT_EQ(expected, actual->u8s_string());
}

void ClientLibraryTestBase::ComputeAndCompareTuple(
    ComputationBuilder* builder, const Literal& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  auto actual_status = ExecuteAndTransfer(builder, arguments);
  EXPECT_IS_OK(actual_status.status());
  if (!actual_status.ok()) {
    return;
  }
  auto actual = actual_status.ConsumeValueOrDie();
  LiteralTestUtil::ExpectEqualTuple(expected, *actual);
}

void ClientLibraryTestBase::ComputeAndCompareTuple(
    ComputationBuilder* builder, const Literal& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  auto actual_status = ExecuteAndTransfer(builder, arguments);
  EXPECT_IS_OK(actual_status.status());
  if (!actual_status.ok()) {
    return;
  }
  auto actual = actual_status.ConsumeValueOrDie();
  LiteralTestUtil::ExpectNearTuple(expected, *actual, error);
}

Computation ClientLibraryTestBase::CreateScalarRelu() {
  ComputationBuilder builder(client_, "relu");
  auto z_value = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "z_value");
  auto zero = builder.ConstantR0<float>(0.0);
  builder.Max(z_value, zero);
  auto computation_status = builder.Build();
  TF_CHECK_OK(computation_status.status());
  return computation_status.ConsumeValueOrDie();
}

Computation ClientLibraryTestBase::CreateScalarMax() {
  ComputationBuilder builder(client_, "max");
  auto x = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
  builder.Max(x, y);
  auto computation_status = builder.Build();
  TF_CHECK_OK(computation_status.status());
  return computation_status.ConsumeValueOrDie();
}

Computation ClientLibraryTestBase::CreateScalarReluSensitivity() {
  ComputationBuilder builder(client_, "relu_sensitivity");
  auto activation =
      builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "activation");
  auto backprop =
      builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "backprop");
  auto zero = builder.ConstantR0<float>(0.0);
  auto activation_gtz = builder.Gt(activation, zero);
  builder.Select(activation_gtz, /*on_true=*/backprop, /*on_false=*/zero);

  auto computation_status = builder.Build();
  TF_CHECK_OK(computation_status.status());
  return computation_status.ConsumeValueOrDie();
}

std::unique_ptr<Array2D<float>> ClientLibraryTestBase::CreatePatternedMatrix(
    int rows, int cols, float offset) {
  auto array = MakeUnique<Array2D<float>>(rows, cols);
  for (int64 row = 0; row < rows; ++row) {
    for (int64 col = 0; col < cols; ++col) {
      (*array)(row, col) = col + (row * 1000.0f) + offset;
    }
  }
  return array;
}

std::unique_ptr<Array2D<float>>
ClientLibraryTestBase::CreatePatternedMatrixWithZeroPadding(int rows, int cols,
                                                            int rows_padded,
                                                            int cols_padded) {
  CHECK_GE(rows_padded, rows);
  CHECK_GE(cols_padded, cols);
  auto array = MakeUnique<Array2D<float>>(rows_padded, cols_padded, 0.0);
  for (int64 row = 0; row < rows; ++row) {
    for (int64 col = 0; col < cols; ++col) {
      (*array)(row, col) = col + (row * 1000.0f);
    }
  }
  return array;
}

}  // namespace xla
