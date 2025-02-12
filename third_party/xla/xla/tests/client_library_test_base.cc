/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/tests/client_library_test_base.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/execution_options_util.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace {

// Name of the interpreter backend.
constexpr char kInterpreter[] = "interpreter";

// Wrapper function that creates a nicer error message (than a bare
// value()) if the platform we intend to test is not available.
LocalClient* GetOrCreateLocalClientOrDie(
    const LocalClientOptions& client_options) {
  absl::StatusOr<LocalClient*> result =
      ClientLibrary::GetOrCreateLocalClient(client_options);
  TF_CHECK_OK(result.status()) << " could not create local client for testing";
  return result.value();
}

// Helper functions to get the reference platform.
se::Platform* GetReferencePlatform() {
  auto result = PlatformUtil::GetPlatform(kInterpreter);
  TF_CHECK_OK(result.status()) << "could not get interpreter platform";
  return result.value();
}

}  // namespace

ClientLibraryTestBase::ClientLibraryTestBase(
    se::Platform* platform, const LocalClientOptions& client_options)
    : client_(GetOrCreateLocalClientOrDie(client_options)),
      execution_options_(CreateDefaultExecutionOptions()) {
  CHECK_EQ(platform, client_options.platform());

  LocalClientOptions ref_options;
  ref_options.set_platform(GetReferencePlatform());
  ref_client_ = GetOrCreateLocalClientOrDie(ref_options);

  // Disabling constant_folding so that tests (usually written using Constants)
  // will exercise the intended code paths, instead of being constant folded.
  //
  // TODO(b/38354253): Constant folding is currently disabled. Change tests to
  // use Parameters instead of Constants, and re-enable constant folding by
  // default.
  execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
      "constant_folding");

  execution_options_.mutable_debug_options()
      ->set_xla_hlo_evaluator_use_fast_path(true);
}

ClientLibraryTestBase::ClientLibraryTestBase(se::Platform* platform)
    : execution_options_(CreateDefaultExecutionOptions()) {
  LocalClientOptions default_options;
  default_options.set_platform(platform);
  client_ = GetOrCreateLocalClientOrDie(default_options);

  LocalClientOptions ref_options;
  ref_options.set_platform(GetReferencePlatform());
  ref_client_ = GetOrCreateLocalClientOrDie(ref_options);

  execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
      "constant_folding");

  execution_options_.mutable_debug_options()
      ->set_xla_hlo_evaluator_use_fast_path(true);
}

std::string ClientLibraryTestBase::SuiteName() const {
  return ::testing::UnitTest::GetInstance()
      ->current_test_info()
      ->test_suite_name();
}

std::string ClientLibraryTestBase::TestName() const {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

absl::StatusOr<std::unique_ptr<GlobalData>> ClientLibraryTestBase::Execute(
    XlaBuilder* builder, absl::Span<GlobalData* const> arguments) {
  // Build the computation, as a convenience.
  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  return client_->Execute(computation, arguments, &execution_options_);
}

absl::StatusOr<Literal> ClientLibraryTestBase::ExecuteAndTransfer(
    const XlaComputation& computation, absl::Span<GlobalData* const> arguments,
    const Shape* shape_with_output_layout) {
  ExecutionOptions execution_options = execution_options_;
  if (shape_with_output_layout != nullptr) {
    *execution_options.mutable_shape_with_output_layout() =
        shape_with_output_layout->ToProto();
  }
  return client_->ExecuteAndTransfer(computation, arguments,
                                     &execution_options);
}

absl::StatusOr<Literal> ClientLibraryTestBase::ExecuteAndTransfer(
    XlaBuilder* builder, absl::Span<GlobalData* const> arguments,
    const Shape* shape_with_output_layout) {
  // Build the computation, as a convenience.
  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  return ExecuteAndTransfer(computation, arguments, shape_with_output_layout);
}

absl::StatusOr<Literal> ClientLibraryTestBase::ExecuteAndTransferReference(
    const XlaComputation& computation, absl::Span<GlobalData* const> arguments,
    const Shape* shape_with_output_layout) {
  ExecutionOptions execution_options = execution_options_;
  if (shape_with_output_layout != nullptr) {
    *execution_options.mutable_shape_with_output_layout() =
        shape_with_output_layout->ToProto();
  }
  execution_options.clear_device_handles();
  return ref_client_->ExecuteAndTransfer(computation, arguments,
                                         &execution_options);
}

std::string ClientLibraryTestBase::ExecuteToString(
    XlaBuilder* builder, absl::Span<GlobalData* const> arguments) {
  auto computation_status = builder->Build();
  if (!computation_status.ok()) {
    return computation_status.status().ToString();
  }
  auto computation = std::move(computation_status).value();

  auto result =
      client_->ExecuteAndTransfer(computation, arguments, &execution_options_);
  if (!result.ok()) {
    return result.status().ToString();
  } else {
    return result.value().ToString();
  }
}

void ClientLibraryTestBase::ComputeAndCompareR1(
    XlaBuilder* builder, const tsl::core::Bitmap& expected,
    absl::Span<GlobalData* const> arguments) {
  Literal expected_literal = LiteralUtil::CreateR1(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

void ClientLibraryTestBase::ComputeAndCompareLiteral(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments, const Shape* shape_with_layout) {
  EXPECT_IS_OK(ComputeAndCompareLiteralWithStatus(builder, expected, arguments,
                                                  shape_with_layout));
}

void ClientLibraryTestBase::ComputeAndCompareLiteral(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error,
    const Shape* shape_with_layout) {
  EXPECT_IS_OK(ComputeAndCompareLiteralWithStatus(builder, expected, arguments,
                                                  error, shape_with_layout));
}

absl::Status
ClientLibraryTestBase::ComputeAndCompareLiteralWithAllOutputLayouts(
    const xla::XlaComputation& computation, const Literal& expected,
    absl::Span<GlobalData* const> arguments,
    const std::function<void(const Literal& actual,
                             const std::string& error_message)>&
        verify_output) {
  // Try with no layout requirement.
  TF_ASSIGN_OR_RETURN(auto actual, ExecuteAndTransfer(computation, arguments));
  verify_output(actual, "");

  // Try with all output layouts.
  std::vector<int64_t> minor_to_major(expected.shape().rank());
  std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
  do {
    auto layout = ShapeUtil::MakeShapeWithDenseLayout(
        expected.shape().element_type(), expected.shape().dimensions(),
        minor_to_major);
    TF_ASSIGN_OR_RETURN(auto actual,
                        ExecuteAndTransfer(computation, arguments, &layout));
    verify_output(actual,
                  absl::StrCat("Test with output layout: ",
                               ShapeUtil::HumanStringWithLayout(layout)));
  } while (std::next_permutation(minor_to_major.begin(), minor_to_major.end()));
  return absl::OkStatus();
}

absl::Status ClientLibraryTestBase::ComputeAndCompareLiteralWithAllInputLayouts(
    const xla::XlaComputation& computation, const Literal& /*expected*/,
    absl::Span<GlobalData* const> arguments,
    const std::function<void(const Literal& actual,
                             const std::string& error_message)>& verify_output,
    const Shape* output_with_layout) {
  std::vector<GlobalData*> arguments_with_layout;
  std::vector<std::string> layout_strings;
  // This is a recursive function. It's an std::function instead of a lambda
  // because it needs to capture itself. The index is the index of the argument
  // to try all layouts for.
  std::function<absl::Status(int64_t)> choose;
  choose = [&, this](int64_t index) -> absl::Status {
    if (index < arguments.size()) {
      // Try out all layouts for the operand.
      TF_ASSIGN_OR_RETURN(auto literal,
                          client_->Transfer(*arguments[index], nullptr));
      // Skip tuples because they don't have a rank.
      if (literal.shape().IsTuple()) {
        layout_strings.push_back(
            ShapeUtil::HumanStringWithLayout(literal.shape()));
        arguments_with_layout.push_back(arguments[index]);
        TF_RETURN_IF_ERROR(choose(index + 1));
        arguments_with_layout.pop_back();
        layout_strings.pop_back();
        return absl::OkStatus();
      }

      std::vector<int64_t> minor_to_major(literal.shape().rank());
      std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
      do {
        auto literal_relayout =
            literal.Relayout(LayoutUtil::MakeLayout(minor_to_major));
        layout_strings.push_back(
            ShapeUtil::HumanStringWithLayout(literal_relayout.shape()));
        TF_ASSIGN_OR_RETURN(auto data,
                            client_->TransferToServer(literal_relayout));
        arguments_with_layout.push_back(data.get());
        TF_RETURN_IF_ERROR(choose(index + 1));
        arguments_with_layout.pop_back();
        layout_strings.pop_back();
      } while (
          std::next_permutation(minor_to_major.begin(), minor_to_major.end()));
      return absl::OkStatus();
    }

    // Every argument has an assigned layout.
    TF_ASSIGN_OR_RETURN(
        auto actual,
        ExecuteAndTransfer(computation,
                           absl::Span<GlobalData* const>(arguments_with_layout),
                           output_with_layout));
    std::string error_message = "Test with input layouts: ";
    for (const auto& str : layout_strings) {
      absl::StrAppend(&error_message, str, " ");
    }
    verify_output(actual, error_message);
    return absl::OkStatus();
  };

  return choose(0);
}

absl::StatusOr<Literal> ClientLibraryTestBase::ComputeAndTransfer(
    XlaBuilder* builder, absl::Span<GlobalData* const> arguments_passed_in,
    const Shape* shape_with_layout) {
  std::vector<GlobalData*> arguments(arguments_passed_in.begin(),
                                     arguments_passed_in.end());

  // Transfer and use elements of arguments_, if the AddParam() API was used.
  std::vector<std::unique_ptr<GlobalData>> owning_arguments;
  if (!arguments_.empty()) {
    CHECK(arguments.empty());
    for (const auto& argument : arguments_) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<GlobalData> owned_argument,
          client_->TransferToServer(MaybeConvertLiteralToTestType(argument)));
      owning_arguments.push_back(std::move(owned_argument));
      arguments.push_back(owning_arguments.back().get());
    }
  }

  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  return ExecuteAndTransfer(computation, arguments, shape_with_layout);
}

absl::Status ClientLibraryTestBase::ComputeAndCompareLiteralWithStatus(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments_passed_in,
    const Shape* shape_with_layout) {
  std::vector<GlobalData*> arguments(arguments_passed_in.begin(),
                                     arguments_passed_in.end());

  // Transfer and use elements of arguments_, if the AddParam() API was used.
  std::vector<std::unique_ptr<GlobalData>> owning_arguments;
  if (!arguments_.empty()) {
    CHECK(arguments.empty());
    for (const auto& argument : arguments_) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<GlobalData> owned_argument,
          client_->TransferToServer(MaybeConvertLiteralToTestType(argument)));
      owning_arguments.push_back(std::move(owned_argument));
      arguments.push_back(owning_arguments.back().get());
    }
  }

  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  if (ShapeUtil::ElementIsFloating(expected.shape()) ||
      ShapeUtil::ElementIsComplex(expected.shape())) {
    LOG(WARNING) << "performing exact comparison of floating point numbers";
  }
  // We allow using a float expected literal for non float outputs. In this
  // case, we need to convert the expected literal to test_type_.
  const Literal* expected_ptr = &expected;
  Literal converted_expected;
  Shape layout_shape;
  if (test_type_ != F32) {
    converted_expected = MaybeConvertLiteralToTestType(expected);
    expected_ptr = &converted_expected;
    if (shape_with_layout != nullptr) {
      layout_shape = *shape_with_layout;
      ShapeUtil::ForEachMutableSubshape(
          &layout_shape, [&](Shape* subshape, const ShapeIndex& /*index*/) {
            if (subshape->element_type() == F32) {
              subshape->set_element_type(test_type_);
            }
          });
      shape_with_layout = &layout_shape;
    }
  }
  auto expect_equal = [&](const Literal& actual,
                          const std::string& error_message) {
    EXPECT_TRUE(LiteralTestUtil::Equal(*expected_ptr, actual)) << error_message;
  };
  if (execution_options_.debug_options().xla_test_all_output_layouts()) {
    return ComputeAndCompareLiteralWithAllOutputLayouts(
        computation, *expected_ptr, arguments, expect_equal);
  }
  if (execution_options_.debug_options().xla_test_all_input_layouts()) {
    return ComputeAndCompareLiteralWithAllInputLayouts(
        computation, *expected_ptr, arguments, expect_equal, shape_with_layout);
  }
  TF_ASSIGN_OR_RETURN(auto actual, ExecuteAndTransfer(computation, arguments,
                                                      shape_with_layout));
  EXPECT_TRUE(LiteralTestUtil::Equal(*expected_ptr, actual));
  return absl::OkStatus();
}

absl::Status ClientLibraryTestBase::ComputeAndCompareLiteralWithStatus(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments_passed_in, ErrorSpec error,
    const Shape* shape_with_layout) {
  std::vector<GlobalData*> arguments(arguments_passed_in.begin(),
                                     arguments_passed_in.end());

  // Transfer and use elements of arguments_, if the AddParam() API was used.
  std::vector<std::unique_ptr<GlobalData>> owning_arguments;
  if (!arguments_.empty()) {
    CHECK(arguments.empty());
    for (const auto& argument : arguments_) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<GlobalData> owned_argument,
          client_->TransferToServer(MaybeConvertLiteralToTestType(argument)));
      owning_arguments.push_back(std::move(owned_argument));
      arguments.push_back(owning_arguments.back().get());
    }
  }

  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  // We allow using a float expected literal for a non float outputs. In this
  // case, we need to convert the expected literal to type_test_.
  const Literal* expected_ptr = &expected;
  Literal converted_expected;
  Shape layout_shape;
  if (test_type_ != F32) {
    converted_expected = MaybeConvertLiteralToTestType(expected);
    expected_ptr = &converted_expected;
    if (shape_with_layout != nullptr) {
      layout_shape = *shape_with_layout;
      ShapeUtil::ForEachMutableSubshape(
          &layout_shape, [&](Shape* subshape, const ShapeIndex& /*index*/) {
            if (subshape->element_type() == F32) {
              subshape->set_element_type(test_type_);
            }
          });
      shape_with_layout = &layout_shape;
    }
  }
  auto expect_near = [&](const Literal& actual,
                         const std::string& error_message) {
    EXPECT_TRUE(LiteralTestUtil::Near(*expected_ptr, actual, error))
        << error_message;
  };
  if (execution_options_.debug_options().xla_test_all_output_layouts()) {
    return ComputeAndCompareLiteralWithAllOutputLayouts(
        computation, *expected_ptr, arguments, expect_near);
  }
  if (execution_options_.debug_options().xla_test_all_input_layouts()) {
    return ComputeAndCompareLiteralWithAllInputLayouts(
        computation, *expected_ptr, arguments, expect_near, shape_with_layout);
  }
  TF_ASSIGN_OR_RETURN(auto actual, ExecuteAndTransfer(computation, arguments,
                                                      shape_with_layout));
  EXPECT_TRUE(LiteralTestUtil::Near(*expected_ptr, actual, error));
  return absl::OkStatus();
}

void ClientLibraryTestBase::ComputeAndCompareR1U8(
    XlaBuilder* builder, absl::string_view expected,
    absl::Span<GlobalData* const> arguments) {
  auto actual_status = ExecuteAndTransfer(builder, arguments);
  EXPECT_IS_OK(actual_status.status());
  if (!actual_status.ok()) {
    return;
  }
  auto actual = std::move(actual_status).value();

  // Turn the expected value into a literal.
  Literal expected_literal = LiteralUtil::CreateR1U8(expected);

  VLOG(1) << "expected: " << expected_literal.ToString();
  VLOG(1) << "actual:   " << actual.ToString();

  EXPECT_EQ(expected, actual.GetR1U8AsString());
}

void ClientLibraryTestBase::ComputeAndCompareTuple(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments) {
  auto actual_status = ExecuteAndTransfer(builder, arguments);
  EXPECT_IS_OK(actual_status.status());
  if (!actual_status.ok()) {
    return;
  }
  auto actual = std::move(actual_status).value();
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

void ClientLibraryTestBase::ComputeAndCompareTuple(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
  auto actual_status = ExecuteAndTransfer(builder, arguments);
  EXPECT_IS_OK(actual_status.status());
  if (!actual_status.ok()) {
    return;
  }
  auto actual = std::move(actual_status).value();
  EXPECT_TRUE(LiteralTestUtil::Near(expected, actual, error));
}

void ClientLibraryTestBase::ComputeAndCompare(
    XlaBuilder* builder, absl::Span<const Literal> arguments) {
  auto status_or_data = ComputeValueAndReference(builder, arguments);
  EXPECT_IS_OK(status_or_data);
  if (!status_or_data.ok()) {
    return;
  }
  Literal reference, result;
  std::tie(reference, result) = std::move(status_or_data).value();
  EXPECT_TRUE(LiteralTestUtil::Equal(reference, result));
}

void ClientLibraryTestBase::ComputeAndCompare(
    XlaBuilder* builder, absl::Span<const Literal> arguments, ErrorSpec error) {
  auto status_or_data = ComputeValueAndReference(builder, arguments);
  EXPECT_IS_OK(status_or_data);
  if (!status_or_data.ok()) {
    return;
  }
  Literal reference, result;
  std::tie(reference, result) = std::move(status_or_data).value();
  EXPECT_TRUE(LiteralTestUtil::Near(reference, result, error));
}

absl::StatusOr<std::pair<Literal, Literal>>
ClientLibraryTestBase::ComputeValueAndReference(
    XlaBuilder* builder, absl::Span<const Literal> arguments) {
  // Transfer the arguments to the executor service. We put the unique_ptr's
  // into a vector to keep the data alive on the service until the end of this
  // function.
  std::vector<std::unique_ptr<GlobalData>> argument_data;
  std::vector<std::unique_ptr<GlobalData>> ref_argument_data;

  // Use `arguments_` if the AddParam() API was used.  Otherwise, use
  // plain `arguments`.
  if (!arguments_.empty()) {
    CHECK_EQ(arguments.size(), 0);
    arguments = arguments_;
  }

  for (const auto& arg : arguments) {
    TF_ASSIGN_OR_RETURN(auto data, client_->TransferToServer(arg.Clone()));
    TF_ASSIGN_OR_RETURN(auto ref_data, ref_client_->TransferToServer(arg));
    argument_data.push_back(std::move(data));
    ref_argument_data.push_back(std::move(ref_data));
  }

  // Create raw pointers to the GlobalData for the rest of the call stack.
  std::vector<GlobalData*> argument_data_ptr;
  std::transform(
      argument_data.begin(), argument_data.end(),
      std::back_inserter(argument_data_ptr),
      [](const std::unique_ptr<GlobalData>& data) { return data.get(); });
  std::vector<GlobalData*> ref_argument_data_ptr;
  std::transform(
      ref_argument_data.begin(), ref_argument_data.end(),
      std::back_inserter(ref_argument_data_ptr),
      [](const std::unique_ptr<GlobalData>& data) { return data.get(); });

  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());

  TF_ASSIGN_OR_RETURN(auto result,
                      ExecuteAndTransfer(computation, argument_data_ptr));

  TF_ASSIGN_OR_RETURN(auto reference, ExecuteAndTransferReference(
                                          computation, ref_argument_data_ptr));

  return std::make_pair(std::move(reference), std::move(result));
}

XlaComputation ClientLibraryTestBase::CreateScalarReluF32() {
  XlaBuilder builder("relu");
  auto shape = ShapeUtil::MakeShape(F32, {});
  auto z_value = Parameter(&builder, 0, shape, "z_value");
  auto zero = ConstantR0<float>(&builder, 0.0f);
  Max(z_value, zero);
  auto computation_status = builder.Build();
  TF_CHECK_OK(computation_status.status());
  return std::move(computation_status).value();
}

XlaComputation ClientLibraryTestBase::CreateScalarMax() {
  XlaBuilder builder("max");
  auto shape = ShapeUtil::MakeShape(test_type_, {});
  auto x = Parameter(&builder, 0, shape, "x");
  auto y = Parameter(&builder, 1, shape, "y");
  Max(x, y);
  auto computation_status = builder.Build();
  TF_CHECK_OK(computation_status.status());
  return std::move(computation_status).value();
}

std::unique_ptr<Array2D<float>> ClientLibraryTestBase::CreatePatternedMatrix(
    int rows, int cols, float offset) {
  auto array = std::make_unique<Array2D<float>>(rows, cols);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
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
  auto array = std::make_unique<Array2D<float>>(rows_padded, cols_padded, 0.0);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      (*array)(row, col) = col + (row * 1000.0f);
    }
  }
  return array;
}

XlaOp ClientLibraryTestBase::AddParam(const Literal& argument,
                                      XlaBuilder* builder) {
  arguments_.push_back(argument.Clone());
  return Parameter(builder, /*parameter_number=*/arguments_.size() - 1,
                   MaybeConvertShapeToTestType(argument.shape()), "");
}

XlaOp ClientLibraryTestBase::CreateConstantFromLiteral(const Literal& literal,
                                                       XlaBuilder* builder) {
  return ConstantLiteral(builder, MaybeConvertLiteralToTestType(literal));
}

absl::StatusOr<std::unique_ptr<GlobalData>>
ClientLibraryTestBase::CreateParameterAndTransferLiteral(
    int64_t parameter_number, const Literal& literal, const std::string& name,
    XlaBuilder* builder, XlaOp* data_handle) {
  return CreateParameterAndTransferLiteral(parameter_number, literal, name,
                                           nullptr, builder, data_handle);
}

Shape ClientLibraryTestBase::MaybeConvertShapeToTestType(const Shape& shape) {
  if (test_type_ == F32) {
    return shape;
  }
  Shape new_shape = shape;
  ShapeUtil::ForEachMutableSubshape(
      &new_shape, [test_type = test_type_](Shape* subshape, const ShapeIndex&) {
        if (subshape->element_type() == F32) {
          subshape->set_element_type(test_type);
        }
      });
  return new_shape;
}

Literal ClientLibraryTestBase::MaybeConvertLiteralToTestType(
    const Literal& literal) {
  switch (test_type_) {
    case BF16:
      return LiteralUtil::ConvertF32ToBF16(literal);
    case F32:
      return literal.Clone();
    case F8E5M2:
      return LiteralUtil::ConvertF32ToF8E5M2(literal);
    case F8E4M3FN:
      return LiteralUtil::ConvertF32ToF8E4M3FN(literal);
    default:
      LOG(FATAL) << "Unsupported test type: " << test_type_;
  }
}

absl::StatusOr<std::unique_ptr<GlobalData>>
ClientLibraryTestBase::CreateParameterAndTransferLiteral(
    int64_t parameter_number, const Literal& literal, const std::string& name,
    const DeviceHandle* device_handle, XlaBuilder* builder,
    XlaOp* data_handle) {
  Literal param_literal = MaybeConvertLiteralToTestType(literal);
  TF_ASSIGN_OR_RETURN(auto data,
                      client_->TransferToServer(param_literal, device_handle));
  *data_handle =
      Parameter(builder, parameter_number, param_literal.shape(), name);
  return data;
}

}  // namespace xla
