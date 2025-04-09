/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_TESTS_CLIENT_LIBRARY_TEST_RUNNER_MIXIN_H_
#define XLA_TESTS_CLIENT_LIBRARY_TEST_RUNNER_MIXIN_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/error_spec.h"
#include "xla/execution_options_util.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_utils.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/bitmap.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

template <typename T>
constexpr inline bool is_floating_or_complex_v =
    std::disjunction_v<is_specialized_floating_point<T>, is_complex<T>>;

// This class is designed to be used as a mixin for tests that formerly extended
// ClientLibraryTestBase. It is a partial re-implementation of
// ClientLibraryTestBase, but explicitly backed by an implementation of
// HloRunnerAgnosticTestBase. It also requires the use of
// HloRunnerAgnosticReferenceMixin, as it relies on RunAndCompare functionality.
//
// This class serves as a crucial bridging mechanism during the migration
// towards a single test base class and a migration away from stream executor.
//
// The reliance on templates / implementation as a mixin lets us switch out the
// underlying test base class and reference runner implementations incrementally
// and on a per-test basis instead of all at once.
template <typename T>
class ClientLibraryTestRunnerMixin : public T {
  static_assert(
      std::is_base_of_v<HloRunnerAgnosticTestBase, T> &&
          T::has_reference_runner_mixin::value,
      "Mixin must be used with a subclass of HloRunnerAgnosticTestBase and "
      "HloRunnerAgnosticReferenceMixin.");

  template <typename NativeT>
  void CheckErrorSpec(std::optional<ErrorSpec> error) {
    if (error.has_value()) {
      CHECK(is_floating_or_complex_v<NativeT>)
          << "Float or complex type required when specifying an ErrorSpec";
    }
  }

 protected:
  template <typename... BaseArgs>
  explicit ClientLibraryTestRunnerMixin(BaseArgs&&... base_args)
      : T(std::forward<BaseArgs>(base_args)...) {}
  ~ClientLibraryTestRunnerMixin() override = default;

  // The float type used in this test.
  PrimitiveType FloatType() const { return test_type_; }
  void set_float_type(PrimitiveType type) { test_type_ = type; }

  absl::StatusOr<Literal> ExecuteAndTransfer(
      const XlaComputation& computation,
      const absl::Span<Literal* const> arguments,
      const Shape* const shape_with_output_layout = nullptr) {
    ExecutionOptions execution_options = execution_options_;
    if (shape_with_output_layout != nullptr) {
      *execution_options.mutable_shape_with_output_layout() =
          shape_with_output_layout->ToProto();
    }
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        BuildAndVerifyHloModule(computation, &execution_options));
    return this->Execute(std::move(module), arguments);
  }

  absl::StatusOr<Literal> ExecuteAndTransfer(
      XlaBuilder* const builder, const absl::Span<Literal* const> arguments,
      const Shape* shape_with_output_layout = nullptr) {
    // Build the computation, as a convenience.
    TF_ASSIGN_OR_RETURN(XlaComputation computation, builder->Build());
    return ExecuteAndTransfer(std::move(computation), arguments,
                              shape_with_output_layout);
  }

  // Run a computation and return its value as a string. If an error
  // occurs, then instead return the error as a string.
  std::string ExecuteToString(XlaBuilder* const builder,
                              const absl::Span<Literal* const> arguments) {
    const absl::StatusOr<Literal> result =
        ExecuteAndTransfer(builder, arguments);
    if (!result.ok()) {
      return result.status().ToString();
    } else {
      return result.value().ToString();
    }
  }

  // Compare with reference.
  // Side effect: EXPECT_OK
  void ComputeAndCompare(XlaBuilder* const builder,
                         const absl::Span<Literal* const> arguments,
                         const std::optional<ErrorSpec> error = std::nullopt) {
    TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder->Build());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            BuildAndVerifyHloModule(computation));
    EXPECT_TRUE(this->RunAndCompare(std::move(module), arguments, error));
  }

  // Compare with literal.
  // Side effect: EXPECT_OK
  void ComputeAndCompareLiteral(XlaBuilder* const builder,
                                const Literal& expected,
                                const absl::Span<Literal* const> arguments,
                                const Shape* shape_with_layout) {
    return ComputeAndCompareLiteral(builder, expected, arguments, std::nullopt,
                                    shape_with_layout);
  }

  // Compare with literal.
  // Side effect: EXPECT_OK
  void ComputeAndCompareLiteral(
      XlaBuilder* const builder, const Literal& expected,
      const absl::Span<Literal* const> arguments,
      const std::optional<ErrorSpec> error = std::nullopt,
      const Shape* shape_with_layout = nullptr) {
    if (error == std::nullopt) {
      if (ShapeUtil::ElementIsFloating(expected.shape()) ||
          ShapeUtil::ElementIsComplex(expected.shape())) {
        LOG(WARNING) << "performing exact comparison of floating point numbers";
      }
    }
    // We allow using a float expected literal for a non float outputs. In this
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
    TF_ASSERT_OK_AND_ASSIGN(
        Literal actual,
        this->ExecuteAndTransfer(builder, arguments, shape_with_layout));
    if (error.has_value()) {
      EXPECT_TRUE(LiteralTestUtil::Near(*expected_ptr, actual, *error));
    } else {
      EXPECT_TRUE(LiteralTestUtil::Equal(*expected_ptr, actual));
    }
  }

  // Compare with literal.
  // Side effect: EXPECT_OK
  void ComputeAndCompareTuple(XlaBuilder* builder, const Literal& expected,
                              absl::Span<Literal* const> arguments,
                              std::optional<ErrorSpec> error = std::nullopt) {
    return ComputeAndCompareLiteral(builder, expected, arguments, error);
  }

  template <typename NativeT>
  void ComputeAndCompareR0(XlaBuilder* builder, NativeT expected,
                           absl::Span<Literal* const> arguments,
                           std::optional<ErrorSpec> error = std::nullopt) {
    CheckErrorSpec<NativeT>(error);
    Literal expected_literal = LiteralUtil::CreateR0<NativeT>(expected);
    ComputeAndCompareLiteral(builder, expected_literal, arguments, error);
  }

  template <typename NativeT>
  void ComputeAndCompareR1(XlaBuilder* builder,
                           absl::Span<const NativeT> expected,
                           absl::Span<Literal* const> arguments,
                           std::optional<ErrorSpec> error = std::nullopt) {
    CheckErrorSpec<NativeT>(error);
    Literal expected_literal = LiteralUtil::CreateR1<NativeT>(expected);
    ComputeAndCompareLiteral(builder, expected_literal, arguments, error);
  }

  void ComputeAndCompareR1(XlaBuilder* builder,
                           const tsl::core::Bitmap& expected,
                           absl::Span<Literal* const> arguments,
                           std::optional<ErrorSpec> error = std::nullopt) {
    Literal expected_literal = LiteralUtil::CreateR1(expected);
    ComputeAndCompareLiteral(builder, expected_literal, arguments, error);
  }

  template <typename NativeT>
  void ComputeAndCompareR2(XlaBuilder* builder,
                           const Array2D<NativeT>& expected,
                           absl::Span<Literal* const> arguments,
                           std::optional<ErrorSpec> error = std::nullopt) {
    CheckErrorSpec<NativeT>(error);
    Literal expected_literal =
        LiteralUtil::CreateR2FromArray2D<NativeT>(expected);
    ComputeAndCompareLiteral(builder, expected_literal, arguments, error);
  }

  template <typename NativeT>
  void ComputeAndCompareR3(XlaBuilder* builder,
                           const Array3D<NativeT>& expected,
                           absl::Span<Literal* const> arguments,
                           std::optional<ErrorSpec> error = std::nullopt) {
    CheckErrorSpec<NativeT>(error);
    Literal expected_literal =
        LiteralUtil::CreateR3FromArray3D<NativeT>(expected);
    ComputeAndCompareLiteral(builder, expected_literal, arguments, error);
  }

  template <typename NativeT>
  void ComputeAndCompareR4(XlaBuilder* builder,
                           const Array4D<NativeT>& expected,
                           absl::Span<Literal* const> arguments,
                           std::optional<ErrorSpec> error = std::nullopt) {
    CheckErrorSpec<NativeT>(error);
    Literal expected_literal =
        LiteralUtil::CreateR4FromArray4D<NativeT>(expected);
    ComputeAndCompareLiteral(builder, expected_literal, arguments, error);
  }

  XlaComputation CreateScalarMax() { return xla::CreateScalarMax(test_type_); }

  Literal CreateParameterAndTransferLiteral(const int64_t parameter_number,
                                            const Literal& literal,
                                            const std::string& name,
                                            XlaBuilder* const builder,
                                            XlaOp* const data_handle) {
    Literal param_literal = MaybeConvertLiteralToTestType(literal);
    *data_handle =
        Parameter(builder, parameter_number, param_literal.shape(), name);
    return param_literal;
  }

  template <typename NativeT>
  Literal CreateR0Parameter(NativeT value, int64_t parameter_number,
                            const std::string& name, XlaBuilder* builder,
                            XlaOp* data_handle) {
    Literal literal = LiteralUtil::CreateR0(value);
    literal = MaybeConvertLiteralToTestType(literal);
    *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
    return literal;
  }

  template <typename NativeT>
  Literal CreateR1Parameter(absl::Span<const NativeT> values,
                            int64_t parameter_number, const std::string& name,
                            XlaBuilder* builder, XlaOp* data_handle) {
    Literal literal = LiteralUtil::CreateR1(values);
    literal = MaybeConvertLiteralToTestType(literal);
    *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
    return literal;
  }

  template <typename NativeT>
  Literal CreateR2Parameter(const Array2D<NativeT>& array_2d,
                            int64_t parameter_number, const std::string& name,
                            XlaBuilder* builder, XlaOp* data_handle) {
    Literal literal = LiteralUtil::CreateR2FromArray2D(array_2d);
    literal = MaybeConvertLiteralToTestType(literal);
    *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
    return literal;
  }

  template <typename NativeT>
  Literal CreateR3Parameter(const Array3D<NativeT>& array_3d,
                            int64_t parameter_number, const std::string& name,
                            XlaBuilder* builder, XlaOp* data_handle) {
    Literal literal = LiteralUtil::CreateR3FromArray3D(array_3d);
    literal = MaybeConvertLiteralToTestType(literal);
    *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
    return literal;
  }

  Literal MaybeConvertLiteralToTestType(const Literal& literal) const {
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

  void SetFastMathDisabled(const bool disabled) {
    auto* opts = execution_options_.mutable_debug_options();
    opts->set_xla_cpu_enable_fast_math(!disabled);
    opts->set_xla_cpu_enable_fast_min_max(!disabled);
    opts->set_xla_gpu_enable_fast_min_max(!disabled);
  }

  // Provides mutable access to the execution DebugOptions field; this lets
  // tests tweak the options that will be used to compile/run the graph.
  DebugOptions* mutable_debug_options() {
    return execution_options_.mutable_debug_options();
  }

 private:
  absl::StatusOr<std::unique_ptr<HloModule>> BuildAndVerifyHloModule(
      const XlaComputation& computation,
      const ExecutionOptions* execution_options = nullptr) const {
    if (execution_options == nullptr) {
      execution_options = &execution_options_;
    }
    TF_ASSIGN_OR_RETURN(
        HloModuleConfig module_config,
        HloModule::CreateModuleConfigFromProto(
            computation.proto(), execution_options->debug_options(),
            execution_options));
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProto(computation.proto(), module_config));
    TF_RETURN_IF_ERROR(this->verifier().Run(module.get()).status());
    return module;
  }

  PrimitiveType test_type_ = F32;
  ExecutionOptions execution_options_ = CreateDefaultExecutionOptions();
};

}  // namespace xla

#endif  // XLA_TESTS_CLIENT_LIBRARY_TEST_RUNNER_MIXIN_H_
