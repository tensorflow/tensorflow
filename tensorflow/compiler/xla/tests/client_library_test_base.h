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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_CLIENT_LIBRARY_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_CLIENT_LIBRARY_TEST_BASE_H_

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/manifest_checking_test.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

// Sets the use_bfloat16 on a container of test cases according to the values in
// use_bfloat16_params. Generates one set of test cases for each values in
// use_bfloat16_params with that value. Returns the result.
template <typename TestCase>
std::vector<TestCase> ExpandUseBfloat16(
    absl::Span<const bool> use_bfloat16_params,
    absl::Span<const TestCase> specs) {
  std::vector<TestCase> expanded;
  for (bool use_bfloat16 : use_bfloat16_params) {
    for (const auto& spec : specs) {
      expanded.push_back(spec);
      expanded.back().use_bfloat16 = use_bfloat16;
    }
  }
  return expanded;
}

// A client library test establishes an in-process XLA client connection.
class ClientLibraryTestBase : public ManifestCheckingTest {
 protected:
  explicit ClientLibraryTestBase(se::Platform* platform = nullptr);

  // Creates a new ClientLibraryTestBase with custom client options.
  ClientLibraryTestBase(se::Platform* platform,
                        const LocalClientOptions& client_options);

  // Returns the name of the test currently being run.
  std::string TestName() const;

  void SetFastMathDisabled(bool disabled) {
    auto* opts = execution_options_.mutable_debug_options();
    opts->set_xla_cpu_enable_fast_math(!disabled);
    opts->set_xla_cpu_enable_fast_min_max(!disabled);
    opts->set_xla_gpu_enable_fast_min_max(!disabled);
  }

  void SetSeed(uint64_t seed) { execution_options_.set_seed(seed); }

  // Provides mutable access to the execution DebugOptions field; this lets
  // tests tweak the options that will be used to compile/run the graph.
  DebugOptions* mutable_debug_options() {
    return execution_options_.mutable_debug_options();
  }

  // TODO(b/25566808): Add helper that populates a literal from a testdata file.

  // Convenience methods for building and running a computation with the member
  // execution options. Modify execution_options_ in your test if you want to
  // customize the options.
  StatusOr<std::unique_ptr<GlobalData>> Execute(
      XlaBuilder* builder, absl::Span<GlobalData* const> arguments);

  StatusOr<Literal> ExecuteAndTransfer(
      XlaBuilder* builder, absl::Span<GlobalData* const> arguments,
      const Shape* shape_with_output_layout = nullptr);

  StatusOr<Literal> ExecuteAndTransfer(
      const XlaComputation& computation,
      absl::Span<GlobalData* const> arguments,
      const Shape* shape_with_output_layout = nullptr);

  // This executes the computation via the reference client (which connects a
  // interpreter backend). The result is used as the expected value of the
  // computation.
  StatusOr<Literal> ExecuteAndTransferReference(
      const XlaComputation& computation,
      absl::Span<GlobalData* const> arguments,
      const Shape* shape_with_output_layout = nullptr);

  // Run a computation and return its value as a string. If an error
  // occurs, then instead return the error as a string.
  std::string ExecuteToString(XlaBuilder* builder,
                              absl::Span<GlobalData* const> arguments);

  // Convenience methods for building and running a computation, transferring
  // the result, and comparing it to the expected value(s). Methods are
  // templated on the native host type which maps to specific XLA types (See
  // XlaBuilder for details). For each rank, two forms are
  // provided: one for floating point types with an ErrorSpec parameter, and one
  // for integral types without the ErrorSpec parameter.
  template <typename NativeT>
  void ComputeAndCompareR0(XlaBuilder* builder, NativeT expected,
                           absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompareR0(XlaBuilder* builder, NativeT expected,
                           absl::Span<GlobalData* const> arguments,
                           ErrorSpec error);

  template <typename NativeT>
  void ComputeAndCompareR1(XlaBuilder* builder,
                           absl::Span<const NativeT> expected,
                           absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompareR1(XlaBuilder* builder,
                           absl::Span<const NativeT> expected,
                           absl::Span<GlobalData* const> arguments,
                           ErrorSpec error);

  // As above, but uses a bitmap to hold the predicate vector to avoid
  // deficiencies of vector<bool>.
  void ComputeAndCompareR1(XlaBuilder* builder,
                           const tensorflow::core::Bitmap& expected,
                           absl::Span<GlobalData* const> arguments);

  template <typename NativeT>
  void ComputeAndCompareR2(XlaBuilder* builder,
                           const Array2D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompareR2(XlaBuilder* builder,
                           const Array2D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments,
                           ErrorSpec error);

  template <typename NativeT>
  void ComputeAndCompareR3(XlaBuilder* builder,
                           const Array3D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompareR3(XlaBuilder* builder,
                           const Array3D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments,
                           ErrorSpec error);

  template <typename NativeT>
  void ComputeAndCompareR4(XlaBuilder* builder,
                           const Array4D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompareR4(XlaBuilder* builder,
                           const Array4D<NativeT>& expected,
                           absl::Span<GlobalData* const> arguments,
                           ErrorSpec error);

  // Build and run the computation and compare the result with the given
  // literal. shape_with_layout indicates the result layout to request when
  // calling Execute.
  void ComputeAndCompareLiteral(XlaBuilder* builder, const Literal& expected,
                                absl::Span<GlobalData* const> arguments,
                                const Shape* shape_with_layout = nullptr);
  void ComputeAndCompareLiteral(XlaBuilder* builder, const Literal& expected,
                                absl::Span<GlobalData* const> arguments,
                                ErrorSpec error,
                                const Shape* shape_with_layout = nullptr);

  // Build and run the computation and return the result as a literal.
  // shape_with_layout indicates the result layout to request when calling
  // Execute.
  StatusOr<Literal> ComputeAndTransfer(
      XlaBuilder* builder, absl::Span<GlobalData* const> arguments,
      const Shape* shape_with_layout = nullptr);

  // ComputeAndCompare variant which returns an error status.
  Status ComputeAndCompareLiteralWithStatus(
      XlaBuilder* builder, const Literal& expected,
      absl::Span<GlobalData* const> arguments,
      const Shape* shape_with_layout = nullptr);
  Status ComputeAndCompareLiteralWithStatus(
      XlaBuilder* builder, const Literal& expected,
      absl::Span<GlobalData* const> arguments, ErrorSpec error,
      const Shape* shape_with_layout = nullptr);

  // Compare the result of the computation to a strings. In XLA strings are
  // represented using rank-1 U8 shapes.
  void ComputeAndCompareR1U8(XlaBuilder* builder, absl::string_view expected,
                             absl::Span<GlobalData* const> arguments);

  // Convenience method for running a built computation, transferring the
  // result, and comparing it to the expected tuple literal.
  void ComputeAndCompareTuple(XlaBuilder* builder, const Literal& expected,
                              absl::Span<GlobalData* const> arguments);
  void ComputeAndCompareTuple(XlaBuilder* builder, const Literal& expected,
                              absl::Span<GlobalData* const> arguments,
                              ErrorSpec error);

  // Convenience method for running a built computation and comparing the result
  // with the reference result.
  void ComputeAndCompare(XlaBuilder* builder,
                         absl::Span<const Literal> arguments);
  void ComputeAndCompare(XlaBuilder* builder,
                         absl::Span<const Literal> arguments, ErrorSpec error);
  template <typename NativeT>
  void ComputeAndCompare(XlaBuilder* builder, const Array<NativeT>& expected,
                         absl::Span<GlobalData* const> arguments);
  template <typename NativeT>
  void ComputeAndCompare(XlaBuilder* builder, const Array<NativeT>& expected,
                         absl::Span<GlobalData* const> arguments,
                         ErrorSpec error);
  // Create scalar operations for use in reductions.
  XlaComputation CreateScalarRelu();
  XlaComputation CreateScalarMax();
  XlaComputation CreateScalarReluSensitivity();

  // Special case convenience functions for creating filled arrays.

  // Creates an array of pseudorandom values lying between the given minimum and
  // maximum values.
  template <typename NativeT>
  std::vector<NativeT> CreatePseudorandomR1(const int width, NativeT min_value,
                                            NativeT max_value, uint32_t seed);
  template <typename NativeT>
  std::unique_ptr<Array2D<NativeT>> CreatePseudorandomR2(const int rows,
                                                         const int cols,
                                                         NativeT min_value,
                                                         NativeT max_value,
                                                         uint32_t seed);

  // Creates a (rows x cols) array filled in the following form:
  //
  //  [      0              1 ...                   cols-1]
  //  [  1,000          1,001 ...          1000.0 + cols-1]
  //  [    ...            ... ...                      ...]
  //  [(rows-1)*1000.0    ... ... (rows-1)*1000.0 + cols-1]
  //
  // If provided, offset is added uniformly to every element (e.g. an offset of
  // 64 would cause 0 in the above to be 64, 1 to be 65, 1000 to be 1064, etc.)
  std::unique_ptr<Array2D<float>> CreatePatternedMatrix(const int rows,
                                                        const int cols,
                                                        float offset = 0.0);

  // Creates a (rows x cols) array as above, padded out to
  // (rows_padded x cols_padded) with zeroes.  Requires rows_padded >= rows
  // and cols_padded > cols.
  std::unique_ptr<Array2D<float>> CreatePatternedMatrixWithZeroPadding(
      const int rows, const int cols, const int rows_padded,
      const int cols_padded);

  // Creates a parameter instruction, transfers the literal for the parameter to
  // server, then stores into "data_handle" the global handle for that
  // parameter. When the use_bfloat16 flag is set but the literal has F32
  // elements, the literal will be converted to BF16 before being transferred.
  StatusOr<std::unique_ptr<GlobalData>> CreateParameterAndTransferLiteral(
      int64_t parameter_number, const Literal& literal, const std::string& name,
      XlaBuilder* builder, XlaOp* data_handle);

  // As above, but the caller can specify the device that the literal is
  // transferred to. If device_handle is nullptr, the literal will be
  // transferred to the default device.
  StatusOr<std::unique_ptr<GlobalData>> CreateParameterAndTransferLiteral(
      int64_t parameter_number, const Literal& literal, const std::string& name,
      const DeviceHandle* device_handle, XlaBuilder* builder,
      XlaOp* data_handle);

  // Creates a parameter instruction and sets the value that will be passed to
  // the computation as specified. This function must be used for all parameters
  // or none and no parameters must be passed when invoking the computation if
  // using this mechanism. If using this mechanism, then each parameter must be
  // set exactly once. The first added parameter gets index 0, then 1 and so on.
  XlaOp AddParam(const Literal& argument, XlaBuilder* builder);

  template <class T>
  XlaOp AddParam(const Array<T>& argument, XlaBuilder* builder) {
    return AddParam(LiteralUtil::CreateFromArray(argument), builder);
  }

  // Creates a constant instruction with the given literal. When the
  // use_bfloat16 flag is set but the literal has F32 elements, the elements
  // will be converted to BF16s.
  XlaOp CreateConstantFromLiteral(const Literal& literal, XlaBuilder* builder);

  // Creates a constant instruction with the given array. When the use_bfloat16
  // flag is set but the array has float elements, the elements will be
  // converted to bfloat16s.

  template <typename NativeT>
  XlaOp CreateConstantFromArray(const Array<NativeT>& array,
                                XlaBuilder* builder) {
    return CreateConstantFromLiteral(LiteralUtil::CreateFromArray(array),
                                     builder);
  }

  // Same as CreateConstantFromArray, but for scalars.
  template <typename NativeT>
  XlaOp CreateConstantFromScalar(NativeT value, XlaBuilder* builder) {
    return CreateConstantFromLiteral(LiteralUtil::CreateR0<NativeT>(value),
                                     builder);
  }

  // Creates a parameter instruction that wraps a given value and then stores
  // into "data_handle" the global handle for that parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR0Parameter(NativeT value,
                                                int64_t parameter_number,
                                                const std::string& name,
                                                XlaBuilder* builder,
                                                XlaOp* data_handle);

  // Creates a parameter instruction that wraps the given values and then stores
  // into "data_handle" the global handle for that parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR1Parameter(
      absl::Span<const NativeT> values, int64_t parameter_number,
      const std::string& name, XlaBuilder* builder, XlaOp* data_handle);

  // Creates a parameter instruction that wraps the given constant array
  // "array_2d" and then stores it to the global handle for that parameter
  // "data_handle".
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR2Parameter(
      const Array2D<NativeT>& array_2d, int64_t parameter_number,
      const std::string& name, XlaBuilder* builder, XlaOp* data_handle);

  // Creates a parameter instruction that wraps the given constant array
  // "array_3d" and then stores it to the global handle for that parameter
  // "data_handle".
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR3Parameter(
      const Array3D<NativeT>& array_3d, int64_t parameter_number,
      const std::string& name, XlaBuilder* builder, XlaOp* data_handle);

  // Creates a parameter instruction that wraps the given constant array
  // "array_4d" and then stores it to the global handle for that parameter
  // "data_handle".
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  //
  // When the use_bfloat16 flag is set but NativeT is float, the data will be
  // converted to bfloat16.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR4Parameter(
      const Array4D<NativeT>& array_4d, int64_t parameter_number,
      const std::string& name, XlaBuilder* builder, XlaOp* data_handle);

  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateParameter(const Array<NativeT>& array_4d,
                                              int64_t parameter_number,
                                              const std::string& name,
                                              XlaBuilder* builder,
                                              XlaOp* data_handle);

  // Getter and setter for the use_bfloat16 flag, which indicates whether to run
  // tests with all float-type input/output converted to bfloat16.
  bool use_bfloat16() const { return use_bfloat16_; }
  void set_use_bfloat16(bool value) { use_bfloat16_ = value; }

  // The float type used in this test, BF16 or F32 according to use_bfloat16.
  PrimitiveType FloatType() const { return use_bfloat16_ ? BF16 : F32; }

  // Executes the computation and calculates the expected reference value using
  // the reference client. Returns two literals in the order of (expected,
  // actual).
  StatusOr<std::pair<Literal, Literal>> ComputeValueAndReference(
      XlaBuilder* builder, absl::Span<const Literal> arguments);

  // Converts an f32 literal to bf16 if use_bfloat16_ is true.
  Literal MaybeConvertLiteralToBfloat16(const Literal& literal);

  LocalClient* client_;
  LocalClient* ref_client_;  // To compute reference result.
  ExecutionOptions execution_options_;

 private:
  Status ComputeAndCompareLiteralWithAllOutputLayouts(
      const xla::XlaComputation& computation, const Literal& expected,
      absl::Span<GlobalData* const> arguments,
      const std::function<void(const Literal& actual,
                               const std::string& error_message)>&
          verify_output);
  Status ComputeAndCompareLiteralWithAllInputLayouts(
      const xla::XlaComputation& computation, const Literal& expected,
      absl::Span<GlobalData* const> arguments,
      const std::function<void(const Literal& actual,
                               const std::string& error_message)>&
          verify_output,
      const Shape* output_with_layout = nullptr);

  // Converts an f32 shape to bf16 if use_bfloat16_ is true.
  Shape MaybeConvertShapeToBfloat16(const Shape& shape);

  // Whether to run tests with all float-type input/output converted to
  // bfloat16.
  bool use_bfloat16_ = false;

  // Arguments to be passed to the computation when it runs.
  std::vector<Literal> arguments_;
};

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR0(
    XlaBuilder* builder, NativeT expected,
    absl::Span<GlobalData* const> arguments) {
  Literal expected_literal = LiteralUtil::CreateR0<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR0(
    XlaBuilder* builder, NativeT expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal = LiteralUtil::CreateR0<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR1(
    XlaBuilder* builder, absl::Span<const NativeT> expected,
    absl::Span<GlobalData* const> arguments) {
  Literal expected_literal = LiteralUtil::CreateR1<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR1(
    XlaBuilder* builder, absl::Span<const NativeT> expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal = LiteralUtil::CreateR1<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR2(
    XlaBuilder* builder, const Array2D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments) {
  Literal expected_literal =
      LiteralUtil::CreateR2FromArray2D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR2(
    XlaBuilder* builder, const Array2D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal =
      LiteralUtil::CreateR2FromArray2D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR3(
    XlaBuilder* builder, const Array3D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments) {
  Literal expected_literal =
      LiteralUtil::CreateR3FromArray3D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR3(
    XlaBuilder* builder, const Array3D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal =
      LiteralUtil::CreateR3FromArray3D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR4(
    XlaBuilder* builder, const Array4D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments) {
  Literal expected_literal =
      LiteralUtil::CreateR4FromArray4D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR4(
    XlaBuilder* builder, const Array4D<NativeT>& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal =
      LiteralUtil::CreateR4FromArray4D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompare(
    XlaBuilder* builder, const Array<NativeT>& expected,
    absl::Span<GlobalData* const> arguments) {
  Literal expected_literal = LiteralUtil::CreateFromArray<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompare(
    XlaBuilder* builder, const Array<NativeT>& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value ||
                    std::is_same<NativeT, bfloat16>::value ||
                    std::is_same<NativeT, half>::value ||
                    std::is_same<NativeT, complex64>::value ||
                    std::is_same<NativeT, complex128>::value,
                "Float or complex type required when specifying an ErrorSpec");
  Literal expected_literal = LiteralUtil::CreateFromArray<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR0Parameter(
    NativeT value, int64_t parameter_number, const std::string& name,
    XlaBuilder* builder, XlaOp* data_handle) {
  Literal literal = LiteralUtil::CreateR0(value);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR1Parameter(
    absl::Span<const NativeT> values, int64_t parameter_number,
    const std::string& name, XlaBuilder* builder, XlaOp* data_handle) {
  Literal literal = LiteralUtil::CreateR1(values);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR2Parameter(
    const Array2D<NativeT>& array_2d, int64_t parameter_number,
    const std::string& name, XlaBuilder* builder, XlaOp* data_handle) {
  Literal literal = LiteralUtil::CreateR2FromArray2D(array_2d);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR3Parameter(
    const Array3D<NativeT>& array_3d, int64_t parameter_number,
    const std::string& name, XlaBuilder* builder, XlaOp* data_handle) {
  Literal literal = LiteralUtil::CreateR3FromArray3D(array_3d);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR4Parameter(
    const Array4D<NativeT>& array_4d, int64_t parameter_number,
    const std::string& name, XlaBuilder* builder, XlaOp* data_handle) {
  Literal literal = LiteralUtil::CreateR4FromArray4D(array_4d);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateParameter(
    const Array<NativeT>& array, int64_t parameter_number,
    const std::string& name, XlaBuilder* builder, XlaOp* data_handle) {
  Literal literal = LiteralUtil::CreateFromArray(array);
  if (use_bfloat16_ && literal.shape().element_type() == F32) {
    literal = LiteralUtil::ConvertF32ToBF16(literal);
  }
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(literal).ConsumeValueOrDie();
  *data_handle = Parameter(builder, parameter_number, literal.shape(), name);
  return data;
}

template <typename NativeT>
std::vector<NativeT> ClientLibraryTestBase::CreatePseudorandomR1(
    const int width, NativeT min_value, NativeT max_value, uint32_t seed) {
  std::vector<NativeT> result(width);
  PseudorandomGenerator<NativeT> generator(min_value, max_value, seed);
  for (int i = 0; i < width; ++i) {
    result[i] = generator.get();
  }
  return result;
}

template <typename NativeT>
std::unique_ptr<Array2D<NativeT>> ClientLibraryTestBase::CreatePseudorandomR2(
    const int rows, const int cols, NativeT min_value, NativeT max_value,
    uint32_t seed) {
  auto result = absl::make_unique<Array2D<NativeT>>(rows, cols);
  PseudorandomGenerator<NativeT> generator(min_value, max_value, seed);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      (*result)(y, x) = generator.get();
    }
  }
  return result;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_CLIENT_LIBRARY_TEST_BASE_H_
