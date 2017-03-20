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

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// A client library test establishes an in-process XLA client connection.
class ClientLibraryTestBase : public ::testing::Test {
 protected:
  explicit ClientLibraryTestBase(
      perftools::gputools::Platform* platform = nullptr,
      tensorflow::gtl::ArraySlice<string> disabled_pass_names = {});

  // Returns the name of the test currently being run.
  string TestName() const;

  void SetFastMathDisabled(bool disabled) {
    execution_options_.set_disable_fast_math(disabled);
  }

  // TODO(b/25566808): Add helper that populates a literal from a testdata file.

  // Convenience methods for building and running a computation from a builder.
  StatusOr<std::unique_ptr<GlobalData>> Execute(
      ComputationBuilder* builder,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  StatusOr<std::unique_ptr<Literal>> ExecuteAndTransfer(
      ComputationBuilder* builder,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_output_layout = nullptr);

  // Convenience OrDie variants of above methods.
  std::unique_ptr<GlobalData> ExecuteOrDie(
      ComputationBuilder* builder,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  std::unique_ptr<Literal> ExecuteAndTransferOrDie(
      ComputationBuilder* builder,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments);

  // Run a computation and return its value as a string. If an error
  // occurs, then instead return the error as a string.
  string ExecuteToString(ComputationBuilder* builder,
                         tensorflow::gtl::ArraySlice<GlobalData*> arguments);

  // Convenience methods for building and running a computation, transferring
  // the result, and comparing it to the expected value(s). Methods are
  // templated on the native host type which maps to specific XLA types (See
  // ComputationBuilder for details). For each rank, two forms are provided: one
  // for floating point types with an ErrorSpec parameter, and one for integral
  // types without the ErrorSpec parameter.
  template <typename NativeT>
  void ComputeAndCompareR0(ComputationBuilder* builder, NativeT expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename NativeT>
  void ComputeAndCompareR0(ComputationBuilder* builder, NativeT expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments,
                           ErrorSpec error);

  template <typename NativeT>
  void ComputeAndCompareR1(ComputationBuilder* builder,
                           tensorflow::gtl::ArraySlice<NativeT> expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename NativeT>
  void ComputeAndCompareR1(ComputationBuilder* builder,
                           tensorflow::gtl::ArraySlice<NativeT> expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments,
                           ErrorSpec error);

  // As above, but uses a bitmap to hold the predicate vector to avoid
  // deficiencies of vector<bool>.
  void ComputeAndCompareR1(ComputationBuilder* builder,
                           const tensorflow::core::Bitmap& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);

  template <typename NativeT>
  void ComputeAndCompareR2(ComputationBuilder* builder,
                           const Array2D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename NativeT>
  void ComputeAndCompareR2(ComputationBuilder* builder,
                           const Array2D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments,
                           ErrorSpec error);

  template <typename NativeT>
  void ComputeAndCompareR3(ComputationBuilder* builder,
                           const Array3D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename NativeT>
  void ComputeAndCompareR3(ComputationBuilder* builder,
                           const Array3D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments,
                           ErrorSpec error);

  template <typename NativeT>
  void ComputeAndCompareR4(ComputationBuilder* builder,
                           const Array4D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  template <typename NativeT>
  void ComputeAndCompareR4(ComputationBuilder* builder,
                           const Array4D<NativeT>& expected,
                           tensorflow::gtl::ArraySlice<GlobalData*> arguments,
                           ErrorSpec error);

  // Build and run the computation and compare the result with the given
  // literal. shape_with_layout indicates the result layout to request when
  // calling Execute.
  void ComputeAndCompareLiteral(
      ComputationBuilder* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_layout = nullptr);
  void ComputeAndCompareLiteral(
      ComputationBuilder* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error,
      const Shape* shape_with_layout = nullptr);

  // ComputeAndCompare variant which returns an error status.
  tensorflow::Status ComputeAndCompareLiteralWithStatus(
      ComputationBuilder* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_layout = nullptr);
  tensorflow::Status ComputeAndCompareLiteralWithStatus(
      ComputationBuilder* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error,
      const Shape* shape_with_layout = nullptr);

  // Compare the result of the computation to a strings. In XLA strings are
  // represented using rank-1 U8 shapes.
  void ComputeAndCompareR1U8(
      ComputationBuilder* builder, tensorflow::StringPiece expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments);

  // Convenience method for running a built computation, transferring the
  // result, and comparing it to the expected tuple literal.
  void ComputeAndCompareTuple(
      ComputationBuilder* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments);
  void ComputeAndCompareTuple(
      ComputationBuilder* builder, const Literal& expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec abs_error);

  // Create scalar operations for use in reductions.
  Computation CreateScalarRelu();
  Computation CreateScalarMax();
  Computation CreateScalarReluSensitivity();

  // Special case convenience functions for creating filled arrays.

  // Creates an array of pseudorandom values lying between the given minimum and
  // maximum values.
  template <typename NativeT>
  std::vector<NativeT> CreatePseudorandomR1(const int width, NativeT min_value,
                                            NativeT max_value, uint32 seed);
  template <typename NativeT>
  std::unique_ptr<Array2D<NativeT>> CreatePseudorandomR2(const int rows,
                                                         const int cols,
                                                         NativeT min_value,
                                                         NativeT max_value,
                                                         uint32 seed);

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

  // Create a parameter instruction that wraps the given values and then stores
  // into "data_handle" the global handle for that parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR1Parameter(
      tensorflow::gtl::ArraySlice<NativeT> values, int64 parameter_number,
      const string& name, ComputationBuilder* builder,
      ComputationDataHandle* data_handle);

  // Create a parameter instruction that wraps the given constant array
  // "array_2d" and then stores to "data_handle" the global handle for that
  // parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR2Parameter(
      const Array2D<NativeT>& array_2d, int64 parameter_number,
      const string& name, ComputationBuilder* builder,
      ComputationDataHandle* data_handle);

  // Create a parameter instruction that wraps the given constant array
  // "array_3d" and then stores to "data_handle" the global handle for that
  // parameter.
  //
  // "parameter_number" is the parameter number.
  // "name" is the name of the parameter instruction.
  template <typename NativeT>
  std::unique_ptr<GlobalData> CreateR3Parameter(
      const Array3D<NativeT>& array_3d, int64 parameter_number,
      const string& name, ComputationBuilder* builder,
      ComputationDataHandle* data_handle);

  Client* client_;
  ExecutionOptions execution_options_;
};

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR0(
    ComputationBuilder* builder, NativeT expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR0<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR0(
    ComputationBuilder* builder, NativeT expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value,
                "Floating point type required when specifying an ErrorSpec");
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR0<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR1(
    ComputationBuilder* builder, tensorflow::gtl::ArraySlice<NativeT> expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR1<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR1(
    ComputationBuilder* builder, tensorflow::gtl::ArraySlice<NativeT> expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value,
                "Floating point type required when specifying an ErrorSpec");
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR1<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR2(
    ComputationBuilder* builder, const Array2D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR2FromArray2D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR2(
    ComputationBuilder* builder, const Array2D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value,
                "Floating point type required when specifying an ErrorSpec");
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR2FromArray2D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR3(
    ComputationBuilder* builder, const Array3D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR3FromArray3D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR3(
    ComputationBuilder* builder, const Array3D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value,
                "Floating point type required when specifying an ErrorSpec");
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR3FromArray3D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR4(
    ComputationBuilder* builder, const Array4D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR4FromArray4D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments);
}

template <typename NativeT>
void ClientLibraryTestBase::ComputeAndCompareR4(
    ComputationBuilder* builder, const Array4D<NativeT>& expected,
    tensorflow::gtl::ArraySlice<GlobalData*> arguments, ErrorSpec error) {
  static_assert(std::is_same<NativeT, float>::value ||
                    std::is_same<NativeT, double>::value,
                "Floating point type required when specifying an ErrorSpec");
  std::unique_ptr<Literal> expected_literal =
      LiteralUtil::CreateR4FromArray4D<NativeT>(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, *expected_literal,
                                                  arguments, error);
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR1Parameter(
    tensorflow::gtl::ArraySlice<NativeT> values, int64 parameter_number,
    const string& name, ComputationBuilder* builder,
    ComputationDataHandle* data_handle) {
  std::unique_ptr<Literal> literal = LiteralUtil::CreateR1(values);
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(*literal).ConsumeValueOrDie();
  *data_handle = builder->Parameter(parameter_number, literal->shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR2Parameter(
    const Array2D<NativeT>& array_2d, int64 parameter_number,
    const string& name, ComputationBuilder* builder,
    ComputationDataHandle* data_handle) {
  std::unique_ptr<Literal> literal = LiteralUtil::CreateR2FromArray2D(array_2d);
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(*literal).ConsumeValueOrDie();
  *data_handle = builder->Parameter(parameter_number, literal->shape(), name);
  return data;
}

template <typename NativeT>
std::unique_ptr<GlobalData> ClientLibraryTestBase::CreateR3Parameter(
    const Array3D<NativeT>& array_3d, int64 parameter_number,
    const string& name, ComputationBuilder* builder,
    ComputationDataHandle* data_handle) {
  std::unique_ptr<Literal> literal = LiteralUtil::CreateR3FromArray3D(array_3d);
  std::unique_ptr<GlobalData> data =
      client_->TransferToServer(*literal).ConsumeValueOrDie();
  *data_handle = builder->Parameter(parameter_number, literal->shape(), name);
  return data;
}

template <typename NativeT>
std::vector<NativeT> ClientLibraryTestBase::CreatePseudorandomR1(
    const int width, NativeT min_value, NativeT max_value, uint32 seed) {
  std::vector<NativeT> result(width);
  test_utils::PseudorandomGenerator<NativeT> generator(min_value, max_value,
                                                       seed);
  for (int i = 0; i < width; ++i) {
    result[i] = generator.get();
  }
  return result;
}

template <typename NativeT>
std::unique_ptr<Array2D<NativeT>> ClientLibraryTestBase::CreatePseudorandomR2(
    const int rows, const int cols, NativeT min_value, NativeT max_value,
    uint32 seed) {
  auto result = MakeUnique<Array2D<NativeT>>(rows, cols);
  test_utils::PseudorandomGenerator<NativeT> generator(min_value, max_value,
                                                       seed);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      (*result)(y, x) = generator.get();
    }
  }
  return result;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_CLIENT_LIBRARY_TEST_BASE_H_
