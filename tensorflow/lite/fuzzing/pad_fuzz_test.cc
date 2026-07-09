/* Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"
#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace {

enum class PadKind { kPad, kPadV2 };
enum class PaddingShapeKind { kValid, kWrongRows, kWrongColumns, kRankOne };

struct PadCase {
  std::vector<int32_t> input_shape;
  std::vector<int64_t> padding_values;
  PadKind pad_kind;
  PaddingShapeKind padding_shape_kind;
  TensorType input_type;
  TensorType padding_type;
  bool dynamic_paddings;
  bool quantized_int8;
  bool non_scalar_pad_value;
  bool invoke;
};

enum class RunResult { kSuccess, kRejected, kHarnessFailure };

// Keep individual allocations small so malformed cases exercise validation
// paths without turning a fuzz iteration into an accidental stress test.
constexpr size_t kMaxInputElements = 256;

size_t ElementCount(const std::vector<int32_t>& shape) {
  size_t result = 1;
  for (int32_t dimension : shape) {
    if (dimension < 0 || dimension > kMaxInputElements) {
      return kMaxInputElements + 1;
    }
    if (dimension != 0 &&
        result > kMaxInputElements / static_cast<size_t>(dimension)) {
      return kMaxInputElements + 1;
    }
    result *= static_cast<size_t>(dimension);
  }
  return result;
}

size_t TypeSize(TensorType type) {
  switch (type) {
    case TensorType_FLOAT32:
      return sizeof(float);
    case TensorType_INT8:
      return sizeof(int8_t);
    case TensorType_INT16:
      return sizeof(int16_t);
    case TensorType_INT32:
      return sizeof(int32_t);
    case TensorType_INT64:
      return sizeof(int64_t);
    case TensorType_BOOL:
      return sizeof(bool);
    default:
      return 0;
  }
}

template <typename T>
void FillValues(std::vector<uint8_t>* bytes, size_t count, int64_t seed) {
  if constexpr (std::is_same_v<T, bool>) {
    std::vector<uint8_t> values(count);
    for (size_t i = 0; i < count; ++i) {
      values[i] = ((seed + static_cast<int64_t>(i % 7)) % 7) != 0;
    }
    *bytes = std::move(values);
  } else {
    std::vector<T> values(count);
    for (size_t i = 0; i < count; ++i) {
      values[i] = static_cast<T>((seed + static_cast<int64_t>(i % 7)) % 7);
    }
    bytes->resize(values.size() * sizeof(T));
    if (!values.empty()) {
      std::memcpy(bytes->data(), values.data(), bytes->size());
    }
  }
}

std::vector<uint8_t> MakeValues(TensorType type, size_t count, int64_t seed) {
  std::vector<uint8_t> bytes;
  switch (type) {
    case TensorType_FLOAT32:
      FillValues<float>(&bytes, count, seed);
      break;
    case TensorType_INT8:
      FillValues<int8_t>(&bytes, count, seed);
      break;
    case TensorType_INT16:
      FillValues<int16_t>(&bytes, count, seed);
      break;
    case TensorType_INT32:
      FillValues<int32_t>(&bytes, count, seed);
      break;
    case TensorType_INT64:
      FillValues<int64_t>(&bytes, count, seed);
      break;
    case TensorType_BOOL:
      FillValues<bool>(&bytes, count, seed);
      break;
    default:
      break;
  }
  return bytes;
}

std::vector<uint8_t> MakePaddingValues(TensorType type,
                                       const std::vector<int64_t>& values) {
  // Padding values are deliberately converted to the tensor's storage type.
  // This lets the fuzzer test truncation and sign-conversion behavior in the
  // kernel, while the INT64 cases retain the original extreme values.
  std::vector<uint8_t> bytes(values.size() * TypeSize(type), 0);
  auto store = [&bytes](size_t index, auto value) {
    std::memcpy(bytes.data() + index * sizeof(value), &value, sizeof(value));
  };
  for (size_t i = 0; i < values.size(); ++i) {
    switch (type) {
      case TensorType_INT8:
        store(i, static_cast<int8_t>(values[i]));
        break;
      case TensorType_INT16:
        store(i, static_cast<int16_t>(values[i]));
        break;
      case TensorType_INT32:
        store(i, static_cast<int32_t>(values[i]));
        break;
      case TensorType_INT64:
        store(i, values[i]);
        break;
      case TensorType_BOOL:
        store(i, values[i] != 0);
        break;
      case TensorType_FLOAT32:
        store(i, static_cast<float>(values[i]));
        break;
      default:
        break;
    }
  }
  return bytes;
}

std::vector<int32_t> PaddingShape(const PadCase& test_case, size_t rank,
                                  size_t* rows, size_t* columns) {
  *rows = rank;
  *columns = 2;
  // Most cases use [rank, 2]. The other shapes model malformed model files
  // and are expected to be rejected by Prepare or Invoke.
  switch (test_case.padding_shape_kind) {
    case PaddingShapeKind::kWrongRows:
      *rows = rank + 1;
      break;
    case PaddingShapeKind::kWrongColumns:
      *columns = 1 + (test_case.input_shape.size() % 3);
      if (*columns == 2) *columns = 3;
      break;
    case PaddingShapeKind::kRankOne:
      return {static_cast<int32_t>(rank * 2)};
    case PaddingShapeKind::kValid:
      break;
  }
  return {static_cast<int32_t>(*rows), static_cast<int32_t>(*columns)};
}

std::vector<int64_t> MaterializePaddings(const PadCase& test_case,
                                         size_t count) {
  std::vector<int64_t> result(count, 0);
  if (test_case.padding_values.empty()) return result;
  for (size_t i = 0; i < count; ++i) {
    result[i] = test_case.padding_values[i % test_case.padding_values.size()];
  }
  return result;
}

RunResult RunPadCase(const PadCase& test_case) {
  if (ElementCount(test_case.input_shape) > kMaxInputElements ||
      TypeSize(test_case.input_type) == 0 ||
      TypeSize(test_case.padding_type) == 0) {
    return RunResult::kRejected;
  }

  const size_t rank = test_case.input_shape.size();
  size_t padding_rows = 0;
  size_t padding_columns = 0;
  const std::vector<int32_t> padding_shape =
      PaddingShape(test_case, rank, &padding_rows, &padding_columns);
  const std::vector<int64_t> padding_values =
      MaterializePaddings(test_case, padding_rows * padding_columns);

  // Build a minimal one-node FlatBuffer model instead of calling the kernel's
  // internal Eval function. This preserves the model-parser, allocator, and
  // Prepare paths that an untrusted model would exercise.
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>{}))};

  // A dynamic paddings tensor has no constant buffer and is populated after
  // AllocateTensors, matching normal runtime use.
  if (!test_case.dynamic_paddings) {
    const std::vector<uint8_t> bytes =
        MakePaddingValues(test_case.padding_type, padding_values);
    buffers.push_back(CreateBuffer(
        builder, builder.CreateVector(bytes.data(), bytes.size())));
  }

  uint32_t padding_buffer = test_case.dynamic_paddings ? 0 : 1;
  const auto input_shape = builder.CreateVector(test_case.input_shape);
  const auto padding_shape_offset = builder.CreateVector(padding_shape);
  const auto output_shape = builder.CreateVector(test_case.input_shape);

  const bool quantized =
      test_case.quantized_int8 && test_case.input_type == TensorType_INT8;
  flatbuffers::Offset<QuantizationParameters> quantization = 0;
  if (quantized) {
    quantization = CreateQuantizationParameters(
        builder, 0, 0, builder.CreateVector<float>({0.25f}),
        builder.CreateVector<int64_t>({0}));
  }

  const auto input_tensor = CreateTensor(
      builder, input_shape, test_case.input_type, 0, 0, quantization);
  const auto padding_tensor = CreateTensor(
      builder, padding_shape_offset, test_case.padding_type, padding_buffer);

  std::vector<flatbuffers::Offset<Tensor>> tensors = {input_tensor,
                                                      padding_tensor};
  std::vector<int32_t> inputs = {0, 1};
  if (test_case.pad_kind == PadKind::kPadV2) {
    const std::vector<int32_t> value_shape = {
        test_case.non_scalar_pad_value ? 2 : 1};
    const auto value_bytes = MakeValues(
        test_case.input_type, value_shape[0],
        test_case.padding_values.empty() ? 0 : test_case.padding_values[0]);
    const uint32_t value_buffer = buffers.size();
    buffers.push_back(CreateBuffer(
        builder, builder.CreateVector(value_bytes.data(), value_bytes.size())));
    const auto value_tensor =
        CreateTensor(builder, builder.CreateVector(value_shape),
                     test_case.input_type, value_buffer, 0, quantization);
    tensors.push_back(value_tensor);
    inputs.push_back(2);
  }

  const int output_index = tensors.size();
  tensors.push_back(CreateTensor(builder, output_shape, test_case.input_type, 0,
                                 0, quantization));
  const std::vector<int32_t> outputs = {output_index};
  const BuiltinOperator op_code = test_case.pad_kind == PadKind::kPad
                                      ? BuiltinOperator_PAD
                                      : BuiltinOperator_PADV2;
  const auto opcode = CreateOperatorCode(builder, op_code, 0, 1);
  const auto options = test_case.pad_kind == PadKind::kPad
                           ? CreatePadOptions(builder).Union()
                           : CreatePadV2Options(builder).Union();
  const auto builtin_options = test_case.pad_kind == PadKind::kPad
                                   ? BuiltinOptions_PadOptions
                                   : BuiltinOptions_PadV2Options;
  const auto op =
      CreateOperator(builder, 0, builder.CreateVector(inputs),
                     builder.CreateVector(outputs), builtin_options, options);
  const auto subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors), builder.CreateVector(inputs),
      builder.CreateVector(outputs), builder.CreateVector(&op, 1));
  const auto model = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&opcode, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("pad_fuzz"),
      builder.CreateVector(buffers));
  builder.Finish(model);

  // Register only the operation under test so failures are attributable to
  // PAD/PADV2 and the test remains independent of the full resolver.
  MutableOpResolver resolver;
  resolver.AddBuiltin(op_code, op_code == BuiltinOperator_PAD
                                   ? ops::builtin::Register_PAD()
                                   : ops::builtin::Register_PADV2());
  std::unique_ptr<Interpreter> interpreter;
  const Model* model_view = GetModel(builder.GetBufferPointer());
  if (InterpreterBuilder(model_view, resolver)(&interpreter) != kTfLiteOk ||
      interpreter == nullptr) {
    return RunResult::kHarnessFailure;
  }
  if (interpreter->ResizeInputTensor(0, test_case.input_shape) != kTfLiteOk) {
    return RunResult::kRejected;
  }
  if (test_case.dynamic_paddings &&
      interpreter->ResizeInputTensor(1, padding_shape) != kTfLiteOk) {
    return RunResult::kRejected;
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) return RunResult::kRejected;

  const size_t input_elements = ElementCount(test_case.input_shape);
  const std::vector<uint8_t> input_bytes =
      MakeValues(test_case.input_type, input_elements, 1);
  if (!input_bytes.empty()) {
    std::memcpy(interpreter->tensor(0)->data.raw, input_bytes.data(),
                input_bytes.size());
  }
  if (test_case.dynamic_paddings) {
    const std::vector<uint8_t> padding_bytes =
        MakePaddingValues(test_case.padding_type, padding_values);
    if (!padding_bytes.empty()) {
      std::memcpy(interpreter->tensor(1)->data.raw, padding_bytes.data(),
                  padding_bytes.size());
    }
  }
  if (!test_case.invoke) return RunResult::kSuccess;
  return interpreter->Invoke() == kTfLiteOk ? RunResult::kSuccess
                                            : RunResult::kRejected;
}

auto PaddingValueDomain() {
  // Bias generation toward boundary values: these are the values most likely
  // to expose integer narrowing, addition, and output-size overflow bugs.
  return fuzztest::OneOf(
      fuzztest::InRange<int64_t>(-4, 4), fuzztest::Just<int64_t>(INT32_MAX),
      fuzztest::Just<int64_t>(INT32_MIN),
      fuzztest::Just<int64_t>(static_cast<int64_t>(INT32_MAX) + 1),
      fuzztest::Just<int64_t>(static_cast<int64_t>(INT32_MIN) - 1));
}

auto PadCaseDomain() {
  // The product domain combines ordinary models with deliberately malformed
  // tensor metadata, unsupported types, and unusual invocation states.
  return fuzztest::StructOf<PadCase>(
      fuzztest::VectorOf(fuzztest::InRange<int32_t>(0, 2))
          .WithMinSize(0)
          .WithMaxSize(8),
      fuzztest::VectorOf(PaddingValueDomain()).WithMinSize(1).WithMaxSize(8),
      fuzztest::ElementOf<PadKind>({PadKind::kPad, PadKind::kPadV2}),
      fuzztest::ElementOf<PaddingShapeKind>(
          {PaddingShapeKind::kValid, PaddingShapeKind::kWrongRows,
           PaddingShapeKind::kWrongColumns, PaddingShapeKind::kRankOne}),
      fuzztest::ElementOf<TensorType>(
          {TensorType_FLOAT32, TensorType_INT8, TensorType_INT32}),
      fuzztest::ElementOf<TensorType>({TensorType_INT8, TensorType_INT16,
                                       TensorType_INT32, TensorType_INT64,
                                       TensorType_BOOL, TensorType_FLOAT32}),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>(),
      fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>());
}

void PadNeverCrashes(const PadCase& test_case) {
  // Rejection is expected for invalid models. A harness failure means the
  // fuzzer itself constructed an invalid test setup, so treat that as a bug.
  EXPECT_NE(RunPadCase(test_case), RunResult::kHarnessFailure);
}

FUZZ_TEST(PadFuzzTest, PadNeverCrashes).WithDomains(PadCaseDomain());

}  // namespace
}  // namespace tflite
