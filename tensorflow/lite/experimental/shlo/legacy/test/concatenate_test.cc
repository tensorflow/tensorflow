/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/legacy/include/shlo.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/debug.h"  // IWYU pragma: keep, b/321245930
#include "tensorflow/lite/experimental/shlo/legacy/src/storage.h"
#include "tensorflow/lite/experimental/shlo/legacy/test/util.h"

namespace stablehlo {
namespace testing {

template <ElementType element_type>
struct TensorConst {
  std::initializer_list<DimensionSize>&& shape;
  std::vector<typename Storage<element_type>::Type>&& values;
};

template <typename T>
std::string ToString(absl::string_view name,
                     const std::vector<const T*>& tensors) {
  std::ostringstream result;
  for (size_t i = 0; i < tensors.size(); ++i) {
    result << name << "[" << i << "]: " << *tensors[i] << "\n";
  }
  return result.str();
}

template <ElementType element_type>
void test(std::initializer_list<TensorConst<element_type>>&& inputs_,
          DimensionSize dimension, TensorConst<element_type>&& expected_) {
  std::vector<Tensor> inputs_storage;
  for (auto& x : inputs_) {
    inputs_storage.emplace_back(
        Tensor(TensorType(Shape(x.shape), element_type), x.values.data()));
  }
  std::vector<const Tensor*> inputs;
  for (auto& x : inputs_storage) {
    inputs.push_back(&x);
  }

  Tensor expected(TensorType(Shape(expected_.shape), element_type),
                  expected_.values.data());

  std::vector<typename Storage<element_type>::Type> result_values(
      expected.num_elements());
  Tensor result(TensorType(expected.type()), result_values.data());

  ASSERT_OK(Concatenate(absl::Span<const Tensor*>(inputs), dimension, result));
  EXPECT_EQ(result, expected)
      << ToString("inputs", inputs) << "dimension: " << dimension;
}

template <ElementType storage_type, ElementType expressed_type>
void test(QuantizedParameter&& quantized_parameter,
          std::initializer_list<TensorConst<expressed_type>>&& inputs_,
          DimensionSize dimension, TensorConst<expressed_type>&& expected_) {
  std::vector<std::vector<typename Storage<storage_type>::Type>> inputs_storage;
  std::vector<QuantizedTensor> input_tensors;
  std::vector<const QuantizedTensor*> inputs;
  for (auto& x : inputs_) {
    inputs_storage.emplace_back(QuantizeVector<storage_type, expressed_type>(
        x.values, quantized_parameter));
    QuantizedTensorElementType element_type(
        storage_type, expressed_type, QuantizedParameter(quantized_parameter));
    QuantizedTensorType type(Shape(x.shape), std::move(element_type));
    input_tensors.emplace_back(
        QuantizedTensor(std::move(type), inputs_storage.back().data()));
  }
  for (auto& t : input_tensors) {
    inputs.push_back(&t);
  }

  auto quantized_expected_values = QuantizeVector<storage_type, expressed_type>(
      expected_.values, quantized_parameter);
  QuantizedTensor expected(
      QuantizedTensorType(
          Shape(expected_.shape),
          QuantizedTensorElementType(storage_type, expressed_type,
                                     QuantizedParameter(quantized_parameter))),
      quantized_expected_values.data());

  std::vector<typename Storage<storage_type>::Type> result_values(
      expected.num_elements());
  QuantizedTensor result(QuantizedTensorType(expected.type()),
                         result_values.data());

  ASSERT_OK(Concatenate(absl::Span<const QuantizedTensor*>(inputs), dimension,
                        result));
  EXPECT_EQ(result, expected)
      << ToString("inputs", inputs) << "dimension: " << dimension;
}

TEST(Concatenate, Unquantized) {
  test<ElementType::kI1>(
      {{{3, 2}, {true, false, true, false, true, false}},
       {{1, 2}, {false, true}}},
      0, {{4, 2}, {true, false, true, false, true, false, false, true}});
  test<ElementType::kSI8>({{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
                          {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kSI16>({{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
                           {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kSI32>({{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
                           {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kBF16>({{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
                           {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kF16>({{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
                          {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kF32>({{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
                          {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});

  test<ElementType::kI1>(
      {{{2, 3}, {true, false, true, false, true, false}},
       {{2, 1}, {true, false}}},
      1, {{2, 4}, {true, false, true, true, false, true, false, false}});
  test<ElementType::kSI8>({{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
                          {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kSI16>({{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
                           {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kSI32>({{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
                           {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kBF16>({{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
                           {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kF16>({{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
                          {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kF32>({{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
                          {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
}

TEST(Concatenate, Quantized) {
  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
      {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
      {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0},
      {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
      {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});

  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
      {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
      {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0},
      {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
      {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});

  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
      {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
      {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0},
      {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{1, 2}, {7, 8}}}, 0,
      {{4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}});

  test<ElementType::kSI8, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
      {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kSI8, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
      {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kSI8, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0},
      {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
      {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});

  test<ElementType::kSI16, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
      {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kSI16, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
      {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kSI16, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0},
      {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
      {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});

  test<ElementType::kSI32, ElementType::kBF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
      {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kSI32, ElementType::kF16>(
      {.scale = 0.1, .zero_point = 0},
      {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
      {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
  test<ElementType::kSI32, ElementType::kF32>(
      {.scale = 0.1, .zero_point = 0},
      {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 1}, {7, 8}}}, 1,
      {{2, 4}, {1, 2, 3, 7, 4, 5, 6, 8}});
}

}  // namespace testing
}  // namespace stablehlo
