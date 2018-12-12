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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_TESTS_TEST_UTILS_H
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_TESTS_TEST_UTILS_H

#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {
#ifdef XLA_TEST_BACKEND_POPLAR
#define POPLAR_TEST_P(X, Y) TEST_P(X, Y)
#else
#define POPLAR_TEST_P(X, Y)
#endif

bool HasOperand(const HloInstruction* parent, const HloInstruction* arg) {
  for (const auto* inst : parent->operands()) {
    if (inst == arg) return true;
  }
  return false;
}

namespace reference_util {
// Implementations of 3D functions which are missing from the reference util.
std::vector<float> Reduce3DTo1D(
    const Array3D<float>& array, float init, absl::Span<const int64> dims,
    const std::function<float(float, float)>& reduce_function) {
  std::vector<float> result;
  CHECK_EQ(dims.size(), 2);
  const std::set<int64> dim_set(dims.begin(), dims.end());
  CHECK_EQ(dim_set.size(), 2);
  for (int64 a0 = 0; a0 == 0 || (!dim_set.count(0) && a0 < array.n1()); ++a0) {
    for (int64 a1 = 0; a1 == 0 || (!dim_set.count(1) && a1 < array.n2());
         ++a1) {
      for (int64 a2 = 0; a2 == 0 || (!dim_set.count(2) && a2 < array.n3());
           ++a2) {
        float accumulator = init;
        for (int64 i0 = 0; i0 == 0 || (dim_set.count(0) && i0 < array.n1());
             ++i0) {
          for (int64 i1 = 0; i1 == 0 || (dim_set.count(1) && i1 < array.n2());
               ++i1) {
            for (int64 i2 = 0; i2 == 0 || (dim_set.count(2) && i2 < array.n3());
                 ++i2) {
              // Handle zero-sized arrays.
              if (array.n1() > 0 && array.n2() > 0 && array.n3() > 0) {
                accumulator = reduce_function(accumulator,
                                              array(a0 + i0, a1 + i1, a2 + i2));
              }
            }
          }
        }
        result.push_back(accumulator);
      }
    }
  }
  return result;
}

std::unique_ptr<Array3D<float>> Broadcast1DTo3D(
    const std::vector<float>& array, const std::vector<int64>& bounds,
    int64 broadcast_from_dim) {
  auto result =
      absl::make_unique<Array3D<float>>(bounds[0], bounds[1], bounds[2]);
  for (int64 i = 0; i < result->n1(); ++i) {
    for (int64 j = 0; j < result->n2(); ++j) {
      for (int64 k = 0; k < result->n3(); ++k) {
        switch (broadcast_from_dim) {
          case 0:
            (*result)(i, j, k) = array[i];
            break;
          case 1:
            (*result)(i, j, k) = array[j];
            break;
          case 2:
            (*result)(i, j, k) = array[k];
            break;
          default:
            break;
        }
      }
    }
  }
  return result;
}

// Applies map_function to each element in the input (3D array) and returns
// the result.
// (n1, n2, n3) index of each element is also provided as
// arguments to map_function.
template <typename F>
static std::unique_ptr<Array3D<float>> MapWithIndexArray3D(
    const Array3D<float>& input, F&& map_function) {
  auto result =
      absl::make_unique<Array3D<float>>(input.n1(), input.n2(), input.n3());
  for (int64 n1 = 0; n1 < input.n1(); ++n1) {
    for (int64 n2 = 0; n2 < input.n2(); ++n2) {
      for (int64 n3 = 0; n3 < input.n3(); ++n3) {
        (*result)(n1, n2, n3) = map_function(input(n1, n2, n3), n1, n2, n3);
      }
    }
  }
  return result;
}

// Applies map_function to each element in the input (3D array) and returns
// the result.
template <typename F>
static std::unique_ptr<Array3D<float>> MapArray3D(const Array3D<float>& input,
                                                  F&& map_function) {
  return MapWithIndexArray3D(input, [&](float value, int64, int64, int64) {
    return map_function(value);
  });
}

// Applies map_function to each pair of element in lhs and rhs (3D array) and
// returns the result.
// (n1, n2, n3) index of each element is also provided as
// arguments to map_function.
template <typename F>
static std::unique_ptr<Array3D<float>> MapWithIndexArray3D(
    const Array3D<float>& lhs, const Array3D<float>& rhs, F&& map_function) {
  auto result = absl::make_unique<Array3D<float>>(lhs.n1(), lhs.n2(), lhs.n3());
  for (int64 n1 = 0; n1 < lhs.n1(); ++n1) {
    for (int64 n2 = 0; n2 < lhs.n2(); ++n2) {
      for (int64 n3 = 0; n3 < lhs.n3(); ++n3) {
        (*result)(n1, n2, n3) =
            map_function(lhs(n1, n2, n3), rhs(n1, n2, n3), n1, n2, n3);
      }
    }
  }
  return result;
}

// Applies map_function to each pair of elements in the input lhs and rhs
// (3D array) and returns the result.
template <typename F>
static std::unique_ptr<Array3D<float>> MapArray3D(const Array3D<float>& lhs,
                                                  const Array3D<float>& rhs,
                                                  F&& map_function) {
  return MapWithIndexArray3D(lhs, rhs,
                             [&](float lhs, float rhs, int64, int64, int64) {
                               return map_function(lhs, rhs);
                             });
}

static std::unique_ptr<Array3D<float>> BatchNorm3D(const Array3D<float>& input,
                                                   const Array3D<float>& mean,
                                                   const Array3D<float>& var,
                                                   const Array3D<float>& scale,
                                                   const Array3D<float>& offset,
                                                   float epsilon) {
  auto normalized = *reference_util::MapArray3D(
      input, mean, [](float a, float b) { return a - b; });
  normalized = *reference_util::MapArray3D(
      normalized, var,
      [&](float a, float b) { return a / std::sqrt(b + epsilon); });
  normalized = *reference_util::MapArray3D(
      normalized, scale, [](float a, float b) { return a * b; });
  return reference_util::MapArray3D(normalized, offset,
                                    [](float a, float b) { return a + b; });
}

}  // namespace reference_util
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_TEST_TEST_UTILS_H
