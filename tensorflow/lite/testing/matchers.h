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

#ifndef TENSORFLOW_LITE_TESTING_MATCHERS_H_
#define TENSORFLOW_LITE_TESTING_MATCHERS_H_

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

// gMock matchers for TfLiteTensors.
//
// EXPECT_THAT(a, EqualsTensor(b));
// EXPECT_THAT(a, Approximately(EqualsTensor(b)));
// EXPECT_THAT(a, Approximately(EqualsTensor(b), /*margin*/));
// EXPECT_THAT(a, Approximately(EqualsTensor(b), /*margin=*/0, /*fraction*/));
//
// TODO: who/impjdi - Expand to more dtypes than just float.
// TODO: who/impjdi - Add cross-dtype matchers.

inline void PrintTo(const TfLiteTensor& tensor, std::ostream* os) {
  *os << "\n" << ::tflite::GetTensorDebugString(&tensor);
}

namespace testing {
namespace tflite {
namespace internal {

enum class FloatComparison { kExact, kApproximate };

struct TensorComparison {
  FloatComparison float_comp = FloatComparison::kExact;
  bool custom_margin = false;
  bool custom_fraction = false;
  double margin = 0.0;    // only used if custom_margin == true
  double fraction = 0.0;  // only used if custom_fraction == true
};

class TensorMatcher {
 public:
  TensorMatcher(const TensorComparison& comp, const TfLiteTensor& expected)
      : comp_(comp), expected_(expected) {}

  bool MatchAndExplain(const TfLiteTensor& actual,
                       MatchResultListener* listener) const {
    const bool match = Match(actual);
    if (listener->IsInterested() && !match) *listener << DescribeDiff(actual);
    return match;
  }

  void DescribeTo(std::ostream* os) const { Describe(os, "is "); }
  void DescribeNegationTo(std::ostream* os) const { Describe(os, "is not "); }

  void SetCompareApproximately() {
    comp_.float_comp = FloatComparison::kApproximate;
  }

  void SetMargin(double margin) {
    ABSL_QCHECK_GE(margin, 0.0)  // Crash OK
        << "Using a negative margin for Approximately";
    comp_.custom_margin = true;
    comp_.margin = margin;
  }

  void SetFraction(double fraction) {
    ABSL_QCHECK(0.0 <= fraction && fraction < 1.0)  // Crash OK
        << "Fraction for Approximately must be >= 0.0 and < 1.0";
    comp_.custom_fraction = true;
    comp_.fraction = fraction;
  }

 private:
  static std::string TensorIndex(int index, const TfLiteIntArray* dims) {
    if (!dims->size) return "";
    std::vector<int> index_nd(dims->size);
    for (int i = dims->size - 1; i >= 0; --i) {
      index_nd[i] = index % dims->data[i];
      index /= dims->data[i];
    }
    return absl::StrCat("[", absl::StrJoin(index_nd, "]["), "]");
  }

  bool CompareFloat(float x, float y) const {
    switch (comp_.float_comp) {
      case FloatComparison::kExact:
        return x == y;
      case FloatComparison::kApproximate:
        if (x == y) return true;
        float fraction, margin;
        if (comp_.custom_margin || comp_.custom_fraction) {
          fraction = comp_.fraction;
          margin = comp_.margin;
        } else {
          constexpr float kEpsilon = 32 * FLT_EPSILON;
          if (std::fabs(x) <= kEpsilon && std::fabs(y) <= kEpsilon) return true;
          fraction = kEpsilon;
          margin = kEpsilon;
        }
        if (!std::isfinite(x) || !std::isfinite(y)) return false;
        float relative_margin = fraction * std::max(std::fabs(x), std::fabs(y));
        return std::fabs(x - y) <= std::max(margin, relative_margin);
    }
    return false;
  }

  void Describe(std::ostream* os, std::string_view prefix) const {
    *os << prefix;
    if (comp_.float_comp == FloatComparison::kApproximate) {
      *os << "approximately ";
      if (comp_.custom_margin || comp_.custom_fraction) {
        *os << "(";
        if (comp_.custom_margin) {
          std::stringstream ss;
          ss << std::setprecision(std::numeric_limits<double>::digits10 + 2)
             << comp_.margin;
          *os << "absolute error of float values <= " << ss.str();
        }
        if (comp_.custom_margin && comp_.custom_fraction) {
          *os << " or ";
        }
        if (comp_.custom_fraction) {
          std::stringstream ss;
          ss << std::setprecision(std::numeric_limits<double>::digits10 + 2)
             << comp_.fraction;
          *os << "relative error of float values <= " << ss.str();
        }
        *os << ") ";
      }
    }
    *os << "equal to ";
    PrintTo(expected_, os);
  }

  std::string DescribeDiff(const TfLiteTensor& actual) const {
    if (actual.type != expected_.type) {
      return absl::StrCat(
          "dtypes don't match: ", TfLiteTypeGetName(actual.type), " vs ",
          TfLiteTypeGetName(expected_.type));
    }
    if (!actual.dims) return "actual.dims is null.";
    if (!expected_.dims) return "expected.dims is null.";
    if (actual.dims->size != expected_.dims->size) {
      return absl::StrCat("dims don't match: ", actual.dims->size, "D vs ",
                          expected_.dims->size, "D");
    }
    if (int n = actual.dims->size;
        std::memcmp(actual.dims->data, expected_.dims->data, n * sizeof(int))) {
      return absl::StrCat(
          "shapes don't match: ", ::tflite::GetShapeDebugString(actual.dims),
          " vs ", ::tflite::GetShapeDebugString(expected_.dims));
    }
    if (!actual.data.raw) return "actual.data is null.";
    if (!expected_.data.raw) return "expected.data is null.";
    if (actual.bytes != expected_.bytes) {
      return absl::StrCat("bytes don't match: ", actual.bytes, " vs ",
                          expected_.bytes);
    }
    std::string error = "\n";
    TfLiteIntArray* dims = actual.dims;
    int n = ::tflite::NumElements(dims);
    constexpr int kMaxMismatches = 20;
    for (int i = 0, j = 0; i < n; ++i) {
      if (!CompareFloat(actual.data.f[i], expected_.data.f[i])) {
        absl::StrAppend(&error, "data", TensorIndex(i, dims),
                        " don't match: ", actual.data.f[i], " vs ",
                        expected_.data.f[i], "\n");
        ++j;
      }
      if (j == kMaxMismatches) {
        absl::StrAppend(&error, "Too many mismatches; stopping after ", j,
                        ".\n");
        break;
      }
    }
    return error;
  }

  bool Match(const TfLiteTensor& actual) const {
    if (actual.type != expected_.type) return false;
    if (!actual.dims) return false;
    if (!expected_.dims) return false;
    if (actual.dims->size != expected_.dims->size) return false;
    if (int n = actual.dims->size;
        std::memcmp(actual.dims->data, expected_.dims->data, n * sizeof(int))) {
      return false;
    }
    if (!actual.data.raw) return false;
    if (!expected_.data.raw) return false;
    if (actual.bytes != expected_.bytes) return false;
    switch (comp_.float_comp) {
      case FloatComparison::kExact:
        if (int n = actual.bytes;
            std::memcmp(actual.data.raw, expected_.data.raw, n)) {
          return false;
        }
        break;
      case FloatComparison::kApproximate:
        for (int i = 0, n = ::tflite::NumElements(actual.dims); i < n; ++i) {
          if (!CompareFloat(actual.data.f[i], expected_.data.f[i])) {
            return false;
          }
        }
        break;
    };
    return true;
  }

  TensorComparison comp_;
  TfLiteTensor expected_;
};

}  // namespace internal

// A struct that simplifies the creation and management of constant
// `TfLiteTensor` objects, automatically deallocating the memory (including
// dims) at destruction time.
//
// Example:
//  float data[] = {2.71828f, 3.14159f};
//  SimpleConstTensor a(TfLiteType::kTfLiteFloat32, {1, 2},
//    absl::MakeSpan(data));
struct SimpleConstTensor : public TfLiteTensor {
  template <typename T>
  SimpleConstTensor(TfLiteType dtype, const std::vector<int>& shape,
                    absl::Span<T> buf) {
    type = dtype;
    dims = TfLiteIntArrayCreate(shape.size());
    std::memcpy(dims->data, shape.data(), shape.size() * sizeof(int));
    data = {.data = buf.data()};
    bytes = buf.size() * sizeof(T);
  }
  ~SimpleConstTensor() { TfLiteIntArrayFree(dims); }
};

// Delegate pretty print to PrintTo(TfLiteTensor&).
inline void PrintTo(const SimpleConstTensor& tensor,
                    std::ostream* os) {  // NOLINT
  PrintTo(absl::implicit_cast<const TfLiteTensor&>(tensor), os);
}

inline PolymorphicMatcher<internal::TensorMatcher> EqualsTensor(
    const TfLiteTensor& expected) {
  internal::TensorComparison comp;
  return MakePolymorphicMatcher(internal::TensorMatcher(comp, expected));
}

template <class InnerTensorMatcherT>
inline InnerTensorMatcherT Approximately(InnerTensorMatcherT m) {
  m.mutable_impl().SetCompareApproximately();
  return m;
}

template <class InnerTensorMatcherT>
inline InnerTensorMatcherT Approximately(InnerTensorMatcherT m, double margin) {
  m.mutable_impl().SetCompareApproximately();
  m.mutable_impl().SetMargin(margin);
  return m;
}

template <class InnerTensorMatcherT>
inline InnerTensorMatcherT Approximately(InnerTensorMatcherT m, double margin,
                                         double fraction) {
  m.mutable_impl().SetCompareApproximately();
  m.mutable_impl().SetMargin(margin);
  m.mutable_impl().SetFraction(fraction);
  return m;
}

}  // namespace tflite
}  // namespace testing

#endif  // TENSORFLOW_LITE_TESTING_MATCHERS_H_
