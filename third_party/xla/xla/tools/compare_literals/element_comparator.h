/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_TOOLS_COMPARE_LITERALS_ELEMENT_COMPARATOR_H_
#define XLA_TOOLS_COMPARE_LITERALS_ELEMENT_COMPARATOR_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/tools/compare_literals/compare_literals.h"

namespace xla::compare_literals {

namespace internal {
// Type trait helpers for complex values.
template <typename T>
struct IsComplex : std::false_type {};
template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_complex_v = IsComplex<T>::value;
}  // namespace internal

template <typename T>
struct ValueTraits {
  static bool IsNan(T val) {
    if constexpr (std::is_floating_point_v<T> || !std::is_integral_v<T>) {
      return std::isnan(static_cast<double>(val));
    }
    return false;
  }
  static bool IsInf(T val) {
    if constexpr (std::is_floating_point_v<T> || !std::is_integral_v<T>) {
      return std::isinf(static_cast<double>(val));
    }
    return false;
  }
  static double AbsDiff(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
      if (a == b) {
        return 0.0;
      }
      uint64_t diff = (a > b)
                          ? static_cast<uint64_t>(a) - static_cast<uint64_t>(b)
                          : static_cast<uint64_t>(b) - static_cast<uint64_t>(a);
      return std::max(1.0, static_cast<double>(diff));
    } else {
      return std::abs(static_cast<double>(a) - static_cast<double>(b));
    }
  }
  static double Magnitude(T val) { return std::abs(static_cast<double>(val)); }
  static std::string Format(T val) {
    if constexpr (std::is_same_v<T, bool>) {
      return val ? "true" : "false";
    } else if constexpr (std::is_integral_v<T>) {
      return absl::StrCat(+val);
    } else {
      return absl::StrCat(static_cast<double>(val));
    }
  }
};

template <typename U>
struct ValueTraits<std::complex<U>> {
  static bool IsNan(const std::complex<U>& val) {
    return std::isnan(val.real()) || std::isnan(val.imag());
  }
  static bool IsInf(const std::complex<U>& val) {
    return std::isinf(val.real()) || std::isinf(val.imag());
  }
  static double AbsDiff(const std::complex<U>& a, const std::complex<U>& b) {
    return std::abs(std::complex<double>(a.real(), a.imag()) -
                    std::complex<double>(b.real(), b.imag()));
  }
  static double Magnitude(const std::complex<U>& val) {
    return std::abs(std::complex<double>(val.real(), val.imag()));
  }
  static std::string Format(const std::complex<U>& val) {
    return absl::StrFormat("(%s, %s)", absl::StrCat(val.real()),
                           absl::StrCat(val.imag()));
  }
};

// Element-level comparator and metric accumulator.
template <typename NativeT>
class ElementComparator {
 public:
  ElementComparator(const ComparisonOptions& options, int64_t total_elements)
      : options_(options) {
    result_.total_elements = total_elements;

    ErrorHeatmap& heatmap = result_.heatmap;
    heatmap.total_elements = total_elements;
    heatmap.target_abs = options.abs_error_bound;
    heatmap.target_rel = options.rel_error_bound;
    heatmap.abs_thresholds = BuildThresholds(options.abs_error_bound);
    heatmap.rel_thresholds = BuildThresholds(options.rel_error_bound);
    heatmap.yellow_threshold_pct = options.heatmap_yellow_pct;

    auto abs_it = absl::c_find(heatmap.abs_thresholds, options.abs_error_bound);
    heatmap.target_abs_idx =
        abs_it != heatmap.abs_thresholds.end()
            ? std::distance(heatmap.abs_thresholds.begin(), abs_it)
            : -1;

    auto rel_it = absl::c_find(heatmap.rel_thresholds, options.rel_error_bound);
    heatmap.target_rel_idx =
        rel_it != heatmap.rel_thresholds.end()
            ? std::distance(heatmap.rel_thresholds.begin(), rel_it)
            : -1;

    const int num_abs_thresh = heatmap.abs_thresholds.size();
    const int num_rel_thresh = heatmap.rel_thresholds.size();
    hist_2d_.assign(num_rel_thresh + 1,
                    std::vector<int64_t>(num_abs_thresh + 1, 0));

    result_.histogram.bins = CreateDefaultRelBins();
  }

  // Compares an individual element pair at linear index `idx`.
  void RecordElement(int64_t idx, NativeT clean_val, NativeT dirty_val) {
    constexpr double kInfinity = std::numeric_limits<double>::infinity();
    double abs_diff = 0.0;
    double rel_diff = 0.0;
    double signed_rel = 0.0;
    bool is_nan = false;
    bool is_inf = false;
    bool has_rel_error = false;

    if (ValueTraits<NativeT>::IsNan(clean_val) ||
        ValueTraits<NativeT>::IsNan(dirty_val)) {
      if (ValueTraits<NativeT>::IsNan(clean_val) &&
          ValueTraits<NativeT>::IsNan(dirty_val)) {
        result_.exact_matches++;
        abs_diff = 0.0;
        rel_diff = 0.0;
        signed_rel = 0.0;
      } else {
        is_nan = true;
        result_.nan_mismatches++;
        result_.mismatches++;
        abs_diff = kInfinity;
        rel_diff = kInfinity;
        signed_rel = kInfinity;
        result_.max_abs_error = std::max(result_.max_abs_error, abs_diff);
        result_.max_rel_error = std::max(result_.max_rel_error, rel_diff);
        if (result_.top_mismatches.size() < options_.max_mismatches_to_record) {
          result_.top_mismatches.push_back(
              {idx, ValueTraits<NativeT>::Format(clean_val),
               ValueTraits<NativeT>::Format(dirty_val), abs_diff, rel_diff});
        }
      }
    } else if (ValueTraits<NativeT>::IsInf(clean_val) ||
               ValueTraits<NativeT>::IsInf(dirty_val)) {
      if (clean_val == dirty_val) {
        result_.exact_matches++;
        abs_diff = 0.0;
        rel_diff = 0.0;
        signed_rel = 0.0;
      } else {
        is_inf = true;
        result_.inf_mismatches++;
        result_.mismatches++;
        abs_diff = kInfinity;
        rel_diff = kInfinity;
        signed_rel = kInfinity;
        result_.max_abs_error = std::max(result_.max_abs_error, abs_diff);
        result_.max_rel_error = std::max(result_.max_rel_error, rel_diff);
        if (result_.top_mismatches.size() < options_.max_mismatches_to_record) {
          result_.top_mismatches.push_back(
              {idx, ValueTraits<NativeT>::Format(clean_val),
               ValueTraits<NativeT>::Format(dirty_val), abs_diff, rel_diff});
        }
      }
    } else {
      if (clean_val == dirty_val) {
        result_.exact_matches++;
        abs_diff = 0.0;
        rel_diff = 0.0;
        signed_rel = 0.0;

        double clean_mag = ValueTraits<NativeT>::Magnitude(clean_val);
        if (clean_mag != 0.0) {
          has_rel_error = true;
          // Online Welford update for exact matches (signed_rel = 0.0)
          finite_rel_samples_++;
          double delta = 0.0 - mean_signed_rel_;
          mean_signed_rel_ += delta / finite_rel_samples_;
          double delta2 = 0.0 - mean_signed_rel_;
          m2_signed_rel_ += delta * delta2;

          min_signed_rel_ = std::min(min_signed_rel_, 0.0);
          max_signed_rel_ = std::max(max_signed_rel_, 0.0);
        }
      } else {
        abs_diff = ValueTraits<NativeT>::AbsDiff(dirty_val, clean_val);
        result_.max_abs_error = std::max(result_.max_abs_error, abs_diff);

        double clean_mag = ValueTraits<NativeT>::Magnitude(clean_val);
        if (clean_mag != 0.0) {
          has_rel_error = true;
          if constexpr (internal::is_complex_v<NativeT>) {
            rel_diff = abs_diff / clean_mag;
            signed_rel = rel_diff;
          } else {
            rel_diff = abs_diff / clean_mag;
            signed_rel = (dirty_val >= clean_val ? rel_diff : -rel_diff);
          }
          result_.max_rel_error = std::max(result_.max_rel_error, rel_diff);

          // Online Welford update guarded against subnormal overflow
          if (std::isfinite(signed_rel)) {
            finite_rel_samples_++;
            double delta = signed_rel - mean_signed_rel_;
            mean_signed_rel_ += delta / finite_rel_samples_;
            double delta2 = signed_rel - mean_signed_rel_;
            m2_signed_rel_ += delta * delta2;

            min_signed_rel_ = std::min(min_signed_rel_, signed_rel);
            max_signed_rel_ = std::max(max_signed_rel_, signed_rel);
          }
        }

        // Mismatch check against target bounds:
        // When clean is non-zero, both bounds must be exceeded.
        // When clean is zero, relative error is undefined so only abs bound
        // applies.
        bool is_mismatch =
            (abs_diff > options_.abs_error_bound) &&
            (!has_rel_error || rel_diff > options_.rel_error_bound);

        if (is_mismatch) {
          result_.mismatches++;
          if (result_.top_mismatches.size() <
              options_.max_mismatches_to_record) {
            result_.top_mismatches.push_back(
                {idx, ValueTraits<NativeT>::Format(clean_val),
                 ValueTraits<NativeT>::Format(dirty_val), abs_diff,
                 has_rel_error ? rel_diff : 0.0});
          }
        }
      }
    }

    // 1D histogram binning (only for finite relative errors or exact zeros)
    if (!is_nan && !is_inf && std::isfinite(signed_rel) &&
        (clean_val == dirty_val || has_rel_error)) {
      int bin_idx = FindRelBin(signed_rel, result_.histogram.bins);
      result_.histogram.bins[bin_idx].count++;
      result_.histogram.total_samples++;
    }

    // 2D heatmap binning
    int a_bin = absl::c_lower_bound(result_.heatmap.abs_thresholds, abs_diff) -
                result_.heatmap.abs_thresholds.begin();
    int r_bin =
        has_rel_error
            ? (absl::c_lower_bound(result_.heatmap.rel_thresholds, rel_diff) -
               result_.heatmap.rel_thresholds.begin())
            : (is_nan || is_inf || abs_diff > 0.0
                   ? static_cast<int>(result_.heatmap.rel_thresholds.size())
                   : 0);
    hist_2d_[r_bin][a_bin]++;
  }

  // Finalizes summary statistics (mean, stddev, median, 2D suffix sums)
  // and returns the final ComparisonResult.
  ComparisonResult Finalize() {
    RelErrorHistogram& histogram = result_.histogram;
    if (finite_rel_samples_ > 0) {
      histogram.min_rel_error = min_signed_rel_;
      histogram.max_rel_error = max_signed_rel_;
      histogram.mean_rel_error = mean_signed_rel_;
      histogram.std_dev_rel_error =
          finite_rel_samples_ > 1 ? std::sqrt(std::max(0.0, m2_signed_rel_) /
                                              (finite_rel_samples_ - 1))
                                  : 0.0;
    }

    if (histogram.total_samples > 0) {
      const int64_t target = (histogram.total_samples + 1) / 2;
      int64_t cumulative = 0;
      for (size_t i = 0; i < histogram.bins.size(); ++i) {
        cumulative += histogram.bins[i].count;
        if (cumulative >= target) {
          histogram.median_bin_index = static_cast<int>(i);
          break;
        }
      }
    }

    // 2D Suffix sum to compute mismatch counts
    ErrorHeatmap& heatmap = result_.heatmap;
    const int num_abs_thresh = heatmap.abs_thresholds.size();
    const int num_rel_thresh = heatmap.rel_thresholds.size();
    heatmap.mismatch_counts =
        ComputeHeatmapMismatchCounts(hist_2d_, num_rel_thresh, num_abs_thresh);

    result_.passed = (result_.mismatches == 0 && result_.nan_mismatches == 0 &&
                      result_.inf_mismatches == 0);

    result_.suggested_error_spec = ComputeSuggestedErrorSpec(
        result_.heatmap, result_.max_abs_error, result_.max_rel_error);

    return result_;
  }

  const ComparisonResult& result() const { return result_; }

 private:
  ComparisonOptions options_;
  ComparisonResult result_;
  std::vector<std::vector<int64_t>> hist_2d_;

  double min_signed_rel_ = std::numeric_limits<double>::infinity();
  double max_signed_rel_ = -std::numeric_limits<double>::infinity();
  double mean_signed_rel_ = 0.0;
  double m2_signed_rel_ = 0.0;
  int64_t finite_rel_samples_ = 0;
};

}  // namespace xla::compare_literals

#endif  // XLA_TOOLS_COMPARE_LITERALS_ELEMENT_COMPARATOR_H_
