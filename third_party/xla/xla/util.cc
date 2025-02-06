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

#include "xla/util.h"

#include <stdarg.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/const_init.h"
#include "absl/base/log_severity.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/stacktrace.h"

namespace xla {

std::vector<int64_t> ToMixedRadix(const int64_t n,
                                  absl::Span<const int64_t> bounds) {
  if (bounds.empty()) {
    return {};
  }

  std::vector<int64_t> digits;
  digits.reserve(bounds.size());
  int64_t divisor = Product(bounds);
  CHECK_GT(divisor, 0);
  int64_t remainder = n % divisor;
  for (const int64_t radix : bounds) {
    CHECK_GT(radix, 0);
    divisor /= radix;
    CHECK_GT(divisor, 0);

    // The divisor is always 1 for the last iteration.
    digits.push_back(remainder / divisor);
    remainder = remainder % divisor;
  }
  return digits;
}

absl::Status WithLogBacktrace(const absl::Status& status) {
  CHECK(!status.ok());
  VLOG(1) << status.ToString();
  VLOG(2) << tsl::CurrentStackTrace();
  return status;
}

ScopedLoggingTimer::ScopedLoggingTimer(absl::string_view label, bool enabled,
                                       const char* file, int line,
                                       TimerStats* timer_stats)
    : label_(label),
      file_(file),
      line_(line),
      timer_stats_(timer_stats),
      enabled_(enabled) {
  if (enabled_) {
    start_micros_ = tsl::Env::Default()->NowMicros();
  }
}

void ScopedLoggingTimer::StopAndLog() {
  if (enabled_) {
    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    double secs = (end_micros - start_micros_) / 1000000.0;

    TimerStats& stats = *timer_stats_;
    absl::MutexLock lock(&stats.stats_mutex);
    stats.cumulative_secs += secs;
    if (secs > stats.max_secs) {
      stats.max_secs = secs;
    }
    stats.times_called++;

    LOG(INFO).AtLocation(file_, line_)
        << label_ << " time: " << tsl::strings::HumanReadableElapsedTime(secs)
        << " (cumulative: "
        << tsl::strings::HumanReadableElapsedTime(stats.cumulative_secs)
        << ", max: " << tsl::strings::HumanReadableElapsedTime(stats.max_secs)
        << ", #called: " << stats.times_called << ")";
    enabled_ = false;
  }
}

ScopedLoggingTimer::~ScopedLoggingTimer() { StopAndLog(); }

absl::Status AddStatus(absl::Status prior, absl::string_view context) {
  CHECK(!prior.ok());
  return absl::Status{prior.code(),
                      absl::StrCat(context, ": ", prior.message())};
}

absl::Status AppendStatus(absl::Status prior, absl::string_view context) {
  CHECK(!prior.ok());
  return absl::Status{prior.code(),
                      absl::StrCat(prior.message(), ": ", context)};
}

std::string Reindent(absl::string_view original,
                     const absl::string_view indentation) {
  std::vector<std::string> pieces =
      absl::StrSplit(absl::string_view(original.data(), original.size()), '\n');
  return absl::StrJoin(
      pieces, "\n", [indentation](std::string* out, absl::string_view s) {
        absl::StrAppend(out, indentation, absl::StripAsciiWhitespace(s));
      });
}

template <typename FloatT>
static void RoundTripNanPayload(FloatT value, std::string* result) {
  static_assert(std::numeric_limits<FloatT>::has_quiet_NaN);
  static_assert(!std::is_same<FloatT, tsl::float8_e4m3fn>::value,
                "RoundTripNanPayload does not support E4M3FN");
  static_assert(!std::is_same<FloatT, tsl::float8_e4m3fnuz>::value,
                "RoundTripNanPayload does not support E4M3FNUZ");
  static_assert(!std::is_same<FloatT, tsl::float8_e5m2fnuz>::value,
                "RoundTripNanPayload does not support E5M2FNUZ");
  const int kPayloadBits = NanPayloadBits<FloatT>();
  if (Eigen::numext::isnan(value) && kPayloadBits > 0) {
    auto rep = absl::bit_cast<
        typename UnsignedIntegerTypeForSize<sizeof(FloatT)>::type>(value);
    auto payload = rep & NanPayloadBitMask<FloatT>();
    if (payload != QuietNanWithoutPayload<FloatT>()) {
      absl::StrAppendFormat(result, "(0x%x)", payload);
    }
  }
}

template <typename FloatT>
static std::string GenericRoundTripFpToString(FloatT value) {
  // TODO(majnemer): Remove this temporary variable once Eigen creates a symbol
  // definition for `max_digits10`.
  int max_decimal_digits = std::numeric_limits<FloatT>::max_digits10;
  return absl::StrFormat("%.*g", max_decimal_digits,
                         static_cast<double>(value));
}

std::string RoundTripFpToString(tsl::float4_e2m1fn value) {
  return GenericRoundTripFpToString(value);
}

std::string RoundTripFpToString(tsl::float8_e5m2 value) {
  std::string result = GenericRoundTripFpToString(value);
  RoundTripNanPayload(value, &result);
  return result;
}

std::string RoundTripFpToString(tsl::float8_e4m3 value) {
  std::string result = GenericRoundTripFpToString(value);
  RoundTripNanPayload(value, &result);
  return result;
}

std::string RoundTripFpToString(tsl::float8_e4m3fnuz value) {
  std::string result = GenericRoundTripFpToString(value);
  return result;
}

std::string RoundTripFpToString(tsl::float8_e5m2fnuz value) {
  std::string result = GenericRoundTripFpToString(value);
  return result;
}

std::string RoundTripFpToString(tsl::float8_e4m3fn value) {
  std::string result = GenericRoundTripFpToString(value);
  return result;
}

std::string RoundTripFpToString(tsl::float8_e4m3b11fnuz value) {
  std::string result = GenericRoundTripFpToString(value);
  return result;
}

std::string RoundTripFpToString(tsl::float8_e3m4 value) {
  std::string result = GenericRoundTripFpToString(value);
  RoundTripNanPayload(value, &result);
  return result;
}

std::string RoundTripFpToString(tsl::float8_e8m0fnu value) {
  std::string result = GenericRoundTripFpToString(value);
  return result;
}

std::string RoundTripFpToString(bfloat16 value) {
  std::string result = GenericRoundTripFpToString(value);
  RoundTripNanPayload(value, &result);
  return result;
}

std::string RoundTripFpToString(half value) {
  std::string result = GenericRoundTripFpToString(value);
  RoundTripNanPayload(value, &result);
  return result;
}

std::string RoundTripFpToString(float value) {
  float parsed_result;
  std::string result =
      absl::StrFormat("%.*g", std::numeric_limits<float>::digits10, value);
  if (!absl::SimpleAtof(result, &parsed_result) || parsed_result != value) {
    result = GenericRoundTripFpToString(value);
  }
  RoundTripNanPayload(value, &result);
  return result;
}

std::string RoundTripFpToString(double value) {
  double parsed_result;
  std::string result =
      absl::StrFormat("%.*g", std::numeric_limits<double>::digits10, value);
  if (!absl::SimpleAtod(result, &parsed_result) || parsed_result != value) {
    result = GenericRoundTripFpToString(value);
  }
  RoundTripNanPayload(value, &result);
  return result;
}

PaddingConfig MakeNoPaddingConfig(int64_t rank) {
  PaddingConfig padding_config;
  for (int64_t dnum = 0; dnum < rank; ++dnum) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(0);
    dimension->set_edge_padding_high(0);
    dimension->set_interior_padding(0);
  }
  return padding_config;
}

PaddingConfig MakeEdgePaddingConfig(
    absl::Span<const std::pair<int64_t, int64_t>> padding) {
  PaddingConfig padding_config;
  for (const std::pair<int64_t, int64_t>& dim : padding) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(dim.first);
    dimension->set_edge_padding_high(dim.second);
    dimension->set_interior_padding(0);
  }
  return padding_config;
}

bool HasInteriorPadding(const PaddingConfig& config) {
  for (const auto& dim : config.dimensions()) {
    if (dim.interior_padding() != 0) {
      return true;
    }
  }
  return false;
}

namespace {
std::string HumanReadableNumOps(double flops, double nanoseconds,
                                absl::string_view op_prefix) {
  if (nanoseconds == 0) {
    return absl::StrCat("NaN ", op_prefix, "OP/s");
  }
  double nano_flops = flops / nanoseconds;
  std::string throughput =
      tsl::strings::HumanReadableNum(static_cast<int64_t>(nano_flops * 1e9));
  absl::string_view sp(throughput);
  // Use the more common "G(FLOPS)", rather than "B(FLOPS)"
  if (absl::EndsWith(sp, "B") ||  // Ends in 'B', ignoring case
      absl::EndsWith(sp, "b")) {
    *throughput.rbegin() = 'G';
  }
  throughput += absl::StrCat(op_prefix, "OP/s");
  return throughput;
}
}  // namespace

std::string HumanReadableNumFlops(double flops, double nanoseconds) {
  return HumanReadableNumOps(flops, nanoseconds, "FL");
}

std::string HumanReadableNumTranscendentalOps(double trops,
                                              double nanoseconds) {
  return HumanReadableNumOps(trops, nanoseconds, "TR");
}

void LogLines(absl::LogSeverity sev, absl::string_view text, const char* fname,
              int lineno) {
  const absl::LogSeverity orig_sev = sev;
  if (sev == absl::LogSeverity::kFatal) {
    sev = absl::LogSeverity::kError;
  }

  // Protect calls with a mutex so we don't interleave calls to LogLines from
  // multiple threads.
  static absl::Mutex log_lines_mu(absl::kConstInit);
  absl::MutexLock lock(&log_lines_mu);

  size_t cur = 0;
  while (cur < text.size()) {
    size_t eol = text.find('\n', cur);
    if (eol == absl::string_view::npos) {
      eol = text.size();
    }
    auto msg = text.substr(cur, eol - cur);
    tsl::internal::LogString(fname, lineno, sev,
                             std::string(msg.data(), msg.size()));
    cur = eol + 1;
  }

  if (orig_sev == absl::LogSeverity::kFatal) {
    tsl::internal::LogString(fname, lineno, orig_sev,
                             "Aborting due to errors.");
  }
}

int64_t Product(absl::Span<const int64_t> xs) {
  return std::accumulate(xs.begin(), xs.end(), static_cast<int64_t>(1),
                         std::multiplies<int64_t>());
}

std::vector<int64_t> ElemwiseProduct(absl::Span<const int64_t> a,
                                     absl::Span<const int64_t> b) {
  CHECK_EQ(a.size(), b.size());
  std::vector<int64_t> result;
  std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(result),
                 std::multiplies<int64_t>());
  return result;
}

absl::InlinedVector<std::pair<int64_t, int64_t>, 8> CommonFactors(
    absl::Span<const int64_t> a, absl::Span<const int64_t> b) {
  CHECK_EQ(Product(a), Product(b));
  absl::InlinedVector<std::pair<int64_t, int64_t>, 8> bounds;
  if (absl::c_equal(a, b)) {
    bounds.reserve(a.size() + 1);
    for (int64_t i = 0; i <= a.size(); ++i) {
      bounds.emplace_back(i, i);
    }
    return bounds;
  }
  int64_t i = 0, j = 0, prior_i = -1, prior_j = -1;
  while (i < a.size() && j < b.size() && a[i] == b[j]) {
    std::tie(prior_i, prior_j) = std::make_pair(i, j);
    bounds.emplace_back(i, j);
    ++i;
    ++j;
  }
  // If the product is different after filtering out zeros, return full group.
  // E.g.,:
  // a={0, 10 ,3}
  //       ^
  //      i=1
  //
  // b={0, 3}
  //       ^
  //      j=1
  if (Product(a.subspan(i)) != Product(b.subspan(j))) {
    return {std::make_pair(0, 0), std::make_pair(a.size(), b.size())};
  }
  if (0 == Product(a.subspan(i))) {
    bounds.push_back(std::make_pair(i, j));
    bounds.push_back(std::make_pair(a.size(), b.size()));
    return bounds;
  }

  for (int64_t partial_size_a = 1, partial_size_b = 1;;) {
    if (partial_size_a == partial_size_b && (i > prior_i || j > prior_j)) {
      std::tie(prior_i, prior_j) = std::make_pair(i, j);
      bounds.emplace_back(i, j);
      continue;
    }
    bool in_bounds_i = i < a.size();
    bool in_bounds_j = j < b.size();
    if (!(in_bounds_i || in_bounds_j)) {
      break;
    }
    bool next_a =
        partial_size_a < partial_size_b ||
        (in_bounds_i &&
         (!in_bounds_j || (partial_size_a == partial_size_b && a[i] <= b[j])));
    bool next_b =
        partial_size_b < partial_size_a ||
        (in_bounds_j &&
         (!in_bounds_i || (partial_size_b == partial_size_a && b[j] <= a[i])));
    if (next_a) {
      partial_size_a *= a[i];
      ++i;
    }
    if (next_b) {
      partial_size_b *= b[j];
      ++j;
    }
  }
  return bounds;
}

ConvertedDimensionNumbers ConvertDimensionNumbers(
    absl::Span<const int64_t> from_dimensions,
    absl::Span<const int64_t> from_sizes, absl::Span<const int64_t> to_sizes) {
  ConvertedDimensionNumbers dimensions;
  auto common_factors = CommonFactors(from_sizes, to_sizes);
  for (int64_t i = 0; i < common_factors.size() - 1; ++i) {
    bool any_present = false;
    bool all_present = true;
    for (int64_t d = common_factors[i].first; d < common_factors[i + 1].first;
         ++d) {
      const bool present = absl::c_linear_search(from_dimensions, d);
      any_present |= present;
      all_present &= present;
    }
    if (all_present) {
      for (int64_t d = common_factors[i].second;
           d < common_factors[i + 1].second; ++d) {
        dimensions.to_dimensions.push_back(d);
      }
      for (int64_t d = common_factors[i].first; d < common_factors[i + 1].first;
           ++d) {
        dimensions.transformed_from_dimensions.push_back(d);
      }
    } else if (any_present) {
      // Try to find if there is a to dimension that is like (from) [2,32] ->
      // (to) [4,4,4] to detect that from dimension 1 can be partially mapped
      // into dimension 1 and 2 of the to sizes with a partial size of 2.
      if (common_factors[i].first + 2 == common_factors[i + 1].first &&
          absl::c_linear_search(from_dimensions, common_factors[i].first + 1)) {
        int64_t from_size = from_sizes[common_factors[i + 1].first - 1];
        bool has_to_dim = false;
        for (int64_t to_dim = common_factors[i + 1].second - 1;
             to_dim >= common_factors[i].second; --to_dim) {
          const int64_t to_size = to_sizes[to_dim];
          if (from_size % to_size == 0) {
            has_to_dim = true;
            from_size /= to_size;
            dimensions.to_dimensions.push_back(to_dim);
          } else {
            break;
          }
        }
        if (has_to_dim) {
          dimensions.split_from_sizes.push_back(from_size);
          dimensions.split_from_dimensions.push_back(common_factors[i].first +
                                                     1);
        }
      }
      for (int64_t d = common_factors[i].first; d < common_factors[i + 1].first;
           ++d) {
        if (absl::c_linear_search(from_dimensions, d)) {
          dimensions.untransformed_from_dimensions.push_back(d);
        }
      }
    }
  }
  absl::c_sort(dimensions.to_dimensions);
  return dimensions;
}

DimensionVector GetNonContractingDims(
    int64_t rank, absl::Span<const int64_t> contracting_dim_numbers,
    absl::Span<const int64_t> batch_dim_numbers) {
  DimensionVector non_contracting_dim_numbers;
  for (int64_t i = 0; i < rank; ++i) {
    if (!absl::c_linear_search(contracting_dim_numbers, i) &&
        !absl::c_linear_search(batch_dim_numbers, i)) {
      non_contracting_dim_numbers.push_back(i);
    }
  }
  return non_contracting_dim_numbers;
}

std::string SanitizeFileName(std::string file_name) {
  for (char& c : file_name) {
    if (c == '/' || c == '\\' || c == '[' || c == ']' || c == ' ') {
      c = '_';
    }
  }
  return file_name;
}

bool DistinctNumbersAreConsecutiveIfSorted(absl::Span<const int64_t> seq) {
  return *absl::c_max_element(seq) - *absl::c_min_element(seq) ==
         seq.size() - 1;
}

}  // namespace xla
