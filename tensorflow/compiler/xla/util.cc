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

#include "tensorflow/compiler/xla/util.h"

#include <stdarg.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/stacktrace.h"

namespace xla {

Status WithLogBacktrace(const Status& status) {
  CHECK(!status.ok());
  VLOG(1) << status.ToString();
  VLOG(2) << tensorflow::CurrentStackTrace();
  return status;
}

ScopedLoggingTimer::ScopedLoggingTimer(const std::string& label, bool enabled,
                                       const char* file, int line,
                                       TimerStats* timer_stats)
    : enabled_(enabled),
      file_(file),
      line_(line),
      label_(label),
      timer_stats_(timer_stats) {
  if (enabled_) {
    start_micros_ = tensorflow::Env::Default()->NowMicros();
  }
}

void ScopedLoggingTimer::StopAndLog() {
  if (enabled_) {
    uint64 end_micros = tensorflow::Env::Default()->NowMicros();
    double secs = (end_micros - start_micros_) / 1000000.0;

    TimerStats& stats = *timer_stats_;
    tensorflow::mutex_lock lock(stats.stats_mutex);
    stats.cumulative_secs += secs;
    if (secs > stats.max_secs) {
      stats.max_secs = secs;
    }
    stats.times_called++;

    LOG(INFO).AtLocation(file_, line_)
        << label_
        << " time: " << tensorflow::strings::HumanReadableElapsedTime(secs)
        << " (cumulative: "
        << tensorflow::strings::HumanReadableElapsedTime(stats.cumulative_secs)
        << ", max: "
        << tensorflow::strings::HumanReadableElapsedTime(stats.max_secs)
        << ", #called: " << stats.times_called << ")";
    enabled_ = false;
  }
}

ScopedLoggingTimer::~ScopedLoggingTimer() { StopAndLog(); }

Status AddStatus(Status prior, absl::string_view context) {
  CHECK(!prior.ok());
  return Status{prior.code(),
                absl::StrCat(context, ": ", prior.error_message())};
}

Status AppendStatus(Status prior, absl::string_view context) {
  CHECK(!prior.ok());
  return Status{prior.code(),
                absl::StrCat(prior.error_message(), ": ", context)};
}

string Reindent(absl::string_view original,
                const absl::string_view indentation) {
  std::vector<string> pieces =
      absl::StrSplit(absl::string_view(original.data(), original.size()), '\n');
  return absl::StrJoin(pieces, "\n", [indentation](string* out, string s) {
    absl::StrAppend(out, indentation, absl::StripAsciiWhitespace(s));
  });
}

template <typename IntT, typename FloatT>
static void RoundTripNanPayload(FloatT value, std::string* result) {
  const int kPayloadBits = NanPayloadBits<FloatT>();
  if (std::isnan(value) && kPayloadBits > 0) {
    auto rep = absl::bit_cast<IntT>(value);
    auto payload = rep & NanPayloadBitMask<FloatT>();
    if (payload != QuietNanWithoutPayload<FloatT>()) {
      absl::StrAppendFormat(result, "(0x%x)", payload);
    }
  }
}

string RoundTripFpToString(tensorflow::bfloat16 value) {
  std::string result = absl::StrFormat("%.4g", static_cast<float>(value));
  RoundTripNanPayload<uint16_t>(value, &result);
  return result;
}

string RoundTripFpToString(Eigen::half value) {
  std::string result = absl::StrFormat("%.5g", static_cast<float>(value));
  RoundTripNanPayload<uint16_t>(value, &result);
  return result;
}

string RoundTripFpToString(float value) {
  char buffer[tensorflow::strings::kFastToBufferSize];
  tensorflow::strings::FloatToBuffer(value, buffer);
  std::string result = buffer;
  RoundTripNanPayload<uint32_t>(value, &result);
  return result;
}

string RoundTripFpToString(double value) {
  char buffer[tensorflow::strings::kFastToBufferSize];
  tensorflow::strings::DoubleToBuffer(value, buffer);
  std::string result = buffer;
  RoundTripNanPayload<uint64_t>(value, &result);
  return result;
}

PaddingConfig MakeNoPaddingConfig(int64 rank) {
  PaddingConfig padding_config;
  for (int64 dnum = 0; dnum < rank; ++dnum) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(0);
    dimension->set_edge_padding_high(0);
    dimension->set_interior_padding(0);
  }
  return padding_config;
}

PaddingConfig MakeEdgePaddingConfig(
    absl::Span<const std::pair<int64, int64>> padding) {
  PaddingConfig padding_config;
  for (const std::pair<int64, int64>& dim : padding) {
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
string HumanReadableNumOps(double flops, double nanoseconds,
                           absl::string_view op_prefix) {
  if (nanoseconds == 0) {
    return absl::StrCat("NaN ", op_prefix, "OP/s");
  }
  double nano_flops = flops / nanoseconds;
  string throughput = tensorflow::strings::HumanReadableNum(
      static_cast<int64>(nano_flops * 1e9));
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

string HumanReadableNumFlops(double flops, double nanoseconds) {
  return HumanReadableNumOps(flops, nanoseconds, "FL");
}

string HumanReadableNumTranscendentalOps(double trops, double nanoseconds) {
  return HumanReadableNumOps(trops, nanoseconds, "TR");
}

void LogLines(int sev, absl::string_view text, const char* fname, int lineno) {
  const int orig_sev = sev;
  if (sev == tensorflow::FATAL) {
    sev = tensorflow::ERROR;
  }

  // Protect calls with a mutex so we don't interleave calls to LogLines from
  // multiple threads.
  static tensorflow::mutex log_lines_mu(tensorflow::LINKER_INITIALIZED);
  tensorflow::mutex_lock lock(log_lines_mu);

  size_t cur = 0;
  while (cur < text.size()) {
    size_t eol = text.find('\n', cur);
    if (eol == absl::string_view::npos) {
      eol = text.size();
    }
    auto msg = text.substr(cur, eol - cur);
    tensorflow::internal::LogString(fname, lineno, sev,
                                    string(msg.data(), msg.size()));
    cur = eol + 1;
  }

  if (orig_sev == tensorflow::FATAL) {
    tensorflow::internal::LogString(fname, lineno, orig_sev,
                                    "Aborting due to errors.");
  }
}

int64 Product(absl::Span<const int64> xs) {
  return std::accumulate(xs.begin(), xs.end(), static_cast<int64>(1),
                         std::multiplies<int64>());
}

absl::InlinedVector<std::pair<int64, int64>, 8> CommonFactors(
    absl::Span<const int64> a, absl::Span<const int64> b) {
  CHECK_EQ(Product(a), Product(b));
  absl::InlinedVector<std::pair<int64, int64>, 8> bounds;
  if (absl::c_equal(a, b)) {
    bounds.reserve(a.size() + 1);
    for (int64 i = 0; i <= a.size(); ++i) {
      bounds.emplace_back(i, i);
    }
    return bounds;
  }
  int64 i = 0, j = 0, prior_i = -1, prior_j = -1;
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

  for (int64 partial_size_a = 1, partial_size_b = 1;;) {
    if (partial_size_a == partial_size_b && (i > prior_i || j > prior_j)) {
      std::tie(prior_i, prior_j) = std::make_pair(i, j);
      bounds.emplace_back(i, j);
      continue;
    }
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
    absl::Span<const int64> from_dimensions, absl::Span<const int64> from_sizes,
    absl::Span<const int64> to_sizes) {
  ConvertedDimensionNumbers dimensions;
  auto common_factors = CommonFactors(from_sizes, to_sizes);
  for (int64 i = 0; i < common_factors.size() - 1; ++i) {
    bool any_present = false;
    bool all_present = true;
    for (int64 d = common_factors[i].first; d < common_factors[i + 1].first;
         ++d) {
      const bool present = absl::c_linear_search(from_dimensions, d);
      any_present |= present;
      all_present &= present;
    }
    if (all_present) {
      for (int64 d = common_factors[i].second; d < common_factors[i + 1].second;
           ++d) {
        dimensions.to_dimensions.push_back(d);
      }
      for (int64 d = common_factors[i].first; d < common_factors[i + 1].first;
           ++d) {
        dimensions.transformed_from_dimensions.push_back(d);
      }
    } else if (any_present) {
      for (int64 d = common_factors[i].first; d < common_factors[i + 1].first;
           ++d) {
        if (absl::c_linear_search(from_dimensions, d)) {
          dimensions.untransformed_from_dimensions.push_back(d);
        }
      }
    }
  }
  return dimensions;
}
string SanitizeFileName(string file_name) {
  for (char& c : file_name) {
    if (c == '/' || c == '\\' || c == '[' || c == ']' || c == ' ') {
      c = '_';
    }
  }
  return file_name;
}

// Utility function to split a double-precision float (F64) into a pair of F32s.
// For a p-bit number, and a splitting point (p/2) <= s <= (p - 1), the
// algorithm produces a (p - s)-bit value 'hi' and a non-overlapping (s - 1)-bit
// value 'lo'. See Theorem 4 in [1] (attributed to Dekker) or [2] for the
// original theorem by Dekker.
//
// For double-precision F64s, which contain a 53 bit mantissa (52 of them
// explicit), we can represent the most significant 49 digits as the unevaluated
// sum of two single-precision floats 'hi' and 'lo'. The 'hi' float stores the
// most significant 24 bits and the sign bit of 'lo' together with its mantissa
// store the remaining 25 bits. The exponent of the resulting representation is
// still restricted to 8 bits of F32.
//
// References:
// [1] A. Thall, Extended-Precision Floating-Point Numbers for GPU Computation,
//     SIGGRAPH Research Posters, 2006.
//     (http://andrewthall.org/papers/df64_qf128.pdf)
// [2] T. J. Dekker, A floating point technique for extending the available
//     precision, Numerische Mathematik, vol. 18, pp. 224–242, 1971.
std::pair<float, float> SplitF64ToF32(double x) {
  const float x_f32 = static_cast<float>(x);

  // Early return if x is an infinity or NaN.
  if (!std::isfinite(x_f32)) {
    // Only values within the range of F32 are supported, unless it is infinity.
    // Small values with large negative exponents would be rounded to zero.
    if (std::isfinite(x)) {
      LOG(WARNING) << "Out of range F64 constant detected: " << x;
    }
    return std::make_pair(x_f32, 0.0f);
  }

  // The high float is simply the double rounded to the nearest float. Because
  // we are rounding to nearest with ties to even, the error introduced in
  // rounding is less than half an ULP in the high ULP.
  const float hi = x_f32;
  // We can compute the low term using Sterbenz' lemma: If a and b are two
  // positive floating point numbers and a/2 ≤ b ≤ 2a, then their difference can
  // be computed exactly.
  // Note: the difference is computed exactly but is rounded to the nearest
  // float which will introduce additional error.
  const float lo = static_cast<float>(x - static_cast<double>(hi));
  return std::make_pair(hi, lo);
}

}  // namespace xla
