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

#include <numeric>
#include <stdarg.h>
#include <numeric>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stacktrace.h"

namespace xla {
namespace {

// Logs the provided status message with a backtrace.
Status WithLogBacktrace(const Status& status) {
  CHECK(!status.ok());
  VLOG(1) << status.ToString();
  VLOG(1) << tensorflow::CurrentStackTrace();
  return status;
}

}  // namespace

ScopedLoggingTimer::ScopedLoggingTimer(const string& label, int32 vlog_level)
    : label(label), vlog_level(vlog_level) {
  if (VLOG_IS_ON(vlog_level)) {
    start_micros = tensorflow::Env::Default()->NowMicros();
  }
}

ScopedLoggingTimer::~ScopedLoggingTimer() {
  if (VLOG_IS_ON(vlog_level)) {
    uint64 end_micros = tensorflow::Env::Default()->NowMicros();
    double secs = (end_micros - start_micros) / 1000000.0;

    LOG(INFO) << label << " time: "
              << tensorflow::strings::HumanReadableElapsedTime(secs);
  }
}

Status AddStatus(Status prior, tensorflow::StringPiece context) {
  CHECK(!prior.ok());
  return Status{prior.code(), tensorflow::strings::StrCat(
                                  context, ": ", prior.error_message())};
}

Status AppendStatus(Status prior, tensorflow::StringPiece context) {
  CHECK(!prior.ok());
  return Status{prior.code(), tensorflow::strings::StrCat(prior.error_message(),
                                                          ": ", context)};
}

// Implementation note: we can't common these out (without using macros) because
// they all need to va_start/va_end their varargs in their frame.

Status InvalidArgument(const char* format, ...) {
  string message;
  va_list args;
  va_start(args, format);
  tensorflow::strings::Appendv(&message, format, args);
  va_end(args);
  return WithLogBacktrace(tensorflow::errors::InvalidArgument(message));
}

Status Unimplemented(const char* format, ...) {
  string message;
  va_list args;
  va_start(args, format);
  tensorflow::strings::Appendv(&message, format, args);
  va_end(args);
  return WithLogBacktrace(tensorflow::errors::Unimplemented(message));
}

Status InternalError(const char* format, ...) {
  string message;
  va_list args;
  va_start(args, format);
  tensorflow::strings::Appendv(&message, format, args);
  va_end(args);
  return WithLogBacktrace(tensorflow::errors::Internal(message));
}

Status FailedPrecondition(const char* format, ...) {
  string message;
  va_list args;
  va_start(args, format);
  tensorflow::strings::Appendv(&message, format, args);
  va_end(args);
  return WithLogBacktrace(tensorflow::errors::FailedPrecondition(message));
}

Status Cancelled(const char* format, ...) {
  string message;
  va_list args;
  va_start(args, format);
  tensorflow::strings::Appendv(&message, format, args);
  va_end(args);
  return WithLogBacktrace(tensorflow::errors::Cancelled(message));
}

Status ResourceExhausted(const char* format, ...) {
  string message;
  va_list args;
  va_start(args, format);
  tensorflow::strings::Appendv(&message, format, args);
  va_end(args);
  return WithLogBacktrace(tensorflow::errors::ResourceExhausted(message));
}

Status NotFound(const char* format, ...) {
  string message;
  va_list args;
  va_start(args, format);
  tensorflow::strings::Appendv(&message, format, args);
  va_end(args);
  return WithLogBacktrace(tensorflow::errors::NotFound(message));
}

Status Unavailable(const char* format, ...) {
  string message;
  va_list args;
  va_start(args, format);
  tensorflow::strings::Appendv(&message, format, args);
  va_end(args);
  return WithLogBacktrace(tensorflow::errors::Unavailable(message));
}

string Reindent(tensorflow::StringPiece original,
                const tensorflow::StringPiece indentation) {
  std::vector<string> pieces = tensorflow::str_util::Split(
      tensorflow::StringPiece(original.data(), original.size()), '\n');
  return tensorflow::str_util::Join(
      pieces, "\n", [indentation](string* out, string s) {
        tensorflow::StringPiece piece(s);
        tensorflow::str_util::RemoveWhitespaceContext(&piece);
        tensorflow::strings::StrAppend(out, indentation, piece);
      });
}

bool IsPermutation(tensorflow::gtl::ArraySlice<int64> permutation, int64 rank) {
  if (rank != permutation.size()) {
    return false;
  }
  std::vector<int64> output(permutation.size(), -1);
  for (auto index : permutation) {
    CHECK_GE(index, 0);
    CHECK_LT(index, rank);
    output[index] = 0;
  }
  return std::find(output.begin(), output.end(), -1) == output.end();
}

std::vector<int64> InversePermutation(
    tensorflow::gtl::ArraySlice<int64> input_permutation) {
  DCHECK(IsPermutation(input_permutation, input_permutation.size()));
  std::vector<int64> output_permutation(input_permutation.size(), -1);
  for (size_t i = 0; i < input_permutation.size(); ++i) {
    output_permutation[input_permutation[i]] = i;
  }
  return output_permutation;
}

std::vector<int64> ComposePermutations(tensorflow::gtl::ArraySlice<int64> p1,
                                       tensorflow::gtl::ArraySlice<int64> p2) {
  CHECK_EQ(p1.size(), p2.size());
  std::vector<int64> output;
  for (size_t i = 0; i < p1.size(); ++i) {
    output.push_back(p1[p2[i]]);
  }
  return output;
}

bool IsIdentityPermutation(tensorflow::gtl::ArraySlice<int64> p) {
  for (int64 i = 0; i < p.size(); ++i) {
    if (p[i] != i) {
      return false;
    }
  }
  return true;
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
                           tensorflow::StringPiece op_prefix) {
  if (nanoseconds == 0) {
    return tensorflow::strings::StrCat("NaN ", op_prefix, "OP/s");
  }
  double nano_flops = flops / nanoseconds;
  string throughput = tensorflow::strings::HumanReadableNum(
      static_cast<int64>(nano_flops * 1e9));
  tensorflow::StringPiece sp(throughput);
  // Use the more common "G(FLOPS)", rather than "B(FLOPS)"
  if (sp.ends_with("B") ||  // Ends in 'B', ignoring case
      sp.ends_with("b")) {
    *throughput.rbegin() = 'G';
  }
  throughput += tensorflow::strings::StrCat(op_prefix, "OP/s");
  return throughput;
}
}  // namespace

string HumanReadableNumFlops(double flops, double nanoseconds) {
  return HumanReadableNumOps(flops, nanoseconds, "FL");
}

string HumanReadableNumTranscendentalOps(double trops, double nanoseconds) {
  return HumanReadableNumOps(trops, nanoseconds, "TR");
}

void LogLines(int sev, tensorflow::StringPiece text, const char* fname,
              int lineno) {
  const int orig_sev = sev;
  if (sev == tensorflow::FATAL) {
    sev = tensorflow::ERROR;
  }

  size_t cur = 0;
  while (cur < text.size()) {
    size_t eol = text.find('\n', cur);
    if (eol == tensorflow::StringPiece::npos) {
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

int64 Product(tensorflow::gtl::ArraySlice<int64> xs) {
  return std::accumulate(xs.begin(), xs.end(), 1, std::multiplies<int64>());
}

std::vector<std::pair<int64, int64>> CommonFactors(
    tensorflow::gtl::ArraySlice<int64> a,
    tensorflow::gtl::ArraySlice<int64> b) {
  CHECK_EQ(Product(a), Product(b));
  if (0 == Product(a)) {
    return {std::make_pair(0, 0), std::make_pair(a.size(), b.size())};
  }

  std::vector<std::pair<int64, int64>> bounds;
  for (int64 i = 0, j = 0, prior_i = -1, prior_j = -1, partial_size_a = 1,
             partial_size_b = 1;
       ;) {
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

}  // namespace xla
