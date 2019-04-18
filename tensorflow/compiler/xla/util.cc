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
#include <numeric>

#include "absl/container/inlined_vector.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stacktrace.h"

namespace xla {

Status WithLogBacktrace(const Status& status) {
  CHECK(!status.ok());
  VLOG(1) << status.ToString();
  VLOG(1) << tensorflow::CurrentStackTrace();
  return status;
}

ScopedLoggingTimer::ScopedLoggingTimer(const string& label, bool enabled)
    : enabled(enabled), label(label) {
  if (enabled) {
    start_micros = tensorflow::Env::Default()->NowMicros();
  }
}

ScopedLoggingTimer::~ScopedLoggingTimer() {
  if (enabled) {
    uint64 end_micros = tensorflow::Env::Default()->NowMicros();
    double secs = (end_micros - start_micros) / 1000000.0;

    LOG(INFO) << label << " time: "
              << tensorflow::strings::HumanReadableElapsedTime(secs);
  }
}

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

bool IsPermutation(absl::Span<const int64> permutation, int64 rank) {
  if (rank != permutation.size()) {
    return false;
  }
  absl::InlinedVector<int64, 8> trivial_permutation(rank);
  absl::c_iota(trivial_permutation, 0);
  return absl::c_is_permutation(permutation, trivial_permutation);
}

std::vector<int64> InversePermutation(
    absl::Span<const int64> input_permutation) {
  DCHECK(IsPermutation(input_permutation, input_permutation.size()));
  std::vector<int64> output_permutation(input_permutation.size(), -1);
  for (size_t i = 0; i < input_permutation.size(); ++i) {
    output_permutation[input_permutation[i]] = i;
  }
  return output_permutation;
}

std::vector<int64> ComposePermutations(absl::Span<const int64> p1,
                                       absl::Span<const int64> p2) {
  CHECK_EQ(p1.size(), p2.size());
  std::vector<int64> output;
  for (size_t i = 0; i < p1.size(); ++i) {
    output.push_back(p1[p2[i]]);
  }
  return output;
}

bool IsIdentityPermutation(absl::Span<const int64> permutation) {
  for (int64 i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != i) {
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

std::vector<std::pair<int64, int64>> CommonFactors(absl::Span<const int64> a,
                                                   absl::Span<const int64> b) {
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

string SanitizeFileName(string file_name) {
  for (char& c : file_name) {
    if (c == '/' || c == '\\' || c == '[' || c == ']' || c == ' ') {
      c = '_';
    }
  }
  return file_name;
}

}  // namespace xla
