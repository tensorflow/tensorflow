// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_text/core/kernels/boise_offset_converter.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace text {

bool IsRightOutsideSpan(int token_start, int token_end, int span_start,
                        int span_end) {
  // Token:        |------)
  // Span: |-----)
  return token_start >= span_end;
}

bool IsLeftOutsideSpan(int token_start, int token_end, int span_start,
                       int span_end) {
  // Token:        |------)
  // Span:                   |-----)
  return token_end <= span_start;
}

bool IsStartOfSpan(int token_start, int token_end, int span_start,
                   int span_end) {
  // Returns true if the token overlaps with the span from the
  // left side (i.e. start) of the span, but not have the span inside.
  // Token:        |-------)
  // Span:              |-----)
  return token_start <= span_start && token_end > span_start &&
         token_end <= span_end;
}

bool IsEndOfSpan(int token_start, int token_end, int span_start, int span_end) {
  // Returns true if the token overlaps with the span from the
  // right side (i.e. end) of the span, but not have the span inside.
  // Token:                  |------)
  // Span:               |------)
  return token_start < span_end && token_end >= span_end &&
         token_start >= span_start;
}

bool IsInsideSpan(int token_start, int token_end, int span_start,
                  int span_end) {
  // Token:        |------)
  // Span:       |-----------)
  return token_start >= span_start && token_end <= span_end;
}

absl::StatusOr<std::vector<std::string>> OffsetsToBoiseTags(
    const std::vector<int>& token_begin_offsets,
    const std::vector<int>& token_end_offsets,
    const std::vector<int>& span_begin_offsets,
    const std::vector<int>& span_end_offsets,
    const std::vector<std::string>& span_type,
    const bool use_strict_boundary_mode) {
  // Verify that token vectors are all the same size
  if (token_begin_offsets.size() != token_end_offsets.size()) {
    return absl::InvalidArgumentError("Token offsets must have the same size");
  }
  if (span_begin_offsets.size() != span_end_offsets.size() ||
      span_begin_offsets.size() != span_type.size()) {
    return absl::InvalidArgumentError("Span offsets must have the same size");
  }

  // Iterate through tokens
  std::vector<std::string> results;
  int span_index = 0;
  for (int i = 0; i < token_begin_offsets.size(); ++i) {
    int token_start = token_begin_offsets[i];
    int token_end = token_end_offsets[i];
    std::string potential_span_type = "O";
    bool recorded = false;

    while (span_index < span_begin_offsets.size() && !recorded) {
      int span_start = span_begin_offsets[span_index];
      int span_end = span_end_offsets[span_index];

      if (IsLeftOutsideSpan(token_start, token_end, span_start, span_end)) {
        results.push_back(potential_span_type);
        recorded = true;
      } else if (IsRightOutsideSpan(token_start, token_end, span_start,
                                    span_end)) {
        span_index++;
      } else if (IsStartOfSpan(token_start, token_end, span_start, span_end)) {
        if (IsEndOfSpan(token_start, token_end, span_start, span_end)) {
          results.push_back(absl::StrCat("S-", span_type[span_index]));
          span_index++;
          recorded = true;
        } else {
          if (use_strict_boundary_mode && token_start != span_start) {
            results.push_back(potential_span_type);
            recorded = true;
          } else {
            results.push_back(absl::StrCat("B-", span_type[span_index]));
            recorded = true;
          }
        }
      } else if (IsEndOfSpan(token_start, token_end, span_start, span_end)) {
        if (use_strict_boundary_mode && token_end != span_end) {
          results.push_back(potential_span_type);
          recorded = true;
        } else {
          potential_span_type = absl::StrCat("E-", span_type[span_index]);
        }
        span_index++;
      } else if (IsInsideSpan(token_start, token_end, span_start, span_end)) {
        // token:     |--)
        // span:   |---------)
        results.push_back(absl::StrCat("I-", span_type[span_index]));
        recorded = true;
      } else {
        // token:  |----------)
        // span:      |----)
        potential_span_type = absl::StrCat("B-", span_type[span_index]);
        span_index++;
      }
    }
    if (!recorded) {
      results.push_back(potential_span_type);
    }
  }
  return results;
}

std::string ExtractSpanType(const std::string& tag) {
  return std::string(absl::ClippedSubstr(tag, 2).data());
}

absl::StatusOr<
    std::tuple<std::vector<int>, std::vector<int>, std::vector<std::string>>>
BoiseTagsToOffsets(const std::vector<int>& token_begin_offsets,
                   const std::vector<int>& token_end_offsets,
                   const std::vector<std::string>& per_token_boise_tags) {
  // Verify that input vectors are all the same size
  if (token_begin_offsets.size() != token_end_offsets.size()) {
    return absl::InvalidArgumentError("Tokens must have the same size");
  }
  if (token_begin_offsets.size() != per_token_boise_tags.size()) {
    return absl::InvalidArgumentError(
        "Tokens and BOISE tags must have the same size");
  }

  std::vector<int> span_start, span_end;
  std::vector<std::string> span_type;
  // Iterate through each token
  int potential_span_start = -1;
  std::string potential_span_type;
  bool started_span = false;

  for (int i = 0; i < token_begin_offsets.size(); ++i) {
    // If we find a (B)egin, (I)nside, (E)nd, or (S)ingleton tag then
    // record a span start.
    const std::string& tag = per_token_boise_tags[i];

    if (!started_span) {
      if (absl::StartsWith(tag, "B-") || absl::StartsWith(tag, "I-")) {
        potential_span_start = token_begin_offsets[i];
        started_span = true;
        potential_span_type = ExtractSpanType(tag);
      }

      if (absl::StartsWith(tag, "E-") || absl::StartsWith(tag, "S-")) {
        // Treat this as a singleton
        span_start.push_back(token_begin_offsets[i]);
        span_end.push_back(token_end_offsets[i]);
        span_type.push_back(ExtractSpanType(tag));
        started_span = false;
        potential_span_type.clear();
      }
    } else {
      // If we have found a Outside, but we previously had a span start (from
      // a Begin, or Inside) then treat this as a singleton and record an span
      // end
      if (absl::StartsWith(tag, "O")) {
        span_start.push_back(potential_span_start);
        span_end.push_back(token_end_offsets[i - 1]);
        span_type.push_back(potential_span_type);
        started_span = false;
        potential_span_type.clear();
      }

      // If we find a End or Singleton then also record an end.
      if (absl::StartsWith(tag, "E-") || absl::StartsWith(tag, "S-")) {
        span_start.push_back(potential_span_start);
        span_end.push_back(token_end_offsets[i]);
        // Also record a span type.
        span_type.push_back(ExtractSpanType(tag));
        started_span = false;
      }

      // If we find a Begin,
      if (absl::StartsWith(tag, "B-") || absl::StartsWith(tag, "I-")) {
        // potential_span_start = token_begin_offsets[i];
        started_span = true;
        potential_span_type = ExtractSpanType(tag);
      }
    }
  }

  // Record span that has started but not closed.
  if (started_span) {
    span_start.push_back(potential_span_start);
    span_end.push_back(token_end_offsets.back());
    span_type.push_back(potential_span_type);
  }

  return std::tuple<std::vector<int>, std::vector<int>,
                    std::vector<std::string>>(span_start, span_end, span_type);
}

std::unordered_set<std::string> GetAllBoiseTagsFromSpanType(
    const std::vector<std::string>& span_type) {
  std::unordered_set<std::string> res{"O"};
  const std::unordered_set<std::string> deduped_span_type(span_type.begin(),
                                                          span_type.end());
  const std::vector<std::string> boise_prefixes = {"B-", "I-", "S-", "E-"};

  for (const std::string& cur_span_type : deduped_span_type) {
    if (cur_span_type.empty() || cur_span_type == "O") {
      continue;
    }
    for (const std::string& prefix : boise_prefixes) {
      std::string tag = absl::StrCat(prefix, cur_span_type);
      res.insert(tag);
    }
  }

  return res;
}

}  // namespace text
}  // namespace tensorflow
