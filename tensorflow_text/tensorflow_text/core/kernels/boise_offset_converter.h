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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BOISE_OFFSET_CONVERTER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BOISE_OFFSET_CONVERTER_H_

#include <tuple>
#include <unordered_set>

#include "absl/status/statusor.h"

namespace tensorflow {
namespace text {
// Translates span begin/end offsets and token begin/end offsets into a
// BOISE scheme.
//
// In the BOISE scheme there is a set of 5 labels for each type:
// - (B)egin: meaning the beginning of the span type.
// - (O)utside: meaning the token is outside of any span type
// - (I)nside: the token is inside the span
// - (S)ingleton: the entire span consists of this single token.
// - (E)nd: this token is the end of the span.
//
// When given the span begin & end offsets along with a set of token begin & end
// offsets, this function helps translate which each token into one of the 5
// labels.
//
// For example, given the following example inputs:
//
//  std::string content = "Who let the dogs out";
//  std::string entity = "dogs";
//  std::vector<string> tokens = { "Who", "let", "the", "dogs", "out" }
//  std::vector<int> token_begin_offsets = { 0, 4, 8, 12, 17 };
//  std::vector<int> token_end_offsets = { 3, 7, 11, 16, 20 };
//  std::vector<int> span_begin_offsets = { 12 };
//  std::vector<int> span_end_offsets = { 16 };
//  std::vector<string> span_type = { "animal" }
//
// Foo will produce the following labels:
//  { "O", "O", "O",  "S-animal", "O", }
//     |    |    |        |        |
//    Who  let  the      dogs     out
//
// Special Case 1: Loose or Strict Boundary Criteria:
// By default, loose boundary criteria are used to decide token start and end,
// given a entity span. In the above example, say if we have
//
//  std::vector<int> span_begin_offsets = { 13 };
//  std::vector<int> span_end_offsets = { 16 };
//
// we still get { "O", "O", "O",  "S-animal", "O", }, even though the span
// begin offset (13) is not exactly aligned with the token begin offset (12).
// Partial overlap between a token and a BOISE tag still qualify the token to
// be labeled with this tag.
//
// You can choose to use strict boundary criteria by passing in
// use_strict_boundary_mode = false argument, with which Foo will produce
// { "O", "O", "O",  "O", "O", } for the case described above.
//
// Special Case 2: One Token Mapped to Multiple BOISE Tags:
// In cases where a token is overlapped with multiple BOISE tags, we label the
// token with the last tag. For example, given the following example inputs:
//
//  std::string content = "Getty Center";
//  std::vector<string> tokens = { "Getty Center" };
//  std::vector<int> token_begin_offsets = { 0 };
//  std::vector<int> token_end_offsets = { 12 };
//  std::vector<int> span_begin_offsets = { 0, 6 };
//  std::vector<int> span_end_offsets = { 5, 12 };
//  std::vector<string> span_type = { "per", "loc" }
//
// Foo will produce the following labels:
//  { "B-loc", }
absl::StatusOr<std::vector<std::string>> OffsetsToBoiseTags(
    const std::vector<int>& token_begin_offsets,
    const std::vector<int>& token_end_offsets,
    const std::vector<int>& span_begin_offsets,
    const std::vector<int>& span_end_offsets,
    const std::vector<std::string>& span_type,
    const bool use_strict_boundary_mode = false);

// Given the token offsets and BOISE tags per token, perform a translation
// that marks start offset, end offset and span type per entity.
//
// For example, given the following example inputs:
//
//  std::vector<int> token_begin_offsets = { 0, 4, 8, 12, 17 };
//  std::vector<int> token_end_offsets = { 3, 7, 11, 16, 20 };
//  std::vector<std::string> per_token_boise_tags = { "O", "O", "O", "S-animal",
//  "O" };
//
// Foo will produce the following offsets and labels vectors:
//  start offsets: { 12, }
//  end offsets: { 16, }
//  span types: { "animal", }
absl::StatusOr<
    std::tuple<std::vector<int>, std::vector<int>, std::vector<std::string>>>
BoiseTagsToOffsets(const std::vector<int>& token_begin_offsets,
                   const std::vector<int>& token_end_offsets,
                   const std::vector<std::string>& per_token_boise_tags);

// Get all possible BOISE tags for given span types. For example,
//
// std::vector<string> span_type = { "loc", "per" }
//
// Foo will produce an unordered set:
//  { "O", "B-loc", "I-loc", "S-loc", "E-loc", "B-per", "I-per", "S-per",
//  "E-per", }.
std::unordered_set<std::string> GetAllBoiseTagsFromSpanType(
    const std::vector<std::string>& span_type);

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_BOISE_OFFSET_CONVERTER_H_
