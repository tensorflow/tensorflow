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

#include "xla/tools/hlo_dump/hlo_dump_utils.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "re2/re2.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/zlib/zlib_writer.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tools/hlo_dump/hlo_dump_assets.h"
#include "xla/tools/hlo_dump/hlo_lexer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/coding.h"
#include "tsl/platform/path.h"

namespace xla::numerics::debug_info {

namespace {

std::string JsStringEscape(absl::string_view s);

bool IsSpaceToken(const Token& t) {
  return t.kind == TokKind::kText &&
         std::all_of(t.value.begin(), t.value.end(),
                     [](unsigned char c) { return std::isspace(c); });
}

size_t SkipSpaces(absl::Span<const Token> tokens, size_t idx) {
  while (idx < tokens.size() && IsSpaceToken(tokens[idx])) {
    idx++;
  }
  return idx;
}

std::vector<std::pair<size_t, size_t>> FindShapeTokensRangesRecursive(
    absl::Span<const Token> tokens, absl::Span<const int64_t> shape_index,
    size_t base_offset) {
  size_t start_idx = 0;
  size_t end_idx = tokens.size();
  while (start_idx < end_idx && IsSpaceToken(tokens[start_idx])) {
    start_idx++;
  }
  while (end_idx > start_idx && IsSpaceToken(tokens[end_idx - 1])) {
    end_idx--;
  }

  if (start_idx >= end_idx) {
    return {};
  }

  absl::Span<const Token> tokens_no_space =
      tokens.subspan(start_idx, end_idx - start_idx);
  base_offset += start_idx;

  size_t matching_paren_idx = absl::string_view::npos;
  bool is_tuple = false;

  if (!tokens_no_space.empty() && tokens_no_space.front().value == "(") {
    int paren_level = 0;
    for (size_t i = 0; i < tokens_no_space.size(); ++i) {
      if (tokens_no_space[i].value == "(") {
        paren_level++;
      } else if (tokens_no_space[i].value == ")") {
        paren_level--;
        if (paren_level == 0) {
          matching_paren_idx = i;
          break;
        }
      }
    }
    if (paren_level == 0) {
      is_tuple = true;
    }
  }

  if (!is_tuple) {
    if (shape_index.empty()) {
      return {{base_offset, base_offset + tokens_no_space.size()}};
    }
    return {};
  }

  if (shape_index.empty()) {
    return {{base_offset, base_offset + tokens_no_space.size()}};
  }

  // Tuple case with index.
  absl::Span<const Token> inner_tokens_list =
      tokens_no_space.subspan(1, matching_paren_idx - 1);
  std::vector<std::pair<size_t, size_t>> elements_indices;
  size_t start = 0;
  int paren_level = 0;
  int bracket_level = 0;
  int brace_level = 0;
  for (size_t i = 0; i < inner_tokens_list.size(); ++i) {
    const std::string& val = inner_tokens_list[i].value;
    if (val == "(") {
      paren_level++;
    } else if (val == ")") {
      paren_level--;
    } else if (val == "[") {
      bracket_level++;
    } else if (val == "]") {
      bracket_level--;
    } else if (val == "{") {
      brace_level++;
    } else if (val == "}") {
      brace_level--;
    } else if (val == "," && paren_level == 0 && bracket_level == 0 &&
               brace_level == 0) {
      elements_indices.push_back({start, i});
      start = i + 1;
    }
  }
  elements_indices.push_back({start, inner_tokens_list.size()});

  int64_t idx_to_find = shape_index[0];
  if (idx_to_find < 0 ||
      static_cast<size_t>(idx_to_find) >= elements_indices.size()) {
    return {};
  }

  auto [elem_start, elem_end] = elements_indices[idx_to_find];
  return FindShapeTokensRangesRecursive(
      inner_tokens_list.subspan(elem_start, elem_end - elem_start),
      shape_index.subspan(1), base_offset + 1 + elem_start);
}

std::string HtmlEscape(absl::string_view s) {
  std::string escaped;
  escaped.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      case '"':
        escaped += "&quot;";
        break;
      case '\'':
        escaped += "&apos;";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

std::string JsStringEscape(absl::string_view s) {
  std::string escaped = absl::Utf8SafeCHexEscape(s);
  return absl::StrReplaceAll(escaped, {{"<", "\\x3c"}, {">", "\\x3e"}});
}

std::string SerializeGraphDataCompressed(const GraphData& data) {
  std::string binary_data;

  tsl::core::PutFixed32(&binary_data, data.nodes.size());

  for (const auto& node : data.nodes) {
    int32_t id = node.id;
    float x = node.x;
    float y = node.y;
    float diff_score = node.diff_score;
    int32_t anchor_id = node.anchor_id;
    uint16_t key_len = node.key.size();

    tsl::core::PutFixed32(&binary_data, id);
    tsl::core::PutFixed32(&binary_data, absl::bit_cast<uint32_t>(x));
    tsl::core::PutFixed32(&binary_data, absl::bit_cast<uint32_t>(y));
    tsl::core::PutFixed32(&binary_data, absl::bit_cast<uint32_t>(diff_score));
    tsl::core::PutFixed32(&binary_data, anchor_id);
    tsl::core::PutFixed16(&binary_data, key_len);
    binary_data.append(node.key.data(), key_len);
  }

  tsl::core::PutFixed32(&binary_data, data.edges.size());

  for (const auto& edge : data.edges) {
    int32_t supplier_id = edge.supplier_id;
    int32_t consumer_id = edge.consumer_id;

    tsl::core::PutFixed32(&binary_data, supplier_id);
    tsl::core::PutFixed32(&binary_data, consumer_id);
  }

  std::string compressed;
  riegeli::ZlibWriter writer(
      riegeli::StringWriter(&compressed),
      riegeli::ZlibWriterBase::Options().set_compression_level(9).set_header(
          riegeli::ZlibWriterBase::Header::kGzip));
  writer.Write(binary_data);
  if (!writer.Close()) {
    LOG(ERROR) << "Failed to close ZlibWriter: " << writer.status();
    return "";
  }

  std::string encoded;
  encoded = absl::Base64Escape(compressed);
  return encoded;
}

std::string QuantizeColor(absl::string_view color) {
  if (color.size() == 7 && color[0] == '#') {
    return absl::StrFormat("#%c%c%c%c%c%c", std::tolower(color[1]),
                           std::tolower(color[1]), std::tolower(color[3]),
                           std::tolower(color[3]), std::tolower(color[5]),
                           std::tolower(color[5]));
  }
  return std::string(color);
}

std::string GenerateBackgroundStyles(
    const absl::flat_hash_map<TensorKey, TensorAnnotation>& annotations) {
  absl::flat_hash_set<std::string> unique_colors;
  // NOLINTNEXTLINE
  for (const auto& [key, ann] : annotations) {
    if (ann.background_color) {
      unique_colors.insert(QuantizeColor(*ann.background_color));
    }
  }

  std::string background_styles;
  // NOLINTNEXTLINE
  for (const auto& color : unique_colors) {
    std::string class_id = color;
    if (absl::StartsWith(class_id, "#")) {
      class_id = class_id.substr(1);
    }
    absl::StrAppend(&background_styles, ".bg-", class_id,
                    " { background-color: ", color, "; }\n");
  }
  return background_styles;
}

struct ShapeId {
  std::string name;
  ShapeIndex shape_index;
  bool operator<(const ShapeId& other) const {
    if (name != other.name) {
      return name < other.name;
    }
    return shape_index < other.shape_index;
  }
};

struct TokenAnnotationMapping {
  absl::flat_hash_map<size_t, const TensorAnnotation*> token_to_annotation;
  absl::flat_hash_map<size_t, std::vector<const TensorAnnotation*>> span_starts;
  absl::flat_hash_map<size_t, std::vector<const TensorAnnotation*>> span_ends;
  absl::flat_hash_map<size_t, std::string> token_anchors;
  absl::flat_hash_map<size_t, std::string> token_links;
  absl::flat_hash_map<size_t, int32_t> token_stack_frame_ids;
  absl::flat_hash_map<size_t, std::string> token_op_names;
  absl::flat_hash_set<size_t> tokens_to_skip;
};

TokenAnnotationMapping GetTokenAnnotationMapping(
    const std::vector<Token>& tokens,
    const absl::flat_hash_map<TensorKey, TensorAnnotation>& annotations) {
  std::map<std::string,
           std::vector<std::pair<ShapeIndex, const TensorAnnotation*>>>
      annotations_by_name;
  // NOLINTNEXTLINE
  for (const auto& [key, ann] : annotations) {
    annotations_by_name[key.instruction_name].push_back(
        {key.shape_index, &ann});
  }

  for (auto& [name, anns] : annotations_by_name) {
    std::sort(anns.begin(), anns.end(), [](const auto& a, const auto& b) {
      return a.first.size() > b.first.size();
    });
  }

  TokenAnnotationMapping mapping;
  std::map<ShapeId, std::vector<size_t>> shape_tokens_indices;

  static constexpr absl::string_view kComputationAttributes[] = {
      "to_apply",
      "condition",
      "body",
      "select",
      "scatter",
      "true_computation",
      "false_computation",
      "branch_computations",
      "called_computations",
      "calls"};

  auto get_id = [](absl::string_view s, const char* prefix) {
    std::string clean_s(s);
    std::replace(clean_s.begin(), clean_s.end(), '#', '_');
    if (!clean_s.empty() && (clean_s[0] == '%' || clean_s[0] == '@')) {
      return absl::StrCat(prefix, clean_s.substr(1));
    }
    return absl::StrCat(prefix, clean_s);
  };

  auto is_name_kind = [](TokKind kind) {
    return kind == TokKind::kNameVariable ||
           kind == TokKind::kNameComputation || kind == TokKind::kName;
  };

  size_t current_instr_name_idx = absl::string_view::npos;
  size_t current_opcode_idx = absl::string_view::npos;

  bool in_dictionary_attribute = false;
  int dictionary_brace_level = 0;
  bool seen_instruction_on_line = false;

  for (size_t i = 0; i < tokens.size(); ++i) {
    const auto& tok = tokens[i];

    // Special handling for metadata to extract info and remove it.
    if (!in_dictionary_attribute && tok.kind == TokKind::kName &&
        tok.value == "metadata") {
      size_t j = SkipSpaces(tokens, i + 1);
      if (j < tokens.size() && tokens[j].value == "=") {
        size_t k = SkipSpaces(tokens, j + 1);
        if (k < tokens.size() && tokens[k].value == "{") {
          // Find preceding comma.
          size_t look_back = i;
          bool found_comma = false;
          while (look_back > 0) {
            look_back--;
            if (IsSpaceToken(tokens[look_back])) {
              continue;
            }
            if (tokens[look_back].value == ",") {
              found_comma = true;
            }
            break;
          }
          size_t skip_start = found_comma ? look_back : i;

          // Find closing brace.
          size_t look_ahead = k;
          int brace_level = 0;
          while (look_ahead < tokens.size()) {
            if (tokens[look_ahead].value == "{") {
              brace_level++;
            } else if (tokens[look_ahead].value == "}") {
              brace_level--;
              if (brace_level == 0) {
                break;
              }
            }
            look_ahead++;
          }

          // Extract sfid and op_name.
          int32_t sfid = -1;
          std::string op_name;
          for (size_t m = i; m <= look_ahead && m < tokens.size(); ++m) {
            if (tokens[m].kind == TokKind::kName) {
              if (tokens[m].value == "stack_frame_id") {
                size_t n = SkipSpaces(tokens, m + 1);
                if (n < tokens.size() && tokens[n].value == "=") {
                  size_t p = SkipSpaces(tokens, n + 1);
                  if (p < tokens.size() && tokens[p].kind == TokKind::kNumber) {
                    if (!absl::SimpleAtoi(tokens[p].value, &sfid)) {
                      sfid = -1;
                    }
                  }
                }
              } else if (tokens[m].value == "op_name") {
                size_t n = SkipSpaces(tokens, m + 1);
                if (n < tokens.size() && tokens[n].value == "=") {
                  size_t p = SkipSpaces(tokens, n + 1);
                  if (p < tokens.size() && tokens[p].kind == TokKind::kString) {
                    op_name = tokens[p].value;
                    if (op_name.size() >= 2 && op_name.front() == '"' &&
                        op_name.back() == '"') {
                      op_name = op_name.substr(1, op_name.size() - 2);
                    }
                  }
                }
              }
            }
          }

          // Assign to opcode.
          if (current_opcode_idx != absl::string_view::npos) {
            if (sfid != -1) {
              mapping.token_stack_frame_ids[current_opcode_idx] = sfid;
            }
            if (!op_name.empty()) {
              mapping.token_op_names[current_opcode_idx] = op_name;
            }
          }

          // Mark tokens to skip.
          for (size_t s = skip_start; s <= look_ahead && s < tokens.size();
               ++s) {
            mapping.tokens_to_skip.insert(s);
          }

          // Advance i.
          i = look_ahead;
          continue;
        }
      }
    }

    // Keep track of whether we are inside dictionary-like attributes (like
    // metadata or frontend_attributes) to avoid false positive instruction
    // identification.
    if (!in_dictionary_attribute && tok.kind == TokKind::kName) {
      size_t j = SkipSpaces(tokens, i + 1);
      if (j < tokens.size() && tokens[j].value == "=") {
        size_t k = SkipSpaces(tokens, j + 1);
        if (k < tokens.size() && tokens[k].value == "{") {
          in_dictionary_attribute = true;
          dictionary_brace_level = 0;
        }
      }
    }

    if (in_dictionary_attribute) {
      if (tok.value == "{") {
        dictionary_brace_level++;
      }
      if (tok.value == "}") {
        dictionary_brace_level--;
        if (dictionary_brace_level == 0) {
          in_dictionary_attribute = false;
        }
      }
    }

    if (IsSpaceToken(tok)) {
      if (absl::StrContains(tok.value, '\n')) {
        current_instr_name_idx = absl::string_view::npos;
        current_opcode_idx = absl::string_view::npos;
        seen_instruction_on_line = false;
      }
      continue;
    }

    // Identify instruction definitions.
    if (!in_dictionary_attribute && !seen_instruction_on_line &&
        is_name_kind(tok.kind)) {
      size_t j = SkipSpaces(tokens, i + 1);
      if (j < tokens.size() && tokens[j].value == "=" &&
          tok.value != "stack_frame_id" && tok.value != "op_name" &&
          tok.value != "op_type") {
        mapping.token_anchors[i] = get_id(tok.value, "instr_");
        current_instr_name_idx = i;
        current_opcode_idx = absl::string_view::npos;
        seen_instruction_on_line = true;
      }
    }

    // Identify the opcode/function call part.
    if (!in_dictionary_attribute &&
        current_instr_name_idx != absl::string_view::npos &&
        current_opcode_idx == absl::string_view::npos) {
      if (tok.kind == TokKind::kNameFunction || tok.kind == TokKind::kKeyword ||
          tok.kind == TokKind::kName) {
        size_t j = SkipSpaces(tokens, i + 1);
        if (j < tokens.size() && tokens[j].value == "(") {
          current_opcode_idx = i;
        }
      }
    }

    // Identify computation definitions.
    if (!in_dictionary_attribute && is_name_kind(tok.kind)) {
      size_t j = SkipSpaces(tokens, i + 1);
      if (j < tokens.size() && tokens[j].value == "{") {
        mapping.token_anchors[i] = get_id(tok.value, "comp_");
      }
    }
    if (tok.kind == TokKind::kKeyword && tok.value == "ENTRY") {
      size_t j = SkipSpaces(tokens, i + 1);
      if (j < tokens.size() && is_name_kind(tokens[j].kind)) {
        mapping.token_anchors[j] = get_id(tokens[j].value, "comp_");
      }
    }

    // Identify operands and link them.
    if (tok.kind == TokKind::kNameFunction || tok.kind == TokKind::kKeyword ||
        tok.kind == TokKind::kKeywordType) {
      size_t j = SkipSpaces(tokens, i + 1);
      if (j < tokens.size() && tokens[j].value == "(") {
        size_t k = j + 1;
        int paren_level = 1;
        while (k < tokens.size() && paren_level > 0) {
          if (tokens[k].value == "(") {
            paren_level++;
          } else if (tokens[k].value == ")") {
            paren_level--;
          } else if (paren_level == 1 && is_name_kind(tokens[k].kind)) {
            mapping.token_links[k] = get_id(tokens[k].value, "instr_");
          }
          k++;
        }
      }
    }

    // Identify computation references in attributes.
    if (tok.kind == TokKind::kName &&
        absl::c_linear_search(kComputationAttributes, tok.value)) {
      size_t j = SkipSpaces(tokens, i + 1);
      if (j < tokens.size() && tokens[j].value == "=") {
        size_t k = SkipSpaces(tokens, j + 1);
        if (k < tokens.size()) {
          if (is_name_kind(tokens[k].kind)) {
            mapping.token_links[k] = get_id(tokens[k].value, "comp_");
          } else if (tokens[k].value == "{") {
            size_t l = k + 1;
            int brace_level = 1;
            while (l < tokens.size() && brace_level > 0) {
              if (tokens[l].value == "{") {
                brace_level++;
              } else if (tokens[l].value == "}") {
                brace_level--;
              } else if (brace_level == 1 && is_name_kind(tokens[l].kind)) {
                mapping.token_links[l] = get_id(tokens[l].value, "comp_");
              }
              l++;
            }
          }
        }
      }
    }

    if (tok.kind == TokKind::kNameVariable &&
        absl::StrContains(tok.value, '#')) {
      std::string gte_name = tok.value;
      if (absl::StartsWith(gte_name, "%")) {
        gte_name = gte_name.substr(1);
      }
      std::replace(gte_name.begin(), gte_name.end(), '#', '_');
      TensorKey key = TensorKey::Create(gte_name, {});
      auto it = annotations.find(key);
      if (it != annotations.end()) {
        const TensorAnnotation* ann = &it->second;
        mapping.token_to_annotation[i] = ann;
        if (ann->stack_frame_id) {
          mapping.token_stack_frame_ids[i] = *ann->stack_frame_id;
        }
        if ((ann->tooltip_data && !ann->tooltip_data->empty()) ||
            ann->anchor_id) {
          mapping.span_starts[i].push_back(ann);
          mapping.span_ends[i].push_back(ann);
        }
      }
    }

    std::string name = tok.value;
    if (absl::StartsWith(name, "%")) {
      name = name.substr(1);
    }
    std::replace(name.begin(), name.end(), '#', '_');

    if (annotations_by_name.count(name) &&
        (tok.kind == TokKind::kNameVariable || tok.kind == TokKind::kName)) {
      size_t j = SkipSpaces(tokens, i + 1);
      if (j < tokens.size() && tokens[j].value == "=") {
        size_t shape_start_idx = SkipSpaces(tokens, j + 1);
        int paren_level = 0;
        int bracket_level = 0;
        int brace_level = 0;
        size_t k = shape_start_idx;
        size_t shape_end_idx = tokens.size();
        while (k < tokens.size()) {
          const auto& tok_k = tokens[k];
          bool is_opcode_start =
              (tok_k.kind == TokKind::kName ||
               tok_k.kind == TokKind::kNameFunction ||
               (tok_k.kind == TokKind::kKeywordType &&
                tok_k.value == "tuple") ||
               (tok_k.kind == TokKind::kKeyword && tok_k.value != "true" &&
                tok_k.value != "false" && tok_k.value != "inf" &&
                tok_k.value != "maximal" && tok_k.value != "replicated" &&
                tok_k.value != "manual" &&
                tok_k.value != "last_tile_dim_replicate"));

          if (is_opcode_start && paren_level == 0 && bracket_level == 0 &&
              brace_level == 0) {
            shape_end_idx = k;
            break;
          }
          if (tok_k.value == "(") {
            paren_level++;
          } else if (tok_k.value == ")") {
            paren_level--;
          } else if (tok_k.value == "[") {
            bracket_level++;
          } else if (tok_k.value == "]") {
            bracket_level--;
          } else if (tok_k.value == "{") {
            brace_level++;
          } else if (tok_k.value == "}") {
            brace_level--;
          }
          k++;
        }
        shape_end_idx = std::min(shape_end_idx, tokens.size());

        absl::Span<const Token> shape_tokens = absl::MakeSpan(tokens).subspan(
            shape_start_idx, shape_end_idx - shape_start_idx);
        for (auto& [shape_index, annotation] : annotations_by_name[name]) {
          auto ranges =
              FindShapeTokensRangesRecursive(shape_tokens, shape_index, 0);
          ShapeId sid = {name, shape_index};
          for (auto [start, end] : ranges) {
            for (size_t token_idx_in_shape = start; token_idx_in_shape < end;
                 ++token_idx_in_shape) {
              if (!IsSpaceToken(shape_tokens[token_idx_in_shape])) {
                size_t abs_token_idx = shape_start_idx + token_idx_in_shape;
                // Deeper shape index (longer path) wins for background/border.
                mapping.token_to_annotation.insert({abs_token_idx, annotation});
                shape_tokens_indices[sid].push_back(abs_token_idx);
                if (annotation->stack_frame_id) {
                  mapping.token_stack_frame_ids[abs_token_idx] =
                      *annotation->stack_frame_id;
                }
              }
            }
          }
        }
      }
    }
  }

  for (const auto& [sid, indices] : shape_tokens_indices) {
    if (indices.empty()) {
      continue;
    }
    auto [min_it, max_it] = std::minmax_element(indices.begin(), indices.end());
    size_t min_idx = *min_it;
    size_t max_idx = *max_it;

    const TensorAnnotation* ann = mapping.token_to_annotation.at(min_idx);
    if ((ann->tooltip_data && !ann->tooltip_data->empty()) || ann->anchor_id) {
      mapping.span_starts[min_idx].push_back(ann);
      mapping.span_ends[max_idx].push_back(ann);
    }
  }

  return mapping;
}

std::string GenerateHloHtmlContent(
    absl::Span<const Token> tokens, const TokenAnnotationMapping& mapping,
    absl::flat_hash_map<std::string, std::string>& tooltip_data) {
  std::string parts;
  int tt_counter = 0;
  absl::flat_hash_map<const TensorAnnotation*, std::string> ann_to_id;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (mapping.tokens_to_skip.count(i)) {
      continue;
    }

    if (mapping.span_starts.count(i)) {
      for (const auto* ann : mapping.span_starts.at(i)) {
        std::string id_attr;
        if (ann->anchor_id) {
          id_attr = absl::StrCat(" id=\"", *ann->anchor_id, "\"");
        }
        std::string tooltip_attr;
        if (ann->tooltip_data) {
          std::string tt_id;
          auto it = ann_to_id.find(ann);
          if (it != ann_to_id.end()) {
            tt_id = it->second;
          } else {
            tt_id = absl::StrCat("tt", tt_counter++);
            ann_to_id[ann] = tt_id;
            tooltip_data[tt_id] = *ann->tooltip_data;
          }
          tooltip_attr = absl::StrCat(" data-tooltip-id=\"", tt_id, "\"");
        }
        absl::StrAppend(&parts, "<span class=\"tooltip\"", id_attr,
                        tooltip_attr, ">");
      }
    }
    std::string anchor_attr;
    if (mapping.token_anchors.count(i)) {
      anchor_attr = absl::StrCat(" id=\"", mapping.token_anchors.at(i), "\"");
    }

    std::string sfid_attr;
    if (mapping.token_stack_frame_ids.count(i)) {
      sfid_attr = absl::StrCat(" data-stack-frame-id=\"",
                               mapping.token_stack_frame_ids.at(i), "\"");
    }

    std::string op_name_attr;
    if (mapping.token_op_names.count(i)) {
      op_name_attr =
          absl::StrCat(" data-op-name=\"", mapping.token_op_names.at(i), "\"");
    }

    std::string extra_attrs = absl::StrCat(sfid_attr, op_name_attr);

    if (mapping.token_links.count(i)) {
      absl::StrAppend(&parts, "<a href=\"#", mapping.token_links.at(i), "\">");
    }

    if (mapping.token_to_annotation.count(i)) {
      const TensorAnnotation* ann = mapping.token_to_annotation.at(i);
      std::string bg_class;
      if (ann->background_color) {
        std::string q_color = QuantizeColor(*ann->background_color);
        if (absl::StartsWith(q_color, "#")) {
          absl::StrAppend(&bg_class, " bg-", q_color.substr(1));
        } else {
          absl::StrAppend(&bg_class, " bg-", q_color);
        }
      }
      std::string style_attr;
      if (ann->border_style) {
        style_attr =
            absl::StrFormat(" style=\"border: %s\"", *ann->border_style);
      }

      const char* css_class = TokKindToClass(tokens[i].kind);
      absl::StrAppend(&parts, "<span class=\"", css_class, bg_class, "\"",
                      anchor_attr, extra_attrs, style_attr, ">",
                      HtmlEscape(tokens[i].value), "</span>");
    } else {
      const char* css_class = TokKindToClass(tokens[i].kind);
      if (css_class[0] != '\0' || !anchor_attr.empty() ||
          !extra_attrs.empty()) {
        absl::StrAppend(&parts, "<span class=\"", css_class, "\"", anchor_attr,
                        extra_attrs, ">", HtmlEscape(tokens[i].value),
                        "</span>");
      } else {
        absl::StrAppend(&parts, HtmlEscape(tokens[i].value));
      }
    }

    if (mapping.token_links.count(i)) {
      parts += "</a>";
    }

    if (mapping.span_ends.count(i)) {
      for (size_t k = 0; k < mapping.span_ends.at(i).size(); ++k) {
        absl::StrAppend(&parts, "</span>");
      }
    }
  }
  return parts;
}

std::string GenerateOriginalValueRecoveryStatsBox(
    const OriginalValueRecoveryInfo& info) {
  if (info.histogram.empty()) {
    return "";
  }
  std::vector<std::string> stats_parts;
  if (info.percentage_recoverable && info.percentage_recovered) {
    stats_parts.push_back(absl::StrFormat(
        "Recoverable tensors (green+yellow): %.2f%%<br/>\nRecovered tensors "
        "(green): %.2f%%",
        *info.percentage_recoverable, *info.percentage_recovered));
  }
  int64_t total_lost = 0;
  for (const auto& [name, count] : info.histogram) {
    total_lost += count;
  }
  std::string hist_html = "<b>Tensors lost per pass:</b><br/><table>";
  for (const auto& [name, count] : info.histogram) {
    double pct =
        total_lost > 0 ? (static_cast<double>(count) / total_lost) * 100 : 0;
    absl::StrAppend(&hist_html, "<tr><td>", name, "</td><td align=\"right\">",
                    count, "</td><td align=\"right\">(",
                    absl::StrFormat("%.2f", pct), "%)</td></tr>");
  }
  absl::StrAppend(&hist_html, "</table>");
  stats_parts.push_back(hist_html);

  return absl::StrCat("<div class=\"stats-box\">\n",
                      absl::StrJoin(stats_parts, "<br/><br/>"), "\n</div>\n");
}

std::string GenerateStackFrameIndexDataJs(
    const xla::StackFrameIndexProto& index) {
  std::string js = "window.stackFrameIndex = {\n";

  auto append_quoted_list = [&](const char* name, auto& list) {
    absl::StrAppend(&js, "    ", name, ": [");
    for (int i = 0; i < list.size(); ++i) {
      absl::StrAppend(&js, "\"", JsStringEscape(list[i]), "\"");
      if (i + 1 < list.size()) {
        absl::StrAppend(&js, ", ");
      }
    }
    absl::StrAppend(&js, "],\n");
  };

  append_quoted_list("fileNames", index.file_names());
  append_quoted_list("functionNames", index.function_names());

  absl::StrAppend(&js, "    fileLocations: [\n");
  for (const auto& loc : index.file_locations()) {
    absl::StrAppendFormat(&js, "      {f: %d, fn: %d, l: %d, c: %d},\n",
                          loc.file_name_id(), loc.function_name_id(),
                          loc.line(), loc.column());
  }
  absl::StrAppend(&js, "    ],\n");

  absl::StrAppend(&js, "    stackFrames: [\n");
  for (const auto& frame : index.stack_frames()) {
    absl::StrAppendFormat(&js, "      {l: %d, p: %d},\n",
                          frame.file_location_id(), frame.parent_frame_id());
  }
  absl::StrAppend(&js, "    ]\n  };\n");

  return js;
}

std::string GenerateConfigInjectionJs() {
  bool is_internal = true;
#ifdef LIBTPU_ON_GCE
  is_internal = false;
#endif
  return absl::StrFormat("window.HloDumpConfig = { isInternal: %s };\n",
                         is_internal ? "true" : "false");
}

}  // namespace

std::string ConvertHloToHtml(
    absl::string_view dump_name, absl::string_view hlo_text,
    const absl::flat_hash_map<TensorKey, TensorAnnotation>& annotations,
    OriginalValueRecoveryInfo recovery_info,
    const xla::StackFrameIndexProto* stack_frame_index,
    const GraphData* graph_data) {
  std::string hlo_dump_ui_js;
  std::string html_template;
  std::string hlo_dump_style_css;
  for (const FileToc* p = ::xla::tools::hlo_dump::hlo_dump_assets_create();
       p->name != nullptr; ++p) {
    absl::string_view name(p->name);
    if (absl::EndsWith(name, "hlo_dump_ui_bin_sanitized.js")) {
      hlo_dump_ui_js = std::string(p->data, p->size);
    } else if (absl::EndsWith(name, "hlo_dump_template.html")) {
      html_template = std::string(p->data, p->size);
    } else if (absl::EndsWith(name, "hlo_dump_style.css")) {
      hlo_dump_style_css = std::string(p->data, p->size);
    }
  }

  std::string background_styles = GenerateBackgroundStyles(annotations);

  std::string processed_hlo_text(hlo_text);
  RE2::GlobalReplace(&processed_hlo_text,
                     "(?m)^\\s*FileNames\r?\n(?:\\s*[0-9]+[^\n]*\r?\n)*", "");
  RE2::GlobalReplace(&processed_hlo_text,
                     "(?m)^\\s*FunctionNames\r?\n(?:\\s*[0-9]+[^\n]*\r?\n)*",
                     "");
  RE2::GlobalReplace(&processed_hlo_text,
                     "(?m)^\\s*FileLocations\r?\n(?:\\s*[0-9]+[^\n]*\r?\n)*",
                     "");
  RE2::GlobalReplace(&processed_hlo_text,
                     "(?m)^\\s*StackFrames\r?\n(?:\\s*[0-9]+[^\n]*\r?\n)*", "");

  std::vector<Token> tokens = LexHlo(processed_hlo_text);

  TokenAnnotationMapping mapping =
      GetTokenAnnotationMapping(tokens, annotations);

  absl::flat_hash_map<std::string, std::string> tooltip_data;
  std::string hlo_content =
      GenerateHloHtmlContent(tokens, mapping, tooltip_data);

  std::string stats_box_html =
      GenerateOriginalValueRecoveryStatsBox(recovery_info);

  std::string graph_content;
  std::string compressed_data_str;
  if (graph_data != nullptr) {
    compressed_data_str = SerializeGraphDataCompressed(*graph_data);
    graph_content =
        "<canvas id=\"dag-canvas\" style=\"width: 100%; height: 100%; display: "
        "block;\"></canvas>";
  }

  std::string data_injection_script;
  absl::StrAppend(&data_injection_script, GenerateConfigInjectionJs());

  absl::StrAppend(&data_injection_script, "window.tooltipData = {\n");
  // NOLINTNEXTLINE
  for (const auto& [id, json_str] : tooltip_data) {
    absl::StrAppend(&data_injection_script, "  \"", id, "\": ", json_str,
                    ",\n");
  }
  absl::StrAppend(&data_injection_script, "};\n");

  if (!compressed_data_str.empty()) {
    absl::StrAppendFormat(&data_injection_script,
                          "window.compressedGraphData = \"%s\";\n",
                          compressed_data_str);
  }

  if (stack_frame_index != nullptr) {
    absl::StrAppend(&data_injection_script,
                    GenerateStackFrameIndexDataJs(*stack_frame_index));
  }

  return absl::StrReplaceAll(
      html_template, {{"{{DUMP_NAME}}", dump_name},
                      {"{{HLO_DUMP_STYLE_CSS}}", hlo_dump_style_css},
                      {"{{BACKGROUND_STYLES}}", background_styles},
                      {"{{HLO_DUMP_UI_JS}}", hlo_dump_ui_js},
                      {"{{GRAPH_CONTENT}}", graph_content},
                      {"{{STATS_BOX}}", stats_box_html},
                      {"{{HLO_CONTENT}}", hlo_content},
                      {"{{DATA_INJECTION_SCRIPT}}", data_injection_script}});
}

absl::flat_hash_map<TensorKey, TensorAnnotation> PopulateMismatchAnnotations(
    const HloModule& module, absl::Span<const MismatchDetails> mismatches) {
  absl::flat_hash_map<TensorKey, TensorAnnotation> annotations;

  absl::flat_hash_map<const HloComputation*, const HloInstruction*>
      comp_to_fusion;
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kFusion) {
        comp_to_fusion[instr->fused_instructions_computation()] = instr;
      }
      TensorKey key = TensorKey::Create(instr->name(), ShapeIndex{});
      TensorAnnotation ann;
      ann.anchor_id = absl::StrCat("step", instr->unique_id());
      annotations[key] = std::move(ann);
    }
  }

  absl::flat_hash_map<std::string, const HloInstruction*> name_to_instr;
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      name_to_instr[instr->name()] = instr;
    }
  }

  for (const MismatchDetails& mismatch : mismatches) {
    auto it = name_to_instr.find(mismatch.target_instruction_name);
    if (it == name_to_instr.end()) {
      continue;
    }
    const HloInstruction* target_instr = it->second;
    TensorKey key;
    key.instruction_name = target_instr->name();
    if (mismatch.output_shape_index.has_value()) {
      key.shape_index.push_back(*mismatch.output_shape_index);
    }

    TensorAnnotation ann;
    ann.anchor_id = absl::StrCat("step", target_instr->unique_id());
    ann.background_color = "pink";

    std::vector<std::string> tooltip_parts;
    tooltip_parts.push_back("<b>Numeric Mismatch:</b>");
    if (mismatch.custom_description.has_value()) {
      tooltip_parts.push_back(*mismatch.custom_description);
    } else {
      tooltip_parts.push_back(absl::StrFormat("Actual: %g", mismatch.actual));
      tooltip_parts.push_back(
          absl::StrFormat("Expected: %g", mismatch.expected));
      tooltip_parts.push_back(
          absl::StrFormat("Rel Error: %g", mismatch.rel_error));
      if (mismatch.percentage_of_elems_exceeding_abs_error.has_value()) {
        tooltip_parts.push_back(
            absl::StrFormat("Elems exceeding abs error: %.2f%%",
                            *mismatch.percentage_of_elems_exceeding_abs_error));
      }
      if (mismatch.percentage_of_elems_exceeding_rel_error.has_value()) {
        tooltip_parts.push_back(
            absl::StrFormat("Elems exceeding rel error: %.2f%%",
                            *mismatch.percentage_of_elems_exceeding_rel_error));
      }
      if (mismatch.percentage_of_elems_exceeding_both_errors.has_value()) {
        tooltip_parts.push_back(absl::StrFormat(
            "Elems exceeding both errors: %.2f%%",
            *mismatch.percentage_of_elems_exceeding_both_errors));
      }
      if (mismatch.result_of_reduce.has_value()) {
        tooltip_parts.push_back(
            absl::StrFormat("Result of reduce: %s",
                            *mismatch.result_of_reduce ? "True" : "False"));
      }
    }

    ann.tooltip_data = absl::StrCat(
        "\"", JsStringEscape(absl::StrJoin(tooltip_parts, "<br/>")), "\"");
    std::optional<std::string> tooltip_data = ann.tooltip_data;
    annotations[key] = std::move(ann);

    const HloInstruction* cur = target_instr;
    while (cur->opcode() == HloOpcode::kFusion) {
      cur = cur->fused_instructions_computation()->root_instruction();
      TensorKey inner_key = TensorKey::Create(cur->name(), ShapeIndex{});
      TensorAnnotation inner_ann;
      inner_ann.anchor_id = absl::StrCat("step", cur->unique_id());
      inner_ann.background_color = "pink";
      inner_ann.tooltip_data = tooltip_data;
      annotations[inner_key] = std::move(inner_ann);
    }
  }

  return annotations;
}

GraphData PopulateMismatchGraphData(
    const HloModule& module, absl::Span<const MismatchDetails> mismatches) {
  GraphData graph_data;

  const HloComputation* entry = module.entry_computation();
  const HloInstruction* root = entry ? entry->root_instruction() : nullptr;

  absl::flat_hash_map<const HloComputation*, const HloInstruction*>
      comp_to_fusion;
  std::vector<const HloInstruction*> all_instructions;
  absl::flat_hash_set<const HloInstruction*> valid_instrs;

  for (const HloComputation* comp : module.computations()) {
    auto instrs = comp->MakeInstructionPostOrder();
    all_instructions.insert(all_instructions.end(), instrs.begin(),
                            instrs.end());
    for (const HloInstruction* instr : instrs) {
      valid_instrs.insert(instr);
      if (instr->opcode() == HloOpcode::kFusion) {
        comp_to_fusion[instr->fused_instructions_computation()] = instr;
      }
    }
  }

  double root_score = 100.0;
  if (!mismatches.empty()) {
    double max_rel = 0.0;
    for (const auto& m : mismatches) {
      if (m.rel_error > max_rel) {
        max_rel = m.rel_error;
      }
    }
    if (max_rel > 0.0) {
      root_score = max_rel * 100.0;
    }
  }

  absl::flat_hash_set<const HloInstruction*> root_instrs;
  if (root != nullptr) {
    const HloInstruction* cur = root;
    root_instrs.insert(cur);
    while (cur->opcode() == HloOpcode::kFusion) {
      cur = cur->fused_instructions_computation()->root_instruction();
      root_instrs.insert(cur);
    }
  }

  auto get_suppliers = [&](const HloInstruction* instr) {
    std::vector<const HloInstruction*> suppliers;
    if (instr->opcode() == HloOpcode::kFusion) {
      suppliers.push_back(
          instr->fused_instructions_computation()->root_instruction());
    } else if (instr->opcode() == HloOpcode::kParameter) {
      auto it = comp_to_fusion.find(instr->parent());
      if (it != comp_to_fusion.end()) {
        const HloInstruction* fusion = it->second;
        if (instr->parameter_number() < fusion->operand_count()) {
          suppliers.push_back(fusion->operand(instr->parameter_number()));
        }
      }
    } else {
      suppliers.assign(instr->operands().begin(), instr->operands().end());
    }
    return suppliers;
  };

  absl::flat_hash_map<const HloInstruction*, int64_t> depth;
  std::function<int64_t(const HloInstruction*)> get_depth =
      [&](const HloInstruction* instr) -> int64_t {
    auto it = depth.find(instr);
    if (it != depth.end()) {
      return it->second;
    }
    int64_t cur_depth = 0;
    for (const HloInstruction* sup : get_suppliers(instr)) {
      int64_t d = get_depth(sup) + 1;
      if (d > cur_depth) {
        cur_depth = d;
      }
    }
    depth[instr] = cur_depth;
    return cur_depth;
  };

  absl::flat_hash_map<int64_t, double> depth_to_next_y;

  for (const HloInstruction* instr : all_instructions) {
    int64_t id = instr->unique_id();
    int64_t d = get_depth(instr);
    double x = static_cast<double>(d) * 2.0;
    double& next_y = depth_to_next_y[d];
    double y = next_y;
    next_y += 2.0;

    double score = root_instrs.contains(instr) ? root_score : 0.0;

    std::vector<std::string> scopes;
    const HloComputation* cur_comp = instr->parent();
    while (true) {
      auto it = comp_to_fusion.find(cur_comp);
      if (it == comp_to_fusion.end()) {
        break;
      }
      const HloInstruction* fusion = it->second;
      scopes.push_back(std::string(fusion->name()));
      cur_comp = fusion->parent();
    }
    std::reverse(scopes.begin(), scopes.end());
    scopes.push_back(std::string(instr->name()));
    std::string key = absl::StrJoin(scopes, "/");

    graph_data.nodes.push_back(GraphNode{/*id=*/id,
                                         /*x=*/x,
                                         /*y=*/y,
                                         /*diff_score=*/score,
                                         /*key=*/key,
                                         /*anchor_id=*/id});

    for (const HloInstruction* sup : get_suppliers(instr)) {
      if (valid_instrs.contains(sup)) {
        graph_data.edges.push_back(GraphEdge{/*supplier_id=*/sup->unique_id(),
                                             /*consumer_id=*/id});
      }
    }
  }

  return graph_data;
}

absl::StatusOr<std::string> DumpHloModuleMismatchWithGraphData(
    const HloModule& module, absl::Span<const MismatchDetails> mismatches,
    absl::string_view output_filename) {
  absl::flat_hash_map<TensorKey, TensorAnnotation> annotations =
      PopulateMismatchAnnotations(module, mismatches);
  GraphData graph_data = PopulateMismatchGraphData(module, mismatches);

  xla::StackFrameIndexProto stack_frame_index = module.stack_frames().proto();
  std::string html =
      ConvertHloToHtml(module.name(), module.ToString(), annotations, {},
                       &stack_frame_index, &graph_data);

  const char* env_dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  std::string outdir;
  std::string html_filename;
  if (env_dir != nullptr && env_dir[0] != '\0') {
    outdir = env_dir;
    html_filename = tsl::io::JoinPath(outdir, output_filename);
  } else if (tsl::io::GetTestUndeclaredOutputsDir(&outdir)) {
    html_filename = tsl::io::JoinPath(outdir, output_filename);
  } else {
    html_filename = tsl::io::GetTempFilename(std::string(output_filename));
  }

  auto status =
      tsl::WriteStringToFile(tsl::Env::Default(), html_filename, html);
  if (!status.ok()) {
    return status;
  }
  return html_filename;
}

}  // namespace xla::numerics::debug_info
