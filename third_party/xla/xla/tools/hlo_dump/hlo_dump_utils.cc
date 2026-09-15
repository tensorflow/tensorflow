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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
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
#include "xla/index_util.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tools/hlo_dump/hlo_dump_assets.h"
#include "xla/tools/hlo_dump/hlo_lexer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
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

std::string FormatJsDouble(double val) {
  if (std::isnan(val)) {
    return "NaN";
  }
  if (std::isinf(val)) {
    return val > 0 ? "Infinity" : "-Infinity";
  }
  return absl::StrFormat("%g", val);
}

std::string SerializeGraphDataCompressed(const GraphData& data) {
  std::string binary_data;

  tsl::core::PutFixed32(&binary_data, data.nodes.size());

  for (const auto& node : data.nodes) {
    int64_t id = node.id;
    float x = node.x;
    float y = node.y;
    float diff_score = node.diff_score;
    int64_t anchor_id = node.anchor_id;
    uint16_t key_len = node.key.size();

    tsl::core::PutFixed64(&binary_data, id);
    tsl::core::PutFixed32(&binary_data, absl::bit_cast<uint32_t>(x));
    tsl::core::PutFixed32(&binary_data, absl::bit_cast<uint32_t>(y));
    tsl::core::PutFixed32(&binary_data, absl::bit_cast<uint32_t>(diff_score));
    tsl::core::PutFixed64(&binary_data, anchor_id);
    tsl::core::PutFixed16(&binary_data, key_len);
    binary_data.append(node.key.data(), key_len);
  }

  tsl::core::PutFixed32(&binary_data, data.edges.size());

  for (const auto& edge : data.edges) {
    int64_t supplier_id = edge.supplier_id;
    int64_t consumer_id = edge.consumer_id;

    tsl::core::PutFixed64(&binary_data, supplier_id);
    tsl::core::PutFixed64(&binary_data, consumer_id);
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
  absl::flat_hash_map<std::string, std::string> tooltip_str_to_id;
  bool in_block = false;
  int line_count = 0;
  constexpr int kLinesPerBlock = 50;

  auto open_block_if_needed = [&]() {
    if (!in_block) {
      absl::StrAppend(&parts, "<div class=\"hlo-block\">");
      in_block = true;
      line_count = 0;
    }
  };

  auto close_block_if_open = [&]() {
    if (in_block) {
      absl::StrAppend(&parts, "</div>");
      in_block = false;
      line_count = 0;
    }
  };

  for (size_t i = 0; i < tokens.size(); ++i) {
    if (mapping.tokens_to_skip.count(i)) {
      continue;
    }

    if (tokens[i].kind == TokKind::kText &&
        absl::StrContains(tokens[i].value, '\n')) {
      std::string val = tokens[i].value;
      size_t pos = 0;
      while (pos < val.size()) {
        size_t nl_pos = val.find('\n', pos);
        if (nl_pos == std::string::npos) {
          open_block_if_needed();
          absl::StrAppend(&parts, HtmlEscape(val.substr(pos)));
          break;
        }
        open_block_if_needed();
        absl::StrAppend(&parts, HtmlEscape(val.substr(pos, nl_pos - pos)),
                        "\n");
        line_count++;
        if (line_count >= kLinesPerBlock) {
          close_block_if_open();
        }
        pos = nl_pos + 1;
      }
      continue;
    }

    open_block_if_needed();

    if (mapping.span_starts.count(i)) {
      for (const auto* ann : mapping.span_starts.at(i)) {
        std::string id_attr;
        if (ann->anchor_id) {
          id_attr = absl::StrCat(" id=\"", *ann->anchor_id, "\"");
        }
        std::string tooltip_attr;
        if (ann->tooltip_data) {
          std::string tt_id;
          auto it = tooltip_str_to_id.find(*ann->tooltip_data);
          if (it != tooltip_str_to_id.end()) {
            tt_id = it->second;
          } else {
            tt_id = absl::StrCat("tt", tt_counter++);
            tooltip_str_to_id[*ann->tooltip_data] = tt_id;
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
      bool needs_span = !anchor_attr.empty() || !extra_attrs.empty() ||
                        mapping.token_links.count(i) > 0;
      if (!needs_span) {
        TokKind kind = tokens[i].kind;
        if (kind == TokKind::kComment || kind == TokKind::kCommentSpecial ||
            kind == TokKind::kString || kind == TokKind::kKeyword ||
            kind == TokKind::kKeywordType || kind == TokKind::kNameFunction ||
            kind == TokKind::kNameComputation || kind == TokKind::kNumber) {
          needs_span = true;
        }
      }
      if (needs_span) {
        const char* css_class = TokKindToClass(tokens[i].kind);
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
  close_block_if_open();
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

std::string ClassifyMismatchPattern(const MismatchBoundingBox& bbox) {
  if (bbox.mismatch_count <= 0) {
    return "";
  }
  if (!bbox.tensor_shape.empty() &&
      (bbox.box_min.empty() || bbox.box_max.empty())) {
    return "";
  }

  size_t rank = bbox.box_min.size();
  int64_t box_volume = 1;
  for (size_t d = 0; d < rank; ++d) {
    int64_t span = std::max<int64_t>(1, bbox.box_max[d] - bbox.box_min[d] + 1);
    box_volume *= span;
  }
  double density = box_volume > 0 ? static_cast<double>(bbox.mismatch_count) /
                                        static_cast<double>(box_volume)
                                  : 0.0;

  // 1. SPARSE_OUTLIERS (density < 2% and count < 16)
  if (bbox.mismatch_count < 16 &&
      (density < 0.02 ||
       (bbox.total_elements > 0 &&
        static_cast<double>(bbox.mismatch_count) / bbox.total_elements < 0.02 &&
        box_volume <= bbox.mismatch_count))) {
    return "SPARSE_OUTLIERS";
  }

  // 2. BOUNDARY (mismatches concentrated at min and max bounds along one or
  // more dimensions with a hollow interior)
  if (rank >= 1 && !bbox.top_mismatch_coords.empty()) {
    for (size_t d = 0; d < rank; ++d) {
      if (bbox.box_max[d] - bbox.box_min[d] >= 2) {
        bool has_min_bound = false;
        bool has_max_bound = false;
        bool has_interior = false;
        for (const auto& coord : bbox.top_mismatch_coords) {
          if (coord.size() == rank) {
            if (coord[d] == bbox.box_min[d]) {
              has_min_bound = true;
            } else if (coord[d] == bbox.box_max[d]) {
              has_max_bound = true;
            } else if (coord[d] > bbox.box_min[d] &&
                       coord[d] < bbox.box_max[d]) {
              has_interior = true;
              break;
            }
          }
        }
        if (has_min_bound && has_max_bound && !has_interior) {
          return "BOUNDARY";
        }
      }
    }
  }

  // 3. STRIDED (periodic delta between mismatch coordinates in the minor
  // dimension, e.g. stride 4, 8, etc.)
  if (rank >= 1 && bbox.top_mismatch_coords.size() >= 2) {
    size_t dim_minor = rank - 1;
    std::vector<int64_t> minor_coords;
    minor_coords.reserve(bbox.top_mismatch_coords.size());
    for (const auto& coord : bbox.top_mismatch_coords) {
      if (coord.size() == rank) {
        minor_coords.push_back(coord[dim_minor]);
      }
    }
    std::sort(minor_coords.begin(), minor_coords.end());
    minor_coords.erase(std::unique(minor_coords.begin(), minor_coords.end()),
                       minor_coords.end());

    if (minor_coords.size() >= 2) {
      int64_t stride = minor_coords[1] - minor_coords[0];
      for (size_t i = 2; i < minor_coords.size(); ++i) {
        stride = std::gcd(stride, minor_coords[i] - minor_coords[i - 1]);
      }
      if (stride >= 2) {
        bool is_periodic = true;
        for (size_t i = 1; i < minor_coords.size(); ++i) {
          int64_t diff = minor_coords[i] - minor_coords[i - 1];
          if (diff % stride != 0 || diff > 4 * stride) {
            is_periodic = false;
            break;
          }
        }
        if (is_periodic) {
          return absl::StrFormat("STRIDED (stride %d)", stride);
        }
      }
    }
  }

  // 4. SINGLE_SLICE (span is 1 along one or more dimensions)
  if (rank >= 1) {
    for (size_t d = 0; d < rank; ++d) {
      if (bbox.box_max[d] == bbox.box_min[d]) {
        if (bbox.tensor_shape.empty() ||
            (d < bbox.tensor_shape.size() && bbox.tensor_shape[d] > 1)) {
          return "SINGLE_SLICE";
        }
      }
    }
  }

  // 5. DENSE_BLOCK (density >= 70%)
  if (density >= 0.70) {
    return "DENSE_BLOCK";
  }

  return "";
}

MismatchBoundingBox ComputeBoundingBoxFromLiteralMask(
    const LiteralSlice& mismatches) {
  MismatchBoundingBox result;
  const Shape& shape = mismatches.shape();
  if (!shape.IsArray()) {
    return result;
  }
  int64_t rank = shape.dimensions().size();
  result.tensor_shape.assign(shape.dimensions().begin(),
                             shape.dimensions().end());
  result.total_elements = ShapeUtil::ElementsIn(shape);
  result.box_min.assign(rank, 0);
  result.box_max.assign(rank, 0);
  result.mismatch_count = 0;

  if (result.total_elements == 0) {
    return result;
  }

  if (rank == 0) {
    if (mismatches.Get<bool>({})) {
      result.mismatch_count = 1;
      result.top_mismatch_coords.push_back({});
      result.pattern = "DENSE_BLOCK";
    }
    return result;
  }

  absl::Span<const bool> data_span = mismatches.data<bool>();
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data_span.data());

  int64_t dim_minor = LayoutUtil::Minor(shape.layout(), 0);
  int64_t W = shape.dimensions(dim_minor);
  int64_t num_rows = result.total_elements / W;

  bool found_any = false;
  std::vector<int64_t> current_min(rank, std::numeric_limits<int64_t>::max());
  std::vector<int64_t> current_max(rank, std::numeric_limits<int64_t>::min());

  constexpr size_t kMaxTopMismatches = 16;
  std::minstd_rand rng(0x1337);
  auto uniform = [&]() -> double {
    return (rng() - rng.min() + 1.0) /
           (static_cast<double>(rng.max() - rng.min()) + 2.0);
  };
  double w = 1.0;
  int64_t skip = 0;
  auto advance_skip = [&]() {
    double u = uniform();
    w *= std::exp(std::log(u) / static_cast<double>(kMaxTopMismatches));
    double u2 = uniform();
    skip = static_cast<int64_t>(std::log(u2) / std::log1p(-w));
    skip = std::max<int64_t>(0, skip);
  };

  for (int64_t r = 0; r < num_rows; ++r) {
    const uint8_t* row_bytes = bytes + r * W;
    int64_t c = 0;
    int64_t first_in_row = -1;
    int64_t last_in_row = -1;
    int64_t row_mismatches = 0;

    std::optional<DimensionVector> row_coords;
    auto get_row_coords = [&]() -> const DimensionVector& {
      if (!row_coords.has_value()) {
        row_coords =
            IndexUtil::LinearIndexToMultidimensionalIndex(shape, r * W);
      }
      return *row_coords;
    };

    auto record_mismatch = [&](int64_t col) {
      row_mismatches++;
      if (first_in_row == -1) {
        first_in_row = col;
      }
      last_in_row = col;

      auto compute_sample_coord = [&]() {
        const DimensionVector& outer_coords = get_row_coords();
        std::vector<int64_t> sample_coord(outer_coords.begin(),
                                          outer_coords.end());
        sample_coord[dim_minor] = col;
        return sample_coord;
      };

      if (result.top_mismatch_coords.size() < kMaxTopMismatches) {
        result.top_mismatch_coords.push_back(compute_sample_coord());
        if (result.top_mismatch_coords.size() == kMaxTopMismatches) {
          advance_skip();
        }
      } else if (skip > 0) {
        --skip;
      } else {
        size_t j = rng() % kMaxTopMismatches;
        result.top_mismatch_coords[j] = compute_sample_coord();
        advance_skip();
      }
    };

    while (c + 8 <= W) {
      uint64_t word;
      std::memcpy(&word, row_bytes + c, sizeof(uint64_t));
      if (word != 0) {
        for (int j = 0; j < 8; ++j) {
          if (row_bytes[c + j]) {
            record_mismatch(c + j);
          }
        }
      }
      c += 8;
    }
    while (c < W) {
      if (row_bytes[c]) {
        record_mismatch(c);
      }
      c++;
    }

    if (row_mismatches > 0) {
      found_any = true;
      result.mismatch_count += row_mismatches;

      const DimensionVector& outer_coords = get_row_coords();
      auto update_bounds = [&](std::vector<int64_t>& box_min,
                               std::vector<int64_t>& box_max) {
        box_min[dim_minor] = std::min(box_min[dim_minor], first_in_row);
        box_max[dim_minor] = std::max(box_max[dim_minor], last_in_row);
        for (int64_t d = 0; d < rank; ++d) {
          if (d != dim_minor) {
            box_min[d] =
                std::min(box_min[d], static_cast<int64_t>(outer_coords[d]));
            box_max[d] =
                std::max(box_max[d], static_cast<int64_t>(outer_coords[d]));
          }
        }
      };

      update_bounds(current_min, current_max);
      if (rank >= 4) {
        int64_t slice_idx = static_cast<int64_t>(outer_coords[rank - 4]);
        if (std::find(result.mismatched_slices.begin(),
                      result.mismatched_slices.end(),
                      slice_idx) == result.mismatched_slices.end()) {
          result.mismatched_slices.push_back(slice_idx);
        }

        std::vector<int64_t> slice_coords;
        slice_coords.reserve(rank - 3);
        for (int64_t d = 0; d <= rank - 4; ++d) {
          slice_coords.push_back(static_cast<int64_t>(outer_coords[d]));
        }
        std::string slice_key = absl::StrJoin(slice_coords, ",");

        auto& sbox = result.slice_boxes[slice_key];
        if (sbox.box_min.empty()) {
          sbox.slice_key = slice_key;
          sbox.slice_coords = slice_coords;
          sbox.slice_index = slice_idx;
          sbox.box_min.assign(rank, std::numeric_limits<int64_t>::max());
          sbox.box_max.assign(rank, std::numeric_limits<int64_t>::min());
        }
        sbox.mismatch_count += row_mismatches;
        update_bounds(sbox.box_min, sbox.box_max);
      }
    }
  }

  if (found_any) {
    result.box_min = std::move(current_min);
    result.box_max = std::move(current_max);
    if (!result.mismatched_slices.empty()) {
      std::sort(result.mismatched_slices.begin(),
                result.mismatched_slices.end());
    }
    result.pattern = ClassifyMismatchPattern(result);
  }
  return result;
}

namespace {

void ApplyBoundingBoxTail(const MismatchBoundingBox& bbox, double rel_error,
                          TensorVisualizationInfo& info) {
  info.mismatch_count = bbox.mismatch_count;
  info.total_elements =
      bbox.total_elements > 0 ? bbox.total_elements : info.total_elements;
  for (const auto& coord : bbox.top_mismatch_coords) {
    info.top_mismatches.push_back({coord, rel_error});
  }
  if (!bbox.mismatched_slices.empty()) {
    info.mismatched_slices = bbox.mismatched_slices;
  }
  if (!bbox.slice_boxes.empty()) {
    info.slice_boxes = bbox.slice_boxes;
  }
  info.pattern =
      !bbox.pattern.empty() ? bbox.pattern : ClassifyMismatchPattern(bbox);
}

}  // namespace

absl::flat_hash_map<std::string, TensorVisualizationInfo>
PopulateTensorVisualizations(const HloModule& module,
                             absl::Span<const MismatchDetails> mismatches) {
  absl::flat_hash_map<std::string, TensorVisualizationInfo> visualizations;

  absl::flat_hash_map<std::string, const MismatchDetails*> instr_to_mismatch;
  for (const MismatchDetails& mismatch : mismatches) {
    instr_to_mismatch[mismatch.target_instruction_name] = &mismatch;
  }

  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      TensorVisualizationInfo info;
      info.instruction_name = std::string(instr->name());
      info.opcode = std::string(HloOpcodeString(instr->opcode()));

      const Shape& instr_shape = instr->shape();
      if (instr_shape.IsArray()) {
        info.shape.assign(instr_shape.dimensions().begin(),
                          instr_shape.dimensions().end());
        info.total_elements = ShapeUtil::ElementsIn(instr_shape);
      } else if (instr_shape.IsTuple() && !instr_shape.tuple_shapes().empty()) {
        const auto it = instr_to_mismatch.find(instr->name());
        int64_t tuple_idx = 0;
        if (it != instr_to_mismatch.end() &&
            it->second->output_shape_index.has_value() &&
            *it->second->output_shape_index <
                instr_shape.tuple_shapes().size()) {
          tuple_idx = *it->second->output_shape_index;
        }
        const Shape& sub = instr_shape.tuple_shapes(tuple_idx);
        if (sub.IsArray()) {
          info.shape.assign(sub.dimensions().begin(), sub.dimensions().end());
          info.total_elements = ShapeUtil::ElementsIn(sub);
        }
      }

      info.box_min.assign(info.shape.size(), 0);
      info.box_max.assign(info.shape.size(), 0);

      auto it = instr_to_mismatch.find(instr->name());
      if (it != instr_to_mismatch.end()) {
        const MismatchDetails* mismatch = it->second;
        info.has_mismatch = true;
        if (mismatch->bounding_box.has_value()) {
          const auto& bbox = *mismatch->bounding_box;
          if (!bbox.tensor_shape.empty()) {
            info.shape = bbox.tensor_shape;
          }
          if (bbox.box_min.size() == info.shape.size()) {
            info.box_min = bbox.box_min;
          }
          if (bbox.box_max.size() == info.shape.size()) {
            info.box_max = bbox.box_max;
          }
          ApplyBoundingBoxTail(bbox, mismatch->rel_error, info);
        } else {
          info.mismatch_count = 1;
        }
      }

      std::string anchor_id = absl::StrCat("step", instr->unique_id());
      visualizations[anchor_id] = info;
      visualizations[instr->name()] = std::move(info);
    }
  }

  for (const MismatchDetails& mismatch : mismatches) {
    auto [it, inserted] =
        visualizations.try_emplace(mismatch.target_instruction_name);
    if (inserted) {
      TensorVisualizationInfo& info = it->second;
      info.instruction_name = mismatch.target_instruction_name;
      info.has_mismatch = true;
      if (mismatch.bounding_box.has_value()) {
        const auto& bbox = *mismatch.bounding_box;
        info.shape = bbox.tensor_shape;
        info.box_min = bbox.box_min;
        info.box_max = bbox.box_max;
        ApplyBoundingBoxTail(bbox, mismatch.rel_error, info);
      } else {
        info.mismatch_count = 1;
      }
    }
  }

  return visualizations;
}

namespace {

template <typename MapT>
std::vector<std::string> SortedKeys(const MapT& map) {
  std::vector<std::string> keys;
  keys.reserve(map.size());
  // NOLINTNEXTLINE
  for (const auto& [key, _] : map) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

void AppendInt64Array(std::string* js, absl::string_view indent,
                      absl::string_view name,
                      absl::Span<const int64_t> values) {
  absl::StrAppend(js, indent, "\"", name, "\": [", absl::StrJoin(values, ", "),
                  "],\n");
}

}  // namespace

std::string SerializeTensorVisualizationsJs(
    const absl::flat_hash_map<std::string, TensorVisualizationInfo>&
        visualizations) {
  std::string js;
  absl::StrAppend(&js, "window.tensorVisualizations = {\n");
  std::vector<std::string> keys = SortedKeys(visualizations);

  for (size_t k = 0; k < keys.size(); ++k) {
    const auto& key = keys[k];
    const auto& info = visualizations.at(key);
    absl::StrAppendFormat(&js, "  \"%s\": {\n", JsStringEscape(key));
    absl::StrAppendFormat(&js, "    \"instruction_name\": \"%s\",\n",
                          JsStringEscape(info.instruction_name));
    absl::StrAppendFormat(&js, "    \"opcode\": \"%s\",\n",
                          JsStringEscape(info.opcode));
    AppendInt64Array(&js, "    ", "shape", info.shape);
    absl::StrAppendFormat(&js, "    \"has_mismatch\": %s,\n",
                          info.has_mismatch ? "true" : "false");
    AppendInt64Array(&js, "    ", "box_min", info.box_min);
    AppendInt64Array(&js, "    ", "box_max", info.box_max);
    if (!info.pattern.empty()) {
      absl::StrAppendFormat(&js, "    \"pattern\": \"%s\",\n",
                            JsStringEscape(info.pattern));
    }
    if (!info.mismatched_slices.empty()) {
      AppendInt64Array(&js, "    ", "mismatched_slices",
                       info.mismatched_slices);
    }
    if (!info.slice_boxes.empty()) {
      absl::StrAppend(&js, "    \"slice_boxes\": {\n");
      std::vector<std::string> slice_keys = SortedKeys(info.slice_boxes);
      for (size_t si = 0; si < slice_keys.size(); ++si) {
        const auto& s_key = slice_keys[si];
        const auto& sbox = info.slice_boxes.at(s_key);
        absl::StrAppendFormat(&js, "      \"%s\": {\n", JsStringEscape(s_key));
        AppendInt64Array(&js, "        ", "box_min", sbox.box_min);
        AppendInt64Array(&js, "        ", "box_max", sbox.box_max);
        absl::StrAppendFormat(&js, "        \"mismatch_count\": %d\n",
                              sbox.mismatch_count);
        absl::StrAppend(
            &js, si + 1 < slice_keys.size() ? "      },\n" : "      }\n");
      }
      absl::StrAppend(&js, "    },\n");
    }
    absl::StrAppend(&js, "    \"top_mismatches\": [");
    for (size_t i = 0; i < info.top_mismatches.size(); ++i) {
      const auto& p = info.top_mismatches[i];
      absl::StrAppend(&js, "{\"coord\": [", absl::StrJoin(p.coord, ", "),
                      absl::StrFormat("], \"rel_error\": %s}",
                                      FormatJsDouble(p.rel_error)));
      if (i + 1 < info.top_mismatches.size()) {
        absl::StrAppend(&js, ", ");
      }
    }
    absl::StrAppend(&js, "],\n");
    absl::StrAppendFormat(&js, "    \"mismatch_count\": %d,\n",
                          info.mismatch_count);
    absl::StrAppendFormat(&js, "    \"total_elements\": %d\n",
                          info.total_elements);
    absl::StrAppend(&js, k + 1 < keys.size() ? "  },\n" : "  }\n");
  }
  absl::StrAppend(&js, "};\n");
  return js;
}

std::string ConvertHloToHtml(
    absl::string_view dump_name, absl::string_view hlo_text,
    const absl::flat_hash_map<TensorKey, TensorAnnotation>& annotations,
    OriginalValueRecoveryInfo recovery_info,
    const xla::StackFrameIndexProto* stack_frame_index,
    const GraphData* graph_data,
    const absl::flat_hash_map<std::string, TensorVisualizationInfo>*
        tensor_visualizations) {
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
    graph_content = absl::StrCat(
        "<div id=\"graph-controls\" style=\"position: absolute; top: 10px; "
        "right: 10px; z-index: 10; background: rgba(255, 255, 255, 0.9); "
        "padding: 4px; border-radius: 4px; box-shadow: 0 1px 3px rgba(60, 64, "
        "67, 0.15); display: flex; align-items: center; gap: 4px; font-family: "
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, "
        "Arial, sans-serif;\">",
        "<button id=\"zoom-in-btn\" class=\"graph-ctrl-btn\" style=\"cursor: "
        "pointer; width: 24px; height: 24px; padding: 0; border: 1px solid "
        "#dadce0; background: #fff; border-radius: 3px; font-family: "
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, "
        "Arial, sans-serif; font-size: 15px; font-weight: 500; color: #3c4043; "
        "display: flex; align-items: center; justify-content: center; "
        "line-height: 1; box-sizing: border-box;\" title=\"Zoom "
        "in\">+</button>",
        "<button id=\"zoom-out-btn\" class=\"graph-ctrl-btn\" style=\"cursor: "
        "pointer; width: 24px; height: 24px; padding: 0; border: 1px solid "
        "#dadce0; background: #fff; border-radius: 3px; font-family: "
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, "
        "Arial, sans-serif; font-size: 15px; font-weight: 500; color: #3c4043; "
        "display: flex; align-items: center; justify-content: center; "
        "line-height: 1; box-sizing: border-box;\" title=\"Zoom "
        "out\">&minus;</button>",
        "<button id=\"zoom-fit-btn\" class=\"graph-ctrl-btn\" style=\"cursor: "
        "pointer; height: 24px; padding: 0 8px; border: 1px solid #dadce0; "
        "background: #fff; border-radius: 3px; font-family: -apple-system, "
        "BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; "
        "font-size: 11px; font-weight: 500; color: #3c4043; display: flex; "
        "align-items: center; justify-content: center; line-height: 1; "
        "box-sizing: border-box;\" title=\"Fit graph to view\">Fit</button>",
        "</div>",
        "<canvas id=\"dag-canvas\" style=\"width: 100%; height: 100%; "
        "display: block;\"></canvas>");
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

  if (tensor_visualizations != nullptr && !tensor_visualizations->empty()) {
    absl::StrAppend(&data_injection_script,
                    SerializeTensorVisualizationsJs(*tensor_visualizations));
  } else {
    absl::StrAppend(&data_injection_script,
                    "window.tensorVisualizations = {};\n");
  }

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

  // absl::flat_hash_map<const HloComputation*, const HloInstruction*>
  //     comp_to_fusion;
  // for (const HloComputation* comp : module.computations()) {
  //   for (const HloInstruction* instr : comp->instructions()) {
  //     if (instr->opcode() == HloOpcode::kFusion) {
  //       comp_to_fusion[instr->fused_instructions_computation()] = instr;
  //     }
  //     TensorKey key = TensorKey::Create(instr->name(), ShapeIndex{});
  //     TensorAnnotation ann;
  //     ann.anchor_id = absl::StrCat("step", instr->unique_id());
  //     annotations[key] = std::move(ann);
  //   }
  // }

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
    if (target_instr->shape().IsTuple() &&
        mismatch.output_shape_index.has_value()) {
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
    annotations[key] = std::move(ann);
  }

  return annotations;
}

GraphData PopulateMismatchGraphData(
    const HloModule& module, absl::Span<const MismatchDetails> mismatches) {
  GraphData graph_data;

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

  absl::flat_hash_map<std::string, const HloInstruction*> name_to_instr;
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      name_to_instr[instr->name()] = instr;
    }
  }

  absl::flat_hash_map<const HloInstruction*, double> instr_to_score;
  auto update_score = [&](const HloInstruction* instr, double score) {
    auto [it, inserted] = instr_to_score.try_emplace(instr, score);
    if (!inserted) {
      if (score == kNanInfMismatchDiffScore ||
          it->second == kNanInfMismatchDiffScore) {
        it->second = kNanInfMismatchDiffScore;
      } else {
        it->second = std::max(it->second, score);
      }
    }
  };

  for (const MismatchDetails& m : mismatches) {
    double score = 100.0;
    if (std::isnan(m.actual) || std::isinf(m.actual) ||
        std::isnan(m.expected) || std::isinf(m.expected)) {
      score = kNanInfMismatchDiffScore;
    } else if (m.rel_error > 0.0) {
      score = m.rel_error * 100.0;
    }

    const HloInstruction* target_instr = nullptr;
    auto it = name_to_instr.find(m.target_instruction_name);
    if (it != name_to_instr.end()) {
      target_instr = it->second;
      update_score(target_instr, score);
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

    double score = 0.0;
    if (auto it = instr_to_score.find(instr); it != instr_to_score.end()) {
      score = it->second;
    }

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
  auto tensor_visualizations = PopulateTensorVisualizations(module, mismatches);

  xla::StackFrameIndexProto stack_frame_index = module.stack_frames().proto();
  std::string html =
      ConvertHloToHtml(module.name(), module.ToString(), annotations, {},
                       &stack_frame_index, &graph_data, &tensor_visualizations);

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
