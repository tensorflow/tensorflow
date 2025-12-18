#include <deque>
#include <queue>
#include <string>
#include <vector>

#include "xla/service/outer_dimension_propagation_example.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace xla {
using Path = std::string;  // e.g., "0" or "1.2"
using PathMulMap = absl::flat_hash_map<Path, int64_t>;

static std::string MakePath(const Path& prefix, int64_t idx) {
  if (prefix.empty()) return absl::StrCat(idx);
  return absl::StrCat(prefix, ".", idx);
}
// Simplified recursive parser for the serialized mapping format produced by
// SerializeMappingForShape. 
static bool ParseSerializedMappingForShapeRec(const Shape& shape,
                                             absl::string_view s, size_t& pos,
                                             const Path& prefix, PathMulMap& out) {
  auto skip_ws = [&](void) {
    while (pos < s.size() && isspace(static_cast<unsigned char>(s[pos]))) ++pos;
  };
  skip_ws();
  if (!shape.IsTuple()) {
    // parse a number or 'null'
    if (pos >= s.size()) return true;  // empty -> null
    if (s.substr(pos, 4) == "null") {
      pos += 4;
      return true;
    }
    // parse optional sign and digits
    size_t start = pos;
    if (s[pos] == '+' || s[pos] == '-') ++pos;
    while (pos < s.size() && isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
    if (start == pos) return false;  // no number
    int64_t v = 0;
    if (!absl::SimpleAtoi<int64_t>(std::string(s.substr(start, pos - start)), &v)) return false;
    Path key = prefix.empty() ? std::string("0") : prefix;
    out.emplace(key, v);
    skip_ws();
    return true;
  }

  // shape is tuple: expect '['
  skip_ws();
  if (pos >= s.size() || s[pos] != '[') return false;
  ++pos;
  skip_ws();
  int64_t arity = shape.tuple_shapes_size();
  for (int64_t i = 0; i < arity; ++i) {
    const Shape& child_shape = shape.tuple_shapes(i);
    // parse child value (could be nested array, number, or null)
    if (child_shape.IsTuple()) {
      if (!ParseSerializedMappingForShapeRec(child_shape, s, pos, MakePath(prefix, i), out)) return false;
    } else {
      // Non-tuple child: parse number or 'null'
      skip_ws();
      if (pos < s.size() && s.substr(pos, 4) == "null") {
        pos += 4;
      } else {
        size_t start = pos;
        if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
        while (pos < s.size() && isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
        if (start == pos) return false;
        int64_t v = 0;
        if (!absl::SimpleAtoi<int64_t>(std::string(s.substr(start, pos - start)), &v)) return false;
        out.emplace(MakePath(prefix, i), v);
      }
    }
    skip_ws();
    // after each child expect ',' or ']' (if last)
    if (i < arity - 1) {
      if (pos >= s.size() || s[pos] != ',') return false;
      ++pos;
      skip_ws();
    }
  }
  skip_ws();
  if (pos >= s.size() || s[pos] != ']') return false;
  ++pos;
  skip_ws();
  return true;
}

static std::pair<bool, PathMulMap> ParseSerializedMappingForShape(
    const Shape& shape, absl::string_view s, const Path& prefix = "") {
  PathMulMap out;
  size_t pos = 0;
  if (!ParseSerializedMappingForShapeRec(shape, s, pos, prefix, out)) {
    return {false, out};
  }
  // allow trailing spaces but nothing else
  while (pos < s.size() && isspace(static_cast<unsigned char>(s[pos]))) ++pos;
  if (pos != s.size()) return {false, out};
  return {true, out};
}

absl::StatusOr<bool> OuterDimensionPropagationExamplePass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  static const char kMarker[] = "|tf_outer_marker=";
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instr : computation->instructions()) {
      std::string opname = instr->metadata().op_name();
      size_t pos = opname.rfind(kMarker);
      if (pos != std::string::npos) {
        std::string serialized = opname.substr(pos + strlen(kMarker));
        auto parsed =
            ParseSerializedMappingForShape(instr->shape(), serialized, "");
        if (!parsed.first) {
          LOG(ERROR)
              << "Failed to parse tf_outer_marker serialized mapping for "
              << instr->ToString() << ", falling back to {\"0\":1}";
          continue;
        }
        auto shape = instr->shape();
        if (!shape.IsTuple()) {
          LOG(INFO) << "The batch multiplier of operator: " << instr->ToString()
                    << " multiplier is " << shape.outer_multiplier();
        } else {
          int64_t arity = shape.tuple_shapes_size();
          for (int64_t i = 0; i < arity; ++i) {
            Shape* child = shape.mutable_tuple_shapes(i);
            LOG(INFO) << "The batch multiplier of operator: "
                      << instr->ToString() << "\n"
                      << i << "th multiplier is " << shape.outer_multiplier();
          }
        }
      }
    }
  }
  return false;
}
} // namespace xla
