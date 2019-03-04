#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_UTIL_H_

/*
 * These functions are independent of poplar, and are included in the
 * optimizers target within the BUILD file.
 */

#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

class Shape;
class Literal;
class HloSharding;

namespace poplarplugin {
namespace {
template <typename To, typename From>
bool check_convert_ok(const To& to, const From& from) {
  To from_converted = static_cast<To>(from);
  From to_converted = static_cast<From>(to);
  return from_converted == to && from == to_converted;
}
}  // namespace

template <typename To, typename From>
absl::optional<To> convert_array(const From& from) {
  To out;
  for (const auto& e : from) {
    out.push_back(e);
    if (!check_convert_ok(out.back(), e)) {
      return absl::nullopt;
    }
  }
  return out;
};

template <typename To, typename From>
absl::optional<To> convert_scalar(const From& from) {
  To to = static_cast<To>(from);
  return check_convert_ok(to, from) ? absl::optional<To>(to) : absl::nullopt;
};

int64 CountShapes(const Shape& shape);
// Find the index when embedding a shape into a tuple. The tuple_index is the
// index of the shape in the new tuple, and the original_index is the index
// of the tensor in the original shape.
int64 InsertIntoTuple(const Shape& tuple, int64 tuple_index,
                      int64 original_index);

std::vector<Shape> FlattenedXlaShape(const Shape& shape);

template <typename NativeT>
StatusOr<NativeT> LiteralScalarToNativeType(const Literal& lit);
template <typename NativeT>
StatusOr<std::vector<NativeT>> LiteralVectorToNativeType(const Literal& lit);
template <typename NativeT>
StatusOr<std::vector<NativeT>> WideConstToNativeType(
    const HloInstruction* wide_const);

bool IsPopOpsFusion(const HloComputation*, const std::string& postfix = "");
bool IsPopOpsFusion(const HloInstruction*, const std::string& postfix = "");
bool IsRepeatLoop(const HloInstruction*);

bool IsSupportedSharding(const HloSharding&);

bool IsInterIpuCopy(const HloInstruction*);
// This function returns the operand of inst at index operand_idx and if the
// operand is an inter ipu copy then it returns the operand which is being
// copied.
const HloInstruction* GetOperandLookThroughInterIpuCopy(
    const HloInstruction* inst, const int64 operand_idx);

// This function returns true if the environment variable has been set. Using
// synthetic data means that *no data* will be copied to/from the device.
bool UseSyntheticData();

std::string GetDebugName(const HloInstruction*);

void GetAllDeps(const HloInstruction* base, std::vector<HloInstruction*>& deps);

void GetAllDepNames(const HloInstruction* base,
                    std::vector<std::string>& names);

}  // namespace poplarplugin
}  // namespace xla

#endif
