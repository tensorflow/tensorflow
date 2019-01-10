#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_UTIL_H_

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

namespace poplarplugin {
namespace {
template <typename To, typename From>
bool check_convert_ok(const To& to, const From& from) {
  To from_converted = static_cast<To>(from);
  From to_converted = static_cast<From>(to);
  return from_converted == to && from == to_converted;
}
}

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

std::vector<xla::Shape> FlattenedXlaShape(const xla::Shape& shape);

template <typename NativeT>
StatusOr<NativeT> LiteralScalarToNativeType(const xla::Literal& lit);
template <typename NativeT>
StatusOr<std::vector<NativeT>> LiteralVectorToNativeType(
    const xla::Literal& lit);
template <typename NativeT>
StatusOr<std::vector<NativeT>> WideConstToNativeType(
    const xla::HloInstruction* wide_const);

bool IsPopOpsFusion(const xla::HloComputation*,
                    const std::string& postfix = "");
bool IsPopOpsFusion(const xla::HloInstruction*,
                    const std::string& postfix = "");
bool IsRepeatCall(const xla::HloComputation*);
bool IsRepeatCall(const xla::HloInstruction*);
// This functions assumes that IsRepeatCall(inst) is true
xla::HloComputation* GetRepeatBody(xla::HloInstruction* inst);
const xla::HloComputation* GetRepeatBody(const xla::HloInstruction* inst);

// This function returns true if the environment variable has been set. Using
// synthetic data means that *no data* will be copied to/from the device.
bool UseSyntheticData();

std::string GetDebugName(const HloInstruction*);

}  // namespace poplarplugin
}  // namespace xla

#endif
