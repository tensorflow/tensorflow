#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_UTIL_H_

/*
 * These functions are independent of poplar, and are included in the
 * optimizers target within the BUILD file.
 */

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

class Shape;
class Literal;

namespace poplarplugin {

template <typename To, typename From>
To convert_array(const From& from) {
  To out;
  for (const auto& e : from) {
    out.push_back(e);
  }
  return out;
};

int64 CountShapes(const Shape& shape);

std::vector<xla::Shape> FlattenedXlaShape(const xla::Shape& shape);

StatusOr<std::vector<int64>> LiteralVectorToInt64Vector(
    const xla::Literal& lit);

StatusOr<std::vector<int64>> WideConstToInt64Vector(
    const xla::HloInstruction* bcast, const xla::HloInstruction* constant);

StatusOr<int64> LiteralScalarInt64toInt64(const xla::Literal& lit);
StatusOr<double> LiteralScalarDoubleToDouble(const xla::Literal& lit);

bool IsPopOpsCall(const xla::HloComputation*, const std::string& postfix = "");
bool IsPopOpsCall(const xla::HloInstruction*, const std::string& postfix = "");

// This function returns true if the environment variable has been set. Using
// synthetic data means that *no data* will be copied to/from the device.
bool UseSyntheticData();

std::string GetDebugName(const HloInstruction*);

}  // namespace poplarplugin
}  // namespace xla

#endif
