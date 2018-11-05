#include "tensorflow/compiler/plugin/poplar/driver/custom_ops/poplibs_ops.h"

#include <string>
#include "absl/container/flat_hash_map.h"

namespace xla {
namespace poplarplugin {
namespace {
absl::flat_hash_map<std::string, CustomPoplibsCallFn> call_map = {};
}

const absl::flat_hash_map<std::string, CustomPoplibsCallFn>&
GetPoprandCallMap() {
  return call_map;
}

}  // namespace poplarplugin
}  // namespace xla
