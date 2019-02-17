#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poprand_ops.h"

#include <string>
#include "absl/container/flat_hash_map.h"

namespace xla {
namespace poplarplugin {

const absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo>&
GetPoprandOpInfoMap() {
  static absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo> info_map = {};
  return info_map;
}

}  // namespace poplarplugin
}  // namespace xla
