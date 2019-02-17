#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplin_ops.h"

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "absl/container/flat_hash_map.h"

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

#include <string>

namespace xla {
namespace poplarplugin {

const absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo>& GetPoplinOpInfoMap() {
  static absl::flat_hash_map<PoplibsOp, CustomPoplibOpInfo> info_map = {};
  return info_map;
}

}  // namespace poplarplugin
}  // namespace xla
