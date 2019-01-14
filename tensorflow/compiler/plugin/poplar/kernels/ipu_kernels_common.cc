#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

namespace tensorflow {
namespace {
void ValidateLayoutDependencies(
    const absl::flat_hash_map<int64, int64>& layout_dependencies,
    const absl::flat_hash_set<int64>& allocating_indexes) {
  // Verify that a target of a layout dependency is not a dependency on another
  // layout dependency or an allocating index.
  for (auto pair : layout_dependencies) {
    auto target = pair.second;
    if (layout_dependencies.count(target) || allocating_indexes.count(target)) {
      LOG(FATAL) << "The layout dependencies of a custom op are not valid.";
    }
  }
}
}
void IpuOpKernel::AddRequiredAttributesToMap() {
  attribute_map_.AddAttribute("allocating_indexes", AllocatingIndexes());
  attribute_map_.AddAttribute("num_inplace_operands",
                              NumberOfInplaceOperands());
  ValidateLayoutDependencies(LayoutDependencies(), AllocatingIndexes());
  attribute_map_.AddAttribute("layout_dependencies", LayoutDependencies());
}
}  // namespace tensorflow
