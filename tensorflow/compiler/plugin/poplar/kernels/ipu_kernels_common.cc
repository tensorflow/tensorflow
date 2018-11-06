#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

namespace tensorflow {
IpuOpKernel::IpuOpKernel(){};

void IpuOpKernel::AddRequiredAttributesToMap() {
  attribute_map_.AddAttribute("allocating_indexes", AllocatingIndexes());
}
}  // namespace tensorflow
