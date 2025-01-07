/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>
#include <optional>

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/fusions/triton/xla_triton_ops.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::xla {

//--- SparseDotMetaEncodingAttr ---
unsigned SparseDotMetaEncodingAttr::getTotalElemsPerThread(
    ArrayRef<int64_t> shape, Type eltTy) const {
  constexpr int kMetadataElementsPerWarp = 16;
  auto mmaLayout = mlir::cast<gpu::NvidiaMmaEncodingAttr>(getParent());
  return product<int64_t>(shape) /
         (mmaLayout.getWarpsPerCTA()[0] * kMetadataElementsPerWarp);
}

SmallVector<unsigned> SparseDotMetaEncodingAttr::getElemsPerThread(
    ArrayRef<int64_t> shape, Type eltTy) const {
  llvm_unreachable("getElemsPerThread is not supported for sparse dot meta");
  return SmallVector<unsigned>();
}

SmallVector<unsigned> SparseDotMetaEncodingAttr::getCTAsPerCGA() const {
  return gpu::getCTAsPerCGA(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getCTAOrder() const {
  return gpu::getCTAOrder(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getCTASplitNum() const {
  return gpu::getCTASplitNum(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getWarpsPerCTA() const {
  return gpu::getWarpsPerCTA(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getWarpOrder() const {
  return {1, 0};
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getThreadsPerWarp() const {
  return gpu::getThreadsPerWarp(getParent());
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getThreadOrder() const {
  return {1, 0};
}
SmallVector<unsigned> SparseDotMetaEncodingAttr::getSizePerThread() const {
  return gpu::getSizePerThread(getParent());
}
std::optional<LinearLayout> SparseDotMetaEncodingAttr::toLinearLayout(
    ArrayRef<int64_t> shape) const {
  return gpu::toLinearLayout(shape, getParent());
}

SmallVector<unsigned> SparseDotMetaEncodingAttr::getRepOrder() const {
  // TODO: b/381422752 - Maybe we should reuse upstream's implementation from
  // lib/Dialect/TritonGPU/IR/Dialect.cpp, but we would need to make it public
  // first.
  if (auto parent = mlir::dyn_cast<gpu::DistributedEncodingTrait>(getParent()))
    return parent.getRepOrder();
  llvm::report_fatal_error("Unimplemented usage of getRepOrder");
}

}  // namespace mlir::triton::xla
