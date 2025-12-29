/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/lowering_util.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/tma_utils.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu::triton {

absl::StatusOr<stream_executor::ThreadDim> ExtractThreadDims(
    mlir::ModuleOp triton_module, mlir::LLVM::LLVMFuncOp func_op) {
  // Extract the launch information from the Triton module.
  auto threads_per_warp_attr =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.threads-per-warp");
  if (!threads_per_warp_attr) {
    return absl::InternalError("ttg.threads-per-warp attribute not found.");
  }
  auto num_warps_attr =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.num-warps");
  if (!num_warps_attr) {
    return absl::InternalError("ttg.num-warps attribute not found.");
  }
  // AMD/ROCm Triton backend does not support warp specialization.
  // Consequently, `ttg.total-num-warps` and  `nvvm.reqntid` are not added
  // to triton module/function.
  // ThreadDim is therefore calculated from the Module attributes and not
  // retrieved from `nvvm.reqntid`.
  auto target = triton_module->getAttrOfType<mlir::StringAttr>("ttg.target");
  if (!target) {
    return absl::InternalError("ttg.target attribute not found.");
  }
  if (target.getValue().find("gfx") != std::string::npos) {
    stream_executor::ThreadDim thread_dims(
        num_warps_attr.getInt() * threads_per_warp_attr.getInt(), 1, 1);
    return thread_dims;
  }
  auto total_num_warps_attr =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.total-num-warps");
  if (!total_num_warps_attr) {
    return absl::InternalError("ttg.total-num-warps attribute not found.");
  }
  auto reqntid_attr =
      func_op->getAttrOfType<mlir::DenseI32ArrayAttr>("nvvm.reqntid");
  if (!reqntid_attr) {
    return absl::InternalError("nvvm.reqntid attribute not found.");
  }
  auto reqntids = reqntid_attr.asArrayRef();
  if (reqntids.empty()) {
    return absl::InternalError("nvvm.reqntid attribute is empty.");
  }
  if (reqntids.size() > 3) {
    return absl::InternalError(
        "nvvm.reqntid attribute has more than 3 dimensions.");
  }

  // Validate the launch information.
  if (num_warps_attr.getInt() != total_num_warps_attr.getInt()) {
    VLOG(6)
        << "num_warps and total_num_warps are different! This can happen if "
           "Triton compilation decides to use a different number of warps than "
           "configured. e.g. auto warp specialization can do that.";
  }
  int64_t expected_total_threads = xla::Product<int32_t>(reqntids);
  int64_t actual_total_threads =
      total_num_warps_attr.getInt() * threads_per_warp_attr.getInt();
  if (actual_total_threads != expected_total_threads) {
    return absl::InternalError(absl::StrCat(
        "Expected total threads as per reqntid attribute to be ",
        expected_total_threads, " but got ", actual_total_threads,
        " as per ttg.total-num-warps and tt.threads-per-warp attributes."));
  }

  stream_executor::ThreadDim thread_dims(reqntids[0],
                                         reqntids.size() > 1 ? reqntids[1] : 1,
                                         reqntids.size() > 2 ? reqntids[2] : 1);
  return thread_dims;
}

absl::StatusOr<stream_executor::gpu::TmaMetadata> ExtractTmaMetadata(
    mlir::LLVM::LLVMFuncOp func_op) {
  stream_executor::gpu::TmaMetadata tma_metadata;
  for (auto [idx, arg] : llvm::enumerate(func_op.getArguments())) {
    if (auto attr =
            func_op.getArgAttrOfType<mlir::triton::xla::TmaDescriptorAttr>(
                idx, "tt.tma_descriptor")) {
      TF_ASSIGN_OR_RETURN(
          auto tma_desc,
          CreateTmaDescriptor(attr.getGlobalShape(), attr.getTileShape(),
                              attr.getTileStrides(), attr.getLayout(),
                              attr.getElementByteSize(),
                              attr.getSwizzleMode().getValue()));
      tma_metadata.arg_index_to_tma_info.insert({idx, tma_desc});
    }
  }
  return tma_metadata;
}

std::vector<llvm::Metadata*> ExtractNvvmAnnotations(
    llvm::Module* ll_triton_module) {
  std::vector<llvm::Metadata*> captured_nvvm_annotations;
  llvm::NamedMDNode* nvvm_annotations =
      ll_triton_module->getNamedMetadata("nvvm.annotations");
  if (nvvm_annotations) {
    for (llvm::MDNode* operand : nvvm_annotations->operands()) {
      captured_nvvm_annotations.push_back(operand);
    }
    ll_triton_module->eraseNamedMetadata(nvvm_annotations);
  }
  return captured_nvvm_annotations;
}

}  // namespace xla::gpu::triton
