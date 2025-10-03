/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/kernel_call.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

// Helper function to parse kernel type string into enum
absl::StatusOr<KernelCall::KernelType> ParseKernelType(
    const std::string& kernel_type_str) {
  if (kernel_type_str == "ptx") {
    return KernelCall::KernelType::kPtxSource;
  } else if (kernel_type_str == "cubin") {
    return KernelCall::KernelType::kCudaBinary;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown kernel type: ", kernel_type_str,
                     ". Supported types: 'ptx', 'cubin'"));
  }
}

absl::StatusOr<KernelCall> KernelCall::Parse(absl::string_view backend_config,
                                             mlir::MLIRContext* mlir_context) {
  auto attrs = mlir::cast<mlir::DictionaryAttr>(
      mlir::parseAttribute(backend_config, mlir_context));

  // Check for required "name" field
  auto name_attr = attrs.getAs<mlir::StringAttr>("name");
  if (!name_attr) {
    return absl::InvalidArgumentError(
        "Missing required field 'name' in backend_config");
  }
  auto name = name_attr.getValue().str();

  // Check for required "kernel_type" field
  auto kernel_type_attr = attrs.getAs<mlir::StringAttr>("kernel_type");
  if (!kernel_type_attr) {
    return absl::InvalidArgumentError(
        "Missing required field 'kernel_type' in backend_config");
  }
  auto kernel_type_str = kernel_type_attr.getValue().str();
  TF_ASSIGN_OR_RETURN(KernelCall::KernelType kernel_type,
                      ParseKernelType(kernel_type_str));

  // Check for required "kernel_data" field
  auto kernel_data_attr = attrs.getAs<mlir::StringAttr>("kernel_data");
  if (!kernel_data_attr) {
    return absl::InvalidArgumentError(
        "Missing required field 'kernel_data' in backend_config");
  }
  auto kernel_data = kernel_data_attr.getValue().str();

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << "Kernel Call backend_config:";
    for (const auto& namedAttr : attrs) {
      std::string value_str;
      llvm::raw_string_ostream os(value_str);
      namedAttr.getValue().print(os);
      LOG(INFO) << "  " << namedAttr.getName().str() << ": " << value_str;
    }
  }

  auto get_int32_attr =
      [&attrs](const char* attr_name) -> absl::StatusOr<int32_t> {
    auto attr = attrs.getAs<mlir::IntegerAttr>(attr_name);
    if (!attr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Missing required field '", attr_name, "' in backend_config"));
    }
    return static_cast<int32_t>(attr.getValue().getSExtValue());
  };

  TF_ASSIGN_OR_RETURN(int32_t grid_x, get_int32_attr("grid_x"));
  TF_ASSIGN_OR_RETURN(int32_t grid_y, get_int32_attr("grid_y"));
  TF_ASSIGN_OR_RETURN(int32_t grid_z, get_int32_attr("grid_z"));
  TF_ASSIGN_OR_RETURN(int32_t block_x, get_int32_attr("block_x"));
  TF_ASSIGN_OR_RETURN(int32_t block_y, get_int32_attr("block_y"));
  TF_ASSIGN_OR_RETURN(int32_t block_z, get_int32_attr("block_z"));
  TF_ASSIGN_OR_RETURN(int32_t shared_mem, get_int32_attr("shared_mem_bytes"));

  // Optional output_indices field
  mlir::ArrayAttr output_indices =
      attrs.getAs<mlir::ArrayAttr>("output_indices");
  std::vector<int32_t> output_indices_vec;
  if (output_indices) {
    for (const mlir::Attribute& index : output_indices) {
      auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(index);
      if (!int_attr) {
        return absl::InvalidArgumentError(
            "Invalid output_indices: all elements must be integers");
      }
      output_indices_vec.push_back(int_attr.getValue().getSExtValue());
    }
  }

  return KernelCall{
      std::move(name),
      std::move(kernel_data),
      kernel_type,
      stream_executor::BlockDim(grid_x, grid_y, grid_z),
      stream_executor::ThreadDim(block_x, block_y, block_z),
      static_cast<size_t>(shared_mem),
      std::move(output_indices_vec),
  };
}

}  // namespace xla::gpu
