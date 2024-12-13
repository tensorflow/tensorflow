/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/triton_call.h"

#include <cstdint>
#include <utility>

#include "absl/strings/string_view.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

namespace xla::gpu {

TritonCall TritonCall::Parse(absl::string_view backend_config,
                             mlir::MLIRContext* mlir_context) {
  // TODO(slebedev): Plumb through num_ctas and enable_wrap_specialization.
  auto attrs = mlir::cast<mlir::DictionaryAttr>(
      mlir::parseAttribute(backend_config, mlir_context));
  auto name = attrs.getAs<mlir::StringAttr>("name").getValue().str();
  auto ir = attrs.getAs<mlir::StringAttr>("ir").str();
  auto grid_x = static_cast<int32_t>(
      attrs.getAs<mlir::IntegerAttr>("grid_x").getValue().getSExtValue());
  auto grid_y = static_cast<int32_t>(
      attrs.getAs<mlir::IntegerAttr>("grid_y").getValue().getSExtValue());
  auto grid_z = static_cast<int32_t>(
      attrs.getAs<mlir::IntegerAttr>("grid_z").getValue().getSExtValue());
  auto num_stages =
      attrs.getAs<mlir::IntegerAttr>("num_stages").getValue().getSExtValue();
  auto num_warps =
      attrs.getAs<mlir::IntegerAttr>("num_warps").getValue().getSExtValue();
  return TritonCall{std::move(name), std::move(ir), num_stages, num_warps,
                    grid_x,          grid_y,        grid_z};
}

}  // namespace xla::gpu
