/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSLATE_HLO_TO_MHLO_STACK_LOCATION_UTILS_H_
#define XLA_HLO_TRANSLATE_HLO_TO_MHLO_STACK_LOCATION_UTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"

namespace mlir {
namespace hlo {
// Construct MLIR location from frame index.
// Returns unknown location if frame is not presented.
// frame_locations, when given, memoizes the location of every frame of
// hlo_module by frame id across calls: it is sized on first use and must only
// be used with hlo_module and one context.
mlir::Location GetLocationFromFrameIndex(
    int frame_id, mlir::Builder& builder, const xla::HloModule* hlo_module,
    llvm::SmallVectorImpl<mlir::LocationAttr>* frame_locations = nullptr);

}  // namespace hlo
}  // namespace mlir

#endif  // XLA_HLO_TRANSLATE_HLO_TO_MHLO_STACK_LOCATION_UTILS_H_
