/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_TRANSLATE_MHLO_TO_HLO_LOCATION_EXPORTER_H_
#define XLA_TRANSLATE_MHLO_TO_HLO_LOCATION_EXPORTER_H_

#include <string>

#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "xla/translate/mhlo_to_hlo/stack_frame_index_builder.h"
#include "xla/xla_data.pb.h"

namespace mlir {
namespace mhlo {

// Returns a OpMetadata proto based on the location of the op. If the location
// is unknown, an empty proto is returned. `op_name` are populated with the op
// location (converted). FileLineColLoc locations are populated by taking the
// file name and line number, and populating `source_file` and `source_line`
// respectively.
xla::OpMetadata CreateOpMetadataFromLocation(
    Operation* op, StackFrameIndexBuilder* frame_index_builder);

// Returns a name that can be used for debugging purposes, e.g., naming
// variable names in generated IR or producing logging output.
std::string GetDebugNameFromLocation(Location location);

}  // namespace mhlo
}  // namespace mlir

#endif  // XLA_TRANSLATE_MHLO_TO_HLO_LOCATION_EXPORTER_H_
