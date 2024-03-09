/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/location_utils.h"

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project

namespace tensorflow {

mlir::Location GetLocationWithoutOpType(mlir::Location loc) {
  if (auto fused_loc = loc.dyn_cast<mlir::FusedLoc>()) {
    auto locations = fused_loc.getLocations();
    if (!locations.empty()) {
      // Skip locations for propagating op_type metadata.
      if (auto name_loc = locations[0].dyn_cast<mlir::NameLoc>()) {
        if (name_loc.getName().strref().ends_with(":")) {
          if (locations.size() == 2)
            return locations[1];
          else if (locations.size() > 2)
            return mlir::FusedLoc::get(
                fused_loc.getContext(),
                {locations.begin() + 1, locations.end()});
        }
      }
    }
  }
  return loc;
}

}  // namespace tensorflow
