//===- Transforms.h - Linalg dialect Transformations definition -----------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef LINALG2_TRANSFORMS_H_
#define LINALG2_TRANSFORMS_H_

namespace mlir {
class Value;
} // namespace mlir

namespace linalg {

class ViewOp;

/// Takes a `view` of type ViewType (i.e. either a ViewOp or a SliceOp) and
/// composes away all the SliceOp to return a single ViewOp.
/// Inserts the required operations after `view`.
ViewOp emitAndReturnFullyComposedView(mlir::Value *v);

} // namespace linalg

#endif // LINALG2_TRANSFORMS_H_
