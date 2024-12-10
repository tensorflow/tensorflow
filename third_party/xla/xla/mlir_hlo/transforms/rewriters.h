/* Copyright 2021 The OpenXLA Authors.

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

#ifndef MLIR_HLO_TRANSFORMS_REWRITERS_H
#define MLIR_HLO_TRANSFORMS_REWRITERS_H

#include "mlir/IR/MLIRContext.h"

namespace mlir {
class MLIRContext;
class RewritePatternSet;
class TypeConverter;

/// Collects a set of patterns that bufferize operations from the standard and
/// other dialects.
void populateExtraBufferizePatterns(MLIRContext *context,
                                    TypeConverter *converter,
                                    RewritePatternSet *patterns);

}  // namespace mlir

#endif  // MLIR_HLO_TRANSFORMS_REWRITERS_H
