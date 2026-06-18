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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_ODML_CONVERTER_FOLDERS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_ODML_CONVERTER_FOLDERS_H_

namespace mlir::odml {

// Populates the pattern set with all folding patterns. These patterns
// are intended to have precedence over any other patterns added to the set.
void PopulateFolderPatterns(RewritePatternSet &patternSet);

}  // namespace mlir::odml

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_ODML_CONVERTER_FOLDERS_H_
