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

#include <unordered_set>

#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

// TODO(silvasean): Add a FileCheck based testing harness for SavedModel to
// replace the following. The source should be TensorFlow Python code. Then we
// can generate SavedModel directories on the fly and import them. Check
// directives can be embedded into the same file as the source.
TEST(SavedModel, HalfPlusTwo) {
  const char kSavedModel[] = "cc/saved_model/testdata/half_plus_two/00000123";
  const std::string saved_model_dir = tensorflow::io::JoinPath(
      tensorflow::testing::TensorFlowSrcRoot(), kSavedModel);
  std::unordered_set<std::string> tags{tensorflow::kSavedModelTagServe};

  mlir::MLIRContext context;
  auto module = tensorflow::SavedModelToMlirImport(
      saved_model_dir, tags, /*debug_info_file=*/"", &context);
  auto* block = module->getBody();

  // testdata/half_plus_two does not use any functions. So we only have the
  // mandatory module terminator op inside its block.
  EXPECT_TRUE(std::next(block->begin()) == block->end());
}
