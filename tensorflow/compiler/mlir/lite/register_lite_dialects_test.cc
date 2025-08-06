/* Copyright 2025 Google Inc. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/register_lite_dialects.h"

#include <gtest/gtest.h>
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace tflite {
namespace {

TEST(RegisterCommonDialectsTest, DoesntCrash) {
  mlir::DialectRegistry registry;

  tflite::RegisterLiteToolingDialects(registry);

  EXPECT_FALSE(registry.getDialectNames().empty());
}

}  // namespace
}  // namespace tflite
