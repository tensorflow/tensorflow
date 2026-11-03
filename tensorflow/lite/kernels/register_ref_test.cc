/* Copyright 2026 Google LLC.

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
#include "tensorflow/lite/kernels/register_ref.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite::ops::builtin {
namespace {

// Matches kMaxCheckedVersion in
// tensorflow/lite/core/kernels/register_test.cc, which fails when a
// builtin op reaches it.
constexpr int kMaxCheckedVersion = 64;

// The reference kernels serve as an oracle (LiteRT's reference CPU kernel mode,
// the ATS reference backend), so they have to load every model the builtin
// kernels load: both resolvers register the same op versions.
TEST(BuiltinRefOpResolverTest, RegistersTheBuiltinOpVersions) {
  BuiltinOpResolver builtin;
  BuiltinRefOpResolver reference;
  for (int id = BuiltinOperator_MIN; id <= BuiltinOperator_MAX; ++id) {
    auto op = static_cast<BuiltinOperator>(id);
    for (int version = 1; version <= kMaxCheckedVersion; ++version) {
      EXPECT_EQ(builtin.FindOp(op, version) != nullptr,
                reference.FindOp(op, version) != nullptr)
          << EnumNameBuiltinOperator(op) << " version " << version
          << " is registered by only one of BuiltinOpResolver and "
             "BuiltinRefOpResolver";
    }
  }
}

}  // namespace
}  // namespace tflite::ops::builtin
