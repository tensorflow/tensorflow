/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/aot/embedded_constant_buffers.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/TargetSelect.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "tsl/platform/statusor.h"

namespace tensorflow::tfcompile {

namespace {

class EmbeddedConstantBuffersTest : public ::testing::Test {
 protected:
  EmbeddedConstantBuffersTest() {
    // Initialize LLVM's MC layer for the native target.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
};

TEST_F(EmbeddedConstantBuffersTest, CreateEmbeddedConstantBuffers) {
  std::vector<ConstantToEmbed> constants_to_embed(1);

  constants_to_embed[0].SerializeIntoBuffer(std::vector<uint8_t>({1, 2, 3}));
  TF_ASSERT_OK_AND_ASSIGN(
      EmbeddedConstantBuffers buffers,
      CreateEmbeddedConstantBuffers(kTargetTripleForHost,
                                    absl::MakeSpan(constants_to_embed)));

  EXPECT_EQ(buffers.variable_decls.size(), constants_to_embed.size());

  for (const auto& variable_decl : buffers.variable_decls) {
    EXPECT_EQ(variable_decl.variable_name, "_constant_buffer_contents");
    EXPECT_EQ(variable_decl.variable_decl,
              "extern \"C\" char _constant_buffer_contents[];");
    EXPECT_EQ(variable_decl.cpp_access_shim,
              "\n    [](char* buffer) -> std::pair<uint64_t, char*> {\n"
              "      uint64_t buffer_size;\n"
              "      std::memcpy(&buffer_size, buffer, sizeof(uint64_t));\n"
              "      return {buffer_size, buffer + sizeof(uint64_t)};\n"
              "    }(_constant_buffer_contents)\n    ");
  }
}

}  // namespace

}  // namespace tensorflow::tfcompile
