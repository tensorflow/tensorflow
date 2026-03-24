/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/util/embedded_constant_buffers.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/TargetSelect.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class EmbeddedConstantBuffersTest : public ::testing::Test {
 protected:
  EmbeddedConstantBuffersTest() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  }
};

TEST_F(EmbeddedConstantBuffersTest, CreateEmbeddedConstantBuffers) {
  std::vector<uint8_t> data1 = {1, 2, 3};
  std::vector<uint8_t> data2 = {4, 5, 6};

  ConstantToEmbed constant_to_embed1;
  constant_to_embed1.symbol_prefix = "symbol1";
  constant_to_embed1.SerializeIntoBuffer(data1);

  ConstantToEmbed constant_to_embed2;
  constant_to_embed2.symbol_prefix = "symbol2";
  constant_to_embed2.SerializeIntoBuffer(data2);

  std::vector<ConstantToEmbed> constants_to_embed = {constant_to_embed1,
                                                     constant_to_embed2};

  TF_ASSERT_OK_AND_ASSIGN(
      EmbeddedConstantBuffers embedded_constant_buffers,
      CreateEmbeddedConstantBuffers(kTargetTripleForHost,
                                    absl::MakeSpan(constants_to_embed)));

  ASSERT_EQ(embedded_constant_buffers.variable_decls.size(), 2);
  EXPECT_EQ(embedded_constant_buffers.variable_decls[0].variable_name,
            "symbol1_constant_buffer_contents");
  EXPECT_EQ(embedded_constant_buffers.variable_decls[1].variable_name,
            "symbol2_constant_buffer_contents");

  EXPECT_FALSE(embedded_constant_buffers.object_file_data.empty());
}

}  // namespace
}  // namespace xla
