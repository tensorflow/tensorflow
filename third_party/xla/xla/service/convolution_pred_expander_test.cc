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

#include "xla/service/convolution_pred_expander.h"

#include <string>

#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace m = match;

using ConvolutionPredExpanderTest = HloTestBase;

TEST_F(ConvolutionPredExpanderTest, Match) {
  std::string hlo_string = R"(HloModule convolution_pred

ENTRY convolution_computation {
  input = pred[10,10]{1,0} parameter(0)
  kernel = pred[10,10]{1,0} parameter(1)
  ROOT conv = pred[10,10]{1,0} convolution(input, kernel), dim_labels=bf_io->bf
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ConvolutionPredExpander expander_pass;
  ASSERT_TRUE(expander_pass.Run(module.get()).value());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Convert(m::Convolution(m::Op().WithElementType(F16),
                                                   m::Op().WithElementType(F16))
                                        .WithElementType(F16))
                             .WithElementType(PRED)));
}

TEST_F(ConvolutionPredExpanderTest, NoMatch) {
  std::string hlo_string = R"(HloModule convolution_s8

ENTRY convolution_computation {
  input = s8[10,10]{1,0} parameter(0)
  kernel = s8[10,10]{1,0} parameter(1)
  ROOT conv = s8[10,10]{1,0} convolution(input, kernel), dim_labels=bf_io->bf
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ConvolutionPredExpander expander_pass;
  ASSERT_FALSE(expander_pass.Run(module.get()).value());
}

}  // namespace
}  // namespace xla
