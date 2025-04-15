/* Copyright 2017 The OpenXLA Authors.

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

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class FftTextTest : public HloTestBase {};

TEST_F(FftTextTest, Fft) {
  absl::string_view hlo_string = R"(
  HloModule Fft_module

  ENTRY Fft {
    input = c64[8,32]{0,1} parameter(0)
    fft = c64[8,32] fft(input), fft_type=FFT, fft_length={32}
    ROOT transpose = c64[32,8] transpose(fft), dimensions={1,0}
  })";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{4e-3, 4e-3}));
}

}  // namespace
}  // namespace xla
