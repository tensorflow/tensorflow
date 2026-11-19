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
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using FftTextTest = HloInterpreterReferenceMixin<HloTestBase>;

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

TEST_F(FftTextTest, FftNonPowerOfTwoLength) {
  absl::string_view hlo_string = R"(
  HloModule fft

  ENTRY main {
    input = c64[3,20]{1,0} parameter(0)
    ROOT fft = c64[3,20]{1,0} fft(input), fft_type=FFT, fft_length={20}
  })";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(FftTextTest, Rfft) {
  absl::string_view hlo_string = R"(
  HloModule rfft

  ENTRY main {
    input = f32[3,20]{1,0} parameter(0)
    ROOT fft = c64[3,11]{1,0} fft(input), fft_type=RFFT, fft_length={20}
  })";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(FftTextTest, Rfft2D) {
  absl::string_view hlo_string = R"(
  HloModule rfft2d

  ENTRY main {
    input = f32[4,8]{1,0} parameter(0)
    ROOT fft = c64[4,5]{1,0} fft(input), fft_type=RFFT, fft_length={4,8}
  })";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

}  // namespace
}  // namespace xla
