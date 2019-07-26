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

#include "tensorflow/compiler/xla/tests/exhaustive_op_test_utils.h"

namespace xla {

// For f32, f16, and bf16, we need 9, 5, and 4 decimal places of precision to be
// guaranteed that we're printing the full number.
//
// (The general formula is, given a floating-point number with S significand
// bits, the number of decimal digits needed to print it to full precision is
//
//   ceil(1 + S * log_10(2)) ~= ceil(1 + S * 0.30103).
//
// See https://people.eecs.berkeley.edu/~wkahan/Math128/BinDecBin.pdf.)
/*static*/
string ExhaustiveOpTestBase::StringifyNum(float x) {
  return absl::StrFormat("%0.9g (0x%08x)", x, BitCast<uint32>(x));
}

/*static*/
string ExhaustiveOpTestBase::StringifyNum(half x) {
  return absl::StrFormat("%0.5g (0x%04x)", static_cast<float>(x),
                         BitCast<uint16>(x));
}

/*static*/
string ExhaustiveOpTestBase::StringifyNum(bfloat16 x) {
  return absl::StrFormat("%0.4g (0x%04x)", static_cast<float>(x),
                         BitCast<uint16>(x));
}

/*static*/
std::vector<std::pair<int64, int64>>
ExhaustiveOpTestBase::CreateExhaustiveF32Ranges() {
  // We break up the 2^32-element space into small'ish chunks to keep peak
  // memory usage low.
  std::vector<std::pair<int64, int64>> result;
  const int64 step = 1 << 25;
  for (int64 i = 0; i < (1l << 32); i += step) {
    result.push_back({i, i + step});
  }
  return result;
}

namespace {
ExhaustiveOpTestBase::ErrorSpec DefaultF64SpecGenerator(float) {
  return ExhaustiveOpTestBase::ErrorSpec(0.0001, 0.0001);
}

ExhaustiveOpTestBase::ErrorSpec DefaultF32SpecGenerator(float) {
  return ExhaustiveOpTestBase::ErrorSpec(0.0001, 0.0001);
}

ExhaustiveOpTestBase::ErrorSpec DefaultF16SpecGenerator(float) {
  return ExhaustiveOpTestBase::ErrorSpec(0.001, 0.001);
}

ExhaustiveOpTestBase::ErrorSpec DefaultBF16SpecGenerator(float) {
  return ExhaustiveOpTestBase::ErrorSpec(0.002, 0.02);
}
}  // namespace

/*static*/
std::function<ExhaustiveOpTestBase::ErrorSpec(float)>
ExhaustiveOpTestBase::GetDefaultSpecGenerator(PrimitiveType ty) {
  switch (ty) {
    case C128:
    case F64:
      return DefaultF64SpecGenerator;
    case C64:
    case F32:
      return DefaultF32SpecGenerator;
    case F16:
      return DefaultF16SpecGenerator;
    case BF16:
      return DefaultBF16SpecGenerator;
    default:
      LOG(FATAL) << "Unhandled Type";
  }
}

}  // namespace xla
