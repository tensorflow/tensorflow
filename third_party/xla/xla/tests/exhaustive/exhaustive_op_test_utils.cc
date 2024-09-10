/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"

#include "xla/types.h"

namespace xla {
namespace exhaustive_op_test {

bool IsSubnormalReal(xla::complex64 value) { return IsSubnormal(value.real()); }

bool IsSubnormalReal(xla::complex128 value) {
  return IsSubnormal(value.real());
}

bool IsMinNormalReal(xla::complex64 value) { return IsMinNormal(value.real()); }

bool IsMinNormalReal(xla::complex128 value) {
  return IsMinNormal(value.real());
}

bool IsSubnormalImaginary(xla::complex64 value) {
  return IsSubnormal(value.imag());
}

bool IsSubnormalImaginary(xla::complex128 value) {
  return IsSubnormal(value.imag());
}

bool IsMinNormalImaginary(xla::complex64 value) {
  return IsMinNormal(value.imag());
}

bool IsMinPositiveImaginary(xla::complex128 value) {
  return IsMinNormal(value.imag());
}

}  // namespace exhaustive_op_test
}  // namespace xla
