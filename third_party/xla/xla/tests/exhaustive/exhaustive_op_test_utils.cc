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

#include <cstddef>

#include "xla/tests/exhaustive/error_spec.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace exhaustive_op_test {

template <PrimitiveType T, size_t N>
/* static */ typename ExhaustiveOpTestTraits<T, N>::ErrorSpecGen
ExhaustiveOpTestTraits<T, N>::FallbackErrorSpecGen() {
  if constexpr (N == 1) {
    return +[](NativeT) { return ErrorSpec{}; };
  } else if constexpr (N == 2) {
    return +[](NativeT, NativeT) { return ErrorSpec{}; };
  } else {
    static_assert(
        N == 1 || N == 2,
        "ExhaustiveOpTestTraits<T, N>::FallbackErrorSpecGen() is only "
        "implemented for N == 1 and N == 2.");
  }
}

template class ExhaustiveOpTestTraits<C128, 1>;
template class ExhaustiveOpTestTraits<C64, 1>;
template class ExhaustiveOpTestTraits<F64, 1>;
template class ExhaustiveOpTestTraits<F32, 1>;
template class ExhaustiveOpTestTraits<F16, 1>;
template class ExhaustiveOpTestTraits<BF16, 1>;
template class ExhaustiveOpTestTraits<F8E5M2, 1>;
template class ExhaustiveOpTestTraits<F8E4M3FN, 1>;

template class ExhaustiveOpTestTraits<F64, 2>;
template class ExhaustiveOpTestTraits<F32, 2>;
template class ExhaustiveOpTestTraits<F16, 2>;
template class ExhaustiveOpTestTraits<BF16, 2>;
template class ExhaustiveOpTestTraits<F8E5M2, 2>;
template class ExhaustiveOpTestTraits<F8E4M3FN, 2>;

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
