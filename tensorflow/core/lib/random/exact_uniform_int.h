/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Exact uniform integers using rejection sampling

#ifndef TENSORFLOW_CORE_LIB_RANDOM_EXACT_UNIFORM_INT_H_
#define TENSORFLOW_CORE_LIB_RANDOM_EXACT_UNIFORM_INT_H_

#include <type_traits>

namespace tensorflow {
namespace random {

template <typename UintType, typename RandomBits>
UintType ExactUniformInt(const UintType n, const RandomBits& random) {
  static_assert(std::is_unsigned<UintType>::value,
                "UintType must be an unsigned int");
  static_assert(std::is_same<UintType, decltype(random())>::value,
                "random() should return UintType");
  if (n == 0) {
    // Consume a value anyway
    // TODO(irving): Assert n != 0, since this case makes no sense.
    return random() * n;
  } else if (0 == (n & (n - 1))) {
    // N is a power of two, so just mask off the lower bits.
    return random() & (n - 1);
  } else {
    // Reject all numbers that skew the distribution towards 0.

    // random's output is uniform in the half-open interval [0, 2^{bits}).
    // For any interval [m,n), the number of elements in it is n-m.

    const UintType range = ~static_cast<UintType>(0);
    const UintType rem = (range % n) + 1;
    UintType rnd;

    // rem = ((2^bits-1) \bmod n) + 1
    // 1 <= rem <= n

    // NB: rem == n is impossible, since n is not a power of 2 (from
    // earlier check).

    do {
      rnd = random();     // rnd uniform over [0, 2^{bits})
    } while (rnd < rem);  // reject [0, rem)
    // rnd is uniform over [rem, 2^{bits})
    //
    // The number of elements in the half-open interval is
    //
    //  2^{bits} - rem = 2^{bits} - ((2^{bits}-1) \bmod n) - 1
    //                 = 2^{bits}-1 - ((2^{bits}-1) \bmod n)
    //                 = n \cdot \lfloor (2^{bits}-1)/n \rfloor
    //
    // therefore n evenly divides the number of integers in the
    // interval.
    //
    // The function v \rightarrow v % n takes values from [bias,
    // 2^{bits}) to [0, n).  Each integer in the range interval [0, n)
    // will have exactly \lfloor (2^{bits}-1)/n \rfloor preimages from
    // the domain interval.
    //
    // Therefore, v % n is uniform over [0, n).  QED.

    return rnd % n;
  }
}

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_RANDOM_EXACT_UNIFORM_INT_H_
