/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RNG_ALG_H_
#define TENSORFLOW_CORE_FRAMEWORK_RNG_ALG_H_

namespace tensorflow {

enum Algorithm {
  // The Philox algorithm, as described in paper
  // ['Parallel Random Numbers: As Easy as 1, 2, 3']
  // (https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
  RNG_ALG_PHILOX = 1,
  // The ThreeFry algorithm, as described in paper
  // ['Parallel Random Numbers: As Easy as 1, 2, 3']
  // (https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
  RNG_ALG_THREEFRY = 2,
  // An algorithm suitable for TPU. Only available on XLA devices.
  RNG_ALG_XLA_DEFAULT = 3
};

static constexpr int RNG_KEY_SIZE = 1;
static constexpr int RNG_MAX_COUNTER_SIZE = 2;
inline int GetCounterSize(Algorithm alg) {
  if (alg == RNG_ALG_PHILOX) {
    return 2;
  } else if (alg == RNG_ALG_THREEFRY) {
    return 1;
  } else if (alg == RNG_ALG_XLA_DEFAULT) {
    return 1;
  }
  return 2;
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_RNG_ALG_H_
