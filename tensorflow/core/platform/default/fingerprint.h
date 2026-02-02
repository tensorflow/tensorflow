/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_FINGERPRINT_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_FINGERPRINT_H_

#include "farmhash-34c13ddfab0e35422f4c3979f360635a8c050260/src/farmhash.h"

namespace tensorflow {

inline uint64 Fingerprint64(const string& s) {
  return ::util::Fingerprint64(s);
}

inline Fprint128 Fingerprint128(const string& s) {
  const auto fingerprint = ::util::Fingerprint128(s);
  return {::util::Uint128Low64(fingerprint),
          ::util::Uint128High64(fingerprint)};
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_FINGERPRINT_H_
