/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_TSL_UTIL_DETERMINISM_TEST_UTIL_H_
#define XLA_TSL_UTIL_DETERMINISM_TEST_UTIL_H_

#include "xla/tsl/util/determinism.h"

namespace tsl {
namespace test {

// Enables determinism for a single test method.
class DeterministicOpsScope {
 public:
  DeterministicOpsScope() : was_enabled_(OpDeterminismRequired()) {
    EnableOpDeterminism(true);
  }
  ~DeterministicOpsScope() { EnableOpDeterminism(was_enabled_); }

 private:
  const bool was_enabled_;
};

}  // namespace test
}  // namespace tsl

#endif  // XLA_TSL_UTIL_DETERMINISM_TEST_UTIL_H_
