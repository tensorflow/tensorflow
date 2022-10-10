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

#include "tensorflow/tsl/platform/net.h"

#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {
namespace internal {

TEST(Net, PickUnusedPortOrDie) {
  int port0 = PickUnusedPortOrDie();
  int port1 = PickUnusedPortOrDie();
  CHECK_GE(port0, 0);
  CHECK_LT(port0, 65536);
  CHECK_GE(port1, 0);
  CHECK_LT(port1, 65536);
  CHECK_NE(port0, port1);
}

}  // namespace internal
}  // namespace tsl
