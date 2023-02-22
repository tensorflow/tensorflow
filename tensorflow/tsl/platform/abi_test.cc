/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/platform/abi.h"

#include <typeinfo>

#include "tensorflow/tsl/platform/test.h"

namespace tsl {

struct MyRandomPODType {};

TEST(AbiTest, AbiDemangleTest) {
  EXPECT_EQ(port::MaybeAbiDemangle(typeid(int).name()), "int");

#ifdef PLATFORM_WINDOWS
  const char pod_type_name[] = "struct tsl::MyRandomPODType";
#else
  const char pod_type_name[] = "tsl::MyRandomPODType";
#endif
  EXPECT_EQ(port::MaybeAbiDemangle(typeid(MyRandomPODType).name()),
            pod_type_name);

  EXPECT_EQ(
      port::MaybeAbiDemangle("help! i'm caught in a C++ mangle factoryasdf"),
      "help! i'm caught in a C++ mangle factoryasdf");
}

}  // namespace tsl
