/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/core/no_destructor.h"

#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

struct CheckOnDestroy {
  ~CheckOnDestroy() { CHECK(false); }
};

TEST(NoDestructorTest, SkipsDestructors) {
  NoDestructor<CheckOnDestroy> destructor_should_not_run;
}

struct CopyOnly {
  CopyOnly() = default;

  CopyOnly(const CopyOnly&) = default;
  CopyOnly& operator=(const CopyOnly&) = default;

  CopyOnly(CopyOnly&&) = delete;
  CopyOnly& operator=(CopyOnly&&) = delete;
};

struct MoveOnly {
  MoveOnly() = default;

  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;

  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;
};

struct ForwardingTestStruct {
  ForwardingTestStruct(const CopyOnly&, MoveOnly&&) {}
};

TEST(NoDestructorTest, ForwardsArguments) {
  CopyOnly copy_only;
  MoveOnly move_only;

  static NoDestructor<ForwardingTestStruct> test_forwarding(
      copy_only, std::move(move_only));
}

TEST(NoDestructorTest, Accessors) {
  static NoDestructor<std::string> awesome("awesome");

  EXPECT_EQ("awesome", *awesome);
  EXPECT_EQ(0, awesome->compare("awesome"));
  EXPECT_EQ(0, awesome.get()->compare("awesome"));
}

}  // namespace tensorflow
