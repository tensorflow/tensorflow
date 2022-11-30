/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/framework/lazy.h"

#include "tensorflow/tsl/platform/test.h"

namespace tsl {
namespace {

class TestCls {
 public:
  TestCls() : TestCls(456, nullptr, nullptr) {}
  TestCls(int val, bool* destructed, bool* constructed)
      : val_(val), destructed_(destructed) {
    if (constructed) {
      *constructed = true;
    }
  }

  int val() const { return val_; }

  ~TestCls() {
    if (destructed_) {
      *destructed_ = true;
    }
  }

 private:
  int val_;
  bool* destructed_;
};

TEST(Lazy, NotInitialized) {
  bool destructed = false, constructed = false;
  {
    Lazy<TestCls> val(456, &destructed, &constructed);
    EXPECT_FALSE(val.IsInitialized());
    EXPECT_FALSE(constructed);
  }
  EXPECT_FALSE(destructed);
}

TEST(Lazy, InitializedDefaultCtr) {
  Lazy<TestCls> val;
  EXPECT_FALSE(val.IsInitialized());
  EXPECT_EQ(val.Get().val(), 456);
}

TEST(Lazy, ConstructAndDestruct) {
  bool destructed = false, constructed = false;
  {
    Lazy<TestCls> val(789, &destructed, &constructed);
    EXPECT_FALSE(val.IsInitialized());
    EXPECT_FALSE(constructed);
    EXPECT_EQ(val.Get().val(), 789);
    EXPECT_TRUE(constructed);
    EXPECT_FALSE(destructed);
  }
  EXPECT_TRUE(destructed);
}

}  // namespace
}  // namespace tsl
