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
#include <cstdint>

#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/cc/experimental/libtf/value_iostream.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {
namespace impl {

struct Visitor {
  const char* operator()(Int64 i) { return "int64"; }
  const char* operator()(Float32 f) { return "float32"; }
  template <class T>
  const char* operator()(const T& i) {
    return "else";
  }
};

TEST(VisitTest, Test1) {
  TaggedValue a(Int64(1)), b(Float32(1.1f));
  TaggedValue c = TaggedValue::None();

  ASSERT_EQ(a.visit<const char*>(Visitor()), "int64");
  ASSERT_EQ(b.visit<const char*>(Visitor()), "float32");
  ASSERT_EQ(c.visit<const char*>(Visitor()), "else");
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
