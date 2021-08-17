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
#include "tensorflow/cc/experimental/libtf/value.h"

#include <cstdint>

#include "tensorflow/cc/experimental/libtf/value_iostream.h"
#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {
namespace impl {

TEST(ValueTest, TestBasic) {
  TaggedValue valuef(3.f);
  TaggedValue valuei(int64_t(3));
  TaggedValue list = TaggedValue::List();
  TaggedValue tuple = TaggedValue::Tuple();
  tuple.tuple().push_back(TaggedValue(int64_t(310)));
  list.list().push_back(valuei);
  list.list().push_back(valuef);
  list.list().push_back(tuple);
  std::stringstream stream;
  stream << list;
  ASSERT_EQ(stream.str(), "[3, 3, (310, ), ]");
}

TEST(ValueTest, TestString) {
  TaggedValue value1a("string1");
  std::string s = "string";
  s += "1";
  TaggedValue value1b(s.c_str());
  // Verify that interned the pointers are the same.
  ASSERT_EQ(value1b.s(), value1a.s());
  TaggedValue value2("string2");
  ASSERT_NE(value1a.s(), value2.s());
  ASSERT_STREQ(value1a.s(), "string1");
  ASSERT_STREQ(value2.s(), "string2");
}

TEST(Test1, TestDict) {
  TaggedValue s1("test1");
  TaggedValue s2("test2");
  TaggedValue d = TaggedValue::Dict();
  d.dict()[s2] = TaggedValue(6.f);
  std::stringstream stream;
  stream << d;
  ASSERT_EQ(stream.str(), "{test2: 6, }");
}

namespace {
TaggedValue add(TaggedValue args, TaggedValue kwargs) {
  if (args.type() == TaggedValue::TUPLE) {
    return TaggedValue(args.tuple()[0].f32() + args.tuple()[1].f32());
  }
  return TaggedValue::None();
}
}  // namespace
TEST(Test1, TestFunctionCall) {
  TaggedValue f32 = TaggedValue(add);
  TaggedValue args = TaggedValue::Tuple();
  args.tuple().emplace_back(TaggedValue(1.f));
  args.tuple().emplace_back(TaggedValue(2.f));
  TaggedValue c = f32.func()(args, TaggedValue::None()).ValueOrDie();
  ASSERT_EQ(c, TaggedValue(3.f));
}

namespace {
int alloc_count = 0;
class Cool {
 public:
  Cool() { alloc_count++; }
  ~Cool() { alloc_count--; }
};
}  // namespace

TEST(Test1, TestCapsule) {
  TaggedValue test_moved, test_copy;
  ASSERT_EQ(alloc_count, 0);
  void* ptr_value = new Cool();
  {
    TaggedValue capsule =
        TaggedValue::Capsule(static_cast<void*>(ptr_value),
                             [](void* x) { delete static_cast<Cool*>(x); });
    ASSERT_EQ(alloc_count, 1);
    ASSERT_EQ(capsule.capsule(), ptr_value);
    test_moved = std::move(capsule);
    ASSERT_EQ(capsule.type(), TaggedValue::NONE);  // NOLINT
    test_copy = test_moved;
    ASSERT_EQ(test_moved.capsule(), ptr_value);
    ASSERT_EQ(test_copy.capsule(), ptr_value);
  }
  ASSERT_EQ(alloc_count, 1);
  test_moved = TaggedValue::None();
  ASSERT_EQ(alloc_count, 1);
  test_copy = TaggedValue(3.f);
  ASSERT_EQ(alloc_count, 0);
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
