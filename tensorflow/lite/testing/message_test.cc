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
#include "tensorflow/lite/testing/message.h"

#include <map>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace testing {
namespace {

// A hierarchical, key-value store.
class TestMessage : public Message {
 public:
  TestMessage() {}
  explicit TestMessage(const std::string& text_to_parse) {
    std::stringstream ss(text_to_parse);
    finished_ = Message::Read(&ss, this);
  }
  void SetField(const std::string& name, const std::string& value) override {
    fields_[name] = value;
  }
  Message* AddChild(const std::string& name) override {
    TestMessage* m = new TestMessage;
    m->name_ = name;
    return Store(m);
  }
  void Finish() override { finished_ = true; }

  int NumChildren() const { return Children().size(); }

  const TestMessage* GetChild(int i) const {
    return dynamic_cast<TestMessage*>(Children()[i].get());
  }

  int NumFields() const { return fields_.size(); }
  const std::string& GetField(const std::string& key) const {
    return fields_.at(key);
  }

  const std::string& name() const { return name_; }
  bool finished() const { return finished_; }

 protected:
  std::string name_;
  std::map<std::string, std::string> fields_;
  bool finished_ = false;
};

TEST(MessageTest, Simple) {
  TestMessage message("x{a:1 b:2} y{} z{c:3} d:4");
  ASSERT_TRUE(message.finished());

  ASSERT_EQ(message.NumFields(), 1);
  EXPECT_EQ(message.GetField("d"), "4");

  ASSERT_EQ(message.NumChildren(), 3);

  auto* x = message.GetChild(0);
  EXPECT_EQ(x->name(), "x");
  ASSERT_EQ(x->NumFields(), 2);
  EXPECT_EQ(x->GetField("a"), "1");
  EXPECT_EQ(x->GetField("b"), "2");

  auto* y = message.GetChild(1);
  EXPECT_EQ(y->name(), "y");
  ASSERT_EQ(y->NumFields(), 0);

  auto* z = message.GetChild(2);
  EXPECT_EQ(z->name(), "z");
  ASSERT_EQ(z->NumFields(), 1);
  EXPECT_EQ(z->GetField("c"), "3");
}

TEST(MessageTest, Unnamed) {
  TestMessage message("x{c:3} {} y{d:4}");
  ASSERT_FALSE(message.finished());
  EXPECT_EQ(message.NumChildren(), 1);
}

TEST(MessageTest, TooManyBraces) {
  TestMessage message("x{c:3} } y{d:4}");
  ASSERT_FALSE(message.finished());
  EXPECT_EQ(message.NumChildren(), 1);
}

TEST(MessageTest, LeftoverToken) {
  TestMessage message("x{c:3} z{test} y{d:4}");
  ASSERT_FALSE(message.finished());
  EXPECT_EQ(message.NumChildren(), 2);
}

TEST(MessageTest, MissingKey) {
  TestMessage message("x{c:3} z{:test} y{d:4}");
  ASSERT_FALSE(message.finished());
  EXPECT_EQ(message.NumChildren(), 2);
}

TEST(MessageTest, MissingValue) {
  TestMessage message("x{c:3} z{test:} y{d:4}");
  ASSERT_FALSE(message.finished());
  EXPECT_EQ(message.NumChildren(), 2);
}

}  // namespace
}  // namespace testing
}  // namespace tflite
