/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/serdes.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/python/ifrt/serdes.pb.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

struct TestNumber : llvm::RTTIExtends<TestNumber, Serializable> {
  int number;

  explicit TestNumber(int number) : number(number) {}

  static char ID;  // NOLINT
};

char TestNumber::ID = 0;  // NOLINT

class TestNumberSerDes : public llvm::RTTIExtends<TestNumberSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::TestNumber";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable) override {
    const TestNumber& obj = llvm::cast<TestNumber>(serializable);
    return absl::StrCat(obj.number);
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized) override {
    int number;
    if (!absl::SimpleAtoi(serialized, &number)) {
      return absl::DataLossError("Unable to parse serialized TestNumber");
    }
    return std::make_unique<TestNumber>(number);
  }

  static char ID;  // NOLINT
};

char TestNumberSerDes::ID = 0;  // NOLINT

class TestNumberTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    RegisterSerDes<TestNumber>(std::make_unique<TestNumberSerDes>());
  }
};

TEST_F(TestNumberTest, RoundTrip) {
  auto obj = std::make_unique<TestNumber>(1234);
  TF_ASSERT_OK_AND_ASSIGN(Serialized serialized, Serialize(*obj));
  TF_ASSERT_OK_AND_ASSIGN(auto deserialized, Deserialize(serialized));
  EXPECT_EQ(obj->number, llvm::cast<TestNumber>(*deserialized).number);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
