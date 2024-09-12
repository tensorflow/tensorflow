/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/serdes.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::tsl::testing::StatusIs;

struct TestNumberDeserializeOptions;

struct TestNumber : llvm::RTTIExtends<TestNumber, Serializable> {
  using DeserializeOptions = TestNumberDeserializeOptions;

  int number;

  explicit TestNumber(int number) : number(number) {}

  static char ID;  // NOLINT
};

[[maybe_unused]] char TestNumber::ID = 0;  // NOLINT

struct TestNumberDeserializeOptions
    : llvm::RTTIExtends<TestNumberDeserializeOptions, DeserializeOptions> {
  absl::Status injected_failure;

  static char ID;  // NOLINT
};

[[maybe_unused]] char TestNumberDeserializeOptions::ID = 0;  // NOLINT

class TestNumberSerDes : public llvm::RTTIExtends<TestNumberSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::TestNumber";
  }

  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    const TestNumber& obj = llvm::cast<TestNumber>(serializable);
    return absl::StrCat(obj.number);
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    if (options != nullptr) {
      auto* deserialize_options =
          llvm::cast<TestNumberDeserializeOptions>(options.get());
      TF_RETURN_IF_ERROR(deserialize_options->injected_failure);
    }

    int number;
    if (!absl::SimpleAtoi(serialized, &number)) {
      return absl::DataLossError("Unable to parse serialized TestNumber");
    }
    return std::make_unique<TestNumber>(number);
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char TestNumberSerDes::ID = 0;  // NOLINT

class TestNumberTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    RegisterSerDes<TestNumber>(std::make_unique<TestNumberSerDes>());
  }
};

TEST_F(TestNumberTest, RoundTrip) {
  auto obj = std::make_unique<TestNumber>(1234);
  TF_ASSERT_OK_AND_ASSIGN(Serialized serialized, Serialize(*obj));
  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized,
      Deserialize<TestNumber>(serialized, /*options=*/nullptr));
  EXPECT_EQ(obj->number, deserialized->number);
}

TEST_F(TestNumberTest, WithOptions) {
  auto obj = std::make_unique<TestNumber>(1234);
  TF_ASSERT_OK_AND_ASSIGN(Serialized serialized, Serialize(*obj));

  auto options = std::make_unique<TestNumberDeserializeOptions>();
  options->injected_failure = absl::InternalError("injected failure");
  EXPECT_THAT(Deserialize<TestNumber>(serialized, std::move(options)),
              StatusIs(absl::StatusCode::kInternal, "injected failure"));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
