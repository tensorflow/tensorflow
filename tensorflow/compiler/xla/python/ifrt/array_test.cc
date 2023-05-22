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

#include "tensorflow/compiler/xla/python/ifrt/array.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/ExtensibleRTTI.h"

namespace xla {
namespace ifrt {
namespace {

class MockArray : public llvm::RTTIExtends<MockArray, Array> {
 public:
  static tsl::RCReference<Array> Create() { return tsl::MakeRef<MockArray>(); }

  MOCK_METHOD(Client*, client, (), (const, override));

  MOCK_METHOD(DType, dtype, (), (const, override));
  MOCK_METHOD(const Shape&, shape, (), (const, override));
  MOCK_METHOD(const Sharding&, sharding, (), (const, override));
  MOCK_METHOD(std::shared_ptr<const Sharding>, shared_ptr_sharding, (),
              (const, override));

  MOCK_METHOD(StatusOr<std::vector<tsl::RCReference<Array>>>,
              DisassembleIntoSingleDeviceArrays, (ArrayCopySemantics semantics),
              (override));

  MOCK_METHOD(Future<Status>, CopyToHostBuffer,
              (void* data,
               std::optional<absl::Span<const int64_t>> byte_strides,
               ArrayCopySemantics semantics),
              (override));

  MOCK_METHOD(StatusOr<tsl::RCReference<Array>>, Reshard,
              (std::shared_ptr<const Sharding> new_sharding,
               ArrayCopySemantics semantics),
              (override));

  MOCK_METHOD(Future<Status>, GetReadyFuture, (), (const, override));

  MOCK_METHOD(Future<Status>, Delete, (), (override));

  MOCK_METHOD(bool, IsDeleted, (), (const, override));

  MOCK_METHOD(std::string, DebugString, (), (const, override));

  static char ID;  // NOLINT
};

char MockArray::ID ABSL_ATTRIBUTE_UNUSED = 0;

TEST(ArrayTest, MakeArrayPointerListTest) {
  const int kNumArrays = 3;
  std::vector<tsl::RCReference<Array>> arrays;
  arrays.reserve(kNumArrays);
  for (int i = 0; i < kNumArrays; ++i) {
    arrays.push_back(MockArray::Create());
  }

  std::vector<Array*> array_pointer_list = MakeArrayPointerList(arrays);
  ASSERT_THAT(array_pointer_list, testing::SizeIs(kNumArrays));
  for (int i = 0; i < kNumArrays; ++i) {
    EXPECT_THAT(array_pointer_list[i], arrays[i].get());
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
