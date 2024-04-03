/* Copyright 2022 The OpenXLA Authors.

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

#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/tuple.h"
#include "tsl/concurrency/ref_count.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

absl::StatusOr<tsl::RCReference<Array>> MakeArray(Client* client) {
  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device, MemoryKind());

  return client->MakeArrayFromHostBuffer(
      data.data(), dtype, shape,
      /*byte_strides=*/std::nullopt, sharding,
      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
      /*on_done_with_host_buffer=*/{});
}

TEST(TupleImplTest, NullaryTuple) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(auto t, client->MakeTuple({}));

  EXPECT_EQ(t->Arity(), 0);
  std::vector<tsl::RCReference<Value>> elements;
  TF_EXPECT_OK(t->Unpack(absl::MakeSpan(elements)));
  EXPECT_EQ(elements.size(), 0);

  TF_EXPECT_OK(t->GetReadyFuture().Await());

  EXPECT_THAT(t->DebugString(), ::testing::MatchesRegex(".*Tuple\\(\\)"));
  EXPECT_FALSE(t->IsDeleted());

  TF_EXPECT_OK(t->Delete().Await());
  EXPECT_TRUE(t->IsDeleted());
}

TEST(TupleImplTest, TupleOfArrays) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(auto a1, MakeArray(client.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto a2, MakeArray(client.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto a3, MakeArray(client.get()));
  std::vector<tsl::RCReference<Value>> elements_in{a1, a2, a3};
  TF_ASSERT_OK_AND_ASSIGN(auto t,
                          client->MakeTuple(absl::MakeSpan(elements_in)));
  EXPECT_EQ(t->Arity(), 3);
  std::vector<tsl::RCReference<Value>> elements(3);
  TF_EXPECT_OK(t->Unpack(absl::MakeSpan(elements)));
  EXPECT_THAT(elements, ::testing::ElementsAre(a1, a2, a3));

  EXPECT_THAT(t->DebugString(),
              ::testing::MatchesRegex(".*Tuple\\(.*,.*,.*\\)"));

  TF_EXPECT_OK(t->Delete().Await());
  EXPECT_TRUE(t->IsDeleted());
  EXPECT_TRUE(a1->IsDeleted());
  EXPECT_TRUE(a2->IsDeleted());
  EXPECT_TRUE(a3->IsDeleted());
}

TEST(TupleImplTest, DeleteOfElementDeletesTuple) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(auto a1, MakeArray(client.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto a2, MakeArray(client.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto a3, MakeArray(client.get()));
  std::vector<tsl::RCReference<Value>> elements_in{a1, a2, a3};
  TF_ASSERT_OK_AND_ASSIGN(auto t,
                          client->MakeTuple(absl::MakeSpan(elements_in)));

  TF_EXPECT_OK(a1->Delete().Await());
  EXPECT_TRUE(t->IsDeleted());
  EXPECT_FALSE(a2->IsDeleted());
  EXPECT_FALSE(a3->IsDeleted());
}

TEST(TupleImplTest, NestedTuples) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(auto a1, MakeArray(client.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto a2, MakeArray(client.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto a3, MakeArray(client.get()));
  std::vector<tsl::RCReference<Value>> e1{a1, a2};
  TF_ASSERT_OK_AND_ASSIGN(auto t1, client->MakeTuple(absl::MakeSpan(e1)));
  EXPECT_EQ(t1->Arity(), 2);
  std::vector<tsl::RCReference<Value>> e2{};
  TF_ASSERT_OK_AND_ASSIGN(auto t2, client->MakeTuple(absl::MakeSpan(e2)));
  EXPECT_EQ(t2->Arity(), 0);

  std::vector<tsl::RCReference<Value>> e3{t1, t2, a3};
  TF_ASSERT_OK_AND_ASSIGN(auto t3, client->MakeTuple(absl::MakeSpan(e3)));
  EXPECT_EQ(t3->Arity(), 3);

  std::vector<tsl::RCReference<Value>> elements(3);
  TF_EXPECT_OK(t3->Unpack(absl::MakeSpan(elements)));
  EXPECT_THAT(elements, ::testing::ElementsAre(t1, t2, a3));

  t3.reset();

  elements.resize(t1->Arity());
  TF_EXPECT_OK(t1->Unpack(absl::MakeSpan(elements)));
  EXPECT_THAT(elements, ::testing::ElementsAre(a1, a2));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
