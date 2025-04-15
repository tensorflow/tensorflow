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

#include "xla/tsl/distributed_runtime/rpc/grpc_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "xla/tsl/distributed_runtime/rpc/test_request.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace tsl {
namespace {

using tsl::proto_testing::EqualsProto;
using tsl::test::TestRequest;

string ToString(const grpc::ByteBuffer& buf) {
  std::vector<grpc::Slice> slices;
  CHECK(buf.Dump(&slices).ok());
  string result;
  for (const grpc::Slice& s : slices) {
    result.append(reinterpret_cast<const char*>(s.begin()), s.size());
  }
  return result;
}

// Return a ByteBuffer that contains str split up into num_slices slices.
grpc::ByteBuffer MakeBuffer(const string& str, int num_slices) {
  // Convert to a ByteBuffer.
  std::vector<::grpc::Slice> slices;
  const size_t per_slice = (str.size() + num_slices - 1) / num_slices;
  for (size_t pos = 0; pos < str.size();) {
    const size_t n = std::min(str.size() - pos, per_slice);
    slices.emplace_back(&str[pos], n);
    pos += n;
  }
  if (slices.empty()) {
    slices.emplace_back();
  }
  return ::grpc::ByteBuffer(&slices[0], slices.size());
}

// Make a proto with approximately the specified length.
TestRequest MakeProto(int size) {
  int approx_size = 0;
  TestRequest proto;
  int index = 0;
  while (approx_size < size) {
    int item_size = std::min(size - approx_size, 1024);
    proto.add_data(string(item_size, 'a' + static_cast<char>(index % 26)));
    approx_size += item_size + 3;  // +3 for encoding overhead.
    index++;
  }
  return proto;
}

TEST(PayloadSerialization, PayloadsAreTransmitted) {
  absl::Status status = errors::InvalidArgument("invalid arg message");
  status.SetPayload("a", absl::Cord("\\xFF\\x02\\x03"));
  absl::Status status_recovered = FromGrpcStatus(ToGrpcStatus(status));

  ASSERT_TRUE(status_recovered.GetPayload("a").has_value());
  EXPECT_EQ(status_recovered.GetPayload("a").value(), "\\xFF\\x02\\x03");
}

TEST(PayloadSerialization, PayloadsCorrupted) {
  ::grpc::Status status(
      ::grpc::StatusCode::INVALID_ARGUMENT, "invalid arg message",
      "string that can not be serialized to the GrpcPayloadContainer proto");

  absl::Status converted = FromGrpcStatus(status);
  EXPECT_TRUE(converted.GetPayload(kGrpcPayloadsLost).has_value());
}

TEST(GrpcProto, Unparse) {
  TestRequest proto;
  proto.add_data("hello");
  proto.add_data("world");
  grpc::ByteBuffer buf;
  ASSERT_TRUE(GrpcMaybeUnparseProto(proto, &buf).ok());
  TestRequest parsed;
  ASSERT_TRUE(parsed.ParseFromString(ToString(buf)));
  ASSERT_THAT(parsed, EqualsProto(proto));
}

TEST(GrpcProto, UnparseToString) {
  TestRequest proto;
  proto.add_data("hello");
  proto.add_data("world");
  string str;
  CHECK(proto.SerializeToString(&str));
  grpc::ByteBuffer buf;
  ASSERT_TRUE(GrpcMaybeUnparseProto(str, &buf).ok());
  TestRequest parsed;
  ASSERT_TRUE(parsed.ParseFromString(ToString(buf)));
  ASSERT_THAT(parsed, EqualsProto(proto));
}

TEST(GrpcProto, Parse) {
  // Test with serialization broken up into a bunch of slices.
  struct Case {
    int length;
    int slices;
  };
  for (Case c : std::vector<Case>{
           {0, 1},
           {20, 1},
           {100, 1},
           {1 << 20, 1},
           {100, 5},
           {10000, 50},
       }) {
    TestRequest proto = MakeProto(c.length);
    ::grpc::ByteBuffer src = MakeBuffer(proto.SerializeAsString(), c.slices);
    TestRequest parsed;
    ASSERT_TRUE(GrpcMaybeParseProto(&src, &parsed))
        << c.length << " " << c.slices;
    ASSERT_THAT(parsed, EqualsProto(proto));
  }
}

TEST(GrpcProto, ParseFromString) {
  // Test with serialization broken up into a bunch of slices.
  struct Case {
    int length;
    int slices;
  };
  for (Case c : std::vector<Case>{
           {0, 1},
           {20, 1},
           {100, 1},
           {1 << 20, 1},
           {100, 5},
           {10000, 50},
       }) {
    TestRequest proto = MakeProto(c.length);
    ::grpc::ByteBuffer src = MakeBuffer(proto.SerializeAsString(), c.slices);
    string parsed_str;
    TestRequest parsed;
    ASSERT_TRUE(GrpcMaybeParseProto(&src, &parsed_str))
        << c.length << " " << c.slices;
    ASSERT_TRUE(parsed.ParseFromString(parsed_str));
    ASSERT_THAT(parsed, EqualsProto(proto));
  }
}

static void BM_UnparseGrpc(::testing::benchmark::State& state) {
  const int size = state.range(0);

  auto proto = MakeProto(size);
  for (auto s : state) {
    grpc::ByteBuffer buf;
    CHECK(GrpcMaybeUnparseProto(proto, &buf).ok());
  }
}
BENCHMARK(BM_UnparseGrpc)->Arg(1)->Arg(1 << 10)->Arg(1 << 20);

static void BM_UnparseString(::testing::benchmark::State& state) {
  const int size = state.range(0);

  auto proto = MakeProto(size);

  for (auto s : state) {
    string buf;
    proto.SerializeToString(&buf);
  }
}
BENCHMARK(BM_UnparseString)->Arg(1)->Arg(1 << 10)->Arg(1 << 20);

static void BM_ParseGrpc(::testing::benchmark::State& state) {
  const int size = state.range(0);
  const int num_slices = state.range(1);

  TestRequest proto = MakeProto(size);
  auto buf = MakeBuffer(proto.SerializeAsString(), num_slices);

  for (auto s : state) {
    CHECK(GrpcMaybeParseProto(&buf, &proto));
  }
}
BENCHMARK(BM_ParseGrpc)
    ->ArgPair(1, 1)
    ->ArgPair(1 << 10, 1)
    ->ArgPair(1 << 10, 4)
    ->ArgPair(1 << 20, 1)
    ->ArgPair(1 << 20, 4);

static void BM_ParseString(::testing::benchmark::State& state) {
  const int size = state.range(0);

  TestRequest proto = MakeProto(size);
  string serial = proto.SerializeAsString();

  for (auto s : state) {
    CHECK(proto.ParseFromString(serial));
  }
}
BENCHMARK(BM_ParseString)->Arg(1)->Arg(1 << 10)->Arg(1 << 20);
}  // namespace
}  // namespace tsl
