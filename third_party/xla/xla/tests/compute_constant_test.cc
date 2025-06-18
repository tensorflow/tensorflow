/* Copyright 2017 The OpenXLA Authors.

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
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// An enumerator for the client types that we want to iterate over in
// the various tests.
enum class ClientType { kLocal, kCompileOnly };
ClientType client_types[] = {ClientType::kLocal, ClientType::kCompileOnly};

class ComputeConstantTest : public ::testing::Test {
 public:
  explicit ComputeConstantTest(se::Platform* platform = nullptr)
      : platform_(platform) {}

  std::string TestName() const {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  Client* ClientOrDie(se::Platform* platform, ClientType client_type) {
    if (client_type == ClientType::kLocal) {
      absl::StatusOr<Client*> result =
          ClientLibrary::GetOrCreateLocalClient(platform);
      TF_CHECK_OK(result.status())
          << "could not create LocalClient for testing";
      return result.value();
    } else if (client_type == ClientType::kCompileOnly) {
      absl::StatusOr<Client*> result =
          ClientLibrary::GetOrCreateCompileOnlyClient(platform);
      TF_CHECK_OK(result.status())
          << "could not create CompileOnlyClient for testing";
      return result.value();
    }
    LOG(FATAL) << "invalid client_type value";
  }

  absl::StatusOr<Literal> ComputeConstantLiteral(
      Client* client, const XlaOp operand, XlaBuilder* builder,
      Layout* output_layout = nullptr) {
    TF_ASSIGN_OR_RETURN(auto subgraph, builder->BuildConstantSubGraph(operand));
    TF_ASSIGN_OR_RETURN(auto computed,
                        client->ComputeConstant(subgraph, output_layout));
    return std::move(computed);
  }

  template <class Scalar>
  absl::StatusOr<Scalar> ComputeConstantScalar(Client* client,
                                               const XlaOp operand,
                                               XlaBuilder* builder) {
    TF_ASSIGN_OR_RETURN(auto literal, ComputeConstantLiteral(client, operand,
                                                             builder, nullptr));
    return literal.Get<Scalar>({});
  }

  bool IsConstant(const XlaOp operand, XlaBuilder* builder) {
    absl::StatusOr<bool> result = builder->IsConstant(operand);
    EXPECT_TRUE(result.ok()) << result.status();
    return result.ok() ? result.value() : false;
  }

  se::Platform* platform_;
};

TEST_F(ComputeConstantTest, ScalarInt32Literal) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation = ConstantR0<int32_t>(&b, 42);
    EXPECT_TRUE(IsConstant(computation, &b));

    auto value = ComputeConstantScalar<int32_t>(client, computation, &b);
    ASSERT_TRUE(value.ok()) << value.status();
    EXPECT_EQ(value.value(), 42);
  }
}

TEST_F(ComputeConstantTest, ScalarFloatAdd) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation =
        Add(ConstantR0<float>(&b, 42.5f), ConstantR0<float>(&b, 1.5f));
    EXPECT_TRUE(IsConstant(computation, &b));

    auto value = ComputeConstantScalar<float>(client, computation, &b);
    ASSERT_TRUE(value.ok()) << value.status();
    EXPECT_EQ(value.value(), 44.0f);
  }
}

TEST_F(ComputeConstantTest, ScalarRng) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation =
        RngUniform(ConstantR0<float>(&b, 1.1f), ConstantR0<float>(&b, 2.1f),
                   ShapeUtil::MakeShape(F32, {}));
    EXPECT_FALSE(IsConstant(computation, &b));

    auto value = ComputeConstantScalar<float>(client, computation, &b);
    ASSERT_FALSE(value.ok())
        << "computing a RNG value should not be considered a constant";
  }
}

TEST_F(ComputeConstantTest, DirectParamMissing) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "param");
    EXPECT_FALSE(IsConstant(computation, &b));

    auto value = ComputeConstantScalar<float>(client, computation, &b);
    EXPECT_TRUE(
        absl::StrContains(value.status().ToString(), "depends on a parameter"))
        << value.status();
  }
}

TEST_F(ComputeConstantTest, GetDimensionSize) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto add =
        Add(ConstantR1<float>(&b, {1.0f}), ConstantR1<float>(&b, {1.0f}));
    auto get_dimension_size = GetDimensionSize(add, 0);
    EXPECT_TRUE(IsConstant(get_dimension_size, &b));

    TF_ASSERT_OK_AND_ASSIGN(auto value, ComputeConstantScalar<int32_t>(
                                            client, get_dimension_size, &b));
    EXPECT_EQ(value, 1);
  }
}

TEST_F(ComputeConstantTest, MultipleGetDimensionSize) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto add =
        Add(ConstantR2<float>(&b, {{1.0f}}), ConstantR2<float>(&b, {{1.0f}}));
    auto get_dimension_size = GetDimensionSize(add, 0);
    auto get_dimension_size_2 = GetDimensionSize(add, 0);
    auto add_2 = Add(get_dimension_size, get_dimension_size_2);
    EXPECT_TRUE(IsConstant(add_2, &b));

    TF_ASSERT_OK_AND_ASSIGN(auto value,
                            ComputeConstantScalar<int32_t>(client, add_2, &b));
    EXPECT_EQ(value, 2);
  }
}

// Test computation of an expression interspersed with param nodes but
// the expression does not depend on the param nodes.
TEST_F(ComputeConstantTest, UnrelatedParam) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());

    auto param_a = Parameter(&b, 10, ShapeUtil::MakeShape(F32, {}), "param0");
    auto constant_4 =
        Add(ConstantR0<float>(&b, 2.5f), ConstantR0<float>(&b, 1.5f));
    auto not_constant_a = Add(constant_4, param_a);

    auto param_b = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "param1");
    auto constant_9 =
        Mul(ConstantR0<float>(&b, 2.0f), ConstantR0<float>(&b, 4.5f));
    auto not_constant_b = Add(param_b, constant_9);

    auto constant_13 = Add(constant_4, constant_9);
    Add(not_constant_b, Add(constant_13, not_constant_a));

    EXPECT_TRUE(IsConstant(constant_13, &b));

    TF_ASSERT_OK_AND_ASSIGN(
        auto value, ComputeConstantScalar<float>(client, constant_13, &b));
    EXPECT_EQ(value, 13.0f);
  }
}

TEST_F(ComputeConstantTest, NonScalarAdd) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());

    auto computation =
        Add(ConstantR1<int32_t>(&b, {1, 2}), ConstantR1<int32_t>(&b, {3, 4}));
    EXPECT_TRUE(IsConstant(computation, &b));

    TF_ASSERT_OK_AND_ASSIGN(auto computed,
                            ComputeConstantLiteral(client, computation, &b));
    Literal expected_literal = LiteralUtil::CreateR1<int32_t>({4, 6});
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, computed));
  }
}

TEST_F(ComputeConstantTest, IntegerDivide) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation =
        Div(ConstantR0<int32_t>(&b, 15), ConstantR0<int32_t>(&b, 3));
    EXPECT_TRUE(IsConstant(computation, &b));

    TF_ASSERT_OK_AND_ASSIGN(auto computed,
                            ComputeConstantLiteral(client, computation, &b));
    Literal expected_literal = LiteralUtil::CreateR0<int32_t>(5);
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, computed));
  }
}

TEST_F(ComputeConstantTest, Layout) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());

    std::vector<std::vector<int64_t>> layouts = {{0, 1}, {1, 0}};
    for (const std::vector<int64_t>& layout : layouts) {
      auto layout_proto = LayoutUtil::MakeLayout(layout);
      TF_ASSERT_OK_AND_ASSIGN(
          auto computed, ComputeConstantLiteral(
                             client,
                             Add(ConstantR2<int32_t>(&b, {{1, 2}, {3, 4}}),
                                 ConstantR2<int32_t>(&b, {{10, 20}, {30, 40}})),
                             &b, &layout_proto));

      Literal expected_literal = LiteralUtil::CreateR2WithLayout<int32_t>(
          {{11, 22}, {33, 44}}, LayoutUtil::MakeLayout(layout));
      ASSERT_THAT(
          computed.shape().ToProto(),
          tsl::proto_testing::EqualsProto(expected_literal.shape().ToProto()));
      EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, computed));
    }
  }
}

}  // namespace
}  // namespace xla
