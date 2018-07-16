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

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/types.h"

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

  string TestName() const {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  Client* ClientOrDie(se::Platform* platform, ClientType client_type) {
    if (client_type == ClientType::kLocal) {
      StatusOr<Client*> result =
          ClientLibrary::GetOrCreateLocalClient(platform);
      TF_CHECK_OK(result.status())
          << "could not create LocalClient for testing";
      return result.ValueOrDie();
    } else if (client_type == ClientType::kCompileOnly) {
      StatusOr<Client*> result =
          ClientLibrary::GetOrCreateCompileOnlyClient(platform);
      TF_CHECK_OK(result.status())
          << "could not create CompileOnlyClient for testing";
      return result.ValueOrDie();
    }
    LOG(FATAL) << "invalid client_type value";
  }

  StatusOr<std::unique_ptr<Literal>> ComputeConstantLiteral(
      Client* client, const XlaOp& operand, XlaBuilder* builder,
      Layout* output_layout = nullptr) {
    TF_ASSIGN_OR_RETURN(auto subgraph, builder->BuildConstantSubGraph(operand));
    TF_ASSIGN_OR_RETURN(auto computed,
                        client->ComputeConstant(subgraph, output_layout));
    return std::move(computed);
  }

  template <class Scalar>
  StatusOr<Scalar> ComputeConstantScalar(Client* client, const XlaOp& operand,
                                         XlaBuilder* builder) {
    TF_ASSIGN_OR_RETURN(auto literal, ComputeConstantLiteral(client, operand,
                                                             builder, nullptr));
    return literal->Get<Scalar>({});
  }

  bool IsConstant(const XlaOp& operand, XlaBuilder* builder) {
    StatusOr<bool> result = builder->IsConstant(operand);
    EXPECT_TRUE(result.ok()) << result.status();
    return result.ok() ? result.ValueOrDie() : false;
  }

  se::Platform* platform_;
};

TEST_F(ComputeConstantTest, ScalarInt32Literal) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation = ConstantR0<int32>(&b, 42);
    EXPECT_TRUE(IsConstant(computation, &b));

    auto value = ComputeConstantScalar<int32>(client, computation, &b);
    ASSERT_TRUE(value.ok()) << value.status();
    EXPECT_EQ(value.ValueOrDie(), 42);
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
    EXPECT_EQ(value.ValueOrDie(), 44.0f);
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
    EXPECT_TRUE(tensorflow::str_util::StrContains(value.status().ToString(),
                                                  "depends on a parameter"))
        << value.status();
  }
}

TEST_F(ComputeConstantTest, IndirectParamMissing) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation =
        Add(ConstantR0<float>(&b, 1.0f),
            Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "param"));
    EXPECT_FALSE(IsConstant(computation, &b));

    auto value = ComputeConstantScalar<float>(client, computation, &b);
    EXPECT_TRUE(tensorflow::str_util::StrContains(value.status().ToString(),
                                                  "depends on a parameter"))
        << value.status();
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
        Add(ConstantR1<int32>(&b, {1, 2}), ConstantR1<int32>(&b, {3, 4}));
    EXPECT_TRUE(IsConstant(computation, &b));

    TF_ASSERT_OK_AND_ASSIGN(auto computed,
                            ComputeConstantLiteral(client, computation, &b));
    std::unique_ptr<Literal> expected_literal =
        LiteralUtil::CreateR1<int32>({4, 6});
    EXPECT_TRUE(LiteralTestUtil::Equal(*expected_literal, *computed));
  }
}

TEST_F(ComputeConstantTest, IntegerDivide) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());
    auto computation = Div(ConstantR0<int32>(&b, 15), ConstantR0<int32>(&b, 3));
    EXPECT_TRUE(IsConstant(computation, &b));

    TF_ASSERT_OK_AND_ASSIGN(auto computed,
                            ComputeConstantLiteral(client, computation, &b));
    std::unique_ptr<Literal> expected_literal = LiteralUtil::CreateR0<int32>(5);
    EXPECT_TRUE(LiteralTestUtil::Equal(*expected_literal, *computed));
  }
}

XLA_TEST_F(ComputeConstantTest, Layout) {
  for (ClientType client_type : client_types) {
    Client* client = ClientOrDie(platform_, client_type);
    XlaBuilder b(TestName());

    std::vector<std::vector<int64>> layouts = {{0, 1}, {1, 0}};
    for (const std::vector<int64>& layout : layouts) {
      auto layout_proto = LayoutUtil::MakeLayout(layout);
      TF_ASSERT_OK_AND_ASSIGN(
          auto computed, ComputeConstantLiteral(
                             client,
                             Add(ConstantR2<int32>(&b, {{1, 2}, {3, 4}}),
                                 ConstantR2<int32>(&b, {{10, 20}, {30, 40}})),
                             &b, &layout_proto));

      std::unique_ptr<Literal> expected_literal =
          LiteralUtil::CreateR2WithLayout<int32>(
              {{11, 22}, {33, 44}}, LayoutUtil::MakeLayout(layout));
      ASSERT_TRUE(LiteralTestUtil::EqualShapesAndLayouts(
          expected_literal->shape(), computed->shape()));
      EXPECT_TRUE(LiteralTestUtil::Equal(*expected_literal, *computed));
    }
  }
}

}  // namespace
}  // namespace xla
