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

#include "xla/service/compilation_environments.h"

#include <memory>
#include <utility>

#include "google/protobuf/descriptor.pb.h"
#include <gmock/gmock.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/message_lite.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/test_compilation_environment.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {

// In order to use TestCompilationEnvironment* with CompilationEnvironments, we
// must define ProcessNewEnv for them.
std::unique_ptr<google::protobuf::Message> ProcessNewEnv1(
    std::unique_ptr<google::protobuf::Message> msg) {
  std::unique_ptr<test::TestCompilationEnvironment1> env(
      google::protobuf::DownCastMessage<test::TestCompilationEnvironment1>(
          msg.release()));
  if (!env) {
    env = std::make_unique<test::TestCompilationEnvironment1>();
  }
  if (env->some_flag() == 0 || env->some_flag() == 1) {
    env->set_some_flag(100);
  }
  return env;
}
std::unique_ptr<google::protobuf::Message> ProcessNewEnv2(
    std::unique_ptr<google::protobuf::Message> msg) {
  std::unique_ptr<test::TestCompilationEnvironment2> env(
      google::protobuf::DownCastMessage<test::TestCompilationEnvironment2>(
          msg.release()));
  if (!env) {
    env = std::make_unique<test::TestCompilationEnvironment2>();
  }
  if (env->some_other_flag() == 0) {
    env->set_some_other_flag(200);
  }
  return env;
}
std::unique_ptr<google::protobuf::Message> ProcessNewEnv3(
    std::unique_ptr<google::protobuf::Message> msg) {
  std::unique_ptr<test::TestCompilationEnvironment3> env(
      google::protobuf::DownCastMessage<test::TestCompilationEnvironment3>(
          msg.release()));
  if (!env) {
    env = std::make_unique<test::TestCompilationEnvironment3>();
  }
  if (env->a_third_flag() == 0) {
    env->set_a_third_flag(300);
  }
  return env;
}

std::unique_ptr<google::protobuf::Message> ProcessCustomDescInFallbackTest(
    std::unique_ptr<google::protobuf::Message> msg_dynamic) {
  auto new_generated_env =
      std::make_unique<xla::test::TestCompilationEnvironment1>();
  // This value is used to identify that the environment was processed via this
  // specific path for the custom descriptor. It should match the
  // kExpectedFallbackValue in the test that uses this function.
  auto kTestSpecificValue = 555;
  auto kDefaultValueIfInputIsUnexpected = 558;

  new_generated_env->set_some_flag(kDefaultValueIfInputIsUnexpected);

  if (msg_dynamic) {
    const google::protobuf::Reflection* refl = msg_dynamic->GetReflection();
    const google::protobuf::Descriptor* d = msg_dynamic->GetDescriptor();
    const google::protobuf::FieldDescriptor* f = d->FindFieldByName("some_flag");
    // Check if the incoming dynamic message has the flag set to
    // kTestSpecificValue
    if (refl && f && refl->HasField(*msg_dynamic, f) &&
        refl->GetUInt32(*msg_dynamic, f) == kTestSpecificValue) {
      new_generated_env->set_some_flag(kTestSpecificValue);
    }
  }

  return new_generated_env;
}

namespace test {
namespace {

class CompilationEnvironmentsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    CompilationEnvironments::RegisterProcessNewEnvFn(
        test::TestCompilationEnvironment1::descriptor(), ProcessNewEnv1);
    CompilationEnvironments::RegisterProcessNewEnvFn(
        test::TestCompilationEnvironment2::descriptor(), ProcessNewEnv2);
    CompilationEnvironments::RegisterProcessNewEnvFn(
        test::TestCompilationEnvironment3::descriptor(), ProcessNewEnv3);
  }
};

TEST_F(CompilationEnvironmentsTest, GetDefaultEnv) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, GetDefaultMutableEnv) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, GetAddedEnvNotModifiedByProcessNewEnv) {
  CompilationEnvironments envs;
  auto env = std::make_unique<TestCompilationEnvironment1>();
  env->set_some_flag(5);
  TF_ASSERT_OK(envs.AddEnv(std::move(env)));
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 5);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 5);
}

TEST_F(CompilationEnvironmentsTest, GetAddedEnvModifiedByProcessNewEnv) {
  CompilationEnvironments envs;
  auto env = std::make_unique<TestCompilationEnvironment1>();
  env->set_some_flag(1);
  TF_ASSERT_OK(envs.AddEnv(std::move(env)));
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, MultipleEnvs) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment2>().some_other_flag(), 200);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, MultipleMutableEnvs) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment2>().some_other_flag(),
            200);
  envs.GetMutableEnv<TestCompilationEnvironment1>().set_some_flag(101);
  envs.GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(201);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 101);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment2>().some_other_flag(),
            201);
}

TEST_F(CompilationEnvironmentsTest, ReplaceExistingEnv) {
  CompilationEnvironments envs;
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(5);
  TF_ASSERT_OK(envs.AddEnv(std::move(env1)));
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 5);
  {
    auto env2 = std::make_unique<TestCompilationEnvironment1>();
    env2->set_some_flag(6);
    ASSERT_THAT(envs.AddEnv(std::move(env2)),
                absl_testing::StatusIs(absl::StatusCode::kAlreadyExists));
  }
  envs.DeleteEnv<TestCompilationEnvironment1>();
  {
    auto env2 = std::make_unique<TestCompilationEnvironment1>();
    env2->set_some_flag(6);
    TF_ASSERT_OK(envs.AddEnv(std::move(env2)));
    EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 6);
  }
}

TEST_F(CompilationEnvironmentsTest, CopyConstructor) {
  // Setup envs with 2 environments
  auto envs = std::make_unique<CompilationEnvironments>();
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(10);
  TF_ASSERT_OK(envs->AddEnv(std::move(env1)));
  auto env2 = std::make_unique<TestCompilationEnvironment2>();
  TF_ASSERT_OK(envs->AddEnv(std::move(env2)));
  envs->GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(20);

  // Call the copy constructor and delete the original CompilationEnvironments
  auto envs_copy = std::make_unique<CompilationEnvironments>(*envs);
  envs.reset();

  // Verify that envs_copy has the same values with which envs was initialized
  EXPECT_EQ(envs_copy->GetEnv<TestCompilationEnvironment1>().some_flag(), 10);
  EXPECT_EQ(envs_copy->GetEnv<TestCompilationEnvironment2>().some_other_flag(),
            20);
}

TEST_F(CompilationEnvironmentsTest, CopyAssignment) {
  // Setup envs1 with 2 environments
  auto envs1 = std::make_unique<CompilationEnvironments>();
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(10);
  TF_ASSERT_OK(envs1->AddEnv(std::move(env1)));
  auto env2 = std::make_unique<TestCompilationEnvironment2>();
  TF_ASSERT_OK(envs1->AddEnv(std::move(env2)));
  envs1->GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(20);

  // Create envs2 with some environments that should be deleted on copy
  // assignment
  auto envs2 = std::make_unique<CompilationEnvironments>();
  auto env3 = std::make_unique<TestCompilationEnvironment1>();
  env3->set_some_flag(30);
  TF_ASSERT_OK(envs2->AddEnv(std::move(env3)));
  auto env4 = std::make_unique<TestCompilationEnvironment3>();
  env4->set_a_third_flag(40);
  TF_ASSERT_OK(envs2->AddEnv(std::move(env4)));

  // Assign envs1 to envs2, and delete envs1. After assignment, the environments
  // originaly added to envs2 should be deleted, and copies of the environments
  // in envs1 should be added to envs2.
  *envs2 = *envs1;
  envs1.reset();

  // Verify that envs2 has the same values with which envs1 was initialized
  EXPECT_EQ(envs2->GetEnv<TestCompilationEnvironment1>().some_flag(), 10);
  EXPECT_EQ(envs2->GetEnv<TestCompilationEnvironment2>().some_other_flag(), 20);

  // Since envs1 did not have TestCompilationEnvironment3, after copy
  // assignment, envs2 will not have one either. So, we should get the default
  // environment value.
  EXPECT_EQ(envs2->GetEnv<TestCompilationEnvironment3>().a_third_flag(), 300);
}

TEST_F(CompilationEnvironmentsTest, ProtoRoundTrip) {
  // Setup envs with 2 environments.
  auto envs = std::make_unique<CompilationEnvironments>();
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(10);
  TF_ASSERT_OK(envs->AddEnv(std::move(env1)));
  auto env2 = std::make_unique<TestCompilationEnvironment2>();
  TF_ASSERT_OK(envs->AddEnv(std::move(env2)));
  envs->GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(20);

  auto proto = envs->ToProto();
  TF_ASSERT_OK_AND_ASSIGN(auto envs_deserialized,
                          CompilationEnvironments::CreateFromProto(proto));

  // Verify that envs_deserialized has the same values with which envs was
  // initialized.
  EXPECT_EQ(
      envs_deserialized->GetEnv<TestCompilationEnvironment1>().some_flag(), 10);
  EXPECT_EQ(envs_deserialized->GetEnv<TestCompilationEnvironment2>()
                .some_other_flag(),
            20);
}

TEST_F(CompilationEnvironmentsTest, EnvTypePresenceCheck) {
  CompilationEnvironments envs;
  EXPECT_FALSE(envs.HasEnv<TestCompilationEnvironment1>());
  envs.GetEnv<TestCompilationEnvironment1>();
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment1>());
}

TEST_F(CompilationEnvironmentsTest, InitializeAllKnownEnvs) {
  CompilationEnvironments envs;
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(400);
  TF_ASSERT_OK(envs.AddEnv(std::move(env1)));
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment1>());
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 400);
  TF_ASSERT_OK(envs.InitializeAllKnownEnvs());
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment1>());
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 400);
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment2>());
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment2>().some_other_flag(), 200);
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment3>());
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment3>().a_third_flag(), 300);
}

TEST_F(CompilationEnvironmentsTest, GetEnvTriggersFullNameFallback) {
  // Create a custom descriptor pool and load the proto into it.
  const google::protobuf::Descriptor* desc_generated =
      test::TestCompilationEnvironment1::descriptor();
  google::protobuf::FileDescriptorProto file_proto;
  desc_generated->file()->CopyTo(&file_proto);

  google::protobuf::DescriptorPool custom_pool;
  custom_pool.BuildFile(file_proto);

  // Register a custom handler for the descriptor from the custom_pool.
  const google::protobuf::Descriptor* desc_custom =
      custom_pool.FindMessageTypeByName(desc_generated->full_name());
  CompilationEnvironments::RegisterProcessNewEnvFn(
      desc_custom, ProcessCustomDescInFallbackTest);
  // We need to deregister the function to avoid side effects in other tests.
  absl::Cleanup cleanup = [=]() {
    CompilationEnvironments::DeregisterProcessNewEnvFn(desc_custom);
  };

  // Create and populate a dynamic message instance using the custom descriptor.
  google::protobuf::DynamicMessageFactory factory(&custom_pool);
  std::unique_ptr<google::protobuf::Message> dynamic_env_instance(
      factory.GetPrototype(desc_custom)->New());
  const google::protobuf::FieldDescriptor* flag_field =
      desc_custom->FindFieldByName("some_flag");
  auto kExpectedFallbackValue = 555;
  dynamic_env_instance->GetReflection()->SetUInt32(
      dynamic_env_instance.get(), flag_field, kExpectedFallbackValue);

  // Add this dynamic instance to CompilationEnvironments.
  CompilationEnvironments envs;
  TF_ASSERT_OK(envs.AddEnv(std::move(dynamic_env_instance)));

  // Trigger lookup by full_name.
  const auto& retrieved_env = envs.GetEnv<test::TestCompilationEnvironment1>();

  // Verify that the fallback value was used.
  EXPECT_EQ(retrieved_env.some_flag(), kExpectedFallbackValue);
}

TEST_F(CompilationEnvironmentsTest, UnknownEnvTypeRoundTripsViaToProto) {
  // Verify that unknown proto types survive a CreateFromProto -> ToProto
  // round trip, preserving both the type URL and the opaque payload.
  constexpr absl::string_view kUnknownTypeUrlA =
      "type.googleapis.com/some.unknown.ProtoTypeA";
  constexpr absl::string_view kUnknownPayloadA = "payload_a";
  constexpr absl::string_view kUnknownTypeUrlB =
      "type.googleapis.com/some.unknown.ProtoTypeB";
  constexpr absl::string_view kUnknownPayloadB = "payload_b";

  CompilationEnvironmentsProto proto;

  // Add a known environment.
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(7);
  proto.add_environments()->PackFrom(*env1);

  // Add two "unknown" environments.
  google::protobuf::Any* unknown1 = proto.add_environments();
  unknown1->set_type_url(kUnknownTypeUrlA);
  unknown1->set_value(kUnknownPayloadA);

  google::protobuf::Any* unknown2 = proto.add_environments();
  unknown2->set_type_url(kUnknownTypeUrlB);
  unknown2->set_value(kUnknownPayloadB);

  // Round-trip: CreateFromProto -> ToProto.
  ASSERT_OK_AND_ASSIGN(auto envs,
                       CompilationEnvironments::CreateFromProto(proto));
  CompilationEnvironmentsProto output_proto = envs->ToProto();

  // The output should contain the known env + both unknown envs.
  ASSERT_EQ(output_proto.environments_size(), 3);

  // The known env should be first (sorted by full_name).
  EXPECT_THAT(output_proto.environments(0).type_url(),
              ::testing::HasSubstr("TestCompilationEnvironment1"));

  // The unknown envs should follow, preserving type URLs and payloads.
  EXPECT_EQ(output_proto.environments(1).type_url(), kUnknownTypeUrlA);
  EXPECT_EQ(output_proto.environments(1).value(), kUnknownPayloadA);
  EXPECT_EQ(output_proto.environments(2).type_url(), kUnknownTypeUrlB);
  EXPECT_EQ(output_proto.environments(2).value(), kUnknownPayloadB);
}

TEST_F(CompilationEnvironmentsTest, UnknownEnvTypePreservedOnCopy) {
  // Verify that unknown environments are preserved across copy construction.
  constexpr absl::string_view kUnknownTypeUrl =
      "type.googleapis.com/some.unknown.InternalProto";
  constexpr absl::string_view kUnknownPayload = "opaque_config_bytes";

  CompilationEnvironmentsProto proto;

  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(99);
  proto.add_environments()->PackFrom(*env1);

  google::protobuf::Any* unknown = proto.add_environments();
  unknown->set_type_url(kUnknownTypeUrl);
  unknown->set_value(kUnknownPayload);

  ASSERT_OK_AND_ASSIGN(auto envs,
                       CompilationEnvironments::CreateFromProto(proto));

  // Copy construct.
  auto envs_copy = std::make_unique<CompilationEnvironments>(*envs);
  envs.reset();  // Destroy the original.

  // The copy should still have the known env.
  EXPECT_EQ(envs_copy->GetEnv<TestCompilationEnvironment1>().some_flag(), 99);

  // The copy should round-trip the unknown env.
  CompilationEnvironmentsProto copy_proto = envs_copy->ToProto();
  ASSERT_EQ(copy_proto.environments_size(), 2);
  EXPECT_EQ(copy_proto.environments(1).type_url(), kUnknownTypeUrl);
  EXPECT_EQ(copy_proto.environments(1).value(), kUnknownPayload);
}

TEST_F(CompilationEnvironmentsTest, ClearAlsoClearsUnknownEnvs) {
  // Clear() must also drop unknown (opaque) environments, otherwise a stale
  // entry would be re-emitted by a subsequent ToProto().
  constexpr absl::string_view kUnknownTypeUrl =
      "type.googleapis.com/some.unknown.InternalProto";
  constexpr absl::string_view kUnknownPayload = "opaque_config_bytes";

  CompilationEnvironmentsProto proto;
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(5);
  proto.add_environments()->PackFrom(*env1);
  google::protobuf::Any* unknown = proto.add_environments();
  unknown->set_type_url(kUnknownTypeUrl);
  unknown->set_value(kUnknownPayload);

  ASSERT_OK_AND_ASSIGN(auto envs,
                       CompilationEnvironments::CreateFromProto(proto));
  ASSERT_EQ(envs->ToProto().environments_size(), 2);

  envs->Clear();

  // Both the known and the unknown environments should be gone.
  EXPECT_FALSE(envs->HasEnv<TestCompilationEnvironment1>());
  EXPECT_EQ(envs->ToProto().environments_size(), 0);
}

TEST_F(CompilationEnvironmentsTest, DuplicateUnknownEnvTypeReturnsError) {
  // Two unknown entries of the same type must be rejected, mirroring
  // AddEnvImpl's duplicate check for known types.
  constexpr absl::string_view kUnknownTypeUrl =
      "type.googleapis.com/some.unknown.DupProto";

  CompilationEnvironmentsProto proto;
  google::protobuf::Any* first = proto.add_environments();
  first->set_type_url(kUnknownTypeUrl);
  first->set_value("first");
  google::protobuf::Any* second = proto.add_environments();
  second->set_type_url(kUnknownTypeUrl);
  second->set_value("second");

  auto result = CompilationEnvironments::CreateFromProto(proto);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kAlreadyExists);
}

TEST_F(CompilationEnvironmentsTest, UnknownEnvsAreSortedByTypeNameInToProto) {
  // ToProto() must emit unknown environments sorted by their fully-qualified
  // type name regardless of input order, so the output is deterministic.
  constexpr absl::string_view kTypeUrlA =
      "type.googleapis.com/some.unknown.ProtoTypeA";
  constexpr absl::string_view kTypeUrlB =
      "type.googleapis.com/some.unknown.ProtoTypeB";

  CompilationEnvironmentsProto proto;
  // Intentionally add B before A.
  google::protobuf::Any* b = proto.add_environments();
  b->set_type_url(kTypeUrlB);
  b->set_value("payload_b");
  google::protobuf::Any* a = proto.add_environments();
  a->set_type_url(kTypeUrlA);
  a->set_value("payload_a");

  ASSERT_OK_AND_ASSIGN(auto envs,
                       CompilationEnvironments::CreateFromProto(proto));
  CompilationEnvironmentsProto output_proto = envs->ToProto();

  ASSERT_EQ(output_proto.environments_size(), 2);
  EXPECT_EQ(output_proto.environments(0).type_url(), kTypeUrlA);
  EXPECT_EQ(output_proto.environments(1).type_url(), kTypeUrlB);
}

TEST_F(CompilationEnvironmentsTest,
       DuplicateUnknownEnvTypeWithDifferentPrefixReturnsError) {
  // Duplicate detection keys on the fully-qualified type name, not the raw
  // type_url, so the same type under two different Any prefixes is still a
  // duplicate (parity with AddEnvImpl, which dedups by descriptor).
  CompilationEnvironmentsProto proto;
  google::protobuf::Any* first = proto.add_environments();
  first->set_type_url("type.googleapis.com/some.unknown.PrefixProto");
  first->set_value("first");
  google::protobuf::Any* second = proto.add_environments();
  second->set_type_url("custom.prefix.example.com/some.unknown.PrefixProto");
  second->set_value("second");

  auto result = CompilationEnvironments::CreateFromProto(proto);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kAlreadyExists);
}

}  // namespace
}  // namespace test
}  // namespace xla
