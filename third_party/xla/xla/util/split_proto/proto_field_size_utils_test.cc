/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/util/split_proto/proto_field_size_utils.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/cord.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;

TEST(ProtoFieldSizeUtilsTest,
     GetTopKProtoFieldSizes_DefaultMessage_ReturnsSortedTopFields) {
  gpu::GpuExecutableProto proto;
  proto.set_module_name("short_name");
  proto.set_asm_text(std::string(1000, 'a'));
  proto.set_binary(std::string(5000, 'b'));

  std::string report = GetTopKProtoFieldSizes(proto, 2);
  EXPECT_THAT(
      report,
      HasSubstr("Top 2 largest fields in proto [xla.gpu.GpuExecutableProto]"));
  EXPECT_THAT(report, HasSubstr("Total ByteSize:"));
  EXPECT_THAT(report, HasSubstr("1. binary (tag 4, type BYTES):"));
  EXPECT_THAT(report, HasSubstr("2. asm_text (tag 3, type STRING):"));
  // module_name should not be in top 2
  EXPECT_THAT(report, Not(HasSubstr("3. module_name")));
}

TEST(ProtoFieldSizeUtilsTest,
     GetTopKProtoFieldSizes_CustomTopK_LimitsOutputCount) {
  gpu::GpuExecutableProto proto;
  proto.set_module_name("short_name");
  proto.set_asm_text(std::string(1000, 'a'));
  proto.set_binary(std::string(5000, 'b'));

  std::string report = GetTopKProtoFieldSizes(proto, 1);
  EXPECT_THAT(
      report,
      HasSubstr("Top 1 largest fields in proto [xla.gpu.GpuExecutableProto]"));
  EXPECT_THAT(report, HasSubstr("1. binary (tag 4, type BYTES):"));
  EXPECT_THAT(report, Not(HasSubstr("2. asm_text")));
}

TEST(ProtoFieldSizeUtilsTest,
     GetTopKProtoFieldSizes_ZeroOrNegativeTopK_ReturnsHeaderOnly) {
  gpu::GpuExecutableProto proto;
  proto.set_module_name("test");
  proto.set_binary("binary_data");

  std::string report = GetTopKProtoFieldSizes(proto, 0);
  EXPECT_THAT(report, HasSubstr("Top 0 largest fields in proto"));
  EXPECT_THAT(report, Not(HasSubstr("1. binary")));

  report = GetTopKProtoFieldSizes(proto, -5);
  EXPECT_THAT(report, HasSubstr("Top 0 largest fields in proto"));
  EXPECT_THAT(report, Not(HasSubstr("1. binary")));
}

TEST(ProtoFieldSizeUtilsTest,
     GetTopKProtoFieldSizes_NestedMessage_ReturnsHierarchicalSubfields) {
  gpu::GpuExecutableProto proto;
  proto.set_module_name("mod");
  auto* hlo_module =
      proto.mutable_hlo_module_with_config()->mutable_hlo_module();
  hlo_module->set_entry_computation_name(std::string(1024 * 1024, 'a'));
  auto* comp = hlo_module->add_computations();
  comp->set_name(std::string(1024 * 1024, 'b'));
  comp->add_instructions()->set_name("inst_1");

  std::string report = GetTopKProtoFieldSizes(proto, 5);
  EXPECT_THAT(
      report,
      HasSubstr("Top 2 largest fields in proto [xla.gpu.GpuExecutableProto]"));
  EXPECT_THAT(report,
              HasSubstr("1. hlo_module_with_config (tag 1, type MESSAGE):"));
  EXPECT_THAT(
      report,
      HasSubstr("-> hlo_module.computations.name (tag 1, type STRING):"));
  EXPECT_THAT(
      report,
      HasSubstr("-> hlo_module.entry_computation_name (tag 2, type STRING):"));
}

TEST(ProtoFieldSizeUtilsTest,
     GetTopKProtoFieldSizes_RepeatedMessage_DrillsDownRepeatedElements) {
  gpu::GpuExecutableProto proto;
  proto.set_module_name("mod");
  auto* hlo_module =
      proto.mutable_hlo_module_with_config()->mutable_hlo_module();
  auto* comp1 = hlo_module->add_computations();
  comp1->set_name(std::string(1024 * 1024, 'a'));
  auto* comp2 = hlo_module->add_computations();
  comp2->set_name(std::string(1024 * 1024, 'b'));

  std::string report = GetTopKProtoFieldSizes(proto, 5);
  EXPECT_THAT(
      report,
      HasSubstr("-> hlo_module.computations.name (tag 1, type STRING):"));
}

TEST(ProtoFieldSizeUtilsTest,
     AnnotateResourceExhaustedError_NonResourceExhausted_ReturnsUnchanged) {
  gpu::GpuExecutableProto proto;
  absl::Status status = absl::InternalError("internal error");

  absl::Status annotated = AnnotateResourceExhaustedError(status, proto);
  EXPECT_THAT(annotated,
              StatusIs(absl::StatusCode::kInternal, "internal error"));
}

TEST(ProtoFieldSizeUtilsTest,
     AnnotateResourceExhaustedError_EmptyMessage_AppendsAnnotation) {
  gpu::GpuExecutableProto proto;
  proto.set_module_name("test_module");
  absl::Status status = absl::ResourceExhaustedError("");

  absl::Status annotated = AnnotateResourceExhaustedError(status, proto);
  EXPECT_THAT(annotated, StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_THAT(
      annotated.message(),
      HasSubstr("Top 1 largest fields in proto [xla.gpu.GpuExecutableProto]"));
}

TEST(ProtoFieldSizeUtilsTest,
     AnnotateResourceExhaustedError_StatusWithPayload_PreservesPayload) {
  gpu::GpuExecutableProto proto;
  proto.set_module_name("test");
  absl::Status status = absl::ResourceExhaustedError("resource exhausted");
  status.SetPayload("test_url", absl::Cord("test_payload"));

  absl::Status annotated = AnnotateResourceExhaustedError(status, proto);
  EXPECT_THAT(annotated, StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(annotated.GetPayload("test_url"), absl::Cord("test_payload"));
}

}  // namespace
}  // namespace xla
