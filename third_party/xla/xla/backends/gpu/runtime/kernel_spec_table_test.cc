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
#include "xla/backends/gpu/runtime/kernel_spec_table.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.pb.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/stream_executor/kernel_spec.pb.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::OneofDescriptor;
using ::google::protobuf::Reflection;
using ::testing::Not;
using ::tsl::proto_testing::EqualsProto;

stream_executor::KernelLoaderSpecProto CubinSpec(absl::string_view kernel_name,
                                                 absl::string_view cubin) {
  stream_executor::KernelLoaderSpecProto spec;
  spec.set_kernel_name(kernel_name);
  spec.set_arity(2);
  spec.mutable_cubin()->set_data(cubin);
  return spec;
}

stream_executor::KernelLoaderSpecProto SymbolSpec(
    absl::string_view kernel_name) {
  stream_executor::KernelLoaderSpecProto spec;
  spec.set_kernel_name(kernel_name);
  spec.set_arity(1);
  spec.mutable_in_process_symbol()->set_persistent_name(kernel_name);
  return spec;
}

// Builds a top level thunk holding a custom kernel thunk for `spec`.
ThunkProto CustomKernelThunk(
    const stream_executor::KernelLoaderSpecProto& spec) {
  ThunkProto thunk;
  CustomKernelProto& custom_kernel =
      *thunk.mutable_custom_kernel_thunk()->mutable_custom_kernel();
  custom_kernel.set_name(spec.kernel_name());
  *custom_kernel.mutable_kernel_spec() = spec;
  return thunk;
}

// Wraps `thunks` in a sequential thunk.
ThunkProto SequentialThunk(const std::vector<ThunkProto>& thunks) {
  ThunkProto thunk;
  for (const ThunkProto& nested : thunks) {
    *thunk.mutable_sequential_thunk()->add_thunks() = nested;
  }
  return thunk;
}

TEST(KernelSpecTableTest, InternDeduplicatesEqualSpecs) {
  KernelSpecTable table;
  EXPECT_TRUE(table.empty());

  const int32_t first = table.Intern(CubinSpec("kernel", "cubin"));
  const int32_t second = table.Intern(CubinSpec("kernel", "cubin"));
  const int32_t third = table.Intern(CubinSpec("other", "cubin"));

  EXPECT_EQ(first, 0);
  EXPECT_EQ(second, 0);
  EXPECT_EQ(third, 1);
  EXPECT_EQ(table.size(), 2);
}

TEST(KernelSpecTableTest, InternKeepsDistinctSpecsApart) {
  KernelSpecTable table;
  // Enough specs to exercise the hash buckets, all of them distinct.
  constexpr int kNumSpecs = 128;
  for (int i = 0; i < kNumSpecs; ++i) {
    EXPECT_EQ(table.Intern(CubinSpec(absl::StrCat("kernel_", i),
                                     absl::StrCat("cubin_", i))),
              i);
  }
  EXPECT_EQ(table.size(), kNumSpecs);
  for (int i = 0; i < kNumSpecs; ++i) {
    ASSERT_OK_AND_ASSIGN(const stream_executor::KernelLoaderSpecProto* spec,
                         table.Get(i));
    EXPECT_THAT(*spec, EqualsProto(CubinSpec(absl::StrCat("kernel_", i),
                                             absl::StrCat("cubin_", i))));
  }
}

TEST(KernelSpecTableTest, InternInternsInProcessSymbolSpecs) {
  KernelSpecTable table;
  EXPECT_EQ(table.Intern(SymbolSpec("symbol")), 0);
  EXPECT_EQ(table.Intern(SymbolSpec("symbol")), 0);
  EXPECT_EQ(table.size(), 1);
}

TEST(KernelSpecTableTest, GetRejectsOutOfRangeIndices) {
  KernelSpecTable table;
  table.Intern(CubinSpec("kernel", "cubin"));
  EXPECT_THAT(table.Get(1), StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(KernelSpecTableTest, AppendPreservesDuplicates) {
  KernelSpecTable table;
  EXPECT_EQ(table.Append(CubinSpec("kernel", "cubin")), 0);
  EXPECT_EQ(table.Append(CubinSpec("kernel", "cubin")), 1);
  EXPECT_EQ(table.size(), 2);
  // A subsequent intern still finds the first of the two.
  EXPECT_EQ(table.Intern(CubinSpec("kernel", "cubin")), 0);
}

TEST(KernelSpecTableTest, IndexAssignmentIsDeterministic) {
  const ThunkProto thunk = SequentialThunk({
      CustomKernelThunk(CubinSpec("a", "cubin_a")),
      CustomKernelThunk(CubinSpec("b", "cubin_b")),
      CustomKernelThunk(CubinSpec("a", "cubin_a")),
      CustomKernelThunk(CubinSpec("c", "cubin_c")),
      CustomKernelThunk(CubinSpec("b", "cubin_b")),
  });

  std::string first_serialized;
  std::string second_serialized;
  for (std::string* serialized : {&first_serialized, &second_serialized}) {
    ThunkProto copy = thunk;
    KernelSpecTable table;
    ASSERT_THAT(InternKernelSpecs(copy, table), IsOk());
    ASSERT_EQ(table.size(), 3);
    ThunkSequenceProto table_and_thunk;
    // Serialize both the rewritten thunk and the table so that a difference in
    // either shows up.
    *table_and_thunk.add_thunks() = copy;
    std::string thunk_bytes;
    ASSERT_TRUE(
        tsl::SerializeToStringDeterministic(table_and_thunk, &thunk_bytes));
    *serialized = thunk_bytes;
    for (const stream_executor::KernelLoaderSpecProto& spec : table.specs()) {
      std::string spec_bytes;
      ASSERT_TRUE(tsl::SerializeToStringDeterministic(spec, &spec_bytes));
      absl::StrAppend(serialized, spec_bytes);
    }
  }
  EXPECT_EQ(first_serialized, second_serialized);
}

TEST(KernelSpecTableTest, InternAndInlineRoundTrip) {
  ThunkProto thunk = SequentialThunk({
      CustomKernelThunk(CubinSpec("a", "cubin_a")),
      CustomKernelThunk(CubinSpec("b", "cubin_b")),
      CustomKernelThunk(CubinSpec("a", "cubin_a")),
  });
  const ThunkProto original = thunk;

  KernelSpecTable table;
  ASSERT_THAT(InternKernelSpecs(thunk, table), IsOk());
  EXPECT_EQ(table.size(), 2);
  EXPECT_THAT(thunk, Not(EqualsProto(original)));
  for (const ThunkProto& nested : thunk.sequential_thunk().thunks()) {
    EXPECT_TRUE(
        nested.custom_kernel_thunk().custom_kernel().has_kernel_spec_index());
  }
  EXPECT_EQ(thunk.sequential_thunk()
                .thunks(0)
                .custom_kernel_thunk()
                .custom_kernel()
                .kernel_spec_index(),
            thunk.sequential_thunk()
                .thunks(2)
                .custom_kernel_thunk()
                .custom_kernel()
                .kernel_spec_index());

  ASSERT_THAT(InlineKernelSpecs(thunk, table), IsOk());
  EXPECT_THAT(thunk, EqualsProto(original));
}

TEST(KernelSpecTableTest, InterningIsIdempotentAndHandlesMixedInput) {
  ThunkProto thunk = SequentialThunk({
      CustomKernelThunk(CubinSpec("a", "cubin_a")),
      CustomKernelThunk(CubinSpec("b", "cubin_b")),
  });
  KernelSpecTable table;
  ASSERT_THAT(InternKernelSpecs(thunk, table), IsOk());
  const ThunkProto interned = thunk;

  // A thunk that already references the table is left alone, even when a new
  // inline spec is mixed in next to it.
  *thunk.mutable_sequential_thunk()->add_thunks() =
      CustomKernelThunk(CubinSpec("c", "cubin_c"));
  ASSERT_THAT(InternKernelSpecs(thunk, table), IsOk());
  EXPECT_EQ(table.size(), 3);
  EXPECT_THAT(thunk.sequential_thunk().thunks(0),
              EqualsProto(interned.sequential_thunk().thunks(0)));
  EXPECT_THAT(thunk.sequential_thunk().thunks(1),
              EqualsProto(interned.sequential_thunk().thunks(1)));
}

TEST(KernelSpecTableTest, InliningLeavesInlineSpecsAlone) {
  // Simulates a proto produced by a compiler that did not deduplicate.
  ThunkProto thunk = SequentialThunk({
      CustomKernelThunk(CubinSpec("a", "cubin_a")),
      CustomKernelThunk(CubinSpec("b", "cubin_b")),
  });
  const ThunkProto original = thunk;

  const KernelSpecTable empty_table;
  ASSERT_THAT(InlineKernelSpecs(thunk, empty_table), IsOk());
  EXPECT_THAT(thunk, EqualsProto(original));
}

TEST(KernelSpecTableTest, InliningRejectsDanglingIndices) {
  ThunkProto thunk = SequentialThunk({
      CustomKernelThunk(CubinSpec("a", "cubin_a")),
  });
  KernelSpecTable table;
  ASSERT_THAT(InternKernelSpecs(thunk, table), IsOk());

  const KernelSpecTable empty_table;
  EXPECT_THAT(InlineKernelSpecs(thunk, empty_table),
              StatusIs(absl::StatusCode::kOutOfRange));
}

// Returns a mutable `ThunkProto` nested inside `message`, adding the
// intermediate messages as needed, or nullptr if `message` cannot hold a nested
// thunk. Fields are considered in declaration order.
ThunkProto* AddNestedThunk(Message& message, int max_depth) {
  if (max_depth <= 0) {
    return nullptr;
  }
  const Descriptor* descriptor = message.GetDescriptor();
  const Reflection* reflection = message.GetReflection();
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const FieldDescriptor* field = descriptor->field(i);
    if (field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
      continue;
    }
    const Descriptor* field_type = field->message_type();
    if (field_type != ThunkProto::descriptor() &&
        field_type != ThunkSequenceProto::descriptor()) {
      continue;
    }
    Message* nested = field->is_repeated()
                          ? reflection->AddMessage(&message, field)
                          : reflection->MutableMessage(&message, field);
    if (field_type == ThunkProto::descriptor()) {
      return static_cast<ThunkProto*>(nested);
    }
    return static_cast<ThunkSequenceProto*>(nested)->add_thunks();
  }
  // No direct nesting; look one level deeper.
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const FieldDescriptor* field = descriptor->field(i);
    if (field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE ||
        field->is_map()) {
      continue;
    }
    Message* nested = field->is_repeated()
                          ? reflection->AddMessage(&message, field)
                          : reflection->MutableMessage(&message, field);
    if (ThunkProto* thunk = AddNestedThunk(*nested, max_depth - 1);
        thunk != nullptr) {
      return thunk;
    }
    // Undo the speculative mutation so that the caller sees a clean message.
    if (field->is_repeated()) {
      reflection->RemoveLast(&message, field);
    } else {
      reflection->ClearField(&message, field);
    }
  }
  return nullptr;
}

// Guards against the walk missing a nesting thunk kind. Rather than hard coding
// the list of nesting kinds, this enumerates them from the descriptor, so a
// newly added nesting kind is covered automatically.
TEST(KernelSpecTableTest, WalkReachesEveryNestingThunkKind) {
  const Descriptor* thunk_descriptor = ThunkProto::descriptor();
  const OneofDescriptor* impl = thunk_descriptor->FindOneofByName("impl");
  ASSERT_NE(impl, nullptr);

  int nesting_kinds = 0;
  for (int i = 0; i < impl->field_count(); ++i) {
    const FieldDescriptor* field = impl->field(i);
    if (field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
      continue;
    }
    ThunkProto thunk;
    Message* impl_message =
        thunk.GetReflection()->MutableMessage(&thunk, field);
    ThunkProto* nested = AddNestedThunk(*impl_message, /*max_depth=*/3);
    if (nested == nullptr) {
      continue;
    }
    ++nesting_kinds;
    *nested = CustomKernelThunk(CubinSpec("kernel", "cubin"));

    KernelSpecTable table;
    ASSERT_THAT(InternKernelSpecs(thunk, table), IsOk()) << field->name();
    EXPECT_EQ(table.size(), 1)
        << "the walk did not descend into " << field->name();
    EXPECT_TRUE(
        nested->custom_kernel_thunk().custom_kernel().has_kernel_spec_index())
        << "the walk did not rewrite the custom kernel in " << field->name();
  }

  // sequential, conditional, while, async start, collective group and dynamic
  // slice fusion thunks all nest other thunks.
  EXPECT_GE(nesting_kinds, 6);
}

}  // namespace
}  // namespace xla::gpu
