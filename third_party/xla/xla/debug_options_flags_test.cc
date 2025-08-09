/* Copyright 2025 The OpenXLA Authors.

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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "xla/xla.pb.h"
#include "tsl/platform/protobuf.h"

using ::testing::IsEmpty;

namespace xla {
namespace {

TEST(DebugOptions, AllFieldsHavePresence) {
  absl::flat_hash_set<std::string> fields_missing_presence;

  const tsl::protobuf::Descriptor* debug_options = DebugOptions::descriptor();
  for (int i = 0; i < debug_options->field_count(); ++i) {
    const tsl::protobuf::FieldDescriptor* field = debug_options->field(i);
    // Repeated fields don't technically have presence (no has_foo) but
    // foo().empty() is just as good.
    if (!field->is_repeated() && !field->has_presence()) {
      fields_missing_presence.insert(std::string(field->name()));
    }
  }

  EXPECT_THAT(fields_missing_presence, IsEmpty())
      << "All scalar fields in DebugOptions must have presence defined by "
         "being labeled `optional`.";
}

}  // namespace
}  // namespace xla
