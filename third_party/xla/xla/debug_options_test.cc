/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/xla.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(DebugOptions, FieldsHavePresence) {
  for (int i = 0; i < DebugOptions::descriptor()->field_count(); ++i) {
    const tsl::protobuf::FieldDescriptor* field =
        DebugOptions::descriptor()->field(i);
    if (field->is_repeated()) {
      continue;
    }

    EXPECT_TRUE(field->has_presence())
        << "DebugOptions field " << field->name()
        << " does not have presence, ie is not explicitly optional, repeated, "
           "or of message or map type. Please ensure that it does to allow "
           "safe merging of DebugOptions instances, such as between user "
           "XLA_FLAGS and stored HloModuleConfig instances when replaying "
           "compilations.";
  }
}

}  // namespace
}  // namespace xla
