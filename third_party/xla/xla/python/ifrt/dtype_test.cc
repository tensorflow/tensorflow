// Copyright 2024 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt/dtype.h"

#include <gtest/gtest.h>
#include "xla/python/ifrt/dtype.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

TEST(DTypeTest, FromToFromProto) {
  for (int i = 0; i < DTypeProto::Kind_descriptor()->value_count(); ++i) {
    DTypeProto proto;
    proto.set_kind(static_cast<DTypeProto::Kind>(
        DTypeProto::Kind_descriptor()->value(i)->number()));
    TF_ASSERT_OK_AND_ASSIGN(DType dtype, DType::FromProto(proto));
    TF_ASSERT_OK_AND_ASSIGN(DType dtype_copy,
                            DType::FromProto(dtype.ToProto()));
    EXPECT_EQ(dtype_copy, dtype);
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
