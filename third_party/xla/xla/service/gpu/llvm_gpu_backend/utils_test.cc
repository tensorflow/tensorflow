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

#include "xla/service/gpu/llvm_gpu_backend/utils.h"

#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

TEST(UtilsTest, TestReplaceFilenameExtension) {
  ASSERT_EQ(ReplaceFilenameExtension("baz.tx", "cc"), "baz.cc");
  ASSERT_EQ(ReplaceFilenameExtension("/foo/baz.txt", "cc"), "/foo/baz.cc");
  ASSERT_EQ(ReplaceFilenameExtension("/foo/baz.", "-nvptx.dummy"),
            "/foo/baz.-nvptx.dummy");
  ASSERT_EQ(ReplaceFilenameExtension("/foo/baz", "cc"), "/foo/baz.cc");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
