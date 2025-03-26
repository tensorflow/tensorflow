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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/test_util.h"

int main(int argc, char** argv) {
  // TfrtCpuBuffer::ToLiteral() currently does not respect the layout of the
  // destination literal.
  static constexpr absl::string_view kFilter =
      "-ArrayImplTest."
      "MakeArrayFromHostBufferAndCopyToHostBufferWithByteStrides";
  xla::ifrt::test_util::SetTestFilterIfNotUserSpecified(kFilter);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
