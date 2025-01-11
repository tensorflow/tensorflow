/* Copyright 2023 The OpenXLA Authors.

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

#include "absl/strings/string_view.h"
#include "xla/python/ifrt/test_util.h"

// For now, all of the tests we run are provided by IFRT, they use
// NanoIfrtClient via the "register_nanort_for_ifrt_tests" target, which can
// also be used to run NanoIfrtClient in other tests. see the BUILD file for the
// list. We need a main function to filter out one test that doesn't seem worth
// supporting.

int main(int argc, char** argv) {
  // This test expects copies to multiple devices to fail, but we only have one
  // device and it doesn't seem worth pretending that we have more.
  static constexpr absl::string_view kFilter =
      "-ArrayImplTest.CopyMixedSourceDevices";
  xla::ifrt::test_util::SetTestFilterIfNotUserSpecified(kFilter);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
