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

#include "xla/pjrt/plugin/example_plugin/myplugin_c_pjrt.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/extensions/example/example_extension.h"

namespace {

TEST(MypluginCPjRtTest, CreatesPjRtAPI) {
  const PJRT_Api* myplugin = GetPjrtApi();
  EXPECT_THAT(myplugin, ::testing::NotNull());
}

TEST(MypluginCPjRtTest, CallsExampleExtension) {
  const PJRT_Api* myplugin = GetPjrtApi();
  EXPECT_THAT(myplugin, ::testing::NotNull());
  PJRT_Example_Extension* ext_api = pjrt::FindExtension<PJRT_Example_Extension>(
      myplugin, PJRT_Extension_Type::PJRT_Extension_Type_Unknown);
  EXPECT_THAT(ext_api, ::testing::NotNull());

  PJRT_ExampleExtension_CreateExampleExtensionCpp_Args get_args = {};
  ext_api->create(&get_args);

  PJRT_ExampleExtension_ExampleMethod_Args args = {
      /*extension=*/get_args.extension_cpp,
      /*value=*/42,
  };
  ext_api->example_method(&args);

  ext_api->destroy(&get_args);
}

}  // namespace
