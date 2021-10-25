/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/experimental/libexport/save.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace libexport {
namespace {

TEST(SaveTest, TestDirectoryStructure) {
  const string base_dir = tensorflow::io::JoinPath(
      tensorflow::testing::TmpDir(), "test_directory_structure");
  TF_ASSERT_OK(Save(base_dir));
  TF_ASSERT_OK(Env::Default()->IsDirectory(base_dir));
}

}  // namespace
}  // namespace libexport
}  // namespace tensorflow
