/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/public/env.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

struct EnvTest {};

TEST(EnvTest, ReadFileToString) {
  Env* env = Env::Default();
  const string dir = testing::TmpDir();
  for (const int length : {0, 1, 1212, 2553, 4928, 8196, 9000}) {
    const string filename = io::JoinPath(dir, strings::StrCat("file", length));

    // Write a file with the given length
    string input(length, 0);
    for (int i = 0; i < length; i++) input[i] = i;
    WriteStringToFile(env, filename, input);

    // Read the file back and check equality
    string output;
    TF_CHECK_OK(ReadFileToString(env, filename, &output));
    CHECK_EQ(length, output.size());
    CHECK_EQ(input, output);
  }
}

}  // namespace tensorflow
