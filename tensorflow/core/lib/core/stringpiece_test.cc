/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/core/stringpiece.h"

#include <unordered_map>
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(StringPiece, Ctor) {
  {
    // const char* without size.
    const char* hello = "hello";
    StringPiece s20(hello);
    EXPECT_TRUE(s20.data() == hello);
    EXPECT_EQ(5, s20.size());

    // const char* with size.
    StringPiece s21(hello, 4);
    EXPECT_TRUE(s21.data() == hello);
    EXPECT_EQ(4, s21.size());

    // Not recommended, but valid C++
    StringPiece s22(hello, 6);
    EXPECT_TRUE(s22.data() == hello);
    EXPECT_EQ(6, s22.size());
  }

  {
    string hola = "hola";
    StringPiece s30(hola);
    EXPECT_TRUE(s30.data() == hola.data());
    EXPECT_EQ(4, s30.size());

    // std::string with embedded '\0'.
    hola.push_back('\0');
    hola.append("h2");
    hola.push_back('\0');
    StringPiece s31(hola);
    EXPECT_TRUE(s31.data() == hola.data());
    EXPECT_EQ(8, s31.size());
  }
}

TEST(StringPiece, ConversionToString) {
  EXPECT_EQ("", string(StringPiece("")));
  EXPECT_EQ("foo", string(StringPiece("foo")));
}

}  // namespace tensorflow
