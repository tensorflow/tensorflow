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

TEST(StringPiece, Contains) {
  StringPiece a("abcdefg");
  StringPiece b("abcd");
  StringPiece c("efg");
  StringPiece d("gh");
  EXPECT_TRUE(a.contains(b));
  EXPECT_TRUE(a.contains(c));
  EXPECT_TRUE(!a.contains(d));
}

TEST(StringPieceHasher, Equality) {
  StringPieceHasher hasher;

  StringPiece s1("foo");
  StringPiece s2("bar");
  StringPiece s3("baz");
  StringPiece s4("zot");

  EXPECT_TRUE(hasher(s1) != hasher(s2));
  EXPECT_TRUE(hasher(s1) != hasher(s3));
  EXPECT_TRUE(hasher(s1) != hasher(s4));
  EXPECT_TRUE(hasher(s2) != hasher(s3));
  EXPECT_TRUE(hasher(s2) != hasher(s4));
  EXPECT_TRUE(hasher(s3) != hasher(s4));

  EXPECT_TRUE(hasher(s1) == hasher(s1));
  EXPECT_TRUE(hasher(s2) == hasher(s2));
  EXPECT_TRUE(hasher(s3) == hasher(s3));
  EXPECT_TRUE(hasher(s4) == hasher(s4));
}

TEST(StringPieceHasher, HashMap) {
  string s1("foo");
  string s2("bar");
  string s3("baz");

  StringPiece p1(s1);
  StringPiece p2(s2);
  StringPiece p3(s3);

  std::unordered_map<StringPiece, int, StringPieceHasher> map;

  map.insert(std::make_pair(p1, 0));
  map.insert(std::make_pair(p2, 1));
  map.insert(std::make_pair(p3, 2));
  EXPECT_EQ(map.size(), 3);

  bool found[3] = {false, false, false};
  for (auto const& val : map) {
    int x = val.second;
    EXPECT_TRUE(x >= 0 && x < 3);
    EXPECT_TRUE(!found[x]);
    found[x] = true;
  }
  EXPECT_EQ(found[0], true);
  EXPECT_EQ(found[1], true);
  EXPECT_EQ(found[2], true);

  auto new_iter = map.find("zot");
  EXPECT_TRUE(new_iter == map.end());

  new_iter = map.find("bar");
  EXPECT_TRUE(new_iter != map.end());

  map.erase(new_iter);
  EXPECT_EQ(map.size(), 2);

  found[0] = false;
  found[1] = false;
  found[2] = false;
  for (const auto& iter : map) {
    int x = iter.second;
    EXPECT_TRUE(x >= 0 && x < 3);
    EXPECT_TRUE(!found[x]);
    found[x] = true;
  }
  EXPECT_EQ(found[0], true);
  EXPECT_EQ(found[1], false);
  EXPECT_EQ(found[2], true);
}
}  // namespace tensorflow
