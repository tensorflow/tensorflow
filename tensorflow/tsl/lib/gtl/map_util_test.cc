/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/lib/gtl/map_util.h"

#include <map>
#include <set>
#include <string>

#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/types.h"

namespace tsl {

TEST(MapUtil, Find) {
  typedef std::map<string, string> Map;
  Map m;

  // Check that I can use a type that's implicitly convertible to the
  // key or value type, such as const char* -> string.
  EXPECT_EQ("", gtl::FindWithDefault(m, "foo", ""));
  m["foo"] = "bar";
  EXPECT_EQ("bar", gtl::FindWithDefault(m, "foo", ""));
  EXPECT_EQ("bar", *gtl::FindOrNull(m, "foo"));
  EXPECT_TRUE(m.count("foo") > 0);
  EXPECT_EQ(m["foo"], "bar");
}

TEST(MapUtil, LookupOrInsert) {
  typedef std::map<string, string> Map;
  Map m;

  // Check that I can use a type that's implicitly convertible to the
  // key or value type, such as const char* -> string.
  EXPECT_EQ("xyz", gtl::LookupOrInsert(&m, "foo", "xyz"));
  EXPECT_EQ("xyz", gtl::LookupOrInsert(&m, "foo", "abc"));
}

TEST(MapUtil, InsertIfNotPresent) {
  // Set operations
  typedef std::set<int> Set;
  Set s;
  EXPECT_TRUE(gtl::InsertIfNotPresent(&s, 0));
  EXPECT_EQ(s.count(0), 1);
  EXPECT_FALSE(gtl::InsertIfNotPresent(&s, 0));
  EXPECT_EQ(s.count(0), 1);
}

}  // namespace tsl
