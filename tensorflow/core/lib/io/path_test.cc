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

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace io {

TEST(PathTest, JoinPath) {
  EXPECT_EQ("/foo/bar", JoinPath("/foo", "bar"));
  EXPECT_EQ("foo/bar", JoinPath("foo", "bar"));
  EXPECT_EQ("foo/bar", JoinPath("foo", "/bar"));
  EXPECT_EQ("/foo/bar", JoinPath("/foo", "/bar"));

  EXPECT_EQ("/bar", JoinPath("", "/bar"));
  EXPECT_EQ("bar", JoinPath("", "bar"));
  EXPECT_EQ("/foo", JoinPath("/foo", ""));

  EXPECT_EQ("/foo/bar/baz/blah/blink/biz",
            JoinPath("/foo/bar/baz/", "/blah/blink/biz"));
}

TEST(PathTest, IsAbsolutePath) {
  EXPECT_FALSE(IsAbsolutePath(""));
  EXPECT_FALSE(IsAbsolutePath("../foo"));
  EXPECT_FALSE(IsAbsolutePath("foo"));
  EXPECT_FALSE(IsAbsolutePath("./foo"));
  EXPECT_FALSE(IsAbsolutePath("foo/bar/baz/"));
  EXPECT_TRUE(IsAbsolutePath("/foo"));
  EXPECT_TRUE(IsAbsolutePath("/foo/bar/../baz"));
}

TEST(PathTest, Dirname) {
  EXPECT_EQ("/hello", Dirname("/hello/"));
  EXPECT_EQ("/", Dirname("/hello"));
  EXPECT_EQ("hello", Dirname("hello/world"));
  EXPECT_EQ("hello", Dirname("hello/"));
  EXPECT_EQ("", Dirname("world"));
  EXPECT_EQ("/", Dirname("/"));
  EXPECT_EQ("", Dirname(""));
}

TEST(PathTest, Basename) {
  EXPECT_EQ("", Basename("/hello/"));
  EXPECT_EQ("hello", Basename("/hello"));
  EXPECT_EQ("world", Basename("hello/world"));
  EXPECT_EQ("", Basename("hello/"));
  EXPECT_EQ("world", Basename("world"));
  EXPECT_EQ("", Basename("/"));
  EXPECT_EQ("", Basename(""));
}

TEST(PathTest, Extension) {
  EXPECT_EQ("gif", Extension("foo.gif"));
  EXPECT_EQ("", Extension("foo."));
  EXPECT_EQ("", Extension(""));
  EXPECT_EQ("", Extension("/"));
  EXPECT_EQ("", Extension("foo"));
  EXPECT_EQ("", Extension("foo/"));
  EXPECT_EQ("gif", Extension("/a/path/to/foo.gif"));
  EXPECT_EQ("html", Extension("/a/path.bar/to/foo.html"));
  EXPECT_EQ("", Extension("/a/path.bar/to/foo"));
  EXPECT_EQ("baz", Extension("/a/path.bar/to/foo.bar.baz"));
}

}  // namespace io
}  // namespace tensorflow
