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
  EXPECT_EQ("/foo/bar/baz/blah", JoinPath("/foo", "bar", "baz", "blah"));
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
  EXPECT_EQ("hdfs://127.0.0.1:9000/",
            Dirname("hdfs://127.0.0.1:9000/train.csv.tfrecords"));
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

TEST(PathTest, CleanPath) {
  EXPECT_EQ(".", CleanPath(""));
  EXPECT_EQ("x", CleanPath("x"));
  EXPECT_EQ("/a/b/c/d", CleanPath("/a/b/c/d"));
  EXPECT_EQ("/a/b/c/d/*", CleanPath("/a/b/c/d/*"));
  EXPECT_EQ("/a/b/c/d", CleanPath("/a/b/c/d/"));
  EXPECT_EQ("/a/b", CleanPath("/a//b"));
  EXPECT_EQ("/a/b", CleanPath("//a//b/"));
  EXPECT_EQ("/", CleanPath("/.."));
  EXPECT_EQ("/", CleanPath("/././././"));
  EXPECT_EQ("/a", CleanPath("/a/b/.."));
  EXPECT_EQ("/", CleanPath("/a/b/../../.."));
  EXPECT_EQ("/", CleanPath("//a//b/..////../..//"));
  EXPECT_EQ("/x", CleanPath("//a//../x//"));
  EXPECT_EQ("x", CleanPath("x"));
  EXPECT_EQ("../../a/c", CleanPath("../../a/b/../c"));
  EXPECT_EQ("../..", CleanPath("../../a/b/../c/../.."));
  EXPECT_EQ("../../bar", CleanPath("foo/../../../bar"));
}

#define EXPECT_PARSE_URI(uri, scheme, host, path)  \
  do {                                             \
    StringPiece u(uri);                            \
    StringPiece s, h, p;                           \
    ParseURI(u, &s, &h, &p);                       \
    EXPECT_EQ(scheme, s.ToString());               \
    EXPECT_EQ(host, h.ToString());                 \
    EXPECT_EQ(path, p.ToString());                 \
    EXPECT_EQ(uri, CreateURI(scheme, host, path)); \
    EXPECT_LE(u.begin(), s.begin());               \
    EXPECT_GE(u.end(), s.begin());                 \
    EXPECT_LE(u.begin(), s.end());                 \
    EXPECT_GE(u.end(), s.end());                   \
    EXPECT_LE(u.begin(), h.begin());               \
    EXPECT_GE(u.end(), h.begin());                 \
    EXPECT_LE(u.begin(), h.end());                 \
    EXPECT_GE(u.end(), h.end());                   \
    EXPECT_LE(u.begin(), p.begin());               \
    EXPECT_GE(u.end(), p.begin());                 \
    EXPECT_LE(u.begin(), p.end());                 \
    EXPECT_GE(u.end(), p.end());                   \
  } while (0)

TEST(PathTest, CreateParseURI) {
  EXPECT_PARSE_URI("http://foo", "http", "foo", "");
  EXPECT_PARSE_URI("/encrypted/://foo", "", "", "/encrypted/://foo");
  EXPECT_PARSE_URI("/usr/local/foo", "", "", "/usr/local/foo");
  EXPECT_PARSE_URI("file:///usr/local/foo", "file", "", "/usr/local/foo");
  EXPECT_PARSE_URI("local.file:///usr/local/foo", "local.file", "",
                   "/usr/local/foo");
  EXPECT_PARSE_URI("a-b:///foo", "", "", "a-b:///foo");
  EXPECT_PARSE_URI(":///foo", "", "", ":///foo");
  EXPECT_PARSE_URI("9dfd:///foo", "", "", "9dfd:///foo");
  EXPECT_PARSE_URI("file:", "", "", "file:");
  EXPECT_PARSE_URI("file:/", "", "", "file:/");
  EXPECT_PARSE_URI("hdfs://localhost:8020/path/to/file", "hdfs",
                   "localhost:8020", "/path/to/file");
  EXPECT_PARSE_URI("hdfs://localhost:8020", "hdfs", "localhost:8020", "");
  EXPECT_PARSE_URI("hdfs://localhost:8020/", "hdfs", "localhost:8020", "/");
}
#undef EXPECT_PARSE_URI

}  // namespace io
}  // namespace tensorflow
