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

#include "tsl/platform/path.h"

#include <string>

#include "tsl/platform/env.h"
#include "tsl/platform/stringpiece.h"
#include "tsl/platform/test.h"

namespace tsl {
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
    EXPECT_EQ(scheme, s);                          \
    EXPECT_EQ(host, h);                            \
    EXPECT_EQ(path, p);                            \
    EXPECT_EQ(uri, CreateURI(scheme, host, path)); \
    EXPECT_LE(u.begin(), s.begin());               \
    EXPECT_GE(u.end(), s.begin());                 \
    EXPECT_LE(u.begin(), s.end());                 \
    EXPECT_GE(u.end(), s.end());                   \
    EXPECT_LE(u.begin(), h.begin());               \
    EXPECT_GE(u.end(), h.begin());                 \
    EXPECT_LE(u.begin(), h.end());                 \
    EXPECT_GE(u.end(), h.end());                   \
    if (p.empty()) {                               \
      EXPECT_EQ(path, "");                         \
    } else {                                       \
      EXPECT_LE(u.begin(), p.begin());             \
      EXPECT_GE(u.end(), p.begin());               \
      EXPECT_LE(u.begin(), p.end());               \
      EXPECT_GE(u.end(), p.end());                 \
    }                                              \
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

TEST(PathTest, CommonPathPrefix) {
  EXPECT_EQ(CommonPathPrefix({"/alpha/beta/c", "/alpha/beta/g"}),
            "/alpha/beta/");
  EXPECT_EQ(CommonPathPrefix({"/a/b/c", "/a/beta/gamma"}), "/a/");
  EXPECT_EQ(CommonPathPrefix({}), "");
  EXPECT_EQ(CommonPathPrefix({"/a/b/c", "", "/a/b/"}), "");
  EXPECT_EQ(CommonPathPrefix({"alpha", "alphabeta"}), "");
}

TEST(PathTest, GetTestWorkspaceDir) {
  constexpr absl::string_view kOriginalValue = "original value";
  std::string dir;

  dir = kOriginalValue;
  tsl::setenv("TEST_SRCDIR", "/repo/src", /*overwrite=*/true);
  tsl::setenv("TEST_WORKSPACE", "my/workspace", /*overwrite=*/true);
  EXPECT_TRUE(GetTestWorkspaceDir(&dir));
  EXPECT_EQ(dir, "/repo/src/my/workspace");
  EXPECT_TRUE(GetTestWorkspaceDir(nullptr));

  dir = kOriginalValue;
  tsl::unsetenv("TEST_SRCDIR");
  tsl::setenv("TEST_WORKSPACE", "my/workspace", /*overwrite=*/true);
  EXPECT_FALSE(GetTestWorkspaceDir(&dir));
  EXPECT_EQ(dir, kOriginalValue);
  EXPECT_FALSE(GetTestWorkspaceDir(nullptr));

  dir = kOriginalValue;
  tsl::setenv("TEST_SRCDIR", "/repo/src", /*overwrite=*/true);
  tsl::unsetenv("TEST_WORKSPACE");
  EXPECT_FALSE(GetTestWorkspaceDir(&dir));
  EXPECT_EQ(dir, kOriginalValue);
  EXPECT_FALSE(GetTestWorkspaceDir(nullptr));

  dir = kOriginalValue;
  tsl::unsetenv("TEST_SRCDIR");
  tsl::unsetenv("TEST_WORKSPACE");
  EXPECT_FALSE(GetTestWorkspaceDir(&dir));
  EXPECT_EQ(dir, kOriginalValue);
  EXPECT_FALSE(GetTestWorkspaceDir(nullptr));
}

TEST(PathTest, GetTestUndeclaredOutputsDir) {
  constexpr absl::string_view kOriginalValue = "original value";
  std::string dir;

  dir = kOriginalValue;
  tsl::setenv("TEST_UNDECLARED_OUTPUTS_DIR", "/test/outputs",
              /*overwrite=*/true);
  EXPECT_TRUE(GetTestUndeclaredOutputsDir(&dir));
  EXPECT_EQ(dir, "/test/outputs");
  EXPECT_TRUE(GetTestUndeclaredOutputsDir(nullptr));

  dir = kOriginalValue;
  tsl::unsetenv("TEST_UNDECLARED_OUTPUTS_DIR");
  EXPECT_FALSE(GetTestUndeclaredOutputsDir(&dir));
  EXPECT_EQ(dir, kOriginalValue);
  EXPECT_FALSE(GetTestUndeclaredOutputsDir(nullptr));
}

TEST(PathTest, ResolveTestPrefixesKeepsThePathUnchanged) {
  constexpr absl::string_view kOriginalValue = "original value";
  std::string resolved_path;

  resolved_path = kOriginalValue;
  EXPECT_TRUE(ResolveTestPrefixes("", resolved_path));
  EXPECT_EQ(resolved_path, "");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(ResolveTestPrefixes("/", resolved_path));
  EXPECT_EQ(resolved_path, "/");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(ResolveTestPrefixes("alpha/beta", resolved_path));
  EXPECT_EQ(resolved_path, "alpha/beta");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(ResolveTestPrefixes("/alpha/beta", resolved_path));
  EXPECT_EQ(resolved_path, "/alpha/beta");
}

TEST(PathTest, ResolveTestPrefixesCanResolveTestWorkspace) {
  constexpr absl::string_view kOriginalValue = "original value";
  std::string resolved_path;

  tsl::setenv("TEST_SRCDIR", "/repo/src", /*overwrite=*/true);
  tsl::setenv("TEST_WORKSPACE", "my/workspace", /*overwrite=*/true);

  resolved_path = kOriginalValue;
  EXPECT_TRUE(ResolveTestPrefixes("TEST_WORKSPACE", resolved_path));
  EXPECT_EQ(resolved_path, "/repo/src/my/workspace");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(ResolveTestPrefixes("TEST_WORKSPACE/", resolved_path));
  EXPECT_EQ(resolved_path, "/repo/src/my/workspace/");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(ResolveTestPrefixes("TEST_WORKSPACE/a/b", resolved_path));
  EXPECT_EQ(resolved_path, "/repo/src/my/workspace/a/b");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(ResolveTestPrefixes("TEST_WORKSPACEE", resolved_path));
  EXPECT_EQ(resolved_path, "TEST_WORKSPACEE");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(ResolveTestPrefixes("/TEST_WORKSPACE", resolved_path));
  EXPECT_EQ(resolved_path, "/TEST_WORKSPACE");
}

TEST(PathTest, ResolveTestPrefixesCannotResolveTestWorkspace) {
  constexpr absl::string_view kOriginalValue = "original value";
  std::string resolved_path;

  tsl::unsetenv("TEST_SRCDIR");
  tsl::unsetenv("TEST_WORKSPACE");

  resolved_path = kOriginalValue;
  EXPECT_FALSE(ResolveTestPrefixes("TEST_WORKSPACE", resolved_path));
  EXPECT_EQ(resolved_path, kOriginalValue);
}

TEST(PathTest, ResolveTestPrefixesCanResolveTestUndeclaredOutputsDir) {
  constexpr absl::string_view kOriginalValue = "original value";
  std::string resolved_path;

  tsl::setenv("TEST_UNDECLARED_OUTPUTS_DIR", "/test/outputs",
              /*overwrite=*/true);

  resolved_path = kOriginalValue;
  EXPECT_TRUE(
      ResolveTestPrefixes("TEST_UNDECLARED_OUTPUTS_DIR", resolved_path));
  EXPECT_EQ(resolved_path, "/test/outputs");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(
      ResolveTestPrefixes("TEST_UNDECLARED_OUTPUTS_DIR/", resolved_path));
  EXPECT_EQ(resolved_path, "/test/outputs/");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(
      ResolveTestPrefixes("TEST_UNDECLARED_OUTPUTS_DIR/a/b", resolved_path));
  EXPECT_EQ(resolved_path, "/test/outputs/a/b");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(
      ResolveTestPrefixes("TEST_UNDECLARED_OUTPUTS_DIRR", resolved_path));
  EXPECT_EQ(resolved_path, "TEST_UNDECLARED_OUTPUTS_DIRR");

  resolved_path = kOriginalValue;
  EXPECT_TRUE(
      ResolveTestPrefixes("/TEST_UNDECLARED_OUTPUTS_DIR", resolved_path));
  EXPECT_EQ(resolved_path, "/TEST_UNDECLARED_OUTPUTS_DIR");
}

TEST(PathTest, ResolveTestPrefixesCannotResolveTestUndeclaredOutputsDir) {
  constexpr absl::string_view kOriginalValue = "original value";
  std::string resolved_path;

  tsl::unsetenv("TEST_UNDECLARED_OUTPUTS_DIR");

  resolved_path = kOriginalValue;
  EXPECT_FALSE(
      ResolveTestPrefixes("TEST_UNDECLARED_OUTPUTS_DIR", resolved_path));
  EXPECT_EQ(resolved_path, kOriginalValue);
}

}  // namespace io
}  // namespace tsl
