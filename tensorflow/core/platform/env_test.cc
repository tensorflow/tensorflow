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

#include "tensorflow/core/platform/env.h"

#include <sys/stat.h>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

struct EnvTest {};

namespace {
string CreateTestFile(Env* env, const string& filename, int length) {
  string input(length, 0);
  for (int i = 0; i < length; i++) input[i] = i;
  WriteStringToFile(env, filename, input);
  return input;
}
}  // namespace

TEST(EnvTest, ReadFileToString) {
  Env* env = Env::Default();
  const string dir = testing::TmpDir();
  for (const int length : {0, 1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1}) {
    const string filename = io::JoinPath(dir, strings::StrCat("file", length));

    // Write a file with the given length
    const string input = CreateTestFile(env, filename, length);

    // Read the file back and check equality
    string output;
    TF_EXPECT_OK(ReadFileToString(env, filename, &output));
    EXPECT_EQ(length, output.size());
    EXPECT_EQ(input, output);

    // Obtain stats.
    FileStatistics stat;
    TF_EXPECT_OK(env->Stat(filename, &stat));
    EXPECT_EQ(length, stat.length);
    EXPECT_TRUE(S_ISREG(stat.mode));
  }
}

TEST(EnvTest, FileToReadonlyMemoryRegion) {
  Env* env = Env::Default();
  const string dir = testing::TmpDir();
  for (const int length : {1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1}) {
    const string filename = io::JoinPath(dir, strings::StrCat("file", length));

    // Write a file with the given length
    const string input = CreateTestFile(env, filename, length);

    // Create the region.
    std::unique_ptr<ReadOnlyMemoryRegion> region;
    TF_EXPECT_OK(env->NewReadOnlyMemoryRegionFromFile(filename, &region));
    ASSERT_NE(region, nullptr);
    EXPECT_EQ(length, region->length());
    EXPECT_EQ(input, string(reinterpret_cast<const char*>(region->data()),
                            region->length()));
    FileStatistics stat;
    TF_EXPECT_OK(env->Stat(filename, &stat));
    EXPECT_EQ(length, stat.length);
    EXPECT_TRUE(S_ISREG(stat.mode));
  }
}

TEST(EnvTest, DeleteRecursively) {
  Env* env = Env::Default();
  // Build a directory structure rooted at root_dir.
  // root_dir -> dirs: child_dir1, child_dir2; files: root_file1, root_file2
  // child_dir1 -> files: child1_file1
  // child_dir2 -> empty
  const string parent_dir = io::JoinPath(testing::TmpDir(), "root_dir");
  const string child_dir1 = io::JoinPath(parent_dir, "child_dir1");
  const string child_dir2 = io::JoinPath(parent_dir, "child_dir2");
  TF_EXPECT_OK(env->CreateDir(parent_dir));
  const string root_file1 = io::JoinPath(parent_dir, "root_file1");
  const string root_file2 = io::JoinPath(parent_dir, "root_file2");
  CreateTestFile(env, root_file1, 100);
  CreateTestFile(env, root_file2, 100);
  TF_EXPECT_OK(env->CreateDir(child_dir1));
  const string child1_file1 = io::JoinPath(child_dir1, "child1_file1");
  CreateTestFile(env, child1_file1, 100);
  TF_EXPECT_OK(env->CreateDir(child_dir2));

  int64 undeleted_files, undeleted_dirs;
  TF_EXPECT_OK(
      env->DeleteRecursively(parent_dir, &undeleted_files, &undeleted_dirs));
  EXPECT_EQ(0, undeleted_files);
  EXPECT_EQ(0, undeleted_dirs);
  EXPECT_FALSE(env->FileExists(io::JoinPath(parent_dir, "root_file1")));
  EXPECT_FALSE(env->FileExists(io::JoinPath(child_dir1, "child1_file1")));
}

TEST(EnvTest, DeleteRecursivelyFail) {
  // Try to delete a non-existent directory.
  Env* env = Env::Default();
  const string parent_dir = io::JoinPath(testing::TmpDir(), "root_dir");

  int64 undeleted_files, undeleted_dirs;
  Status s =
      env->DeleteRecursively(parent_dir, &undeleted_files, &undeleted_dirs);
  EXPECT_EQ("Not found: Directory doesn't exist", s.ToString());
  EXPECT_EQ(0, undeleted_files);
  EXPECT_EQ(1, undeleted_dirs);
}

TEST(EnvTest, LocalFileSystem) {
  // Test filename with file:// syntax.
  Env* env = Env::Default();
  const string dir = testing::TmpDir();
  for (const int length : {0, 1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1}) {
    string filename = io::JoinPath(dir, strings::StrCat("file", length));

    filename = strings::StrCat("file://", filename);

    // Write a file with the given length
    const string input = CreateTestFile(env, filename, length);

    // Read the file back and check equality
    string output;
    TF_EXPECT_OK(ReadFileToString(env, filename, &output));
    EXPECT_EQ(length, output.size());
    EXPECT_EQ(input, output);

    FileStatistics stat;
    TF_EXPECT_OK(env->Stat(filename, &stat));
    EXPECT_EQ(length, stat.length);
    EXPECT_TRUE(S_ISREG(stat.mode));
  }
}

class InterPlanetaryFileSystem : public NullFileSystem {
 public:
  Status GetChildren(const string& dir, std::vector<string>* result) override {
    std::vector<string> Planets = {"Mercury", "Venus",  "Earth",  "Mars",
                                   "Jupiter", "Saturn", "Uranus", "Neptune"};
    result->insert(result->end(), Planets.begin(), Planets.end());
    return Status::OK();
  }
};

REGISTER_FILE_SYSTEM("ipfs", InterPlanetaryFileSystem);

TEST(EnvTest, IPFS) {
  Env* env = Env::Default();
  std::vector<string> planets;
  TF_EXPECT_OK(env->GetChildren("ipfs://solarsystem", &planets));
  int c = 0;
  std::vector<string> Planets = {"Mercury", "Venus",  "Earth",  "Mars",
                                 "Jupiter", "Saturn", "Uranus", "Neptune"};
  for (auto p : Planets) {
    EXPECT_EQ(p, planets[c++]);
  }
}

TEST(EnvTest, GetSchemeForURI) {
  EXPECT_EQ(GetSchemeFromURI("http://foo"), "http");
  EXPECT_EQ(GetSchemeFromURI("/encrypted/://foo"), "");
  EXPECT_EQ(GetSchemeFromURI("/usr/local/foo"), "");
  EXPECT_EQ(GetSchemeFromURI("file:///usr/local/foo"), "file");
  EXPECT_EQ(GetSchemeFromURI("local.file:///usr/local/foo"), "local.file");
  EXPECT_EQ(GetSchemeFromURI("a-b:///foo"), "");
  EXPECT_EQ(GetSchemeFromURI(":///foo"), "");
  EXPECT_EQ(GetSchemeFromURI("9dfd:///foo"), "");
}

TEST(EnvTest, SleepForMicroseconds) {
  Env* env = Env::Default();
  const int64 start = env->NowMicros();
  const int64 sleep_time = 1e6 + 5e5;
  env->SleepForMicroseconds(sleep_time);
  const int64 delta = env->NowMicros() - start;

  // Subtract 10 from the sleep_time for this check because NowMicros can
  // sometimes give slightly inconsistent values between the start and the
  // finish (e.g. because the two calls run on different CPUs).
  EXPECT_GE(delta, sleep_time - 10);
}

}  // namespace tensorflow
