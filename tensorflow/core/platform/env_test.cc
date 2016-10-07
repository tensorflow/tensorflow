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

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

string CreateTestFile(Env* env, const string& filename, int length) {
  string input(length, 0);
  for (int i = 0; i < length; i++) input[i] = i;
  WriteStringToFile(env, filename, input);
  return input;
}

GraphDef CreateTestProto() {
  GraphDef g;
  NodeDef* node = g.add_node();
  node->set_name("name1");
  node->set_op("op1");
  node = g.add_node();
  node->set_name("name2");
  node->set_op("op2");
  return g;
}

}  // namespace

string BaseDir() { return io::JoinPath(testing::TmpDir(), "base_dir"); }

class DefaultEnvTest : public ::testing::Test {
 protected:
  void SetUp() override { env_->CreateDir(BaseDir()); }

  void TearDown() override {
    int64 undeleted_files, undeleted_dirs;
    env_->DeleteRecursively(BaseDir(), &undeleted_files, &undeleted_dirs);
  }

  Env* env_ = Env::Default();
};

TEST_F(DefaultEnvTest, ReadFileToString) {
  for (const int length : {0, 1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1}) {
    const string filename = strings::StrCat(BaseDir(), "/bar/..//file", length);

    // Write a file with the given length
    const string input = CreateTestFile(env_, filename, length);

    // Read the file back and check equality
    string output;
    TF_EXPECT_OK(ReadFileToString(env_, filename, &output));
    EXPECT_EQ(length, output.size());
    EXPECT_EQ(input, output);

    // Obtain stats.
    FileStatistics stat;
    TF_EXPECT_OK(env_->Stat(filename, &stat));
    EXPECT_EQ(length, stat.length);
    EXPECT_FALSE(stat.is_directory);
  }
}

TEST_F(DefaultEnvTest, ReadWriteBinaryProto) {
  const GraphDef proto = CreateTestProto();
  const string filename = strings::StrCat(BaseDir(), "binary_proto");

  // Write the binary proto
  TF_EXPECT_OK(WriteBinaryProto(env_, filename, proto));

  // Read the binary proto back in and make sure it's the same.
  GraphDef result;
  TF_EXPECT_OK(ReadBinaryProto(env_, filename, &result));
  EXPECT_EQ(result.DebugString(), proto.DebugString());
}

TEST_F(DefaultEnvTest, ReadWriteTextProto) {
  const GraphDef proto = CreateTestProto();
  const string filename = strings::StrCat(BaseDir(), "text_proto");

  // Write the text proto
  string as_text;
  EXPECT_TRUE(protobuf::TextFormat::PrintToString(proto, &as_text));
  TF_EXPECT_OK(WriteStringToFile(env_, filename, as_text));

  // Read the text proto back in and make sure it's the same.
  GraphDef result;
  TF_EXPECT_OK(ReadTextProto(env_, filename, &result));
  EXPECT_EQ(result.DebugString(), proto.DebugString());
}

TEST_F(DefaultEnvTest, FileToReadonlyMemoryRegion) {
  for (const int length : {1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1}) {
    const string filename =
        io::JoinPath(BaseDir(), strings::StrCat("file", length));

    // Write a file with the given length
    const string input = CreateTestFile(env_, filename, length);

    // Create the region.
    std::unique_ptr<ReadOnlyMemoryRegion> region;
    TF_EXPECT_OK(env_->NewReadOnlyMemoryRegionFromFile(filename, &region));
    ASSERT_NE(region, nullptr);
    EXPECT_EQ(length, region->length());
    EXPECT_EQ(input, string(reinterpret_cast<const char*>(region->data()),
                            region->length()));
    FileStatistics stat;
    TF_EXPECT_OK(env_->Stat(filename, &stat));
    EXPECT_EQ(length, stat.length);
    EXPECT_FALSE(stat.is_directory);
  }
}

TEST_F(DefaultEnvTest, DeleteRecursively) {
  // Build a directory structure rooted at root_dir.
  // root_dir -> dirs: child_dir1, child_dir2; files: root_file1, root_file2
  // child_dir1 -> files: child1_file1
  // child_dir2 -> empty
  const string parent_dir = io::JoinPath(BaseDir(), "root_dir");
  const string child_dir1 = io::JoinPath(parent_dir, "child_dir1");
  const string child_dir2 = io::JoinPath(parent_dir, "child_dir2");
  TF_EXPECT_OK(env_->CreateDir(parent_dir));
  const string root_file1 = io::JoinPath(parent_dir, "root_file1");
  const string root_file2 = io::JoinPath(parent_dir, "root_file2");
  const string root_file3 = io::JoinPath(parent_dir, ".root_file3");
  CreateTestFile(env_, root_file1, 100);
  CreateTestFile(env_, root_file2, 100);
  CreateTestFile(env_, root_file3, 100);
  TF_EXPECT_OK(env_->CreateDir(child_dir1));
  const string child1_file1 = io::JoinPath(child_dir1, "child1_file1");
  CreateTestFile(env_, child1_file1, 100);
  TF_EXPECT_OK(env_->CreateDir(child_dir2));

  int64 undeleted_files, undeleted_dirs;
  TF_EXPECT_OK(
      env_->DeleteRecursively(parent_dir, &undeleted_files, &undeleted_dirs));
  EXPECT_EQ(0, undeleted_files);
  EXPECT_EQ(0, undeleted_dirs);
  EXPECT_FALSE(env_->FileExists(root_file1));
  EXPECT_FALSE(env_->FileExists(root_file2));
  EXPECT_FALSE(env_->FileExists(root_file3));
  EXPECT_FALSE(env_->FileExists(child1_file1));
}

TEST_F(DefaultEnvTest, DeleteRecursivelyFail) {
  // Try to delete a non-existent directory.
  const string parent_dir = io::JoinPath(BaseDir(), "root_dir");

  int64 undeleted_files, undeleted_dirs;
  Status s =
      env_->DeleteRecursively(parent_dir, &undeleted_files, &undeleted_dirs);
  EXPECT_EQ("Not found: Directory doesn't exist", s.ToString());
  EXPECT_EQ(0, undeleted_files);
  EXPECT_EQ(1, undeleted_dirs);
}

TEST_F(DefaultEnvTest, RecursivelyCreateDir) {
  const string create_path = io::JoinPath(BaseDir(), "a//b/c/d");
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));  // repeat creation.
  EXPECT_TRUE(env_->FileExists(create_path));
}

TEST_F(DefaultEnvTest, RecursivelyCreateDirEmpty) {
  TF_CHECK_OK(env_->RecursivelyCreateDir(""));
}

TEST_F(DefaultEnvTest, RecursivelyCreateDirSubdirsExist) {
  // First create a/b.
  const string subdir_path = io::JoinPath(BaseDir(), "a/b");
  TF_CHECK_OK(env_->CreateDir(io::JoinPath(BaseDir(), "a")));
  TF_CHECK_OK(env_->CreateDir(subdir_path));
  EXPECT_TRUE(env_->FileExists(subdir_path));

  // Now try to recursively create a/b/c/d/
  const string create_path = io::JoinPath(BaseDir(), "a/b/c/d/");
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));  // repeat creation.
  EXPECT_TRUE(env_->FileExists(create_path));
  EXPECT_TRUE(env_->FileExists(io::JoinPath(BaseDir(), "a/b/c")));
}

TEST_F(DefaultEnvTest, LocalFileSystem) {
  // Test filename with file:// syntax.
  for (const int length : {0, 1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1}) {
    string filename = io::JoinPath(BaseDir(), strings::StrCat("file", length));

    filename = strings::StrCat("file://", filename);

    // Write a file with the given length
    const string input = CreateTestFile(env_, filename, length);

    // Read the file back and check equality
    string output;
    TF_EXPECT_OK(ReadFileToString(env_, filename, &output));
    EXPECT_EQ(length, output.size());
    EXPECT_EQ(input, output);

    FileStatistics stat;
    TF_EXPECT_OK(env_->Stat(filename, &stat));
    EXPECT_EQ(length, stat.length);
    EXPECT_FALSE(stat.is_directory);
  }
}

#define EXPECT_PARSE_URI(uri, scheme, host, path)  \
  do {                                             \
    StringPiece s, h, p;                           \
    ParseURI(uri, &s, &h, &p);                     \
    EXPECT_EQ(scheme, s.ToString());               \
    EXPECT_EQ(host, h.ToString());                 \
    EXPECT_EQ(path, p.ToString());                 \
    EXPECT_EQ(uri, CreateURI(scheme, host, path)); \
  } while (0)

TEST_F(DefaultEnvTest, CreateParseURI) {
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

TEST_F(DefaultEnvTest, SleepForMicroseconds) {
  const int64 start = env_->NowMicros();
  const int64 sleep_time = 1e6 + 5e5;
  env_->SleepForMicroseconds(sleep_time);
  const int64 delta = env_->NowMicros() - start;

  // Subtract 10 from the sleep_time for this check because NowMicros can
  // sometimes give slightly inconsistent values between the start and the
  // finish (e.g. because the two calls run on different CPUs).
  EXPECT_GE(delta, sleep_time - 10);
}

class TmpDirFileSystem : public NullFileSystem {
 public:
  bool FileExists(const string& dir) override {
    StringPiece scheme, host, path;
    ParseURI(dir, &scheme, &host, &path);
    if (path.empty()) return false;
    return Env::Default()->FileExists(io::JoinPath(BaseDir(), path));
  }

  Status CreateDir(const string& dir) override {
    StringPiece scheme, host, path;
    ParseURI(dir, &scheme, &host, &path);
    if (scheme != "tmpdirfs") {
      return errors::FailedPrecondition("scheme must be tmpdirfs");
    }
    if (host != "testhost") {
      return errors::FailedPrecondition("host must be testhost");
    }
    return Env::Default()->CreateDir(io::JoinPath(BaseDir(), path));
  }
};

REGISTER_FILE_SYSTEM("tmpdirfs", TmpDirFileSystem);

TEST_F(DefaultEnvTest, RecursivelyCreateDirWithUri) {
  Env* env = Env::Default();
  const string create_path = "tmpdirfs://testhost/a/b/c/d";
  EXPECT_FALSE(env->FileExists(create_path));
  TF_CHECK_OK(env->RecursivelyCreateDir(create_path));
  TF_CHECK_OK(env->RecursivelyCreateDir(create_path));  // repeat creation.
  EXPECT_TRUE(env->FileExists(create_path));
}

// Creates a new TestEnv that uses Env::Default for all basic ops but
// uses the default implementation for the GetMatchingFiles function instead.
class TestEnv : public EnvWrapper {
 public:
  explicit TestEnv(Env* env) : EnvWrapper(env) {}

  ~TestEnv() override = default;
};

Env* GetTestEnv() {
  static Env* default_env = new TestEnv(Env::Default());
  return default_env;
}

class InterPlanetaryFileSystem : public NullFileSystem {
 public:
  Status IsDirectory(const string& dirname) override {
    if (dirname == "ipfs://solarsystem" ||
        dirname == "ipfs://solarsystem/Earth" ||
        dirname == "ipfs://solarsystem/Jupiter") {
      return Status::OK();
    }
    return Status(tensorflow::error::FAILED_PRECONDITION, "Not a directory");
  }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    std::vector<string> celestial_bodies;
    if (dir == "ipfs://solarsystem") {
      celestial_bodies = {"Mercury",  "Venus",   "Earth",  "Mars",
                          "Jupiter",  "Saturn",  "Uranus", "Neptune",
                          ".PlanetX", "Planet0", "Planet1"};

    } else if (dir == "ipfs://solarsystem/Earth") {
      celestial_bodies = {"Moon"};
    } else if (dir == "ipfs://solarsystem/Jupiter") {
      celestial_bodies = {"Europa", "Io", "Ganymede"};
    }
    result->insert(result->end(), celestial_bodies.begin(),
                   celestial_bodies.end());
    return Status::OK();
  }
};

REGISTER_FILE_SYSTEM_ENV(GetTestEnv(), "ipfs", InterPlanetaryFileSystem);

class TestEnvTest : public ::testing::Test {
 protected:
  void SetUp() override { env_->CreateDir(BaseDir()); }

  void TearDown() override {
    int64 undeleted_files, undeleted_dirs;
    env_->DeleteRecursively(BaseDir(), &undeleted_files, &undeleted_dirs);
  }

  // Returns all the matched entries as a comma separated string removing the
  // common prefix of BaseDir().
  string Match(const string& base_dir, const string& suffix_pattern) {
    std::vector<string> results;
    Status s = env_->GetMatchingPaths(io::JoinPath(base_dir, suffix_pattern),
                                      &results);
    if (!s.ok()) {
      return s.ToString();
    } else {
      std::vector<StringPiece> trimmed_results;
      std::sort(results.begin(), results.end());
      for (const string& result : results) {
        StringPiece trimmed_result(result);
        EXPECT_TRUE(trimmed_result.Consume(base_dir + "/"));
        trimmed_results.push_back(trimmed_result);
      }
      return str_util::Join(trimmed_results, ",");
    }
  }

  Env* env_ = GetTestEnv();
};

TEST_F(TestEnvTest, IPFS) {
  std::vector<string> matched_planets;
  TF_EXPECT_OK(env_->GetChildren("ipfs://solarsystem", &matched_planets));
  std::vector<string> planets = {"Mercury",  "Venus",   "Earth",  "Mars",
                                 "Jupiter",  "Saturn",  "Uranus", "Neptune",
                                 ".PlanetX", "Planet0", "Planet1"};
  int c = 0;
  for (auto p : matched_planets) {
    EXPECT_EQ(p, planets[c++]);
  }
}

TEST_F(TestEnvTest, MatchNonExistentFile) {
  EXPECT_EQ(Match(BaseDir(), "thereisnosuchfile"), "");
}

TEST_F(TestEnvTest, MatchSimple) {
  // Create a few files.
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-00"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-0a"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-01"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-aaa"), ""));

  EXPECT_EQ(Match(BaseDir(), "match-*"),
            "match-00,match-01,match-0a,match-aaa");
  EXPECT_EQ(Match(BaseDir(), "match-0[0-9]"), "match-00,match-01");
  EXPECT_EQ(Match(BaseDir(), "match-?[0-9]"), "match-00,match-01");
  EXPECT_EQ(Match(BaseDir(), "match-?a*"), "match-0a,match-aaa");
  EXPECT_EQ(Match(BaseDir(), "match-??"), "match-00,match-01,match-0a");
}

TEST_F(TestEnvTest, MatchDirectory) {
  // Create some directories.
  TF_EXPECT_OK(
      env_->RecursivelyCreateDir(io::JoinPath(BaseDir(), "match-00/abc")));
  TF_EXPECT_OK(
      env_->RecursivelyCreateDir(io::JoinPath(BaseDir(), "match-0a/abc")));
  TF_EXPECT_OK(
      env_->RecursivelyCreateDir(io::JoinPath(BaseDir(), "match-01/abc")));
  TF_EXPECT_OK(
      env_->RecursivelyCreateDir(io::JoinPath(BaseDir(), "match-aaa/abc")));

  // Create a few files.
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-00/abc/x"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-0a/abc/x"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-01/abc/x"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-aaa/abc/x"), ""));

  EXPECT_EQ(Match(BaseDir(), "match-*/abc/x"),
            "match-00/abc/x,match-01/abc/x,match-0a/abc/x,match-aaa/abc/x");
  EXPECT_EQ(Match(BaseDir(), "match-0[0-9]/abc/x"),
            "match-00/abc/x,match-01/abc/x");
  EXPECT_EQ(Match(BaseDir(), "match-?[0-9]/abc/x"),
            "match-00/abc/x,match-01/abc/x");
  EXPECT_EQ(Match(BaseDir(), "match-?a*/abc/x"),
            "match-0a/abc/x,match-aaa/abc/x");
  EXPECT_EQ(Match(BaseDir(), "match-?[^a]/abc/x"),
            "match-00/abc/x,match-01/abc/x");
}

TEST_F(TestEnvTest, MatchMultipleWildcards) {
  // Create some directories.
  TF_EXPECT_OK(
      env_->RecursivelyCreateDir(io::JoinPath(BaseDir(), "match-00/abc")));
  TF_EXPECT_OK(
      env_->RecursivelyCreateDir(io::JoinPath(BaseDir(), "match-01/abc")));
  TF_EXPECT_OK(
      env_->RecursivelyCreateDir(io::JoinPath(BaseDir(), "match-02/abc")));

  // Create a few files.
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-00/abc/00"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-00/abc/01"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-00/abc/09"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-01/abc/00"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-01/abc/04"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-01/abc/10"), ""));
  TF_EXPECT_OK(
      WriteStringToFile(env_, io::JoinPath(BaseDir(), "match-02/abc/00"), ""));

  EXPECT_EQ(Match(BaseDir(), "match-0[0-1]/abc/0[0-8]"),
            "match-00/abc/00,match-00/abc/01,match-01/abc/00,match-01/abc/04");
}

}  // namespace tensorflow
