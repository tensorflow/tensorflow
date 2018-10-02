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
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/null_file_system.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

string CreateTestFile(Env* env, const string& filename, int length) {
  string input(length, 0);
  for (int i = 0; i < length; i++) input[i] = i;
  TF_CHECK_OK(WriteStringToFile(env, filename, input));
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
  void SetUp() override { TF_CHECK_OK(env_->CreateDir(BaseDir())); }

  void TearDown() override {
    int64 undeleted_files, undeleted_dirs;
    TF_CHECK_OK(
        env_->DeleteRecursively(BaseDir(), &undeleted_files, &undeleted_dirs));
  }

  Env* env_ = Env::Default();
};

TEST_F(DefaultEnvTest, IncompleteReadOutOfRange) {
  const string filename = io::JoinPath(BaseDir(), "out_of_range");
  const string input = CreateTestFile(env_, filename, 2);
  std::unique_ptr<RandomAccessFile> f;
  TF_EXPECT_OK(env_->NewRandomAccessFile(filename, &f));

  // Reading past EOF should give an OUT_OF_RANGE error
  StringPiece result;
  char scratch[3];
  EXPECT_EQ(error::OUT_OF_RANGE, f->Read(0, 3, &result, scratch).code());
  EXPECT_EQ(input, result);

  // Exact read to EOF works.
  TF_EXPECT_OK(f->Read(0, 2, &result, scratch));
  EXPECT_EQ(input, result);
}

TEST_F(DefaultEnvTest, ReadFileToString) {
  for (const int length : {0, 1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1, (256 << 20) + 100}) {
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
  EXPECT_EQ(error::Code::NOT_FOUND, env_->FileExists(root_file1).code());
  EXPECT_EQ(error::Code::NOT_FOUND, env_->FileExists(root_file2).code());
  EXPECT_EQ(error::Code::NOT_FOUND, env_->FileExists(root_file3).code());
  EXPECT_EQ(error::Code::NOT_FOUND, env_->FileExists(child1_file1).code());
}

TEST_F(DefaultEnvTest, DeleteRecursivelyFail) {
  // Try to delete a non-existent directory.
  const string parent_dir = io::JoinPath(BaseDir(), "root_dir");

  int64 undeleted_files, undeleted_dirs;
  Status s =
      env_->DeleteRecursively(parent_dir, &undeleted_files, &undeleted_dirs);
  EXPECT_EQ(error::Code::NOT_FOUND, s.code());
  EXPECT_EQ(0, undeleted_files);
  EXPECT_EQ(1, undeleted_dirs);
}

TEST_F(DefaultEnvTest, RecursivelyCreateDir) {
  const string create_path = io::JoinPath(BaseDir(), "a//b/c/d");
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));  // repeat creation.
  TF_EXPECT_OK(env_->FileExists(create_path));
}

TEST_F(DefaultEnvTest, RecursivelyCreateDirEmpty) {
  TF_CHECK_OK(env_->RecursivelyCreateDir(""));
}

TEST_F(DefaultEnvTest, RecursivelyCreateDirSubdirsExist) {
  // First create a/b.
  const string subdir_path = io::JoinPath(BaseDir(), "a/b");
  TF_CHECK_OK(env_->CreateDir(io::JoinPath(BaseDir(), "a")));
  TF_CHECK_OK(env_->CreateDir(subdir_path));
  TF_EXPECT_OK(env_->FileExists(subdir_path));

  // Now try to recursively create a/b/c/d/
  const string create_path = io::JoinPath(BaseDir(), "a/b/c/d/");
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));  // repeat creation.
  TF_EXPECT_OK(env_->FileExists(create_path));
  TF_EXPECT_OK(env_->FileExists(io::JoinPath(BaseDir(), "a/b/c")));
}

TEST_F(DefaultEnvTest, LocalFileSystem) {
  // Test filename with file:// syntax.
  int expected_num_files = 0;
  std::vector<string> matching_paths;
  for (const int length : {0, 1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1}) {
    string filename = io::JoinPath(BaseDir(), strings::StrCat("len", length));

    filename = strings::StrCat("file://", filename);

    // Write a file with the given length
    const string input = CreateTestFile(env_, filename, length);
    ++expected_num_files;

    // Ensure that GetMatchingPaths works as intended.
    TF_EXPECT_OK(env_->GetMatchingPaths(
        // Try it with the "file://" URI scheme.
        strings::StrCat("file://", io::JoinPath(BaseDir(), "l*")),
        &matching_paths));
    EXPECT_EQ(expected_num_files, matching_paths.size());
    TF_EXPECT_OK(env_->GetMatchingPaths(
        // Try it without any URI scheme.
        io::JoinPath(BaseDir(), "l*"), &matching_paths));
    EXPECT_EQ(expected_num_files, matching_paths.size());

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

TEST_F(DefaultEnvTest, SleepForMicroseconds) {
  const int64 start = env_->NowMicros();
  const int64 sleep_time = 1e6 + 5e5;
  env_->SleepForMicroseconds(sleep_time);
  const int64 delta = env_->NowMicros() - start;

  // Subtract 200 from the sleep_time for this check because NowMicros can
  // sometimes give slightly inconsistent values between the start and the
  // finish (e.g. because the two calls run on different CPUs).
  EXPECT_GE(delta, sleep_time - 200);
}

class TmpDirFileSystem : public NullFileSystem {
 public:
  Status FileExists(const string& dir) override {
    StringPiece scheme, host, path;
    io::ParseURI(dir, &scheme, &host, &path);
    if (path.empty()) return errors::NotFound(dir, " not found");
    // The special "flushed" file exists only if the filesystem's caches have
    // been flushed.
    if (path == "/flushed") {
      if (flushed_) {
        return Status::OK();
      } else {
        return errors::NotFound("FlushCaches() not called yet");
      }
    }
    return Env::Default()->FileExists(io::JoinPath(BaseDir(), path));
  }

  Status CreateDir(const string& dir) override {
    StringPiece scheme, host, path;
    io::ParseURI(dir, &scheme, &host, &path);
    if (scheme != "tmpdirfs") {
      return errors::FailedPrecondition("scheme must be tmpdirfs");
    }
    if (host != "testhost") {
      return errors::FailedPrecondition("host must be testhost");
    }
    return Env::Default()->CreateDir(io::JoinPath(BaseDir(), path));
  }

  void FlushCaches() override { flushed_ = true; }

 private:
  bool flushed_ = false;
};

REGISTER_FILE_SYSTEM("tmpdirfs", TmpDirFileSystem);

TEST_F(DefaultEnvTest, FlushFileSystemCaches) {
  Env* env = Env::Default();
  const string flushed = "tmpdirfs://testhost/flushed";
  EXPECT_EQ(error::Code::NOT_FOUND, env->FileExists(flushed).code());
  TF_EXPECT_OK(env->FlushFileSystemCaches());
  TF_EXPECT_OK(env->FileExists(flushed));
}

TEST_F(DefaultEnvTest, RecursivelyCreateDirWithUri) {
  Env* env = Env::Default();
  const string create_path = "tmpdirfs://testhost/a/b/c/d";
  EXPECT_EQ(error::Code::NOT_FOUND, env->FileExists(create_path).code());
  TF_CHECK_OK(env->RecursivelyCreateDir(create_path));
  TF_CHECK_OK(env->RecursivelyCreateDir(create_path));  // repeat creation.
  TF_EXPECT_OK(env->FileExists(create_path));
}

TEST_F(DefaultEnvTest, GetExecutablePath) {
  Env* env = Env::Default();
  TF_EXPECT_OK(env->FileExists(env->GetExecutablePath()));
}

TEST_F(DefaultEnvTest, LocalTempFilename) {
  Env* env = Env::Default();
  string filename;
  EXPECT_TRUE(env->LocalTempFilename(&filename));
  EXPECT_FALSE(env->FileExists(filename).ok());

  // Write something to the temporary file.
  std::unique_ptr<WritableFile> file_to_write;
  TF_CHECK_OK(env->NewWritableFile(filename, &file_to_write));
#if defined(PLATFORM_GOOGLE)
  TF_CHECK_OK(file_to_write->Append("Nu"));
  TF_CHECK_OK(file_to_write->Append(absl::Cord("ll")));
#else
  // TODO(ebrevdo): Remove this version.
  TF_CHECK_OK(file_to_write->Append("Null"));
#endif
  TF_CHECK_OK(file_to_write->Close());
  TF_CHECK_OK(env->FileExists(filename));

  // Read from the temporary file and check content.
  std::unique_ptr<RandomAccessFile> file_to_read;
  TF_CHECK_OK(env->NewRandomAccessFile(filename, &file_to_read));
  StringPiece content;
  char scratch[1024];
  CHECK_EQ(error::OUT_OF_RANGE,
           file_to_read->Read(0 /* offset */, 1024 /* n */, &content, scratch)
               .code());
  EXPECT_EQ("Null", content);

  // Delete the temporary file.
  TF_CHECK_OK(env->DeleteFile(filename));
  EXPECT_FALSE(env->FileExists(filename).ok());
}

TEST_F(DefaultEnvTest, CreateUniqueFileName) {
  Env* env = Env::Default();

  string prefix = "tempfile-prefix-";
  string suffix = ".tmp";
  string filename = prefix;

  EXPECT_TRUE(env->CreateUniqueFileName(&filename, suffix));

  EXPECT_TRUE(str_util::StartsWith(filename, prefix));
  EXPECT_TRUE(str_util::EndsWith(filename, suffix));
}

}  // namespace tensorflow
