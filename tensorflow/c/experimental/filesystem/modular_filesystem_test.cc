/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/filesystem/modular_filesystem.h"

#include <memory>
#include <random>
#include <string>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/error.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/command_line_flags.h"

#if defined(PLATFORM_WINDOWS)
// Make mkdir resolve to _mkdir to create the test temporary directory.
#include <direct.h>
#define mkdir(name, mode) _mkdir(name)

// Windows defines the following macros to convert foo to fooA or fooW,
// depending on the type of the string argument. We don't use these macros, so
// undefine them here.
#undef LoadLibrary
#undef CopyFile
#undef DeleteFile
#undef TranslateName
#endif  // defined(PLATFORM_WINDOWS)

// The tests defined here test the compliance of filesystems with the API
// defined by `filesystem_interface.h`.
//
// As some filesystems require special setup, these tests are run manually.
//
// Each filesystem implementation can be provided by DSOs, so we provide the
// `--dsos` flag to specify a list of shared objects to be loaded in order.
// If the flag is not used, no shared objects are loaded.
//
// Every filesystem provides support for accessing URIs of form
// `[<scheme>://]<path>` where `<scheme>` is optional (if missing, we are
// accessing local paths). This test suite tests exactly one scheme for each
// invocation. By default, we are testing all schemes available but this can be
// restricted by using `--schemes` to specify a set of schemes to test.
//
// Example invocation:
//  bazel test //tensorflow/c/experimental/filesystem:modular_filesystem_test \\
//  --test_arg=--dso=/path/to/one.so --test_arg=--dso=/path/to/another.so \\
//  --test_arg=--scheme= --test_arg=--scheme=file
//
// Note that to test the local filesystem we use an empty value.

namespace tensorflow {
namespace {

using ::tensorflow::error::Code;

// As we need to test multiple URI schemes we need a parameterized test.
// Furthermore, since each test creates and deletes files, we will use the same
// fixture to create new directories in `SetUp`. Each directory will reside in
// `::testing::TempDir()`, will use a RNG component and the test name. This
// ensures that two consecutive runs are unlikely to clash.
class ModularFileSystemTest : public ::testing::TestWithParam<std::string> {
 public:
  // Initializes `root_dir_` to a unique value made of `::testing::TempDir()`, a
  // static random value unique for all the tests in one invocation, and the
  // current test name.
  //
  // Since the test name contains `/` (due to parameters), this function
  // replaces `/` with `_`.
  //
  // We trade in one extra initialization for readability.
  ModularFileSystemTest() {
    const std::string test_name = tensorflow::str_util::StringReplace(
        ::testing::UnitTest::GetInstance()->current_test_info()->name(), "/",
        "_", /*replace_all=*/true);
    root_dir_ = tensorflow::io::JoinPath(
        ::testing::TempDir(),
        tensorflow::strings::StrCat("tf_fs_", rng_val_, "_", test_name));
    env_ = Env::Default();
  }

  void SetUp() override {
    if (mkdir(root_dir_.c_str(), 0755) != 0) {
      int error_code = errno;
      GTEST_SKIP() << "Cannot create working directory: "
                   << tensorflow::IOError(root_dir_, error_code);
    }
  }

  // Converts path reference to URI reference.
  //
  // If URI scheme is empty, URI reference is `path` relative to current test
  // root directory. Otherwise, we need to add the `<scheme>://` in front of
  // this path.
  //
  // TODO(mihaimaruseac): Note that some filesystem might require a different
  // approach here, for example they might require the root directory path to
  // be in a special format, etc. When we get there, we might decide to move
  // this class to `modular_filesystem_test.h` and extend the instantiation to
  // also take as argument an implementation for this method/a subclass factory
  // (see
  // https://github.com/google/googletest/blob/master/googletest/docs/advanced.md#creating-value-parameterized-abstract-tests)
  std::string GetURIForPath(StringPiece path) {
    const std::string translated_name =
        tensorflow::io::JoinPath(root_dir_, path);
    if (GetParam().empty()) return translated_name;

    return tensorflow::strings::StrCat(GetParam(), "://", translated_name);
  }

  // Converts absolute paths to paths relative to root_dir_.
  StringPiece GetRelativePath(StringPiece absolute_path) {
    return tensorflow::str_util::StripPrefix(absolute_path, root_dir_);
  }

  // Initializes the randomness used to ensure test isolation.
  static void InitializeTestRNG() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribution;
    rng_val_ = distribution(gen);
  }

 protected:
  Env* env_;

 private:
  std::string root_dir_;
  static int rng_val_;
};

int ModularFileSystemTest::rng_val_;

// As some of the implementations might be missing, the tests should still pass
// if the returned `Status` signals the unimplemented state.
bool UnimplementedOrReturnsCode(Status actual_status, Code expected_code) {
  Code actual_code = actual_status.code();
  return (actual_code == Code::UNIMPLEMENTED) || (actual_code == expected_code);
}

TEST_P(ModularFileSystemTest, TestTranslateName) {
  const std::string generic_path = GetURIForPath("some_path");
  FileSystem* fs = nullptr;
  Status s = env_->GetFileSystemForFile(generic_path, &fs);
  if (fs == nullptr || !s.ok())
    GTEST_SKIP() << "No filesystem registered: " << s;

  // First, test some interesting corner cases concerning empty URIs
  if (GetParam().empty()) {
    EXPECT_EQ(fs->TranslateName(""), "");
    EXPECT_EQ(fs->TranslateName("/"), "/");
    EXPECT_EQ(fs->TranslateName("//"), "/");
    // Empty scheme also allows relative paths
    EXPECT_EQ(fs->TranslateName("a_file"), "a_file");
    EXPECT_EQ(fs->TranslateName("a_dir/.."), ".");
  } else {
    EXPECT_EQ(fs->TranslateName(tensorflow::strings::StrCat(GetParam(), "://")),
              "/");
    EXPECT_EQ(
        fs->TranslateName(tensorflow::strings::StrCat(GetParam(), ":///")),
        "/");
    EXPECT_EQ(
        fs->TranslateName(tensorflow::strings::StrCat(GetParam(), ":////")),
        "/");
  }

  // Now test several paths/URIs
  EXPECT_EQ(GetRelativePath(fs->TranslateName(GetURIForPath("a_file"))),
            "/a_file");
  EXPECT_EQ(GetRelativePath(fs->TranslateName(GetURIForPath("a_dir/a_file"))),
            "/a_dir/a_file");
  EXPECT_EQ(GetRelativePath(fs->TranslateName(GetURIForPath("./a_file"))),
            "/a_file");
  EXPECT_EQ(GetRelativePath(fs->TranslateName(
                GetURIForPath("a/convoluted/../path/./to/.//.///a/file"))),
            "/a/path/to/a/file");
}

TEST_P(ModularFileSystemTest, TestCreateFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestCreateFileNonExisting) {
  const std::string filepath = GetURIForPath("dir_not_found/a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestCreateFileExistingDir) {
  const std::string filepath = GetURIForPath("a_file");
  Status status = env_->CreateDir(filepath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  std::unique_ptr<WritableFile> new_file;
  status = env_->NewWritableFile(filepath, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestCreateFilePathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  const std::string new_path = GetURIForPath("a_file/a_file");
  std::unique_ptr<WritableFile> new_file;
  status = env_->NewWritableFile(new_path, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestAppendFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewAppendableFile(filepath, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestAppendFileNonExisting) {
  const std::string filepath = GetURIForPath("dir_not_found/a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewAppendableFile(filepath, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestAppendFileExistingDir) {
  const std::string filepath = GetURIForPath("a_file");
  Status status = env_->CreateDir(filepath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  std::unique_ptr<WritableFile> new_file;
  status = env_->NewAppendableFile(filepath, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestCreateThenAppendFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  std::unique_ptr<WritableFile> same_file;
  status = env_->NewAppendableFile(filepath, &same_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestAppendFilePathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_path = GetURIForPath("a_file/a_file");
  std::unique_ptr<WritableFile> same_file;
  status = env_->NewAppendableFile(new_path, &same_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestReadFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<RandomAccessFile> new_file;
  Status status = env_->NewRandomAccessFile(filepath, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestReadFileNonExisting) {
  const std::string filepath = GetURIForPath("dir_not_found/a_file");
  std::unique_ptr<RandomAccessFile> new_file;
  Status status = env_->NewRandomAccessFile(filepath, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestReadFileExistingDir) {
  const std::string filepath = GetURIForPath("a_file");
  Status status = env_->CreateDir(filepath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  std::unique_ptr<RandomAccessFile> new_file;
  status = env_->NewRandomAccessFile(filepath, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestCreateThenReadFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  std::unique_ptr<RandomAccessFile> same_file;
  status = env_->NewRandomAccessFile(filepath, &same_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestReadFilePathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_path = GetURIForPath("a_file/a_file");
  std::unique_ptr<RandomAccessFile> same_file;
  status = env_->NewRandomAccessFile(new_path, &same_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestCreateMemoryRegion) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<ReadOnlyMemoryRegion> region;
  Status status = env_->NewReadOnlyMemoryRegionFromFile(filepath, &region);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestCreateMemoryRegionNonExisting) {
  const std::string filepath = GetURIForPath("dir_not_found/a_file");
  std::unique_ptr<ReadOnlyMemoryRegion> region;
  Status status = env_->NewReadOnlyMemoryRegionFromFile(filepath, &region);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestCreateMemoryRegionExistingDir) {
  const std::string filepath = GetURIForPath("a_file");
  Status status = env_->CreateDir(filepath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  std::unique_ptr<ReadOnlyMemoryRegion> new_file;
  status = env_->NewReadOnlyMemoryRegionFromFile(filepath, &new_file);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestCreateMemoryRegionFromEmptyFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  std::unique_ptr<ReadOnlyMemoryRegion> region;
  status = env_->NewReadOnlyMemoryRegionFromFile(filepath, &region);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::INVALID_ARGUMENT);
}

TEST_P(ModularFileSystemTest, TestCreateMemoryRegionFromFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string test_data("asdf");
  status = new_file->Append(test_data);
  if (!status.ok()) GTEST_SKIP() << "Append() not supported: " << status;
  status = new_file->Flush();
  if (!status.ok()) GTEST_SKIP() << "Flush() not supported: " << status;
  status = new_file->Close();
  if (!status.ok()) GTEST_SKIP() << "Close() not supported: " << status;

  std::unique_ptr<ReadOnlyMemoryRegion> region;
  status = env_->NewReadOnlyMemoryRegionFromFile(filepath, &region);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok())
    GTEST_SKIP() << "NewReadOnlyMemoryRegionFromFile() not supported: "
                 << status;
  EXPECT_EQ(region->length(), test_data.size());
  EXPECT_STREQ(reinterpret_cast<const char*>(region->data()),
               test_data.c_str());
}

TEST_P(ModularFileSystemTest, TestCreateMemoryRegionFromFilePathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  std::string new_path = GetURIForPath("a_file/a_file");
  std::unique_ptr<ReadOnlyMemoryRegion> region;
  status = env_->NewReadOnlyMemoryRegionFromFile(new_path, &region);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestCreateDir) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestCreateDirNoParent) {
  const std::string dirpath = GetURIForPath("dir_not_found/a_dir");
  Status status = env_->CreateDir(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestCreateDirWhichIsFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = env_->CreateDir(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::ALREADY_EXISTS);
}

TEST_P(ModularFileSystemTest, TestCreateDirTwice) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  status = env_->CreateDir(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::ALREADY_EXISTS);
}

TEST_P(ModularFileSystemTest, TestCreateDirPathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_path = GetURIForPath("a_file/a_dir");
  status = env_->CreateDir(new_path);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestRecursivelyCreateDir) {
  const std::string dirpath = GetURIForPath("a/path/to/a/dir");
  Status status = env_->RecursivelyCreateDir(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestRecursivelyCreateDirInATree) {
  const std::string dirpath = GetURIForPath("a/path/to/a/dir");
  Status status = env_->RecursivelyCreateDir(dirpath);
  if (!status.ok())
    GTEST_SKIP() << "RecursivelyCreateDir() not supported: " << status;

  const std::string new_dirpath = GetURIForPath("a/path/to/a/another/dir");
  status = env_->RecursivelyCreateDir(new_dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestRecursivelyCreateDirWhichIsFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = env_->RecursivelyCreateDir(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestRecursivelyCreateDirTwice) {
  const std::string dirpath = GetURIForPath("a/path/to/a/dir");
  Status status = env_->RecursivelyCreateDir(dirpath);
  if (!status.ok())
    GTEST_SKIP() << "RecursivelyCreateDir() not supported: " << status;

  status = env_->RecursivelyCreateDir(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestRecursivelyCreateDirPathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_path = GetURIForPath("a_file/a_dir");
  status = env_->RecursivelyCreateDir(new_path);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestRecursivelyCreateDirFromNestedDir) {
  const std::string parent_path = GetURIForPath("some/path");
  Status status = env_->RecursivelyCreateDir(parent_path);
  if (!status.ok())
    GTEST_SKIP() << "RecursivelyCreateDir() not supported: " << status;

  const std::string new_dirpath = GetURIForPath("some/path/that/is/extended");
  status = env_->RecursivelyCreateDir(new_dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestRecursivelyCreateDirFromNestedFile) {
  const std::string parent_path = GetURIForPath("some/path");
  Status status = env_->RecursivelyCreateDir(parent_path);
  if (!status.ok())
    GTEST_SKIP() << "RecursivelyCreateDir() not supported: " << status;

  const std::string filepath = GetURIForPath("some/path/to_a_file");
  std::unique_ptr<WritableFile> file;
  status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_dirpath = GetURIForPath("some/path/to_a_file/error");
  status = env_->RecursivelyCreateDir(new_dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestDeleteFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = env_->DeleteFile(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestDeleteFileFromDirectory) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  const std::string filepath = GetURIForPath("a_dir/a_file");
  std::unique_ptr<WritableFile> new_file;
  status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = env_->DeleteFile(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestDeleteFileDoesNotExist) {
  const std::string filepath = GetURIForPath("a_file");
  Status status = env_->DeleteFile(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestDeleteFileWhichIsDirectory) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  status = env_->DeleteFile(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestDeleteFilePathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_path = GetURIForPath("a_file/a_new_file");
  status = env_->DeleteFile(new_path);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestDeleteDirectory) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  status = env_->DeleteDir(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestDeleteDirectoryFromDirectory) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  const std::string target_path = GetURIForPath("a_dir/another_dir");
  EXPECT_EQ(env_->CreateDir(target_path).code(), Code::OK);

  status = env_->DeleteDir(target_path);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestDeleteDirectoryDoesNotExist) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->DeleteDir(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestDeleteDirectoryNotEmpty) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  const std::string filepath = GetURIForPath("a_dir/a_file");
  std::unique_ptr<WritableFile> new_file;
  status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = env_->DeleteDir(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestDeleteDirectoryWhichIsFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = env_->DeleteDir(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestDeleteDirectoryPathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_path = GetURIForPath("a_file/a_dir");
  status = env_->DeleteDir(new_path);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestDeleteRecursivelyEmpty) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  int64 undeleted_files = 0;
  int64 undeleted_dirs = 0;
  status = env_->DeleteRecursively(dirpath, &undeleted_files, &undeleted_dirs);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  EXPECT_EQ(undeleted_files, 0);
  EXPECT_EQ(undeleted_dirs, 0);
}

TEST_P(ModularFileSystemTest, TestDeleteRecursivelyNotEmpty) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  const std::string some_path = GetURIForPath("a_dir/another_dir");
  status = env_->CreateDir(some_path);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  const std::string another_path = GetURIForPath("a_dir/yet_another_dir");
  status = env_->CreateDir(another_path);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  const std::string filepath = GetURIForPath("a_dir/a_file");
  std::unique_ptr<WritableFile> new_file;
  status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  int64 undeleted_files = 0;
  int64 undeleted_dirs = 0;
  status = env_->DeleteRecursively(dirpath, &undeleted_files, &undeleted_dirs);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  EXPECT_EQ(undeleted_files, 0);
  EXPECT_EQ(undeleted_dirs, 0);
}

TEST_P(ModularFileSystemTest, TestDeleteRecursivelyDoesNotExist) {
  const std::string dirpath = GetURIForPath("a_dir");

  int64 undeleted_files = 0;
  int64 undeleted_dirs = 0;
  Status status =
      env_->DeleteRecursively(dirpath, &undeleted_files, &undeleted_dirs);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
  EXPECT_EQ(undeleted_files, 0);
  EXPECT_EQ(undeleted_dirs, 1);
}

TEST_P(ModularFileSystemTest, TestDeleteRecursivelyAFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  int64 undeleted_files = 0;
  int64 undeleted_dirs = 0;
  status = env_->DeleteRecursively(filepath, &undeleted_files, &undeleted_dirs);
  EXPECT_EQ(undeleted_files, 0);
  EXPECT_EQ(undeleted_dirs, 0);
}

TEST_P(ModularFileSystemTest, TestDeleteRecursivelyPathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_path = GetURIForPath("a_file/a_dir");
  int64 undeleted_files, undeleted_dirs;
  status = env_->DeleteRecursively(new_path, &undeleted_files, &undeleted_dirs);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestDeleteRecursivelyANestedDir) {
  const std::string parent_path = GetURIForPath("parent/path");
  Status status = env_->RecursivelyCreateDir(parent_path);
  if (!status.ok())
    GTEST_SKIP() << "RecursivelyCreateDir() not supported: " << status;

  const std::string new_dirpath = GetURIForPath("parent/path/that/is/extended");
  status = env_->RecursivelyCreateDir(new_dirpath);
  if (!status.ok())
    GTEST_SKIP() << "RecursivelyCreateDir() not supported: " << status;

  const std::string path = GetURIForPath("parent/path/that");
  int64 undeleted_files = 0;
  int64 undeleted_dirs = 0;
  status = env_->DeleteRecursively(path, &undeleted_files, &undeleted_dirs);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  EXPECT_EQ(undeleted_files, 0);
  EXPECT_EQ(undeleted_dirs, 0);

  // Parent directory must still exist
  status = env_->FileExists(parent_path);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestDeleteRecursivelyANestedFile) {
  const std::string parent_path = GetURIForPath("some/path");
  Status status = env_->RecursivelyCreateDir(parent_path);
  if (!status.ok())
    GTEST_SKIP() << "RecursivelyCreateDir() not supported: " << status;

  const std::string filepath = GetURIForPath("some/path/to_a_file");
  std::unique_ptr<WritableFile> file;
  status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  int64 undeleted_files = 0;
  int64 undeleted_dirs = 0;
  status = env_->DeleteRecursively(filepath, &undeleted_files, &undeleted_dirs);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  EXPECT_EQ(undeleted_files, 0);
  EXPECT_EQ(undeleted_dirs, 0);

  // Parent directory must still exist
  status = env_->FileExists(parent_path);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestRenameFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_new_file");
  status = env_->RenameFile(filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "RenameFile() not supported: " << status;

  status = env_->FileExists(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
  status = env_->FileExists(new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestRenameFileOverwrite) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_new_file");
  std::unique_ptr<WritableFile> new_file;
  status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = env_->RenameFile(filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "RenameFile() not supported: " << status;

  status = env_->FileExists(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
  status = env_->FileExists(new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestRenameFileSourceNotFound) {
  const std::string filepath = GetURIForPath("a_file");
  const std::string new_filepath = GetURIForPath("a_new_file");
  Status status = env_->RenameFile(filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestRenameFileDestinationParentNotFound) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_dir/a_file");
  status = env_->RenameFile(filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestRenameFileSourceIsDirectory) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_new_file");
  status = env_->RenameFile(dirpath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestRenameFileTargetIsDirectory) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string dirpath = GetURIForPath("a_dir");
  status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  status = env_->RenameFile(filepath, dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestRenameFileSourcePathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string old_filepath = GetURIForPath("a_file/x");
  const std::string new_filepath = GetURIForPath("a_new_file");
  status = env_->RenameFile(old_filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestRenameFileTargetPathIsInvalid) {
  const std::string old_filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> old_file;
  Status status = env_->NewWritableFile(old_filepath, &old_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_file/a_new_file");
  status = env_->RenameFile(old_filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestRenameFileCompareContents) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string test_data("asdf");
  status = file->Append(test_data);
  if (!status.ok()) GTEST_SKIP() << "Append() not supported: " << status;
  status = file->Flush();
  if (!status.ok()) GTEST_SKIP() << "Flush() not supported: " << status;
  status = file->Close();
  if (!status.ok()) GTEST_SKIP() << "Close() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_new_file");
  status = env_->RenameFile(filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "RenameFile() not supported: " << status;

  uint64 size;
  status = env_->GetFileSize(new_filepath, &size);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "GetFileSize() not supported: " << status;
  EXPECT_EQ(size, test_data.size());
}

TEST_P(ModularFileSystemTest, TestCopyFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_new_file");
  status = env_->CopyFile(filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "CopyFile() not supported: " << status;

  status = env_->FileExists(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  status = env_->FileExists(new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestCopyFileOverwrite) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_new_file");
  std::unique_ptr<WritableFile> new_file;
  status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = env_->CopyFile(filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "CopyFile() not supported: " << status;

  status = env_->FileExists(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  status = env_->FileExists(new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestCopyFileSourceNotFound) {
  const std::string filepath = GetURIForPath("a_file");
  const std::string new_filepath = GetURIForPath("a_new_file");
  Status status = env_->CopyFile(filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestCopyFileSourceIsDirectory) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_new_file");
  status = env_->CopyFile(dirpath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestCopyFileTargetIsDirectory) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> new_file;
  Status status = env_->NewWritableFile(filepath, &new_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string dirpath = GetURIForPath("a_dir");
  status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  status = env_->CopyFile(filepath, dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestCopyFileSourcePathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string old_filepath = GetURIForPath("a_file/x");
  const std::string new_filepath = GetURIForPath("a_new_file");
  status = env_->CopyFile(old_filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestCopyFileTargetPathIsInvalid) {
  const std::string old_filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> old_file;
  Status status = env_->NewWritableFile(old_filepath, &old_file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_file/a_new_file");
  status = env_->CopyFile(old_filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestCopyFileCompareContents) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string test_data("asdf");
  status = file->Append(test_data);
  if (!status.ok()) GTEST_SKIP() << "Append() not supported: " << status;
  status = file->Flush();
  if (!status.ok()) GTEST_SKIP() << "Flush() not supported: " << status;
  status = file->Close();
  if (!status.ok()) GTEST_SKIP() << "Close() not supported: " << status;

  const std::string new_filepath = GetURIForPath("a_new_file");
  status = env_->CopyFile(filepath, new_filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "RenameFile() not supported: " << status;

  uint64 size;
  status = env_->GetFileSize(filepath, &size);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "GetFileSize() not supported: " << status;
  EXPECT_EQ(size, test_data.size());

  status = env_->GetFileSize(new_filepath, &size);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "GetFileSize() not supported: " << status;
  EXPECT_EQ(size, test_data.size());
}

TEST_P(ModularFileSystemTest, TestFileExists) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = env_->FileExists(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestFileExistsButIsDirectory) {
  const std::string filepath = GetURIForPath("a_file");
  Status status = env_->CreateDir(filepath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  status = env_->FileExists(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestFileExistsNotFound) {
  const std::string filepath = GetURIForPath("a_file");
  Status status = env_->FileExists(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestFileExistsPathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string target_path = GetURIForPath("a_file/a_new_file");
  status = env_->FileExists(target_path);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestFilesExist) {
  const std::vector<std::string> filenames = {GetURIForPath("a"),
                                              GetURIForPath("b")};
  for (const auto& filename : filenames) {
    std::unique_ptr<WritableFile> file;
    Status status = env_->NewWritableFile(filename, &file);
    if (!status.ok())
      GTEST_SKIP() << "NewWritableFile() not supported: " << status;
  }

  EXPECT_TRUE(env_->FilesExist(filenames, /*status=*/nullptr));

  std::vector<Status> statuses;
  EXPECT_TRUE(env_->FilesExist(filenames, &statuses));
  EXPECT_EQ(statuses.size(), filenames.size());
  for (const auto& status : statuses)
    EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestFilesExistAllFailureModes) {
  // if reordering these, make sure to reorder checks at the end
  const std::vector<std::string> filenames = {
      GetURIForPath("a_dir"),
      GetURIForPath("a_file"),
      GetURIForPath("a_file/a_new_file"),
      GetURIForPath("file_not_found"),
  };

  Status status = env_->CreateDir(filenames[0]);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  std::unique_ptr<WritableFile> file;
  status = env_->NewWritableFile(filenames[1], &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  std::vector<Status> statuses;
  EXPECT_FALSE(env_->FilesExist(filenames, &statuses));
  EXPECT_EQ(statuses.size(), filenames.size());
  EXPECT_PRED2(UnimplementedOrReturnsCode, statuses[0], Code::OK);
  EXPECT_PRED2(UnimplementedOrReturnsCode, statuses[1], Code::OK);
  EXPECT_PRED2(UnimplementedOrReturnsCode, statuses[2],
               Code::FAILED_PRECONDITION);
  EXPECT_PRED2(UnimplementedOrReturnsCode, statuses[3], Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestFilesExistsNoFiles) {
  const std::vector<std::string> filenames = {};
  EXPECT_TRUE(env_->FilesExist(filenames, /*status=*/nullptr));

  std::vector<Status> statuses;
  EXPECT_TRUE(env_->FilesExist(filenames, &statuses));
  EXPECT_TRUE(statuses.empty());
}

TEST_P(ModularFileSystemTest, TestStatEmptyFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  FileStatistics stat;
  status = env_->Stat(filepath, &stat);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Stat() not supported: " << status;
  EXPECT_FALSE(stat.is_directory);
  EXPECT_EQ(stat.length, 0);
}

TEST_P(ModularFileSystemTest, TestStatNonEmptyFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string test_data("asdf");
  status = file->Append(test_data);
  if (!status.ok()) GTEST_SKIP() << "Append() not supported: " << status;
  status = file->Flush();
  if (!status.ok()) GTEST_SKIP() << "Flush() not supported: " << status;
  status = file->Close();
  if (!status.ok()) GTEST_SKIP() << "Close() not supported: " << status;

  FileStatistics stat;
  status = env_->Stat(filepath, &stat);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Stat() not supported: " << status;
  EXPECT_FALSE(stat.is_directory);
  EXPECT_EQ(stat.length, test_data.size());
}

TEST_P(ModularFileSystemTest, TestStatDirectory) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  FileStatistics stat;
  status = env_->Stat(dirpath, &stat);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Stat() not supported: " << status;
  EXPECT_TRUE(stat.is_directory);
}

TEST_P(ModularFileSystemTest, TestStatNotFound) {
  const std::string dirpath = GetURIForPath("a_dir");
  FileStatistics stat;
  Status status = env_->Stat(dirpath, &stat);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestStatPathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string target_path = GetURIForPath("a_file/a_new_file");
  FileStatistics stat;
  status = env_->Stat(target_path, &stat);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestIsDirectory) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  status = env_->IsDirectory(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
}

TEST_P(ModularFileSystemTest, TestIsDirectoryFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = env_->IsDirectory(filepath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestIsDirectoryNotFound) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->IsDirectory(dirpath);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestIsDirectoryPathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string target_path = GetURIForPath("a_file/a_new_file");
  status = env_->IsDirectory(target_path);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestGetFileSizeEmptyFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  uint64 size;
  status = env_->GetFileSize(filepath, &size);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "GetFileSize() not supported: " << status;
  EXPECT_EQ(size, 0);
}

TEST_P(ModularFileSystemTest, TestGetFileSizeNonEmptyFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string test_data("asdf");
  status = file->Append(test_data);
  if (!status.ok()) GTEST_SKIP() << "Append() not supported: " << status;
  status = file->Flush();
  if (!status.ok()) GTEST_SKIP() << "Flush() not supported: " << status;
  status = file->Close();
  if (!status.ok()) GTEST_SKIP() << "Close() not supported: " << status;

  uint64 size;
  status = env_->GetFileSize(filepath, &size);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "GetFileSize() not supported: " << status;
  EXPECT_EQ(size, test_data.size());
}

TEST_P(ModularFileSystemTest, TestGetFileSizeDirectory) {
  const std::string dirpath = GetURIForPath("a_dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  uint64 size;
  status = env_->GetFileSize(dirpath, &size);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestGetFileSizeNotFound) {
  const std::string filepath = GetURIForPath("a_dir");
  uint64 size;
  Status status = env_->GetFileSize(filepath, &size);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestGetFileSizePathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string target_path = GetURIForPath("a_file/a_new_file");
  uint64 size;
  status = env_->GetFileSize(target_path, &size);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestGetChildren) {
  const std::string dirpath = GetURIForPath("dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  // If updating, make sure to update expected_children below.
  const std::vector<std::string> filenames = {
      GetURIForPath("dir/a_file"),
      GetURIForPath("dir/another_file"),
  };
  for (const auto& filename : filenames) {
    std::unique_ptr<WritableFile> file;
    status = env_->NewWritableFile(filename, &file);
    if (!status.ok())
      GTEST_SKIP() << "NewWritableFile() not supported: " << status;
  }

  // If updating, make sure to update expected_children below.
  const std::vector<std::string> dirnames = {
      GetURIForPath("dir/a_dir"),
      GetURIForPath("dir/another_dir"),
  };
  for (const auto& dirname : dirnames) {
    status = env_->CreateDir(dirname);
    if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;
  }

  std::vector<std::string> children;
  status = env_->GetChildren(dirpath, &children);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "GetChildren() not supported: " << status;

  // All entries must show up in the vector.
  // Must contain only the last name in filenames and dirnames.
  const std::vector<std::string> expected_children = {"a_file", "another_file",
                                                      "a_dir", "another_dir"};
  EXPECT_EQ(children.size(), filenames.size() + dirnames.size());
  for (const auto& child : expected_children)
    EXPECT_NE(std::find(children.begin(), children.end(), child),
              children.end());
}

TEST_P(ModularFileSystemTest, TestGetChildrenEmpty) {
  const std::string dirpath = GetURIForPath("dir");
  Status status = env_->CreateDir(dirpath);
  if (!status.ok()) GTEST_SKIP() << "CreateDir() not supported: " << status;

  std::vector<std::string> children;
  status = env_->GetChildren(dirpath, &children);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  EXPECT_EQ(children.size(), 0);
}

TEST_P(ModularFileSystemTest, TestGetChildrenOfFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  std::vector<std::string> children;
  status = env_->GetChildren(filepath, &children);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestGetChildrenPathNotFound) {
  const std::string target_path = GetURIForPath("a_dir");
  std::vector<std::string> children;
  Status status = env_->GetChildren(target_path, &children);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::NOT_FOUND);
}

TEST_P(ModularFileSystemTest, TestGetChildrenPathIsInvalid) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string target_path = GetURIForPath("a_file/a_new_dir");
  std::vector<std::string> children;
  status = env_->GetChildren(target_path, &children);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::FAILED_PRECONDITION);
}

TEST_P(ModularFileSystemTest, TestGetMatchingPaths) {
  const std::vector<std::string> matching_filenames = {
      GetURIForPath("a_file"),
      GetURIForPath("another_file"),
  };
  const std::vector<std::string> other_filenames = {
      GetURIForPath("some_file"),
      GetURIForPath("yet_another_file"),
  };

  for (const auto& filename : matching_filenames) {
    std::unique_ptr<WritableFile> file;
    Status status = env_->NewWritableFile(filename, &file);
    if (!status.ok())
      GTEST_SKIP() << "NewWritableFile() not supported: " << status;
  }

  for (const auto& filename : other_filenames) {
    std::unique_ptr<WritableFile> file;
    Status status = env_->NewWritableFile(filename, &file);
    if (!status.ok())
      GTEST_SKIP() << "NewWritableFile() not supported: " << status;
  }

  std::vector<std::string> results;
  Status status = env_->GetMatchingPaths(GetURIForPath("/a*"), &results);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok())
    GTEST_SKIP() << "GetMatchingPaths() not supported: " << status;
  EXPECT_EQ(results.size(), matching_filenames.size());
  for (const auto& match : matching_filenames)
    EXPECT_NE(std::find(results.begin(), results.end(), match), results.end());
}

TEST_P(ModularFileSystemTest, TestGetMatchingPathsEmptyFileSystem) {
  std::vector<std::string> results;
  Status status = env_->GetMatchingPaths(GetURIForPath("a*"), &results);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  EXPECT_EQ(results.size(), 0);
}

TEST_P(ModularFileSystemTest, TestGetMatchingPathsEmptyPattern) {
  const std::vector<std::string> filenames = {
      GetURIForPath("a_file"),
      GetURIForPath("another_file"),
      GetURIForPath("some_file"),
      GetURIForPath("yet_another_file"),
  };

  for (const auto& filename : filenames) {
    std::unique_ptr<WritableFile> file;
    Status status = env_->NewWritableFile(filename, &file);
    if (!status.ok())
      GTEST_SKIP() << "NewWritableFile() not supported: " << status;
  }

  std::vector<std::string> results;
  Status status = env_->GetMatchingPaths(GetURIForPath(""), &results);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok())
    GTEST_SKIP() << "GetMatchingPaths() not supported: " << status;
  EXPECT_EQ(results.size(), 1);
  EXPECT_NE(std::find(results.begin(), results.end(), GetURIForPath("")),
            results.end());
}

TEST_P(ModularFileSystemTest, TestGetMatchingPathsLiteralMatch) {
  const std::vector<std::string> filenames = {
      GetURIForPath("a_file"),
      GetURIForPath("another_file"),
      GetURIForPath("some_file"),
      GetURIForPath("yet_another_file"),
  };

  for (const auto& filename : filenames) {
    std::unique_ptr<WritableFile> file;
    Status status = env_->NewWritableFile(filename, &file);
    if (!status.ok())
      GTEST_SKIP() << "NewWritableFile() not supported: " << status;
  }

  std::vector<std::string> results;
  Status status = env_->GetMatchingPaths(filenames[0], &results);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok())
    GTEST_SKIP() << "GetMatchingPaths() not supported: " << status;
  EXPECT_EQ(results.size(), 1);
  EXPECT_NE(std::find(results.begin(), results.end(), filenames[0]),
            results.end());
}

TEST_P(ModularFileSystemTest, TestGetMatchingPathsNoMatch) {
  const std::vector<std::string> filenames = {
      GetURIForPath("a_file"),
      GetURIForPath("another_file"),
      GetURIForPath("some_file"),
      GetURIForPath("yet_another_file"),
  };

  for (const auto& filename : filenames) {
    std::unique_ptr<WritableFile> file;
    Status status = env_->NewWritableFile(filename, &file);
    if (!status.ok())
      GTEST_SKIP() << "NewWritableFile() not supported: " << status;
  }

  std::vector<std::string> results;
  Status status = env_->GetMatchingPaths(GetURIForPath("x?y*"), &results);
  if (!status.ok())
    GTEST_SKIP() << "GetMatchingPaths() not supported: " << status;
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  EXPECT_EQ(results.size(), 0);
}

TEST_P(ModularFileSystemTest, TestAppendAndTell) {
  const std::string filename = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filename, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  int64 position;
  status = file->Tell(&position);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Tell() not supported: " << status;
  EXPECT_EQ(position, 0);

  const std::string test_data("asdf");
  status = file->Append(test_data);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Append() not supported: " << status;

  status = file->Tell(&position);
  EXPECT_EQ(status.code(), Code::OK);
  EXPECT_EQ(position, test_data.size());
}

TEST_P(ModularFileSystemTest, TestClose) {
  const std::string filename = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filename, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  status = file->Close();
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Close() not supported: " << status;
}

TEST_P(ModularFileSystemTest, TestRoundTrip) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string test_data("asdf");
  status = file->Append(test_data);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Append() not supported: " << status;

  status = file->Flush();
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Flush() not supported: " << status;

  status = file->Close();
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Close() not supported: " << status;

  std::unique_ptr<RandomAccessFile> read_file;
  status = env_->NewRandomAccessFile(filepath, &read_file);
  if (!status.ok())
    GTEST_SKIP() << "NewRandomAccessFile() not supported: " << status;

  char scratch[64 /* big enough to accommodate test_data */] = {0};
  StringPiece result;
  status = read_file->Read(0, test_data.size(), &result, scratch);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  EXPECT_EQ(test_data, result);
}

TEST_P(ModularFileSystemTest, TestRoundTripWithAppendableFile) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string test_data("asdf");
  status = file->Append(test_data);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Append() not supported: " << status;

  status = file->Flush();
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Flush() not supported: " << status;

  status = file->Close();
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Close() not supported: " << status;

  std::unique_ptr<WritableFile> same_file;
  status = env_->NewAppendableFile(filepath, &same_file);
  if (!status.ok())
    GTEST_SKIP() << "NewAppendableFile() not supported: " << status;

  const std::string more_test_data("qwer");
  EXPECT_EQ(same_file->Append(more_test_data).code(), Code::OK);
  EXPECT_EQ(same_file->Flush().code(), Code::OK);
  EXPECT_EQ(same_file->Close().code(), Code::OK);

  std::unique_ptr<RandomAccessFile> read_file;
  status = env_->NewRandomAccessFile(filepath, &read_file);
  if (!status.ok())
    GTEST_SKIP() << "NewRandomAccessFile() not supported: " << status;

  char scratch[64 /* big enough for test_data and more_test_data */] = {0};
  StringPiece result;
  status = read_file->Read(0, test_data.size() + more_test_data.size(), &result,
                           scratch);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  EXPECT_EQ(test_data + more_test_data, result);
  EXPECT_EQ(
      read_file->Read(test_data.size(), more_test_data.size(), &result, scratch)
          .code(),
      Code::OK);
  EXPECT_EQ(more_test_data, result);
}

TEST_P(ModularFileSystemTest, TestReadOutOfRange) {
  const std::string filepath = GetURIForPath("a_file");
  std::unique_ptr<WritableFile> file;
  Status status = env_->NewWritableFile(filepath, &file);
  if (!status.ok())
    GTEST_SKIP() << "NewWritableFile() not supported: " << status;

  const std::string test_data("asdf");
  status = file->Append(test_data);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Append() not supported: " << status;

  status = file->Flush();
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Flush() not supported: " << status;

  status = file->Close();
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OK);
  if (!status.ok()) GTEST_SKIP() << "Close() not supported: " << status;

  std::unique_ptr<RandomAccessFile> read_file;
  status = env_->NewRandomAccessFile(filepath, &read_file);
  if (!status.ok())
    GTEST_SKIP() << "NewRandomAccessFile() not supported: " << status;

  char scratch[64 /* must be bigger than test_data */] = {0};
  StringPiece result;
  // read at least 1 byte more than test_data
  status = read_file->Read(0, test_data.size() + 1, &result, scratch);
  EXPECT_PRED2(UnimplementedOrReturnsCode, status, Code::OUT_OF_RANGE);
}

// The URI schemes that need to be tested are provided by the user via flags
// (or, if none is supplied, all existing schemes are used). As a scheme can
// become available after a shared object with a filesystem implementation is
// loaded, we can only check for availability after all arguments have been
// parsed.
//
// Furthermore, as `INSTANTIATE_TEST_SUITE_P` needs to be at global level and we
// don't want to have a `std::vector<std::string>` at global level, we use a
// static pointer to such a vector: we construct it via `SchemeVector()` below
// and when tests are instantiated we process it using `GetSchemes()`.
static std::vector<std::string>* SchemeVector() {
  static std::vector<std::string>* schemes = new std::vector<std::string>;
  return schemes;
}

// `INSTANTIATE_TEST_SUITE_P` is called once for every `TEST_P`. However, we
// only want to analyze the user provided schemes and those that are registered
// only once. Hence, this function keeping another static pointer to a vector
// which contains only the schemes under test.
//
// Without this additional step, when there are schemes available but the user
// only requests schemes that don't exist, first instantiation of the test would
// filter out all the user provided schemes (as they are not registered) but
// subsequent instantiations would return all registered schemes (since the
// vector with the user provided schemes is cleared).
static std::vector<std::string>* GetSchemesFromUserOrEnv() {
  std::vector<std::string>* all_schemes = new std::vector<std::string>;
  tensorflow::Status status =
      tensorflow::Env::Default()->GetRegisteredFileSystemSchemes(all_schemes);

  if (status.ok()) {
    std::vector<std::string>* user_schemes = SchemeVector();
    if (!user_schemes->empty()) {
      auto is_requested_scheme = [user_schemes](const auto& scheme) {
        return std::find(user_schemes->begin(), user_schemes->end(), scheme) ==
               user_schemes->end();
      };
      auto end = std::remove_if(all_schemes->begin(), all_schemes->end(),
                                is_requested_scheme);
      all_schemes->erase(end, all_schemes->end());
    }
  }

  return all_schemes;
}

static std::vector<std::string> GetSchemes() {
  static std::vector<std::string>* schemes = GetSchemesFromUserOrEnv();
  return *schemes;
}

INSTANTIATE_TEST_SUITE_P(ModularFileSystem, ModularFileSystemTest,
                         ::testing::ValuesIn(GetSchemes()));

// Loads a shared object implementing filesystem functionality.
static bool LoadDSO(const std::string& dso) {
  tensorflow::Status status = RegisterFilesystemPlugin(dso);
  if (!status.ok())
    VLOG(0) << "Filesystems from '" << dso
            << "' could not be registered: " << status;
  return status.ok();
}

// Tests whether a URI scheme results in a filesystem that is supported.
//
// As we need these URI schemes to instantiate the test suite when
// `testing::InitGoogleTest` gets called, here we just store them to an
// internal scheme registry. See `URISchemeRegister` above.
static bool GetURIScheme(const std::string& scheme) {
  tensorflow::SchemeVector()->push_back(scheme);
  return true;
}

}  // namespace
}  // namespace tensorflow

// Due to the usages of flags for this manual test, we need a special `main` to
// ensure our flags are parsed properly as `testing::InitGoogleTest` silently
// ignores other flags. Furthermore, we need this to ensure that the DSO is
// loaded exactly once, if provided.
GTEST_API_ int main(int argc, char** argv) {
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("dso", tensorflow::LoadDSO, "",
                       "Path to shared object to load"),
      tensorflow::Flag("scheme", tensorflow::GetURIScheme, "",
                       "URI scheme to test")};
  if (!tensorflow::Flags::Parse(&argc, argv, flag_list)) {
    std::cout << tensorflow::Flags::Usage(argv[0], flag_list);
    return -1;
  }

  tensorflow::testing::InstallStacktraceHandler();
  tensorflow::ModularFileSystemTest::InitializeTestRNG();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
