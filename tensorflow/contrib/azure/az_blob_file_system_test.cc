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

#include "tensorflow/contrib/azure/az_blob_file_system.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/cloud/http_request_fake.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class AzBlobFileSystemTest : public ::testing::Test {
 protected:
  AzBlobFileSystemTest() {}

  std::string TmpDir(const std::string& path) {
    auto account = std::getenv("TF_AZURE_ACCOUNT");
    auto container = std::getenv("TF_AZURE_CONTAINER");
    if (account != nullptr && container != nullptr) {
      return "az://" +
             io::JoinPath(std::string(account) + ".blob.core.windows.net",
                          std::string(container), path);
    } else {
      return "az://" + io::JoinPath(testing::TmpDir(), path);
    }
  }

  Status WriteString(const std::string& fname, const std::string& content) {
    std::unique_ptr<WritableFile> writer;
    TF_RETURN_IF_ERROR(azbfs.NewWritableFile(fname, &writer));
    TF_RETURN_IF_ERROR(writer->Append(content));
    TF_RETURN_IF_ERROR(writer->Close());
    return Status::OK();
  }

  Status ReadAll(const std::string& fname, std::string* content) {
    std::unique_ptr<RandomAccessFile> reader;
    TF_RETURN_IF_ERROR(azbfs.NewRandomAccessFile(fname, &reader));

    uint64 file_size = 0;
    TF_RETURN_IF_ERROR(azbfs.GetFileSize(fname, &file_size));

    content->resize(file_size);
    StringPiece result;
    TF_RETURN_IF_ERROR(
        reader->Read(0, file_size, &result, gtl::string_as_array(content)));
    if (file_size != result.size()) {
      return errors::DataLoss("expected ", file_size, " got ", result.size(),
                              " bytes");
    }
    *content = result.ToString();
    return Status::OK();
  }

  AzBlobFileSystem azbfs;
};

TEST_F(AzBlobFileSystemTest, NewRandomAccessFile) {
  const std::string fname = TmpDir("RandomAccessFile");
  const std::string content = "abcdefghijklmn";

  TF_ASSERT_OK(WriteString(fname, content));

  std::unique_ptr<RandomAccessFile> reader;
  TF_EXPECT_OK(azbfs.NewRandomAccessFile(fname, &reader));

  std::string got;
  got.resize(content.size());
  StringPiece result;
  TF_EXPECT_OK(
      reader->Read(0, content.size(), &result, gtl::string_as_array(&got)));
  EXPECT_EQ(content.size(), result.size());
  EXPECT_EQ(content, result);

  got.clear();
  got.resize(4);
  TF_EXPECT_OK(reader->Read(2, 4, &result, gtl::string_as_array(&got)));
  EXPECT_EQ(4, result.size());
  EXPECT_EQ(content.substr(2, 4), result);
}

TEST_F(AzBlobFileSystemTest, NewWritableFile) {
  std::unique_ptr<WritableFile> writer;
  const std::string fname = TmpDir("WritableFile");
  TF_EXPECT_OK(azbfs.NewWritableFile(fname, &writer));
  TF_EXPECT_OK(writer->Append("content1,"));
  TF_EXPECT_OK(writer->Append("content2"));
  TF_EXPECT_OK(writer->Flush());
  TF_EXPECT_OK(writer->Sync());
  TF_EXPECT_OK(writer->Close());

  std::string content;
  TF_EXPECT_OK(ReadAll(fname, &content));
  EXPECT_EQ("content1,content2", content);
}

TEST_F(AzBlobFileSystemTest, NewAppendableFile) {
  std::unique_ptr<WritableFile> writer;

  const std::string fname = TmpDir("AppendableFile");
  TF_ASSERT_OK(WriteString(fname, "test"));

  TF_EXPECT_OK(azbfs.NewAppendableFile(fname, &writer));
  TF_EXPECT_OK(writer->Append("content"));
  TF_EXPECT_OK(writer->Close());
}

TEST_F(AzBlobFileSystemTest, NewReadOnlyMemoryRegionFromFile) {
  const std::string fname = TmpDir("MemoryFile");
  const std::string content = "content";
  TF_ASSERT_OK(WriteString(fname, content));
  std::unique_ptr<ReadOnlyMemoryRegion> region;
  TF_EXPECT_OK(azbfs.NewReadOnlyMemoryRegionFromFile(fname, &region));

  EXPECT_EQ(content, std::string(reinterpret_cast<const char*>(region->data()),
                                 region->length()));
}

TEST_F(AzBlobFileSystemTest, FileExists) {
  const std::string fname = TmpDir("FileExists");
  if (azbfs.FileExists(fname).ok()) {
    // Ensure the file doesn't yet exist.
    TF_ASSERT_OK(azbfs.DeleteFile(fname));
  }
  EXPECT_EQ(error::Code::NOT_FOUND, azbfs.FileExists(fname).code());
  TF_ASSERT_OK(WriteString(fname, "test"));
  TF_EXPECT_OK(azbfs.FileExists(fname));
  TF_ASSERT_OK(azbfs.DeleteFile(fname));
}

TEST_F(AzBlobFileSystemTest, GetChildren) {
  const std::string base = TmpDir("GetChildren");
  TF_EXPECT_OK(azbfs.CreateDir(base));

  const std::string file = io::JoinPath(base, "TestFile.csv");
  TF_EXPECT_OK(WriteString(file, "test"));

  const std::string subdir = io::JoinPath(base, "SubDir");
  TF_EXPECT_OK(azbfs.CreateDir(subdir));
  const std::string subfile = io::JoinPath(subdir, "TestSubFile.csv");
  TF_EXPECT_OK(WriteString(subfile, "test"));

  std::vector<string> children;
  TF_EXPECT_OK(azbfs.GetChildren(base, &children));
  std::sort(children.begin(), children.end());
  EXPECT_EQ(std::vector<string>({"SubDir", "TestFile.csv"}), children);
}

TEST_F(AzBlobFileSystemTest, DeleteFile) {
  const std::string fname = TmpDir("DeleteFile");
  TF_ASSERT_OK(WriteString(fname, "test"));
  TF_EXPECT_OK(azbfs.DeleteFile(fname));
}

TEST_F(AzBlobFileSystemTest, GetFileSize) {
  const std::string fname = TmpDir("GetFileSize");
  TF_ASSERT_OK(WriteString(fname, "test"));
  uint64 file_size = 0;
  TF_EXPECT_OK(azbfs.GetFileSize(fname, &file_size));
  EXPECT_EQ(4, file_size);
}

TEST_F(AzBlobFileSystemTest, CreateDir) {
  const std::string dir = TmpDir("CreateDir");
  TF_EXPECT_OK(azbfs.CreateDir(dir));

  const std::string file = io::JoinPath(dir, "CreateDirFile.csv");
  TF_EXPECT_OK(WriteString(file, "test"));
  FileStatistics stat;
  TF_EXPECT_OK(azbfs.Stat(dir, &stat));
  EXPECT_TRUE(stat.is_directory);
}

TEST_F(AzBlobFileSystemTest, DeleteDir) {
  const std::string dir = TmpDir("DeleteDir");
  const std::string file = io::JoinPath(dir, "DeleteDirFile.csv");
  TF_EXPECT_OK(WriteString(file, "test"));
  TF_EXPECT_OK(azbfs.DeleteDir(dir));

  FileStatistics stat;
  // Still OK here as virtual directories always exist
  TF_EXPECT_OK(azbfs.Stat(dir, &stat));
}

TEST_F(AzBlobFileSystemTest, RenameFile) {
  const std::string fname1 = TmpDir("RenameFile1");
  const std::string fname2 = TmpDir("RenameFile2");
  TF_ASSERT_OK(WriteString(fname1, "test"));
  TF_EXPECT_OK(azbfs.RenameFile(fname1, fname2));
  std::string content;
  TF_EXPECT_OK(ReadAll(fname2, &content));
  EXPECT_EQ("test", content);
}

TEST_F(AzBlobFileSystemTest, RenameFile_Overwrite) {
  const std::string fname1 = TmpDir("RenameFile1");
  const std::string fname2 = TmpDir("RenameFile2");

  TF_ASSERT_OK(WriteString(fname2, "test"));
  TF_EXPECT_OK(azbfs.FileExists(fname2));

  TF_ASSERT_OK(WriteString(fname1, "test"));
  TF_EXPECT_OK(azbfs.RenameFile(fname1, fname2));
  std::string content;
  TF_EXPECT_OK(ReadAll(fname2, &content));
  EXPECT_EQ("test", content);
}

TEST_F(AzBlobFileSystemTest, StatFile) {
  const std::string fname = TmpDir("StatFile");
  TF_ASSERT_OK(WriteString(fname, "test"));
  FileStatistics stat;
  TF_EXPECT_OK(azbfs.Stat(fname, &stat));
  EXPECT_EQ(4, stat.length);
  EXPECT_FALSE(stat.is_directory);
}

TEST_F(AzBlobFileSystemTest, GetMatchingPaths_NoWildcard) {
  const auto fname = TmpDir("path/subpath/file2.txt");

  TF_ASSERT_OK(WriteString(fname, "test"));
  std::vector<std::string> results;
  TF_EXPECT_OK(azbfs.GetMatchingPaths(fname, &results));
  EXPECT_EQ(std::vector<std::string>({fname}), results);
}

TEST_F(AzBlobFileSystemTest, GetMatchingPaths_FilenameWildcard) {
  const auto fname1 = TmpDir("path/subpath/file1.txt");
  const auto fname2 = TmpDir("path/subpath/file2.txt");
  const auto fname3 = TmpDir("path/subpath/another.txt");

  TF_ASSERT_OK(WriteString(fname1, "test"));
  TF_ASSERT_OK(WriteString(fname2, "test"));
  TF_ASSERT_OK(WriteString(fname3, "test"));

  const auto pattern = TmpDir("path/subpath/file*.txt");
  std::vector<std::string> results;
  TF_EXPECT_OK(azbfs.GetMatchingPaths(pattern, &results));
  EXPECT_EQ(std::vector<std::string>({fname1, fname2}), results);
}

}  // namespace
}  // namespace tensorflow