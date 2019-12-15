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

#include "tensorflow/core/platform/s3/s3_file_system.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

class S3FileSystemTest : public ::testing::Test {
 protected:
  S3FileSystemTest() {}

  string TmpDir(const string& path) {
    char* test_dir = getenv("S3_TEST_TMPDIR");
    if (test_dir != nullptr) {
      return io::JoinPath(string(test_dir), path);
    } else {
      return "s3://" + io::JoinPath(testing::TmpDir(), path);
    }
  }

  Status WriteString(const string& fname, const string& content) {
    std::unique_ptr<WritableFile> writer;
    TF_RETURN_IF_ERROR(s3fs.NewWritableFile(fname, &writer));
    TF_RETURN_IF_ERROR(writer->Append(content));
    TF_RETURN_IF_ERROR(writer->Close());
    return Status::OK();
  }

  Status ReadAll(const string& fname, string* content) {
    std::unique_ptr<RandomAccessFile> reader;
    TF_RETURN_IF_ERROR(s3fs.NewRandomAccessFile(fname, &reader));

    uint64 file_size = 0;
    TF_RETURN_IF_ERROR(s3fs.GetFileSize(fname, &file_size));

    content->resize(file_size);
    StringPiece result;
    TF_RETURN_IF_ERROR(
        reader->Read(0, file_size, &result, &(*content)[0]));
    if (file_size != result.size()) {
      return errors::DataLoss("expected ", file_size, " got ", result.size(),
                              " bytes");
    }
    return Status::OK();
  }

  S3FileSystem s3fs;
};

TEST_F(S3FileSystemTest, NewRandomAccessFile) {
  const string fname = TmpDir("RandomAccessFile");
  const string content = "abcdefghijklmn";

  TF_ASSERT_OK(WriteString(fname, content));

  std::unique_ptr<RandomAccessFile> reader;
  TF_EXPECT_OK(s3fs.NewRandomAccessFile(fname, &reader));

  string got;
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

TEST_F(S3FileSystemTest, NewWritableFile) {
  std::unique_ptr<WritableFile> writer;
  const string fname = TmpDir("WritableFile");
  TF_EXPECT_OK(s3fs.NewWritableFile(fname, &writer));
  TF_EXPECT_OK(writer->Append("content1,"));
  TF_EXPECT_OK(writer->Append("content2"));
  TF_EXPECT_OK(writer->Flush());
  TF_EXPECT_OK(writer->Sync());
  TF_EXPECT_OK(writer->Close());

  string content;
  TF_EXPECT_OK(ReadAll(fname, &content));
  EXPECT_EQ("content1,content2", content);
}

TEST_F(S3FileSystemTest, NewAppendableFile) {
  std::unique_ptr<WritableFile> writer;

  const string fname = TmpDir("AppendableFile");
  TF_ASSERT_OK(WriteString(fname, "test"));

  TF_EXPECT_OK(s3fs.NewAppendableFile(fname, &writer));
  TF_EXPECT_OK(writer->Append("content"));
  TF_EXPECT_OK(writer->Close());
}

TEST_F(S3FileSystemTest, NewReadOnlyMemoryRegionFromFile) {
  const string fname = TmpDir("MemoryFile");
  const string content = "content";
  TF_ASSERT_OK(WriteString(fname, content));
  std::unique_ptr<ReadOnlyMemoryRegion> region;
  TF_EXPECT_OK(s3fs.NewReadOnlyMemoryRegionFromFile(fname, &region));

  EXPECT_EQ(content, StringPiece(reinterpret_cast<const char*>(region->data()),
                                 region->length()));
}

TEST_F(S3FileSystemTest, FileExists) {
  const string fname = TmpDir("FileExists");
  // Ensure the file doesn't yet exist.
  TF_ASSERT_OK(s3fs.DeleteFile(fname));
  EXPECT_EQ(error::Code::NOT_FOUND, s3fs.FileExists(fname).code());
  TF_ASSERT_OK(WriteString(fname, "test"));
  TF_EXPECT_OK(s3fs.FileExists(fname));
}

TEST_F(S3FileSystemTest, GetChildren) {
  const string base = TmpDir("GetChildren");
  TF_EXPECT_OK(s3fs.CreateDir(base));

  const string file = io::JoinPath(base, "TestFile.csv");
  TF_EXPECT_OK(WriteString(file, "test"));

  const string subdir = io::JoinPath(base, "SubDir");
  TF_EXPECT_OK(s3fs.CreateDir(subdir));
  // s3 object storage doesn't support empty directory, we create file in the
  // directory
  const string subfile = io::JoinPath(subdir, "TestSubFile.csv");
  TF_EXPECT_OK(WriteString(subfile, "test"));

  std::vector<string> children;
  TF_EXPECT_OK(s3fs.GetChildren(base, &children));
  std::sort(children.begin(), children.end());
  EXPECT_EQ(std::vector<string>({"SubDir", "TestFile.csv"}), children);
}

TEST_F(S3FileSystemTest, DeleteFile) {
  const string fname = TmpDir("DeleteFile");
  TF_ASSERT_OK(WriteString(fname, "test"));
  TF_EXPECT_OK(s3fs.DeleteFile(fname));
}

TEST_F(S3FileSystemTest, GetFileSize) {
  const string fname = TmpDir("GetFileSize");
  TF_ASSERT_OK(WriteString(fname, "test"));
  uint64 file_size = 0;
  TF_EXPECT_OK(s3fs.GetFileSize(fname, &file_size));
  EXPECT_EQ(4, file_size);
}

TEST_F(S3FileSystemTest, CreateDir) {
  // s3 object storage doesn't support empty directory, we create file in the
  // directory
  const string dir = TmpDir("CreateDir");
  TF_EXPECT_OK(s3fs.CreateDir(dir));

  const string file = io::JoinPath(dir, "CreateDirFile.csv");
  TF_EXPECT_OK(WriteString(file, "test"));
  FileStatistics stat;
  TF_EXPECT_OK(s3fs.Stat(dir, &stat));
  EXPECT_TRUE(stat.is_directory);
}

TEST_F(S3FileSystemTest, DeleteDir) {
  // s3 object storage doesn't support empty directory, we create file in the
  // directory
  const string dir = TmpDir("DeleteDir");
  const string file = io::JoinPath(dir, "DeleteDirFile.csv");
  TF_EXPECT_OK(WriteString(file, "test"));
  EXPECT_FALSE(s3fs.DeleteDir(dir).ok());

  TF_EXPECT_OK(s3fs.DeleteFile(file));
  TF_EXPECT_OK(s3fs.DeleteDir(dir));
  FileStatistics stat;
  EXPECT_FALSE(s3fs.Stat(dir, &stat).ok());
}

TEST_F(S3FileSystemTest, RenameFile) {
  const string fname1 = TmpDir("RenameFile1");
  const string fname2 = TmpDir("RenameFile2");
  TF_ASSERT_OK(WriteString(fname1, "test"));
  TF_EXPECT_OK(s3fs.RenameFile(fname1, fname2));
  string content;
  TF_EXPECT_OK(ReadAll(fname2, &content));
  EXPECT_EQ("test", content);
}

TEST_F(S3FileSystemTest, RenameFile_Overwrite) {
  const string fname1 = TmpDir("RenameFile1");
  const string fname2 = TmpDir("RenameFile2");

  TF_ASSERT_OK(WriteString(fname2, "test"));
  TF_EXPECT_OK(s3fs.FileExists(fname2));

  TF_ASSERT_OK(WriteString(fname1, "test"));
  TF_EXPECT_OK(s3fs.RenameFile(fname1, fname2));
  string content;
  TF_EXPECT_OK(ReadAll(fname2, &content));
  EXPECT_EQ("test", content);
}

TEST_F(S3FileSystemTest, StatFile) {
  const string fname = TmpDir("StatFile");
  TF_ASSERT_OK(WriteString(fname, "test"));
  FileStatistics stat;
  TF_EXPECT_OK(s3fs.Stat(fname, &stat));
  EXPECT_EQ(4, stat.length);
  EXPECT_FALSE(stat.is_directory);
}

}  // namespace
}  // namespace tensorflow
