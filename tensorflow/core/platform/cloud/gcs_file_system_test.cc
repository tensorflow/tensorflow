/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/gcs_file_system.h"
#include <fstream>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cloud/http_request_fake.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

std::vector<HttpRequest*> CreateGetThreeChildrenRequest() {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/subpath/file2.txt\" },"
      "  { \"name\": \"path/file3.txt\" }]}")});
  return requests;
}

void ExpectGetThreeChildrenFiles(const std::vector<string>& children) {
  EXPECT_EQ(3, children.size());
  EXPECT_EQ("file1.txt", children[0]);
  EXPECT_EQ("subpath/file2.txt", children[1]);
  EXPECT_EQ("file3.txt", children[2]);
}

class FakeAuthProvider : public AuthProvider {
 public:
  Status GetToken(string* token) override {
    *token = "fake_token";
    return Status::OK();
  }
};

TEST(GcsFileSystemTest, NewRandomAccessFile_NoReadAhead) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://bucket.storage.googleapis.com/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-5\n",
           "012345"),
       new FakeHttpRequest(
           "Uri: https://bucket.storage.googleapis.com/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 6-11\n",
           "6789")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(fs.NewRandomAccessFile("gs://bucket/random_access.txt", &file));

  char scratch[6];
  StringPiece result;

  // Read the first chunk.
  TF_EXPECT_OK(file->Read(0, sizeof(scratch), &result, scratch));
  EXPECT_EQ("012345", result);

  // Read the second chunk.
  EXPECT_EQ(
      errors::Code::OUT_OF_RANGE,
      file->Read(sizeof(scratch), sizeof(scratch), &result, scratch).code());
  EXPECT_EQ("6789", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_WithReadAhead) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://bucket.storage.googleapis.com/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-8\n",
           "01234567"),
       new FakeHttpRequest(
           "Uri: https://bucket.storage.googleapis.com/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 6-15\n",
           "6789abcd"),
       new FakeHttpRequest(
           "Uri: https://bucket.storage.googleapis.com/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 6-20\n",
           "6789abcd"),
       new FakeHttpRequest(
           "Uri: https://bucket.storage.googleapis.com/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 15-29\n",
           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   5 /* read ahead bytes */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(fs.NewRandomAccessFile("gs://bucket/random_access.txt", &file));

  char scratch[100];
  StringPiece result;

  // Read the first chunk. The cache will be updated with 4 + 5 = 9 bytes.
  TF_EXPECT_OK(file->Read(0, 4, &result, scratch));
  EXPECT_EQ("0123", result);

  // The second chunk will be fully loaded from the cache, no requests are made.
  TF_EXPECT_OK(file->Read(4, 4, &result, scratch));
  EXPECT_EQ("4567", result);

  // The chunk is only partially cached -- the request will be made to
  // reload the cache. 5 + 5 = 10 bytes will be requested.
  TF_EXPECT_OK(file->Read(6, 5, &result, scratch));
  EXPECT_EQ("6789a", result);

  // The range can only be partially satisfied. An attempt to fill the cache
  // with 10 + 5 = 15 bytes will be made.
  EXPECT_EQ(errors::Code::OUT_OF_RANGE,
            file->Read(6, 10, &result, scratch).code());
  EXPECT_EQ("6789abcd", result);

  // The range cannot be satisfied. An attempt to fill the cache
  // with 10 + 5 = 15 bytes will be made.
  EXPECT_EQ(errors::Code::OUT_OF_RANGE,
            file->Read(15, 10, &result, scratch).code());
  EXPECT_TRUE(result.empty());
}

TEST(GcsFileSystemTest, NewWritableFile) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
      "uploadType=media&name=path%2Fwriteable.txt\n"
      "Auth Token: fake_token\n"
      "Post body: content1,content2\n",
      "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(fs.NewWritableFile("gs://bucket/path/writeable.txt", &file));

  TF_EXPECT_OK(file->Append("content1,"));
  TF_EXPECT_OK(file->Append("content2"));
  TF_EXPECT_OK(file->Close());
}

TEST(GcsFileSystemTest, NewAppendableFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://bucket.storage.googleapis.com/path%2Fappendable.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-1048575\n",
           "content1,"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=media&name=path%2Fappendable.txt\n"
           "Auth Token: fake_token\n"
           "Post body: content1,content2\n",
           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(fs.NewAppendableFile("gs://bucket/path/appendable.txt", &file));

  TF_EXPECT_OK(file->Append("content2"));
  TF_EXPECT_OK(file->Close());
}

TEST(GcsFileSystemTest, NewReadOnlyMemoryRegionFromFile) {
  const string content = "file content";
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Frandom_access.txt?fields=size\n"
           "Auth Token: fake_token\n",
           strings::StrCat("{\"size\": \"", content.size(), "\"}")),
       new FakeHttpRequest(
           strings::StrCat("Uri: https://bucket.storage.googleapis.com/"
                           "path%2Frandom_access.txt\n"
                           "Auth Token: fake_token\n"
                           "Range: 0-",
                           content.size() - 1, "\n"),
           content)});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  std::unique_ptr<ReadOnlyMemoryRegion> region;
  TF_EXPECT_OK(fs.NewReadOnlyMemoryRegionFromFile(
      "gs://bucket/path/random_access.txt", &region));

  EXPECT_EQ(content, StringPiece(reinterpret_cast<const char*>(region->data()),
                                 region->length()));
}

TEST(GcsFileSystemTest, FileExists) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Ffile1.txt?fields=size\n"
           "Auth Token: fake_token\n",
           "{\"size\": \"100\"}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Ffile2.txt?fields=size\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"))});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  EXPECT_TRUE(fs.FileExists("gs://bucket/path/file1.txt"));
  EXPECT_FALSE(fs.FileExists("gs://bucket/path/file2.txt"));
}

TEST(GcsFileSystemTest, FileExists_BucketOnly) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket1\n"
           "Auth Token: fake_token\n",
           "{\"size\": \"100\"}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket2\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"))});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  EXPECT_TRUE(fs.FileExists("gs://bucket1"));
  EXPECT_FALSE(fs.FileExists("gs://bucket2/"));
}

TEST(GcsFileSystemTest, GetChildren_ThreeFiles) {
  auto requests = CreateGetThreeChildrenRequest();
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", &children));

  ExpectGetThreeChildrenFiles(children);
}

TEST(GcsFileSystemTest, GetChildren_ThreeFiles_NoSlash) {
  auto requests = CreateGetThreeChildrenRequest();
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path", &children));

  ExpectGetThreeChildrenFiles(children);
}

TEST(GcsFileSystemTest, GetChildren_Root) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket-a-b-c/o?"
      "fields=items%2Fname%2CnextPageToken\n"
      "Auth Token: fake_token\n",
      "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket-a-b-c", &children));

  EXPECT_EQ(0, children.size());
}

TEST(GcsFileSystemTest, GetChildren_Empty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n",
      "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", &children));

  EXPECT_EQ(0, children.size());
}

TEST(GcsFileSystemTest, GetChildren_Pagination) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
           "Auth Token: fake_token\n",
           "{\"nextPageToken\": \"ABCD==\", "
           " \"items\": [ "
           "  { \"name\": \"path/file1.txt\" },"
           "  { \"name\": \"path/subpath/file2.txt\" },"
           "  { \"name\": \"path/file3.txt\" }]}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F"
           "&pageToken=ABCD==\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"path/file4.txt\" },"
           "  { \"name\": \"path/file5.txt\" }]}")});

  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path", &children));

  EXPECT_EQ(5, children.size());
  EXPECT_EQ("file1.txt", children[0]);
  EXPECT_EQ("subpath/file2.txt", children[1]);
  EXPECT_EQ("file3.txt", children[2]);
  EXPECT_EQ("file4.txt", children[3]);
  EXPECT_EQ("file5.txt", children[4]);
}

TEST(GcsFileSystemTest, DeleteFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile1.txt\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  TF_EXPECT_OK(fs.DeleteFile("gs://bucket/path/file1.txt"));
}

TEST(GcsFileSystemTest, DeleteDir_Empty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n",
      "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  TF_EXPECT_OK(fs.DeleteDir("gs://bucket/path/"));
}

TEST(GcsFileSystemTest, DeleteDir_NonEmpty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  EXPECT_FALSE(fs.DeleteDir("gs://bucket/path/").ok());
}

TEST(GcsFileSystemTest, GetFileSize) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "file.txt?fields=size\n"
      "Auth Token: fake_token\n",
      strings::StrCat("{\"size\": \"1010\"}"))});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  uint64 size;
  TF_EXPECT_OK(fs.GetFileSize("gs://bucket/file.txt", &size));
  EXPECT_EQ(1010, size);
}

TEST(GcsFileSystemTest, RenameFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt/rewriteTo/b/bucket/o/path%2Fdst.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n",
           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt\n"
           "Auth Token: fake_token\n"
           "Delete: yes\n",
           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */);

  TF_EXPECT_OK(
      fs.RenameFile("gs://bucket/path/src.txt", "gs://bucket/path/dst.txt"));
}

}  // namespace
}  // namespace tensorflow
