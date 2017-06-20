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
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-5\n",
           "012345"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 6-11\n",
           "6789")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

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

TEST(GcsFileSystemTest, NewRandomAccessFile_NoReadAhead_differentN) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-2\n",
           "012"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 3-12\n",
           "3456789")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(fs.NewRandomAccessFile("gs://bucket/random_access.txt", &file));

  char small_scratch[3];
  StringPiece result;

  // Read the first chunk.
  TF_EXPECT_OK(file->Read(0, sizeof(small_scratch), &result, small_scratch));
  EXPECT_EQ("012", result);

  // Read the second chunk that is larger. Requires allocation of new buffer.
  char large_scratch[10];

  EXPECT_EQ(errors::Code::OUT_OF_RANGE,
            file->Read(sizeof(small_scratch), sizeof(large_scratch), &result,
                       large_scratch)
                .code());
  EXPECT_EQ("3456789", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_WithReadAhead) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-8\n",
           "012345678"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 6-14\n",
           "6789abcde"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 6-20\n",
           "6789abcd"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 7-21\n",
           "789abcdef"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 20-34\n",
           ""),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-14\n",
           "01234567")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   5 /* read ahead bytes */, 0 /* initial retry delay */);

  char scratch[100];
  StringPiece result;
  {
    // We are instantiating this in an enclosed scope to make sure after the
    // unique ptr goes out of scope, we can still access result.
    std::unique_ptr<RandomAccessFile> file;
    TF_EXPECT_OK(
        fs.NewRandomAccessFile("gs://bucket/random_access.txt", &file));

    // Read the first chunk. The buffer will be updated with 4 + 5 = 9 bytes.
    scratch[5] = 'x';
    TF_EXPECT_OK(file->Read(0, 4, &result, scratch));
    EXPECT_EQ("0123", result);
    EXPECT_EQ(scratch[5], 'x');  // Make sure we only copied 4 bytes.

    // The second chunk will be fully loaded from the buffer, no requests are
    // made.
    TF_EXPECT_OK(file->Read(4, 4, &result, scratch));
    EXPECT_EQ("4567", result);

    // The chunk is only partially buffered -- the request will be made to
    // reload the buffer. 9 bytes will be requested (same as initial buffer
    // size).
    TF_EXPECT_OK(file->Read(6, 5, &result, scratch));
    EXPECT_EQ("6789a", result);

    // The range can only be partially satisfied. An attempt to fill the buffer
    // with 10 + 5 = 15 bytes will be made (buffer is resized for this request).
    EXPECT_EQ(errors::Code::OUT_OF_RANGE,
              file->Read(6, 10, &result, scratch).code());
    EXPECT_EQ("6789abcd", result);

    // The range cannot be satisfied, and the requested offset lies within the
    // buffer, but the end of the range is outside of the buffer.
    // A new request will be made to read 10 + 5 = 15 bytes.
    EXPECT_EQ(errors::Code::OUT_OF_RANGE,
              file->Read(7, 10, &result, scratch).code());
    EXPECT_EQ("789abcdef", result);

    // The range cannot be satisfied, and the requested offset is greater than
    // the
    // buffered range. A new request will be made to read 10 + 5 = 15 bytes.
    EXPECT_EQ(errors::Code::OUT_OF_RANGE,
              file->Read(20, 10, &result, scratch).code());
    EXPECT_TRUE(result.empty());

    TF_EXPECT_OK(file->Read(0, 4, &result, scratch));
  }

  // The beginning of the file is not in the buffer. This call will result
  // in another request. The buffer size is still 15.
  EXPECT_EQ("0123", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   5 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<RandomAccessFile> file;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.NewRandomAccessFile("gs://bucket/", &file).code());
}

TEST(GcsFileSystemTest, NewWritableFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable.txt\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Put body: content1,content2\n",
                           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(fs.NewWritableFile("gs://bucket/path/writeable.txt", &file));

  TF_EXPECT_OK(file->Append("content1,"));
  TF_EXPECT_OK(file->Append("content2"));
  TF_EXPECT_OK(file->Flush());
  // The calls to flush, sync, and close below should not cause uploads because
  // the file is not dirty.
  TF_EXPECT_OK(file->Flush());
  TF_EXPECT_OK(file->Sync());
  TF_EXPECT_OK(file->Close());
}

TEST(GcsFileSystemTest, NewWritableFile_ResumeUploadSucceeds) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable.txt\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Put body: content1,content2\n",
                           "", errors::Unavailable("503"), 503),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes */17\n"
                           "Put: yes\n",
                           "", errors::FailedPrecondition("308"), nullptr,
                           {{"Range", "0-10"}}, 308),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 11-16/17\n"
                           "Put body: ntent2\n",
                           "", errors::Unavailable("503"), 503),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes */17\n"
                           "Put: yes\n",
                           "", errors::FailedPrecondition("308"), nullptr,
                           {{"Range", "bytes=0-12"}}, 308),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 13-16/17\n"
                           "Put body: ent2\n",
                           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(fs.NewWritableFile("gs://bucket/path/writeable.txt", &file));

  TF_EXPECT_OK(file->Append("content1,"));
  TF_EXPECT_OK(file->Append("content2"));
  TF_EXPECT_OK(file->Close());
}

TEST(GcsFileSystemTest, NewWritableFile_ResumeUploadSucceedsOnGetStatus) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable.txt\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Put body: content1,content2\n",
                           "", errors::Unavailable("503"), 503),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes */17\n"
                           "Put: yes\n",
                           "", Status::OK(), nullptr, {}, 201)});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(fs.NewWritableFile("gs://bucket/path/writeable.txt", &file));

  TF_EXPECT_OK(file->Append("content1,"));
  TF_EXPECT_OK(file->Append("content2"));
  TF_EXPECT_OK(file->Close());
}

TEST(GcsFileSystemTest, NewWritableFile_ResumeUploadAllAttemptsFail) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable.txt\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Put body: content1,content2\n",
                           "", errors::Unavailable("503"), 503)});
  for (int i = 0; i < 10; i++) {
    requests.emplace_back(new FakeHttpRequest(
        "Uri: https://custom/upload/location\n"
        "Auth Token: fake_token\n"
        "Header Content-Range: bytes */17\n"
        "Put: yes\n",
        "", errors::FailedPrecondition("important HTTP error 308"), nullptr,
        {{"Range", "0-10"}}, 308));
    requests.emplace_back(new FakeHttpRequest(
        "Uri: https://custom/upload/location\n"
        "Auth Token: fake_token\n"
        "Header Content-Range: bytes 11-16/17\n"
        "Put body: ntent2\n",
        "", errors::Unavailable("important HTTP error 503"), 503));
  }
  // These calls will be made in the Close() attempt from the destructor.
  // Letting the destructor succeed.
  requests.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
      "uploadType=resumable&name=path%2Fwriteable.txt\n"
      "Auth Token: fake_token\n"
      "Header X-Upload-Content-Length: 17\n"
      "Post: yes\n",
      "", {{"Location", "https://custom/upload/location"}}));
  requests.emplace_back(
      new FakeHttpRequest("Uri: https://custom/upload/location\n"
                          "Auth Token: fake_token\n"
                          "Header Content-Range: bytes 0-16/17\n"
                          "Put body: content1,content2\n",
                          ""));
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 2 /* max upload attempts */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(fs.NewWritableFile("gs://bucket/path/writeable.txt", &file));

  TF_EXPECT_OK(file->Append("content1,"));
  TF_EXPECT_OK(file->Append("content2"));
  const auto& status = file->Close();
  EXPECT_EQ(errors::Code::ABORTED, status.code());
  EXPECT_TRUE(StringPiece(status.error_message())
                  .contains("All 10 retry attempts failed. The last failure: "
                            "Unavailable: important HTTP error 503"))
      << status;
}

TEST(GcsFileSystemTest, NewWritableFile_UploadReturns410) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable.txt\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Put body: content1,content2\n",
                           "", errors::NotFound("important HTTP error 410"),
                           410),
       // These calls will be made in the Close() attempt from the destructor.
       // Letting the destructor succeed.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable.txt\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Put body: content1,content2\n",
                           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(fs.NewWritableFile("gs://bucket/path/writeable.txt", &file));

  TF_EXPECT_OK(file->Append("content1,"));
  TF_EXPECT_OK(file->Append("content2"));
  const auto& status = file->Close();
  EXPECT_EQ(errors::Code::UNAVAILABLE, status.code());
  EXPECT_TRUE(
      StringPiece(status.error_message())
          .contains(
              "Upload to gs://bucket/path/writeable.txt failed, caused by: "
              "Not found: important HTTP error 410"))
      << status;
  EXPECT_TRUE(StringPiece(status.error_message())
                  .contains("when uploading gs://bucket/path/writeable.txt"))
      << status;
}

TEST(GcsFileSystemTest, NewWritableFile_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   5 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<WritableFile> file;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.NewWritableFile("gs://bucket/", &file).code());
}

TEST(GcsFileSystemTest, NewAppendableFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fappendable.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-1048575\n",
           "content1,"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fappendable.txt\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Put body: content1,content2\n",
                           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(fs.NewAppendableFile("gs://bucket/path/appendable.txt", &file));

  TF_EXPECT_OK(file->Append("content2"));
  TF_EXPECT_OK(file->Close());
}

TEST(GcsFileSystemTest, NewAppendableFile_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   5 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<WritableFile> file;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.NewAppendableFile("gs://bucket/", &file).code());
}

TEST(GcsFileSystemTest, NewReadOnlyMemoryRegionFromFile) {
  const string content = "file content";
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Frandom_access.txt?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           strings::StrCat("{\"size\": \"", content.size(),
                           "\", \"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           strings::StrCat("Uri: https://storage.googleapis.com/bucket/"
                           "path%2Frandom_access.txt\n"
                           "Auth Token: fake_token\n"
                           "Range: 0-",
                           content.size() - 1, "\n"),
           content)});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<ReadOnlyMemoryRegion> region;
  TF_EXPECT_OK(fs.NewReadOnlyMemoryRegionFromFile(
      "gs://bucket/path/random_access.txt", &region));

  EXPECT_EQ(content, StringPiece(reinterpret_cast<const char*>(region->data()),
                                 region->length()));
}

TEST(GcsFileSystemTest, NewReadOnlyMemoryRegionFromFile_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   5 /* read ahead bytes */, 0 /* initial retry delay */);

  std::unique_ptr<ReadOnlyMemoryRegion> region;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.NewReadOnlyMemoryRegionFromFile("gs://bucket/", &region).code());
}

TEST(GcsFileSystemTest, FileExists_YesAsObject) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "path%2Ffile1.txt?fields=size%2Cupdated\n"
      "Auth Token: fake_token\n",
      strings::StrCat("{\"size\": \"1010\","
                      "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.FileExists("gs://bucket/path/file1.txt"));
}

TEST(GcsFileSystemTest, FileExists_YesAsFolder) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsubfolder?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsubfolder%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"path/subfolder/\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.FileExists("gs://bucket/path/subfolder"));
}

TEST(GcsFileSystemTest, FileExists_YesAsBucket) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket1\n"
           "Auth Token: fake_token\n",
           "{\"size\": \"100\"}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket1\n"
           "Auth Token: fake_token\n",
           "{\"size\": \"100\"}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.FileExists("gs://bucket1"));
  TF_EXPECT_OK(fs.FileExists("gs://bucket1/"));
}

TEST(GcsFileSystemTest, FileExists_NotAsObjectOrFolder) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Ffile1.txt?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Ffile1.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{\"items\": []}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  EXPECT_EQ(errors::Code::NOT_FOUND,
            fs.FileExists("gs://bucket/path/file1.txt").code());
}

TEST(GcsFileSystemTest, FileExists_NotAsBucket) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket2\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket2\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.FileExists("gs://bucket2/").code());
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.FileExists("gs://bucket2").code());
}

TEST(GcsFileSystemTest, GetChildren_NoItems) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&prefix="
      "path%2F\n"
      "Auth Token: fake_token\n",
      "{\"prefixes\": [\"path/subpath/\"]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", &children));

  EXPECT_EQ(std::vector<string>({"subpath/"}), children);
}

TEST(GcsFileSystemTest, GetChildren_ThreeFiles) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&prefix="
      "path%2F\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/file3.txt\" }],"
      "\"prefixes\": [\"path/subpath/\"]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", &children));

  EXPECT_EQ(std::vector<string>({"file1.txt", "file3.txt", "subpath/"}),
            children);
}

TEST(GcsFileSystemTest, GetChildren_SelfDirectoryMarker) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&prefix="
      "path%2F\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/\" },"
      "  { \"name\": \"path/file3.txt\" }],"
      "\"prefixes\": [\"path/subpath/\"]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", &children));

  EXPECT_EQ(std::vector<string>({"file3.txt", "subpath/"}), children);
}

TEST(GcsFileSystemTest, GetChildren_ThreeFiles_NoSlash) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&prefix="
      "path%2F\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/file3.txt\" }],"
      "\"prefixes\": [\"path/subpath/\"]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path", &children));

  EXPECT_EQ(std::vector<string>({"file1.txt", "file3.txt", "subpath/"}),
            children);
}

TEST(GcsFileSystemTest, GetChildren_Root) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket-a-b-c/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F\n"
      "Auth Token: fake_token\n",
      "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket-a-b-c", &children));

  EXPECT_EQ(0, children.size());
}

TEST(GcsFileSystemTest, GetChildren_Empty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&prefix="
      "path%2F\n"
      "Auth Token: fake_token\n",
      "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", &children));

  EXPECT_EQ(0, children.size());
}

TEST(GcsFileSystemTest, GetChildren_Pagination) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&"
           "prefix=path%2F\n"
           "Auth Token: fake_token\n",
           "{\"nextPageToken\": \"ABCD==\", "
           "\"items\": [ "
           "  { \"name\": \"path/file1.txt\" },"
           "  { \"name\": \"path/file3.txt\" }],"
           "\"prefixes\": [\"path/subpath/\"]}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&"
           "prefix=path%2F"
           "&pageToken=ABCD==\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"path/file4.txt\" },"
           "  { \"name\": \"path/file5.txt\" }]}")});

  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path", &children));

  EXPECT_EQ(std::vector<string>({"file1.txt", "file3.txt", "subpath/",
                                 "file4.txt", "file5.txt"}),
            children);
}

TEST(GcsFileSystemTest, GetMatchingPaths_NoWildcard) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsubpath%2F\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/subpath/file2.txt\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> result;
  TF_EXPECT_OK(
      fs.GetMatchingPaths("gs://bucket/path/subpath/file2.txt", &result));
  EXPECT_EQ(std::vector<string>({"gs://bucket/path/subpath/file2.txt"}),
            result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_BucketAndWildcard) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/subpath/file2.txt\" },"
      "  { \"name\": \"path/file3.txt\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/*/*", &result));
  EXPECT_EQ(std::vector<string>({"gs://bucket/path/file1.txt",
                                 "gs://bucket/path/file3.txt",
                                 "gs://bucket/path/subpath"}),
            result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_FolderAndWildcard_Matches) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/subpath/file2.txt\" },"
      "  { \"name\": \"path/file3.txt\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/path/*/file2.txt", &result));
  EXPECT_EQ(std::vector<string>({"gs://bucket/path/subpath/file2.txt"}),
            result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_SelfDirectoryMarker) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/\" },"
      "  { \"name\": \"path/file3.txt\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/path/*", &result));
  EXPECT_EQ(std::vector<string>({"gs://bucket/path/file3.txt"}), result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_FolderAndWildcard_NoMatches) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/subpath/file2.txt\" },"
      "  { \"name\": \"path/file3.txt\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/path/*/file3.txt", &result));
  EXPECT_EQ(std::vector<string>(), result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_OnlyWildcard) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  std::vector<string> result;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.GetMatchingPaths("gs://*", &result).code());
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
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.DeleteFile("gs://bucket/path/file1.txt"));
}

TEST(GcsFileSystemTest, DeleteFile_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   5 /* read ahead bytes */, 0 /* initial retry delay */);

  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.DeleteFile("gs://bucket/").code());
}

TEST(GcsFileSystemTest, DeleteDir_Empty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F&maxResults=2\n"
      "Auth Token: fake_token\n",
      "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.DeleteDir("gs://bucket/path/"));
}

TEST(GcsFileSystemTest, DeleteDir_OnlyDirMarkerLeft) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F&maxResults=2\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"path/\" }]}"),
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2F\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.DeleteDir("gs://bucket/path/"));
}

TEST(GcsFileSystemTest, DeleteDir_BucketOnly) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?fields=items%2F"
      "name%2CnextPageToken&maxResults=2\nAuth Token: fake_token\n",
      "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.DeleteDir("gs://bucket"));
}

TEST(GcsFileSystemTest, DeleteDir_NonEmpty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F&maxResults=2\n"
      "Auth Token: fake_token\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  EXPECT_EQ(error::Code::FAILED_PRECONDITION,
            fs.DeleteDir("gs://bucket/path/").code());
}

TEST(GcsFileSystemTest, GetFileSize) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "file.txt?fields=size%2Cupdated\n"
      "Auth Token: fake_token\n",
      strings::StrCat("{\"size\": \"1010\","
                      "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  uint64 size;
  TF_EXPECT_OK(fs.GetFileSize("gs://bucket/file.txt", &size));
  EXPECT_EQ(1010, size);
}

TEST(GcsFileSystemTest, GetFileSize_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   5 /* read ahead bytes */, 0 /* initial retry delay */);

  uint64 size;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.GetFileSize("gs://bucket/", &size).code());
}

TEST(GcsFileSystemTest, RenameFile_Folder) {
  std::vector<HttpRequest*> requests(
      {// Check if this is a folder or an object.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path1%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"path1/subfolder/file1.txt\" }]}"),
       // Requesting the full list of files in the folder.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path1%2F\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"path1/\" },"  // A directory marker.
           "  { \"name\": \"path1/subfolder/file1.txt\" },"
           "  { \"name\": \"path1/file2.txt\" }]}"),
       // Copying the directory marker.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2F/rewriteTo/b/bucket/o/path2%2F\n"
           "Auth Token: fake_token\n"
           "Post: yes\n",
           "{\"done\": true}"),
       // Deleting the original directory marker.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2F\n"
           "Auth Token: fake_token\n"
           "Delete: yes\n",
           ""),
       // Copying the first file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2Fsubfolder%2Ffile1.txt/rewriteTo/b/bucket/o/"
           "path2%2Fsubfolder%2Ffile1.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n",
           "{\"done\": true}"),
       // Deleting the first original file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2Fsubfolder%2Ffile1.txt\n"
           "Auth Token: fake_token\n"
           "Delete: yes\n",
           ""),
       // Copying the second file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2Ffile2.txt/rewriteTo/b/bucket/o/path2%2Ffile2.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n",
           "{\"done\": true}"),
       // Deleting the second original file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2Ffile2.txt\n"
           "Auth Token: fake_token\n"
           "Delete: yes\n",
           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.RenameFile("gs://bucket/path1", "gs://bucket/path2/"));
}

TEST(GcsFileSystemTest, RenameFile_Object) {
  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsrc.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{}"),
       // IsDirectory is checking if the path exists as an object.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           strings::StrCat("{\"size\": \"1010\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       // Copying to the new location.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt/rewriteTo/b/bucket/o/path%2Fdst.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n",
           "{\"done\": true}"),
       // Deleting the original file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt\n"
           "Auth Token: fake_token\n"
           "Delete: yes\n",
           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(
      fs.RenameFile("gs://bucket/path/src.txt", "gs://bucket/path/dst.txt"));
}

/// Tests the scenario when deletion returns a failure, but actually succeeds.
TEST(GcsFileSystemTest, RenameFile_Object_DeletionRetried) {
  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsrc.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{}"),
       // IsDirectory is checking if the path exists as an object.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           strings::StrCat("{\"size\": \"1010\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       // Copying to the new location.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt/rewriteTo/b/bucket/o/path%2Fdst.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n",
           "{\"done\": true}"),
       // Deleting the original file - the deletion returns a failure.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt\n"
           "Auth Token: fake_token\n"
           "Delete: yes\n",
           "", errors::Unavailable("503"), 503),
       // Deleting the original file again - the deletion returns NOT_FOUND.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt\n"
           "Auth Token: fake_token\n"
           "Delete: yes\n",
           "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(
      fs.RenameFile("gs://bucket/path/src.txt", "gs://bucket/path/dst.txt"));
}

/// Tests the case when rewrite couldn't complete in one RPC.
TEST(GcsFileSystemTest, RenameFile_Object_Incomplete) {
  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsrc.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{}"),
       // IsDirectory is checking if the path exists as an object.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           strings::StrCat("{\"size\": \"1010\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       // Copying to the new location.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt/rewriteTo/b/bucket/o/path%2Fdst.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n",
           "{\"done\": false}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  EXPECT_EQ(
      errors::Code::UNIMPLEMENTED,
      fs.RenameFile("gs://bucket/path/src.txt", "gs://bucket/path/dst.txt")
          .code());
}

TEST(GcsFileSystemTest, Stat_Object) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "file.txt?fields=size%2Cupdated\n"
      "Auth Token: fake_token\n",
      strings::StrCat("{\"size\": \"1010\","
                      "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("gs://bucket/file.txt", &stat));
  EXPECT_EQ(1010, stat.length);
  EXPECT_EQ(1461971724896, stat.mtime_nsec / 1000 / 1000);
  EXPECT_FALSE(stat.is_directory);
}

TEST(GcsFileSystemTest, Stat_Folder) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "subfolder?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=subfolder%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"subfolder/\" }]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("gs://bucket/subfolder", &stat));
  EXPECT_EQ(0, stat.length);
  EXPECT_EQ(0, stat.mtime_nsec);
  EXPECT_TRUE(stat.is_directory);
}

TEST(GcsFileSystemTest, Stat_ObjectOrFolderNotFound) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  FileStatistics stat;
  EXPECT_EQ(error::Code::NOT_FOUND, fs.Stat("gs://bucket/path", &stat).code());
}

TEST(GcsFileSystemTest, Stat_Bucket) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
      "Auth Token: fake_token\n",
      "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("gs://bucket/", &stat));
  EXPECT_EQ(0, stat.length);
  EXPECT_EQ(0, stat.mtime_nsec);
  EXPECT_TRUE(stat.is_directory);
}

TEST(GcsFileSystemTest, Stat_BucketNotFound) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
      "Auth Token: fake_token\n",
      "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  FileStatistics stat;
  EXPECT_EQ(error::Code::NOT_FOUND, fs.Stat("gs://bucket/", &stat).code());
}

TEST(GcsFileSystemTest, IsDirectory_NotFound) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=file.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "file.txt?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  EXPECT_EQ(error::Code::NOT_FOUND,
            fs.IsDirectory("gs://bucket/file.txt").code());
}

TEST(GcsFileSystemTest, IsDirectory_NotDirectoryButObject) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=file.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "file.txt?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           strings::StrCat("{\"size\": \"1010\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  EXPECT_EQ(error::Code::FAILED_PRECONDITION,
            fs.IsDirectory("gs://bucket/file.txt").code());
}

TEST(GcsFileSystemTest, IsDirectory_Yes) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=subfolder%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{\"items\": [{\"name\": \"subfolder/\"}]}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=subfolder%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{\"items\": [{\"name\": \"subfolder/\"}]}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.IsDirectory("gs://bucket/subfolder"));
  TF_EXPECT_OK(fs.IsDirectory("gs://bucket/subfolder/"));
}

TEST(GcsFileSystemTest, IsDirectory_Bucket) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
           "Auth Token: fake_token\n",
           "{}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
           "Auth Token: fake_token\n",
           "{}")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.IsDirectory("gs://bucket"));
  TF_EXPECT_OK(fs.IsDirectory("gs://bucket/"));
}

TEST(GcsFileSystemTest, IsDirectory_BucketNotFound) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
      "Auth Token: fake_token\n",
      "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  EXPECT_EQ(error::Code::NOT_FOUND, fs.IsDirectory("gs://bucket/").code());
}

TEST(GcsFileSystemTest, CreateDir_Folder) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=subpath%2F\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 0\n"
           "Post: yes\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Put body: \n",
                           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=subpath%2F\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 0\n"
           "Post: yes\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Put body: \n",
                           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.CreateDir("gs://bucket/subpath"));
  TF_EXPECT_OK(fs.CreateDir("gs://bucket/subpath/"));
}

TEST(GcsFileSystemTest, CreateDir_Bucket) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
           "Auth Token: fake_token\n",
           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
           "Auth Token: fake_token\n",
           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  TF_EXPECT_OK(fs.CreateDir("gs://bucket/"));
  TF_EXPECT_OK(fs.CreateDir("gs://bucket"));
}

TEST(GcsFileSystemTest, DeleteRecursively_Ok) {
  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"path/file1.txt\" }]}"),
       // GetChildren recursively.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"path/\" },"  // The current directory's marker.
           "  { \"name\": \"path/file1.txt\" },"
           "  { \"name\": \"path/subpath/file2.txt\" },"
           "  { \"name\": \"path/file3.txt\" }]}"),
       // Delete the current directory's marker.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2F\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           ""),
       // Delete the object - fails and will be retried.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile1.txt\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           "", errors::Unavailable("500"), 500),
       // Delete the object again.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile1.txt\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           ""),
       // Delete the object.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Fsubpath%2Ffile2.txt\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           ""),
       // Delete the object.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile3.txt\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           "")});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  int64 undeleted_files, undeleted_dirs;
  TF_EXPECT_OK(fs.DeleteRecursively("gs://bucket/path", &undeleted_files,
                                    &undeleted_dirs));
  EXPECT_EQ(0, undeleted_files);
  EXPECT_EQ(0, undeleted_dirs);
}

TEST(GcsFileSystemTest, DeleteRecursively_DeletionErrors) {
  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"path/file1.txt\" }]}"),
       // Calling GetChildren recursively.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
           "Auth Token: fake_token\n",
           "{\"items\": [ "
           "  { \"name\": \"path/file1.txt\" },"
           "  { \"name\": \"path/subpath/\" },"
           "  { \"name\": \"path/subpath/file2.txt\" },"
           "  { \"name\": \"path/file3.txt\" }]}"),
       // Deleting the object.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile1.txt\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           ""),
       // Deleting the directory marker gs://bucket/path/ - fails with 404.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Fsubpath%2F\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           "", errors::NotFound("404"), 404),
       // Checking if gs://bucket/path/subpath/ is a folder - it is.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsubpath%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           strings::StrCat("{\"items\": [ "
                           "    { \"name\": \"path/subpath/\" }]}")),
       // Deleting the object gs://bucket/path/subpath/file2.txt
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Fsubpath%2Ffile2.txt\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           ""),
       // Deleting the object s://bucket/path/file3.txt - fails with 404.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile3.txt\n"
                           "Auth Token: fake_token\n"
                           "Delete: yes\n",
                           "", errors::NotFound("404"), 404),
       // Checking if gs://bucket/path/file3.txt/ is a folder - it's not.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Ffile3.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{}"),
       // Checking if gs://bucket/path/file3.txt is an object - fails with 404.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Ffile3.txt?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"), 404)});

  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  int64 undeleted_files, undeleted_dirs;
  TF_EXPECT_OK(fs.DeleteRecursively("gs://bucket/path", &undeleted_files,
                                    &undeleted_dirs));
  EXPECT_EQ(1, undeleted_files);
  EXPECT_EQ(1, undeleted_dirs);
}

TEST(GcsFileSystemTest, DeleteRecursively_NotAFolder) {
  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n",
           "{}"),
       // IsDirectory is checking if the path exists as an object.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path?fields=size%2Cupdated\n"
           "Auth Token: fake_token\n",
           "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(std::unique_ptr<AuthProvider>(new FakeAuthProvider),
                   std::unique_ptr<HttpRequest::Factory>(
                       new FakeHttpRequestFactory(&requests)),
                   0 /* read ahead bytes */, 0 /* initial retry delay */);

  int64 undeleted_files, undeleted_dirs;
  EXPECT_EQ(error::Code::NOT_FOUND,
            fs.DeleteRecursively("gs://bucket/path", &undeleted_files,
                                 &undeleted_dirs)
                .code());
  EXPECT_EQ(0, undeleted_files);
  EXPECT_EQ(1, undeleted_dirs);
}

TEST(GcsFileSystemTest, OverrideReadaheadBufferSize) {
  GcsFileSystem fs1;
  EXPECT_EQ(256 * 1024 * 1024, fs1.get_readahead_buffer_size());

  setenv("GCS_READAHEAD_BUFFER_SIZE_BYTES", "123456789", 1);
  GcsFileSystem fs2;
  EXPECT_EQ(123456789L, fs2.get_readahead_buffer_size());
}

}  // namespace
}  // namespace tensorflow
