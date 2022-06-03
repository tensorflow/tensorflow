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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cloud/http_request_fake.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"

// Undef DeleteFile macro defined in wndows.h.
#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#endif

namespace tensorflow {
namespace {

static GcsFileSystem::TimeoutConfig kTestTimeoutConfig(5, 1, 10, 20, 30);
static RetryConfig kTestRetryConfig(0 /* init_delay_time_us */);

// Default (empty) constraint config
static std::unordered_set<string>* kAllowedLocationsDefault =
    new std::unordered_set<string>();
// Constraint config if bucket location constraint is turned on, with no
// custom list
static std::unordered_set<string>* kAllowedLocationsAuto =
    new std::unordered_set<string>({"auto"});

class FakeAuthProvider : public AuthProvider {
 public:
  Status GetToken(string* token) override {
    *token = "fake_token";
    return OkStatus();
  }
};

class FakeZoneProvider : public ZoneProvider {
 public:
  Status GetZone(string* zone) override {
    *zone = "us-east1-b";
    return OkStatus();
  }
};

TEST(GcsFileSystemTest, NewRandomAccessFile_NoBlockCache) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-5\n"
           "Timeouts: 5 1 20\n",
           "012345"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 6-11\n"
           "Timeouts: 5 1 20\n",
           "6789")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  StringPiece filename;
  TF_EXPECT_OK(file->Name(&filename));
  EXPECT_EQ(filename, "gs://bucket/random_access.txt");

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

TEST(GcsFileSystemTest, NewRandomAccessFile_Buffered) {
  std::vector<HttpRequest*> requests({
      new FakeHttpRequest(
          "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
          "Auth Token: fake_token\n"
          "Range: 0-9\n"
          "Timeouts: 5 1 20\n",
          "0123456789"),
      new FakeHttpRequest(
          "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
          "Auth Token: fake_token\n"
          "Range: 10-19\n"
          "Timeouts: 5 1 20\n",
          ""),
  });
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 10 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  StringPiece filename;
  TF_EXPECT_OK(file->Name(&filename));
  EXPECT_EQ(filename, "gs://bucket/random_access.txt");

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

TEST(GcsFileSystemTest, NewRandomAccessFile_Buffered_Errors) {
  std::vector<HttpRequest*> requests({
      new FakeHttpRequest(
          "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
          "Auth Token: fake_token\n"
          "Range: 0-9\n"
          "Timeouts: 5 1 20\n",
          "Server Not", errors::Unavailable("important HTTP error 308"),
          nullptr, {}, 308),
      new FakeHttpRequest(
          "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
          "Auth Token: fake_token\n"
          "Range: 6-15\n"
          "Timeouts: 5 1 20\n",
          "123"),
  });
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 10 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  StringPiece filename;
  TF_EXPECT_OK(file->Name(&filename));
  EXPECT_EQ(filename, "gs://bucket/random_access.txt");

  char scratch[6];
  StringPiece result;

  // Read the first chunk.
  EXPECT_EQ(errors::Code::UNAVAILABLE,
            file->Read(0, sizeof(scratch), &result, scratch).code());
  EXPECT_EQ("", result);

  // Read the second chunk.
  EXPECT_EQ(
      errors::Code::OUT_OF_RANGE,
      file->Read(sizeof(scratch), sizeof(scratch), &result, scratch).code());
  EXPECT_EQ("123", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_Buffered_ReadAtEOF) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-9\n"
           "Timeouts: 5 1 20\n",
           "0123456789"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 10-19\n"
           "Timeouts: 5 1 20\n",
           "")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 10 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  StringPiece filename;
  TF_EXPECT_OK(file->Name(&filename));
  EXPECT_EQ(filename, "gs://bucket/random_access.txt");

  char scratch[10];
  StringPiece result;

  // Read the first chunk.
  TF_EXPECT_OK(file->Read(0, sizeof(scratch), &result, scratch));
  EXPECT_EQ("0123456789", result);

  // Read the second chunk.
  EXPECT_EQ(
      errors::Code::OUT_OF_RANGE,
      file->Read(sizeof(scratch), sizeof(scratch), &result, scratch).code());
  EXPECT_EQ("", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_Buffered_CachedOutOfRange) {
  // In this test, there is only one backend request since we cache the file
  // size.
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
      "Auth Token: fake_token\n"
      "Range: 0-9\n"
      "Timeouts: 5 1 20\n",
      "012345678")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 10 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  StringPiece filename;
  TF_EXPECT_OK(file->Name(&filename));
  EXPECT_EQ(filename, "gs://bucket/random_access.txt");

  char scratch[5];
  StringPiece result;

  // Read the first chunk. Even though the backend response is out-of-range,
  // we should get a OK status since we're just reading the first 5 bytes.
  TF_EXPECT_OK(file->Read(0, sizeof(scratch), &result, scratch));
  EXPECT_EQ("01234", result);

  TF_EXPECT_OK(file->Read(4, sizeof(scratch), &result, scratch));
  EXPECT_EQ("45678", result);

  // Return the cached error once the user starts reading out of range.
  EXPECT_EQ(errors::Code::OUT_OF_RANGE,
            file->Read(5, sizeof(scratch), &result, scratch).code());
  EXPECT_EQ("5678", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_Buffered_CachedNotSequential) {
  // In this test, the second read is seeking backwards, so it should trigger
  // a backend request.
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 1-10\n"
           "Timeouts: 5 1 20\n",
           "12345678"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-9\n"
           "Timeouts: 5 1 20\n",
           "012345678")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 10 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  StringPiece filename;
  TF_EXPECT_OK(file->Name(&filename));
  EXPECT_EQ(filename, "gs://bucket/random_access.txt");

  char scratch[5];
  StringPiece result;

  TF_EXPECT_OK(file->Read(1, sizeof(scratch), &result, scratch));
  EXPECT_EQ("12345", result);
  TF_EXPECT_OK(file->Read(0, sizeof(scratch), &result, scratch));
  EXPECT_EQ("01234", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_Buffered_Growing) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-9\n"
           "Timeouts: 5 1 20\n",
           "012345678"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 9-18\n"
           "Timeouts: 5 1 20\n",
           "9")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 10 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  StringPiece filename;
  TF_EXPECT_OK(file->Name(&filename));
  EXPECT_EQ(filename, "gs://bucket/random_access.txt");

  char scratch[10];
  StringPiece result;

  // Read the first chunk. Since the first read is out-of-range,
  // we don't cache the out-of-range flag and each subsequent read triggers a
  // backend call.
  EXPECT_EQ(errors::Code::OUT_OF_RANGE,
            file->Read(0, sizeof(scratch), &result, scratch).code());
  EXPECT_EQ("012345678", result);

  TF_EXPECT_OK(file->Read(0, sizeof(scratch), &result, scratch));
  EXPECT_EQ("0123456789", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_Buffered_ReadBackwards) {
  // Go backwards in the file. It should trigger a new read.
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 5-14\n"
           "Timeouts: 5 1 20\n",
           "56789"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-9\n"
           "Timeouts: 5 1 20\n",
           "0123456789")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 10 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  StringPiece filename;
  TF_EXPECT_OK(file->Name(&filename));
  EXPECT_EQ(filename, "gs://bucket/random_access.txt");

  char scratch[10];
  StringPiece result;

  // Read the first chunk.
  EXPECT_EQ(errors::Code::OUT_OF_RANGE,
            file->Read(5, sizeof(scratch), &result, scratch).code());
  EXPECT_EQ("56789", result);

  // Go back and read from the beginning of the file.
  TF_EXPECT_OK(file->Read(0, sizeof(scratch), &result, scratch));
  EXPECT_EQ("0123456789", result);
}

TEST(GcsFileSystemTest,
     NewRandomAccessFile_WithLocationConstraintInSameLocation) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      R"(
          {
            "location":"US-EAST1"
          })")});

  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsAuto,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));
}

TEST(GcsFileSystemTest, NewRandomAccessFile_WithLocationConstraintCaching) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           R"(
          {
            "location":"US-EAST1"
          })"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/anotherbucket\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           R"(
          {
            "location":"US-EAST1"
          })"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           R"(
          {
            "location":"US-EAST1"
          })")});

  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsAuto,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;

  string bucket = "gs://bucket/random_access.txt";
  string another_bucket = "gs://anotherbucket/random_access.txt";
  // Multiple calls should only cause one request to the location API.
  TF_EXPECT_OK(fs.NewRandomAccessFile(bucket, nullptr, &file));
  TF_EXPECT_OK(fs.NewRandomAccessFile(bucket, nullptr, &file));

  // A new bucket should have one cache miss
  TF_EXPECT_OK(fs.NewRandomAccessFile(another_bucket, nullptr, &file));
  // And then future calls to both should be cached
  TF_EXPECT_OK(fs.NewRandomAccessFile(bucket, nullptr, &file));
  TF_EXPECT_OK(fs.NewRandomAccessFile(another_bucket, nullptr, &file));

  // Trigger a flush, should then require one more call
  fs.FlushCaches(nullptr);
  TF_EXPECT_OK(fs.NewRandomAccessFile(bucket, nullptr, &file));
}

TEST(GcsFileSystemTest,
     NewRandomAccessFile_WithLocationConstraintInDifferentLocation) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      R"(
          {
            "location":"BARFOO"
          })")});

  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsAuto,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  EXPECT_EQ(
      tensorflow::errors::FailedPrecondition(
          "Bucket 'bucket' is in 'barfoo' location, allowed locations "
          "are: (us-east1)."),
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));
}

TEST(GcsFileSystemTest, NewRandomAccessFile_NoBlockCache_DifferentN) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-2\n"
           "Timeouts: 5 1 20\n",
           "012"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 3-12\n"
           "Timeouts: 5 1 20\n",
           "3456789")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

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

TEST(GcsFileSystemTest, NewRandomAccessFile_WithBlockCache) {
  // Our underlying file in this test is a 15 byte file with contents
  // "0123456789abcde".
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "random_access.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"15\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-8\n"
           "Timeouts: 5 1 20\n",
           "012345678"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 9-17\n"
           "Timeouts: 5 1 20\n",
           "9abcde"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 18-26\n"
           "Timeouts: 5 1 20\n",
           "")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 9 /* block size */,
      18 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  char scratch[100];
  StringPiece result;
  {
    // We are instantiating this in an enclosed scope to make sure after the
    // unique ptr goes out of scope, we can still access result.
    std::unique_ptr<RandomAccessFile> file;
    TF_EXPECT_OK(fs.NewRandomAccessFile("gs://bucket/random_access.txt",
                                        nullptr, &file));

    // Read the first chunk. The cache will be populated with the first block of
    // 9 bytes.
    scratch[5] = 'x';
    TF_EXPECT_OK(file->Read(0, 4, &result, scratch));
    EXPECT_EQ("0123", result);
    EXPECT_EQ(scratch[5], 'x');  // Make sure we only copied 4 bytes.

    // The second chunk will be fully loaded from the cache, no requests are
    // made.
    TF_EXPECT_OK(file->Read(4, 4, &result, scratch));
    EXPECT_EQ("4567", result);

    // The chunk is only partially cached -- the request will be made to fetch
    // the next block. 9 bytes will be requested, starting at offset 9.
    TF_EXPECT_OK(file->Read(6, 5, &result, scratch));
    EXPECT_EQ("6789a", result);

    // The range can only be partially satisfied, as the second block contains
    // only 6 bytes for a total of 9 + 6 = 15 bytes in the file.
    EXPECT_EQ(errors::Code::OUT_OF_RANGE,
              file->Read(6, 10, &result, scratch).code());
    EXPECT_EQ("6789abcde", result);

    // The range cannot be satisfied, and the requested offset is past the end
    // of the cache. A new request will be made to read 9 bytes starting at
    // offset 18. This request will return an empty response, and there will not
    // be another request.
    EXPECT_EQ(errors::Code::OUT_OF_RANGE,
              file->Read(20, 10, &result, scratch).code());
    EXPECT_TRUE(result.empty());

    // The beginning of the file should still be in the LRU cache. There should
    // not be another request. The buffer size is still 15.
    TF_EXPECT_OK(file->Read(0, 4, &result, scratch));
  }

  EXPECT_EQ("0123", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_WithBlockCache_Flush) {
  // Our underlying file in this test is a 15 byte file with contents
  // "0123456789abcde".
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "random_access.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"15\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-8\n"
           "Timeouts: 5 1 20\n",
           "012345678"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "random_access.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"15\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-8\n"
           "Timeouts: 5 1 20\n",
           "012345678")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 9 /* block size */,
      18 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  char scratch[100];
  StringPiece result;
  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));
  // Read the first chunk. The cache will be populated with the first block of
  // 9 bytes.
  scratch[5] = 'x';
  TF_EXPECT_OK(file->Read(0, 4, &result, scratch));
  EXPECT_EQ("0123", result);
  EXPECT_EQ(scratch[5], 'x');  // Make sure we only copied 4 bytes.
  // Flush caches and read the second chunk. This will be a cache miss, and
  // the same block will be fetched again.
  fs.FlushCaches(nullptr);
  TF_EXPECT_OK(file->Read(4, 4, &result, scratch));
  EXPECT_EQ("4567", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_WithBlockCache_MaxStaleness) {
  // Our underlying file in this test is a 16 byte file with contents
  // "0123456789abcdef".
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "object?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"16\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest("Uri: https://storage.googleapis.com/bucket/object\n"
                           "Auth Token: fake_token\n"
                           "Range: 0-7\n"
                           "Timeouts: 5 1 20\n",
                           "01234567"),
       new FakeHttpRequest("Uri: https://storage.googleapis.com/bucket/object\n"
                           "Auth Token: fake_token\n"
                           "Range: 8-15\n"
                           "Timeouts: 5 1 20\n",
                           "89abcdef")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 8 /* block size */,
      16 /* max bytes */, 3600 /* max staleness */,
      3600 /* stat cache max age */, 0 /* stat cache max entries */,
      0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);
  char scratch[100];
  StringPiece result;
  // There should only be two HTTP requests issued to GCS even though we iterate
  // this loop 10 times.  This shows that the underlying FileBlockCache persists
  // across file close/open boundaries.
  for (int i = 0; i < 10; i++) {
    // Create two files. Since these files have the same name and the max
    // staleness of the filesystem is > 0, they will share the same blocks.
    std::unique_ptr<RandomAccessFile> file1;
    std::unique_ptr<RandomAccessFile> file2;
    TF_EXPECT_OK(fs.NewRandomAccessFile("gs://bucket/object", nullptr, &file1));
    TF_EXPECT_OK(fs.NewRandomAccessFile("gs://bucket/object", nullptr, &file2));
    // Reading the first block from file1 should load it once.
    TF_EXPECT_OK(file1->Read(0, 8, &result, scratch));
    EXPECT_EQ("01234567", result);
    // Reading the first block from file2 should not trigger a request to load
    // the first block again, because the FileBlockCache shared by file1 and
    // file2 already has the first block.
    TF_EXPECT_OK(file2->Read(0, 8, &result, scratch));
    EXPECT_EQ("01234567", result);
    // Reading the second block from file2 should load it once.
    TF_EXPECT_OK(file2->Read(8, 8, &result, scratch));
    EXPECT_EQ("89abcdef", result);
    // Reading the second block from file1 should not trigger a request to load
    // the second block again, because the FileBlockCache shared by file1 and
    // file2 already has the second block.
    TF_EXPECT_OK(file1->Read(8, 8, &result, scratch));
    EXPECT_EQ("89abcdef", result);
  }
}

TEST(GcsFileSystemTest,
     NewRandomAccessFile_WithBlockCache_FileSignatureChanges) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "random_access.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"5\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-8\n"
           "Timeouts: 5 1 20\n",
           "01234"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "random_access.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"5\",\"generation\": \"2\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-8\n"
           "Timeouts: 5 1 20\n",
           "43210")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 9 /* block size */,
      18 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  char scratch[5];
  StringPiece result;

  // First read.
  TF_EXPECT_OK(file->Read(0, sizeof(scratch), &result, scratch));
  EXPECT_EQ("01234", result);

  // Second read. File signatures are different.
  TF_EXPECT_OK(file->Read(0, sizeof(scratch), &result, scratch));
  EXPECT_EQ("43210", result);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider),
      0 /* read ahead bytes */, 0 /* max bytes */, 0 /* max staleness */,
      0 /* stat cache max age */, 0 /* stat cache max entries */,
      0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<RandomAccessFile> file;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.NewRandomAccessFile("gs://bucket/", nullptr, &file).code());
}

TEST(GcsFileSystemTest, NewRandomAccessFile_InconsistentRead) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "random_access.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"6\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-5\n"
           "Timeouts: 5 1 20\n",
           "012")});

  // Set stat_cache_max_age to 1000s so that StatCache could work.
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 1e3 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  // Stat the file first so that the file stats are cached.
  FileStatistics stat;
  TF_ASSERT_OK(fs.Stat("gs://bucket/random_access.txt", nullptr, &stat));

  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  char scratch[6];
  StringPiece result;

  EXPECT_EQ(errors::Code::INTERNAL,
            file->Read(0, sizeof(scratch), &result, scratch).code());
}

TEST(GcsFileSystemTest, NewWritableFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fwriteable?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"16\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fwriteable\n"
           "Auth Token: fake_token\n"
           "Range: 0-7\n"
           "Timeouts: 5 1 20\n",
           "01234567"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Timeouts: 5 1 30\n"
                           "Put body: content1,content2\n",
                           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fwriteable?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"33\",\"generation\": \"2\","
                           "\"updated\": \"2016-04-29T23:15:34.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fwriteable\n"
           "Auth Token: fake_token\n"
           "Range: 0-7\n"
           "Timeouts: 5 1 20\n",
           "01234567")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 8 /* block size */,
      8 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  // Read from the file first, to fill the block cache.
  std::unique_ptr<RandomAccessFile> rfile;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/path/writeable", nullptr, &rfile));
  char scratch[100];
  StringPiece result;
  TF_EXPECT_OK(rfile->Read(0, 4, &result, scratch));
  EXPECT_EQ("0123", result);
  // Open the writable file.
  std::unique_ptr<WritableFile> wfile;
  TF_EXPECT_OK(
      fs.NewWritableFile("gs://bucket/path/writeable", nullptr, &wfile));
  TF_EXPECT_OK(wfile->Append("content1,"));
  int64_t pos;
  TF_EXPECT_OK(wfile->Tell(&pos));
  EXPECT_EQ(9, pos);
  TF_EXPECT_OK(wfile->Append("content2"));
  TF_EXPECT_OK(wfile->Flush());
  // Re-reading the file should trigger another HTTP request to GCS.
  TF_EXPECT_OK(rfile->Read(0, 4, &result, scratch));
  EXPECT_EQ("0123", result);
  // The calls to flush, sync, and close below should not cause uploads because
  // the file is not dirty.
  TF_EXPECT_OK(wfile->Flush());
  TF_EXPECT_OK(wfile->Sync());
  TF_EXPECT_OK(wfile->Close());
}

TEST(GcsFileSystemTest, NewWritableFile_ResumeUploadSucceeds) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable.txt\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Timeouts: 5 1 30\n"
                           "Put body: content1,content2\n",
                           "", errors::Unavailable("503"), 503),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Header Content-Range: bytes */17\n"
                           "Put: yes\n",
                           "", errors::Unavailable("308"), nullptr,
                           {{"Range", "0-10"}}, 308),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 11-16/17\n"
                           "Timeouts: 5 1 30\n"
                           "Put body: ntent2\n",
                           "", errors::Unavailable("503"), 503),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Header Content-Range: bytes */17\n"
                           "Put: yes\n",
                           "", errors::Unavailable("308"), nullptr,
                           {{"Range", "bytes=0-12"}}, 308),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 13-16/17\n"
                           "Timeouts: 5 1 30\n"
                           "Put body: ent2\n",
                           "", errors::Unavailable("308"), 308),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Header Content-Range: bytes */17\n"
                           "Put: yes\n",
                           "", errors::Unavailable("308"), nullptr,
                           {{"Range", "bytes=0-14"}}, 308),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 15-16/17\n"
                           "Timeouts: 5 1 30\n"
                           "Put body: t2\n",
                           "")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(
      fs.NewWritableFile("gs://bucket/path/writeable.txt", nullptr, &file));

  TF_EXPECT_OK(file->Append("content1,"));
  TF_EXPECT_OK(file->Append("content2"));
  TF_EXPECT_OK(file->Close());
}

TEST(GcsFileSystemTest, NewWritableFile_ResumeUploadSucceedsOnGetStatus) {
  // This test also verifies that a file's blocks are purged from the cache when
  // the file is written, even when the write takes the "succeeds on get status"
  // path.
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fwriteable?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"16\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fwriteable\n"
           "Auth Token: fake_token\n"
           "Range: 0-7\n"
           "Timeouts: 5 1 20\n",
           "01234567"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Timeouts: 5 1 30\n"
                           "Put body: content1,content2\n",
                           "", errors::Unavailable("503"), 503),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Header Content-Range: bytes */17\n"
                           "Put: yes\n",
                           "", OkStatus(), nullptr, {}, 201),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fwriteable?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"33\",\"generation\": \"2\","
                           "\"updated\": \"2016-04-29T23:19:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fwriteable\n"
           "Auth Token: fake_token\n"
           "Range: 0-7\n"
           "Timeouts: 5 1 20\n",
           "01234567")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 8 /* block size */,
      8 /* max bytes */, 3600 /* max staleness */,
      3600 /* stat cache max age */, 0 /* stat cache max entries */,
      0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);
  // Pull the file's first block into the cache. This will trigger the first
  // HTTP request to GCS.
  std::unique_ptr<RandomAccessFile> rfile;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/path/writeable", nullptr, &rfile));
  char scratch[100];
  StringPiece result;
  TF_EXPECT_OK(rfile->Read(0, 4, &result, scratch));
  EXPECT_EQ("0123", result);
  // Now write to the same file. Once the write succeeds, the cached block will
  // be flushed.
  std::unique_ptr<WritableFile> wfile;
  TF_EXPECT_OK(
      fs.NewWritableFile("gs://bucket/path/writeable", nullptr, &wfile));
  TF_EXPECT_OK(wfile->Append("content1,"));
  TF_EXPECT_OK(wfile->Append("content2"));
  // Appending doesn't invalidate the read cache - only flushing does. This read
  // will not trigger an HTTP request to GCS.
  TF_EXPECT_OK(rfile->Read(4, 4, &result, scratch));
  EXPECT_EQ("4567", result);
  // Closing the file triggers HTTP requests to GCS and invalidates the read
  // cache for the file.
  TF_EXPECT_OK(wfile->Close());
  // Reading the first block of the file goes to GCS again.
  TF_EXPECT_OK(rfile->Read(0, 8, &result, scratch));
  EXPECT_EQ("01234567", result);
}

TEST(GcsFileSystemTest, NewWritableFile_ResumeUploadAllAttemptsFail) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable.txt\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Timeouts: 5 1 30\n"
                           "Put body: content1,content2\n",
                           "", errors::Unavailable("503"), 503)});
  for (int i = 0; i < 10; i++) {
    requests.emplace_back(
        new FakeHttpRequest("Uri: https://custom/upload/location\n"
                            "Auth Token: fake_token\n"
                            "Timeouts: 5 1 10\n"
                            "Header Content-Range: bytes */17\n"
                            "Put: yes\n",
                            "", errors::Unavailable("important HTTP error 308"),
                            nullptr, {{"Range", "0-10"}}, 308));
    requests.emplace_back(new FakeHttpRequest(
        "Uri: https://custom/upload/location\n"
        "Auth Token: fake_token\n"
        "Header Content-Range: bytes 11-16/17\n"
        "Timeouts: 5 1 30\n"
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
      "Post: yes\n"
      "Timeouts: 5 1 10\n",
      "", {{"Location", "https://custom/upload/location"}}));
  requests.emplace_back(
      new FakeHttpRequest("Uri: https://custom/upload/location\n"
                          "Auth Token: fake_token\n"
                          "Header Content-Range: bytes 0-16/17\n"
                          "Timeouts: 5 1 30\n"
                          "Put body: content1,content2\n",
                          ""));
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */,
      RetryConfig(2 /* .init_delay_time_us */), kTestTimeoutConfig,
      *kAllowedLocationsDefault, nullptr /* gcs additional header */,
      false /* compose append */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(
      fs.NewWritableFile("gs://bucket/path/writeable.txt", nullptr, &file));

  TF_EXPECT_OK(file->Append("content1,"));
  TF_EXPECT_OK(file->Append("content2"));
  const auto& status = file->Close();
  EXPECT_EQ(errors::Code::ABORTED, status.code());
  EXPECT_TRUE(
      absl::StrContains(status.error_message(),
                        "All 10 retry attempts failed. The last failure: "
                        "important HTTP error 503"))
      << status;
}

TEST(GcsFileSystemTest, NewWritableFile_UploadReturns410) {
  std::vector<string> results;
  TF_EXPECT_OK(
      Env::Default()->GetMatchingPaths("/tmp/tmp_file_tensorflow*", &results));
  const int64_t tmp_files_before = results.size();

  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fwriteable.txt\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Timeouts: 5 1 30\n"
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
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Timeouts: 5 1 30\n"
                           "Put body: content1,content2\n",
                           "")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  {
    std::unique_ptr<WritableFile> file;
    TF_EXPECT_OK(
        fs.NewWritableFile("gs://bucket/path/writeable.txt", nullptr, &file));

    TF_EXPECT_OK(file->Append("content1,"));
    TF_EXPECT_OK(file->Append("content2"));
    const auto& status = file->Close();
    EXPECT_EQ(errors::Code::UNAVAILABLE, status.code());
    EXPECT_TRUE(
        absl::StrContains(status.error_message(),
                          "Upload to gs://bucket/path/writeable.txt failed, "
                          "caused by: important HTTP error 410"))
        << status;
    EXPECT_TRUE(
        absl::StrContains(status.error_message(),
                          "when uploading gs://bucket/path/writeable.txt"))
        << status;
  }

  // Check that no new tempfiles were left over after failure and destruction
  // of the file.
  results.clear();
  TF_EXPECT_OK(
      Env::Default()->GetMatchingPaths("/tmp/tmp_file_tensorflow*", &results));
  EXPECT_EQ(tmp_files_before, results.size());
}

TEST(GcsFileSystemTest, NewWritableFile_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<WritableFile> file;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.NewWritableFile("gs://bucket/", nullptr, &file).code());
}

TEST(GcsFileSystemTest, NewAppendableFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fappendable?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"8\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fappendable\n"
           "Auth Token: fake_token\n"
           "Range: 0-1048575\n"
           "Timeouts: 5 1 20\n",
           "content1,"),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fappendable\n"
           "Auth Token: fake_token\n"
           "Range: 0-31\n"
           "Timeouts: 5 1 20\n",
           "content1,"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
           "uploadType=resumable&name=path%2Fappendable\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 17\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "", {{"Location", "https://custom/upload/location"}}),
       new FakeHttpRequest("Uri: https://custom/upload/location\n"
                           "Auth Token: fake_token\n"
                           "Header Content-Range: bytes 0-16/17\n"
                           "Timeouts: 5 1 30\n"
                           "Put body: content1,content2\n",
                           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fappendable?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"8\",\"generation\": \"2\","
                           "\"updated\": \"2016-04-29T23:25:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fappendable\n"
           "Auth Token: fake_token\n"
           "Range: 0-31\n"
           "Timeouts: 5 1 20\n",
           "01234567")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 32 /* block size */,
      32 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  // Create an appendable file. This should read the file from GCS, and pull its
  // contents into the block cache.
  std::unique_ptr<WritableFile> wfile;
  TF_EXPECT_OK(
      fs.NewAppendableFile("gs://bucket/path/appendable", nullptr, &wfile));
  TF_EXPECT_OK(wfile->Append("content2"));
  // Verify that the file contents are in the block cache. This read should not
  // trigger an HTTP request to GCS.
  std::unique_ptr<RandomAccessFile> rfile;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/path/appendable", nullptr, &rfile));
  char scratch[100];
  StringPiece result;
  TF_EXPECT_OK(rfile->Read(0, 8, &result, scratch));
  EXPECT_EQ("content1", result);
  // Closing the appendable file will flush its contents to GCS, triggering HTTP
  // requests.
  TF_EXPECT_OK(wfile->Close());
  // Redo the read. The block should be reloaded from GCS, causing one more HTTP
  // request to load it.
  TF_EXPECT_OK(rfile->Read(0, 4, &result, scratch));
  EXPECT_EQ("0123", result);
}

TEST(GcsFileSystemTest, NewAppendableFile_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<WritableFile> file;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.NewAppendableFile("gs://bucket/", nullptr, &file).code());
}

TEST(GcsFileSystemTest, NewAppendableFile_ObjectDoesNotExist) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/filename\n"
           "Auth Token: fake_token\n"
           "Range: 0-1048575\n"
           "Timeouts: 5 1 20\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o"
           "?uploadType=resumable&name=filename\n"
           "Auth Token: fake_token\n"
           "Header X-Upload-Content-Length: 0\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(fs.NewAppendableFile("gs://bucket/filename", nullptr, &file));
}

TEST(GcsFileSystemTest, NewReadOnlyMemoryRegionFromFile) {
  const string content = "file content";
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Frandom_access.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"", content.size(), "\"",
                           ", \"generation\": \"1\"",
                           ", \"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           strings::StrCat("Uri: https://storage.googleapis.com/bucket/"
                           "path%2Frandom_access.txt\n"
                           "Auth Token: fake_token\n"
                           "Range: 0-",
                           content.size() - 1, "\n", "Timeouts: 5 1 20\n"),
           content)});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<ReadOnlyMemoryRegion> region;
  TF_EXPECT_OK(fs.NewReadOnlyMemoryRegionFromFile(
      "gs://bucket/path/random_access.txt", nullptr, &region));

  EXPECT_EQ(content, StringPiece(reinterpret_cast<const char*>(region->data()),
                                 region->length()));
}

TEST(GcsFileSystemTest, NewReadOnlyMemoryRegionFromFile_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<ReadOnlyMemoryRegion> region;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.NewReadOnlyMemoryRegionFromFile("gs://bucket/", nullptr, &region)
                .code());
}

TEST(GcsFileSystemTest, FileExists_YesAsObject) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "path%2Ffile1.txt?fields=size%2Cgeneration%2Cupdated\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                      "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.FileExists("gs://bucket/path/file1.txt", nullptr));
}

TEST(GcsFileSystemTest, FileExists_YesAsFolder) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsubfolder?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsubfolder%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/subfolder/\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.FileExists("gs://bucket/path/subfolder", nullptr));
}

TEST(GcsFileSystemTest, FileExists_YesAsBucket) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"size\": \"100\"}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"size\": \"100\"}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.FileExists("gs://bucket1", nullptr));
  TF_EXPECT_OK(fs.FileExists("gs://bucket1/", nullptr));
}

TEST(GcsFileSystemTest, FileExists_NotAsObjectOrFolder) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Ffile1.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Ffile1.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": []}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  EXPECT_EQ(errors::Code::NOT_FOUND,
            fs.FileExists("gs://bucket/path/file1.txt", nullptr).code());
}

TEST(GcsFileSystemTest, FileExists_NotAsBucket) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket2\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket2\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.FileExists("gs://bucket2/", nullptr).code());
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.FileExists("gs://bucket2", nullptr).code());
}

TEST(GcsFileSystemTest, FileExists_StatCache) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Ffile1.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsubfolder%2F?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsubfolder%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/subfolder/\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  // The stat cache will ensure that repeated lookups don't trigger additional
  // HTTP requests.
  for (int i = 0; i < 10; i++) {
    TF_EXPECT_OK(fs.FileExists("gs://bucket/path/file1.txt", nullptr));
    TF_EXPECT_OK(fs.FileExists("gs://bucket/path/subfolder/", nullptr));
  }
}

TEST(GcsFileSystemTest, FileExists_DirectoryMark) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "dir%2F?fields=size%2Cgeneration%2Cupdated\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      strings::StrCat("{\"size\": \"5\",\"generation\": \"1\","
                      "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.FileExists("gs://bucket/dir/", nullptr));
  TF_EXPECT_OK(fs.IsDirectory("gs://bucket/dir/", nullptr));
}

TEST(GcsFileSystemTest, GetChildren_NoItems) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&prefix="
      "path%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"prefixes\": [\"path/subpath/\"]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", nullptr, &children));

  EXPECT_EQ(std::vector<string>({"subpath/"}), children);
}

TEST(GcsFileSystemTest, GetChildren_ThreeFiles) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&prefix="
      "path%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/file3.txt\" }],"
      "\"prefixes\": [\"path/subpath/\"]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", nullptr, &children));

  EXPECT_EQ(std::vector<string>({"file1.txt", "file3.txt", "subpath/"}),
            children);
}

TEST(GcsFileSystemTest, GetChildren_SelfDirectoryMarker) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&prefix="
      "path%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/\" },"
      "  { \"name\": \"path/file3.txt\" }],"
      "\"prefixes\": [\"path/subpath/\"]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", nullptr, &children));

  EXPECT_EQ(std::vector<string>({"file3.txt", "subpath/"}), children);
}

TEST(GcsFileSystemTest, GetChildren_ThreeFiles_NoSlash) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&prefix="
      "path%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/file3.txt\" }],"
      "\"prefixes\": [\"path/subpath/\"]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path", nullptr, &children));

  EXPECT_EQ(std::vector<string>({"file1.txt", "file3.txt", "subpath/"}),
            children);
}

TEST(GcsFileSystemTest, GetChildren_Root) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket-a-b-c/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket-a-b-c", nullptr, &children));

  EXPECT_EQ(0, children.size());
}

TEST(GcsFileSystemTest, GetChildren_Empty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&prefix="
      "path%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path/", nullptr, &children));

  EXPECT_EQ(0, children.size());
}

TEST(GcsFileSystemTest, GetChildren_Pagination) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2Cprefixes%2CnextPageToken&delimiter=%2F&"
           "prefix=path%2F\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
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
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/file4.txt\" },"
           "  { \"name\": \"path/file5.txt\" }]}")});

  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> children;
  TF_EXPECT_OK(fs.GetChildren("gs://bucket/path", nullptr, &children));

  EXPECT_EQ(std::vector<string>({"file1.txt", "file3.txt", "subpath/",
                                 "file4.txt", "file5.txt"}),
            children);
}

TEST(GcsFileSystemTest, GetMatchingPaths_NoWildcard) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsubpath%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/subpath/file2.txt\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/path/subpath/file2.txt",
                                   nullptr, &result));
  EXPECT_EQ(std::vector<string>({"gs://bucket/path/subpath/file2.txt"}),
            result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_BucketAndWildcard) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/subpath/file2.txt\" },"
      "  { \"name\": \"path/file3.txt\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/*/*", nullptr, &result));
  EXPECT_EQ(std::vector<string>({"gs://bucket/path/file1.txt",
                                 "gs://bucket/path/file3.txt",
                                 "gs://bucket/path/subpath"}),
            result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_FolderAndWildcard_Matches) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/subpath/file2.txt\" },"
      "  { \"name\": \"path/file3.txt\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> result;
  TF_EXPECT_OK(
      fs.GetMatchingPaths("gs://bucket/path/*/file2.txt", nullptr, &result));
  EXPECT_EQ(std::vector<string>({"gs://bucket/path/subpath/file2.txt"}),
            result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_SelfDirectoryMarker) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/\" },"
      "  { \"name\": \"path/file3.txt\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/path/*", nullptr, &result));
  EXPECT_EQ(std::vector<string>({"gs://bucket/path/file3.txt"}), result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_SlashInObjectName) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/\" },"
      "  { \"name\": \"path//foo.txt\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/path/*", nullptr, &result));
  EXPECT_EQ(std::vector<string>(), result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_SlashInObjectNameEscaped) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/\" },"
      "  { \"name\": \"path//foo.txt\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/path/\\/*", nullptr, &result));
  EXPECT_EQ(std::vector<string>({"gs://bucket/path//foo.txt"}), result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_FolderAndWildcard_NoMatches) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" },"
      "  { \"name\": \"path/subpath/file2.txt\" },"
      "  { \"name\": \"path/file3.txt\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> result;
  TF_EXPECT_OK(
      fs.GetMatchingPaths("gs://bucket/path/*/file3.txt", nullptr, &result));
  EXPECT_EQ(std::vector<string>(), result);
}

TEST(GcsFileSystemTest, GetMatchingPaths_OnlyWildcard) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::vector<string> result;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.GetMatchingPaths("gs://*", nullptr, &result).code());
}

TEST(GcsFileSystemTest, GetMatchingPaths_Cache) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsubpath%2F\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/subpath/file2.txt\" }]}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/file1.txt\" },"
           "  { \"name\": \"path/subpath/file2.txt\" },"
           "  { \"name\": \"path/file3.txt\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 3600 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  // Repeated calls to fs.GetMatchingPaths on these patterns should not lead to
  // any additional HTTP requests to GCS.
  for (int i = 0; i < 10; i++) {
    std::vector<string> result;
    TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/path/subpath/file2.txt",
                                     nullptr, &result));
    EXPECT_EQ(std::vector<string>({"gs://bucket/path/subpath/file2.txt"}),
              result);
    TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/*/*", nullptr, &result));
    EXPECT_EQ(std::vector<string>({"gs://bucket/path/file1.txt",
                                   "gs://bucket/path/file3.txt",
                                   "gs://bucket/path/subpath"}),
              result);
  }
}

TEST(GcsFileSystemTest, GetMatchingPaths_Cache_Flush) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsubpath%2F\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/subpath/file2.txt\" }]}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsubpath%2F\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/subpath/file2.txt\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 3600 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  // This loop should trigger the first HTTP request to GCS.
  for (int i = 0; i < 10; i++) {
    std::vector<string> result;
    TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/path/subpath/file2.txt",
                                     nullptr, &result));
    EXPECT_EQ(std::vector<string>({"gs://bucket/path/subpath/file2.txt"}),
              result);
  }
  // After flushing caches, there should be another (identical) request to GCS.
  fs.FlushCaches(nullptr);
  for (int i = 0; i < 10; i++) {
    std::vector<string> result;
    TF_EXPECT_OK(fs.GetMatchingPaths("gs://bucket/path/subpath/file2.txt",
                                     nullptr, &result));
    EXPECT_EQ(std::vector<string>({"gs://bucket/path/subpath/file2.txt"}),
              result);
  }
}

TEST(GcsFileSystemTest, DeleteFile) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Ffile1.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"8\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Ffile1.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-15\n"
           "Timeouts: 5 1 20\n",
           "01234567"),
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile1.txt\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Ffile1.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"8\",\"generation\": \"2\","
                           "\"updated\": \"2016-04-29T23:19:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Ffile1.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-15\n"
           "Timeouts: 5 1 20\n",
           "76543210")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 16 /* block size */,
      16 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  // Do an initial read of the file to load its contents into the block cache.
  char scratch[100];
  StringPiece result;
  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/path/file1.txt", nullptr, &file));
  TF_EXPECT_OK(file->Read(0, 8, &result, scratch));
  EXPECT_EQ("01234567", result);
  // Deleting the file triggers the next HTTP request to GCS.
  TF_EXPECT_OK(fs.DeleteFile("gs://bucket/path/file1.txt", nullptr));
  // Re-reading the file causes its contents to be reloaded from GCS and not
  // from the block cache.
  TF_EXPECT_OK(file->Read(0, 8, &result, scratch));
  EXPECT_EQ("76543210", result);
}

TEST(GcsFileSystemTest, DeleteFile_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.DeleteFile("gs://bucket/", nullptr).code());
}

TEST(GcsFileSystemTest, DeleteFile_StatCacheRemoved) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "file.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/file.txt\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "file.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=file.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 16 /* block size */,
      16 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  // Stats the file first so the stat is cached.
  FileStatistics stat_before_deletion;
  TF_EXPECT_OK(fs.Stat("gs://bucket/file.txt", nullptr, &stat_before_deletion));
  EXPECT_EQ(1010, stat_before_deletion.length);

  TF_EXPECT_OK(fs.DeleteFile("gs://bucket/file.txt", nullptr));

  FileStatistics stat_after_deletion;
  EXPECT_EQ(
      error::Code::NOT_FOUND,
      fs.Stat("gs://bucket/file.txt", nullptr, &stat_after_deletion).code());
}

TEST(GcsFileSystemTest, DeleteDir_Empty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F&maxResults=2\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.DeleteDir("gs://bucket/path/", nullptr));
}

TEST(GcsFileSystemTest, DeleteDir_OnlyDirMarkerLeft) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F&maxResults=2\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/\" }]}"),
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2F\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           "")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.DeleteDir("gs://bucket/path/", nullptr));
}

TEST(GcsFileSystemTest, DeleteDir_BucketOnly) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?fields=items%2F"
      "name%2CnextPageToken&maxResults=2\nAuth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.DeleteDir("gs://bucket", nullptr));
}

TEST(GcsFileSystemTest, DeleteDir_NonEmpty) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
      "fields=items%2Fname%2CnextPageToken&prefix=path%2F&maxResults=2\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{\"items\": [ "
      "  { \"name\": \"path/file1.txt\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  EXPECT_EQ(error::Code::FAILED_PRECONDITION,
            fs.DeleteDir("gs://bucket/path/", nullptr).code());
}

TEST(GcsFileSystemTest, GetFileSize) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "file.txt?fields=size%2Cgeneration%2Cupdated\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                      "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  uint64 size;
  TF_EXPECT_OK(fs.GetFileSize("gs://bucket/file.txt", nullptr, &size));
  EXPECT_EQ(1010, size);
}

TEST(GcsFileSystemTest, GetFileSize_NoObjectName) {
  std::vector<HttpRequest*> requests;
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  uint64 size;
  EXPECT_EQ(errors::Code::INVALID_ARGUMENT,
            fs.GetFileSize("gs://bucket/", nullptr, &size).code());
}

TEST(GcsFileSystemTest, RenameFile_Folder) {
  std::vector<HttpRequest*> requests(
      {// Check if this is a folder or an object.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path1%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path1/subfolder/file1.txt\" }]}"),
       // Requesting the full list of files in the folder.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path1%2F\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path1/\" },"  // A directory marker.
           "  { \"name\": \"path1/subfolder/file1.txt\" },"
           "  { \"name\": \"path1/file2.txt\" }]}"),
       // Copying the directory marker.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2F/rewriteTo/b/bucket/o/path2%2F\n"
           "Auth Token: fake_token\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "{\"done\": true}"),
       // Deleting the original directory marker.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2F\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n"
           "Delete: yes\n",
           ""),
       // Copying the first file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2Fsubfolder%2Ffile1.txt/rewriteTo/b/bucket/o/"
           "path2%2Fsubfolder%2Ffile1.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "{\"done\": true}"),
       // Deleting the first original file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2Fsubfolder%2Ffile1.txt\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n"
           "Delete: yes\n",
           ""),
       // Copying the second file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2Ffile2.txt/rewriteTo/b/bucket/o/path2%2Ffile2.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "{\"done\": true}"),
       // Deleting the second original file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path1%2Ffile2.txt\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n"
           "Delete: yes\n",
           "")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(
      fs.RenameFile("gs://bucket/path1", "gs://bucket/path2/", nullptr));
}

TEST(GcsFileSystemTest, RenameFile_Object) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"8\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fsrc.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-15\n"
           "Timeouts: 5 1 20\n",
           "01234567"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fdst.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"8\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fdst.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-15\n"
           "Timeouts: 5 1 20\n",
           "76543210"),
       // IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsrc.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}"),
       // Copying to the new location.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt/rewriteTo/b/bucket/o/path%2Fdst.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "{\"done\": true}"),
       // Deleting the original file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n"
           "Delete: yes\n",
           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"8\",\"generation\": \"2\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fsrc.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-15\n"
           "Timeouts: 5 1 20\n",
           "89abcdef"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fdst.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"8\",\"generation\": \"2\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://storage.googleapis.com/bucket/path%2Fdst.txt\n"
           "Auth Token: fake_token\n"
           "Range: 0-15\n"
           "Timeouts: 5 1 20\n",
           "fedcba98")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 16 /* block size */,
      64 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);
  // Do an initial read of the source and destination files to load their
  // contents into the block cache.
  char scratch[100];
  StringPiece result;
  std::unique_ptr<RandomAccessFile> src;
  std::unique_ptr<RandomAccessFile> dst;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/path/src.txt", nullptr, &src));
  TF_EXPECT_OK(src->Read(0, 8, &result, scratch));
  EXPECT_EQ("01234567", result);
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/path/dst.txt", nullptr, &dst));
  TF_EXPECT_OK(dst->Read(0, 8, &result, scratch));
  EXPECT_EQ("76543210", result);
  // Now rename src to dst. This should flush the block cache for both files.
  TF_EXPECT_OK(fs.RenameFile("gs://bucket/path/src.txt",
                             "gs://bucket/path/dst.txt", nullptr));
  // Re-read both files. This should reload their contents from GCS.
  TF_EXPECT_OK(src->Read(0, 8, &result, scratch));
  EXPECT_EQ("89abcdef", result);
  TF_EXPECT_OK(dst->Read(0, 8, &result, scratch));
  EXPECT_EQ("fedcba98", result);
}

TEST(GcsFileSystemTest, RenameFile_Object_FlushTargetStatCache) {
  std::vector<HttpRequest*> requests(
      {// Stat the target file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fdst.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1000\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       // IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsrc.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}"),
       // IsDirectory is checking if the path exists as an object.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       // Copying to the new location.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt/rewriteTo/b/bucket/o/path%2Fdst.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "{\"done\": true}"),
       // Deleting the original file.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n"
           "Delete: yes\n",
           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fdst.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);
  // Do an initial stat of the destination file to load their contents into the
  // stat cache.
  FileStatistics stat_before_renaming;
  TF_EXPECT_OK(
      fs.Stat("gs://bucket/path/dst.txt", nullptr, &stat_before_renaming));
  EXPECT_EQ(1000, stat_before_renaming.length);

  TF_EXPECT_OK(fs.RenameFile("gs://bucket/path/src.txt",
                             "gs://bucket/path/dst.txt", nullptr));

  FileStatistics stat_after_renaming;
  TF_EXPECT_OK(
      fs.Stat("gs://bucket/path/dst.txt", nullptr, &stat_after_renaming));
  EXPECT_EQ(1010, stat_after_renaming.length);
}

/// Tests the scenario when deletion returns a failure, but actually succeeds.
TEST(GcsFileSystemTest, RenameFile_Object_DeletionRetried) {
  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsrc.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}"),
       // IsDirectory is checking if the path exists as an object.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       // Copying to the new location.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt/rewriteTo/b/bucket/o/path%2Fdst.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "{\"done\": true}"),
       // Deleting the original file - the deletion returns a failure.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n"
           "Delete: yes\n",
           "", errors::Unavailable("503"), 503),
       // Deleting the original file again - the deletion returns NOT_FOUND.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n"
           "Delete: yes\n",
           "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.RenameFile("gs://bucket/path/src.txt",
                             "gs://bucket/path/dst.txt", nullptr));
}

/// Tests the case when rewrite couldn't complete in one RPC.
TEST(GcsFileSystemTest, RenameFile_Object_Incomplete) {
  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsrc.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}"),
       // IsDirectory is checking if the path exists as an object.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       // Copying to the new location.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Fsrc.txt/rewriteTo/b/bucket/o/path%2Fdst.txt\n"
           "Auth Token: fake_token\n"
           "Post: yes\n"
           "Timeouts: 5 1 10\n",
           "{\"done\": false}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  EXPECT_EQ(errors::Code::UNIMPLEMENTED,
            fs.RenameFile("gs://bucket/path/src.txt",
                          "gs://bucket/path/dst.txt", nullptr)
                .code());
}

TEST(GcsFileSystemTest, Stat_Object) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "file.txt?fields=size%2Cgeneration%2Cupdated\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                      "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("gs://bucket/file.txt", nullptr, &stat));
  EXPECT_EQ(1010, stat.length);
  EXPECT_NEAR(1461971724896, stat.mtime_nsec / 1000 / 1000, 1);
  EXPECT_FALSE(stat.is_directory);
}

TEST(GcsFileSystemTest, Stat_Folder) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "subfolder?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=subfolder%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"subfolder/\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("gs://bucket/subfolder", nullptr, &stat));
  EXPECT_EQ(0, stat.length);
  EXPECT_EQ(0, stat.mtime_nsec);
  EXPECT_TRUE(stat.is_directory);
}

TEST(GcsFileSystemTest, Stat_ObjectOrFolderNotFound) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  FileStatistics stat;
  EXPECT_EQ(error::Code::NOT_FOUND,
            fs.Stat("gs://bucket/path", nullptr, &stat).code());
}

TEST(GcsFileSystemTest, Stat_Bucket) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "{}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("gs://bucket/", nullptr, &stat));
  EXPECT_EQ(0, stat.length);
  EXPECT_EQ(0, stat.mtime_nsec);
  EXPECT_TRUE(stat.is_directory);
}

TEST(GcsFileSystemTest, Stat_BucketNotFound) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  FileStatistics stat;
  EXPECT_EQ(error::Code::NOT_FOUND,
            fs.Stat("gs://bucket/", nullptr, &stat).code());
}

TEST(GcsFileSystemTest, Stat_Cache) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "file.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "subfolder%2F?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=subfolder%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"subfolder/\" }]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  // Repeated calls to fs.Stat on these paths should not lead to any additional
  // HTTP requests to GCS.
  for (int i = 0; i < 10; i++) {
    FileStatistics stat;
    TF_EXPECT_OK(fs.Stat("gs://bucket/file.txt", nullptr, &stat));
    EXPECT_EQ(1010, stat.length);
    EXPECT_NEAR(1461971724896, stat.mtime_nsec / 1000 / 1000, 1);
    EXPECT_FALSE(stat.is_directory);
    TF_EXPECT_OK(fs.Stat("gs://bucket/subfolder/", nullptr, &stat));
    EXPECT_EQ(0, stat.length);
    EXPECT_EQ(0, stat.mtime_nsec);
    EXPECT_TRUE(stat.is_directory);
  }
}

TEST(GcsFileSystemTest, Stat_Cache_Flush) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "file.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "file.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);
  // There should be a single HTTP request to GCS for fs.Stat in this loop.
  for (int i = 0; i < 10; i++) {
    FileStatistics stat;
    TF_EXPECT_OK(fs.Stat("gs://bucket/file.txt", nullptr, &stat));
    EXPECT_EQ(1010, stat.length);
    EXPECT_NEAR(1461971724896, stat.mtime_nsec / 1000 / 1000, 1);
    EXPECT_FALSE(stat.is_directory);
  }
  // After flushing caches, there should be a second request to GCS for fs.Stat.
  fs.FlushCaches(nullptr);
  for (int i = 0; i < 10; i++) {
    FileStatistics stat;
    TF_EXPECT_OK(fs.Stat("gs://bucket/file.txt", nullptr, &stat));
    EXPECT_EQ(1010, stat.length);
    EXPECT_NEAR(1461971724896, stat.mtime_nsec / 1000 / 1000, 1);
    EXPECT_FALSE(stat.is_directory);
  }
}

TEST(GcsFileSystemTest, Stat_FilenameEndingWithSlash) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "dir%2F?fields=size%2Cgeneration%2Cupdated\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      strings::StrCat("{\"size\": \"5\",\"generation\": \"1\","
                      "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("gs://bucket/dir/", nullptr, &stat));
  EXPECT_EQ(5, stat.length);
  EXPECT_TRUE(stat.is_directory);
}

TEST(GcsFileSystemTest, IsDirectory_NotFound) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=file.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "file.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  EXPECT_EQ(error::Code::NOT_FOUND,
            fs.IsDirectory("gs://bucket/file.txt", nullptr).code());
}

TEST(GcsFileSystemTest, IsDirectory_NotDirectoryButObject) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=file.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "file.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                           "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  EXPECT_EQ(error::Code::FAILED_PRECONDITION,
            fs.IsDirectory("gs://bucket/file.txt", nullptr).code());
}

TEST(GcsFileSystemTest, IsDirectory_Yes) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=subfolder%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [{\"name\": \"subfolder/\"}]}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=subfolder%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [{\"name\": \"subfolder/\"}]}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.IsDirectory("gs://bucket/subfolder", nullptr));
  TF_EXPECT_OK(fs.IsDirectory("gs://bucket/subfolder/", nullptr));
}

TEST(GcsFileSystemTest, IsDirectory_Bucket) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}"),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.IsDirectory("gs://bucket", nullptr));
  TF_EXPECT_OK(fs.IsDirectory("gs://bucket/", nullptr));
}

TEST(GcsFileSystemTest, IsDirectory_BucketNotFound) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  EXPECT_EQ(error::Code::NOT_FOUND,
            fs.IsDirectory("gs://bucket/", nullptr).code());
}

TEST(GcsFileSystemTest, CreateDir_Folder) {
  std::vector<HttpRequest*> requests(

      {
          // File doesn't exist.
          new FakeHttpRequest(
              "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
              "subpath%2F?fields=size%2Cgeneration%2Cupdated\n"
              "Auth Token: fake_token\n"
              "Timeouts: 5 1 10\n",
              "{}"),
          // Simple upload.
          new FakeHttpRequest(
              "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
              "uploadType=media&name=subpath%2F&ifGenerationMatch=0\n"
              "Auth Token: fake_token\n"
              "Post: yes\n"
              "Timeouts: 5 1 10\n",
              ""),
          // File exists.
          new FakeHttpRequest(
              "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
              "subpath%2F?fields=size%2Cgeneration%2Cupdated\n"
              "Auth Token: fake_token\n"
              "Timeouts: 5 1 10\n",
              strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                              "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
          // File doesn't exist again.
          new FakeHttpRequest(
              "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
              "subpath%2F?fields=size%2Cgeneration%2Cupdated\n"
              "Auth Token: fake_token\n"
              "Timeouts: 5 1 10\n",
              "{}"),
          // Simulate object uploaded in between.
          new FakeHttpRequest(
              "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
              "uploadType=media&name=subpath%2F&ifGenerationMatch=0\n"
              "Auth Token: fake_token\n"
              "Post: yes\n"
              "Timeouts: 5 1 10\n",
              "", errors::FailedPrecondition("412"), 412),
      });
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.CreateDir("gs://bucket/subpath", nullptr));
  // Check that when GCS returns the object already exists return that the
  // directory already exists.
  EXPECT_EQ(errors::AlreadyExists("gs://bucket/subpath"),
            fs.CreateDir("gs://bucket/subpath", nullptr));
  // Check that when GCS returns the object already has a version (failed
  // precondition) return directory already exists.
  EXPECT_EQ(errors::AlreadyExists("gs://bucket/subpath"),
            fs.CreateDir("gs://bucket/subpath", nullptr));
}

TEST(GcsFileSystemTest, CreateDir_Bucket) {
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           ""),
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TF_EXPECT_OK(fs.CreateDir("gs://bucket/", nullptr));
  TF_EXPECT_OK(fs.CreateDir("gs://bucket", nullptr));
}

TEST(GcsFileSystemTest, DeleteRecursively_Ok) {
  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/file1.txt\" }]}"),
       // GetChildren recursively.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/\" },"  // The current directory's marker.
           "  { \"name\": \"path/file1.txt\" },"
           "  { \"name\": \"path/subpath/file2.txt\" },"
           "  { \"name\": \"path/file3.txt\" }]}"),
       // Delete the current directory's marker.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2F\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           ""),
       // Delete the object - fails and will be retried.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile1.txt\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           "", errors::Unavailable("500"), 500),
       // Delete the object again.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile1.txt\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           ""),
       // Delete the object.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Fsubpath%2Ffile2.txt\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           ""),
       // Delete the object.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile3.txt\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           "")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  int64_t undeleted_files, undeleted_dirs;
  TF_EXPECT_OK(fs.DeleteRecursively("gs://bucket/path", nullptr,
                                    &undeleted_files, &undeleted_dirs));
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
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/file1.txt\" }]}"),
       // Calling GetChildren recursively.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2F\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{\"items\": [ "
           "  { \"name\": \"path/file1.txt\" },"
           "  { \"name\": \"path/subpath/\" },"
           "  { \"name\": \"path/subpath/file2.txt\" },"
           "  { \"name\": \"path/file3.txt\" }]}"),
       // Deleting the object.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile1.txt\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           ""),
       // Deleting the directory marker gs://bucket/path/ - fails with 404.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Fsubpath%2F\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           "", errors::NotFound("404"), 404),
       // Checking if gs://bucket/path/subpath/ is a folder - it is.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Fsubpath%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           strings::StrCat("{\"items\": [ "
                           "    { \"name\": \"path/subpath/\" }]}")),
       // Deleting the object gs://bucket/path/subpath/file2.txt
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Fsubpath%2Ffile2.txt\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           ""),
       // Deleting the object s://bucket/path/file3.txt - fails with 404.
       new FakeHttpRequest("Uri: https://www.googleapis.com/storage/v1/b"
                           "/bucket/o/path%2Ffile3.txt\n"
                           "Auth Token: fake_token\n"
                           "Timeouts: 5 1 10\n"
                           "Delete: yes\n",
                           "", errors::NotFound("404"), 404),
       // Checking if gs://bucket/path/file3.txt/ is a folder - it's not.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o?"
           "fields=items%2Fname%2CnextPageToken&prefix=path%2Ffile3.txt%2F"
           "&maxResults=1\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}"),
       // Checking if gs://bucket/path/file3.txt is an object - fails with 404.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path%2Ffile3.txt?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404)});

  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  int64_t undeleted_files, undeleted_dirs;
  TF_EXPECT_OK(fs.DeleteRecursively("gs://bucket/path", nullptr,
                                    &undeleted_files, &undeleted_dirs));
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
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "{}"),
       // IsDirectory is checking if the path exists as an object.
       new FakeHttpRequest(
           "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
           "path?fields=size%2Cgeneration%2Cupdated\n"
           "Auth Token: fake_token\n"
           "Timeouts: 5 1 10\n",
           "", errors::NotFound("404"), 404)});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  int64_t undeleted_files, undeleted_dirs;
  EXPECT_EQ(error::Code::NOT_FOUND,
            fs.DeleteRecursively("gs://bucket/path", nullptr, &undeleted_files,
                                 &undeleted_dirs)
                .code());
  EXPECT_EQ(0, undeleted_files);
  EXPECT_EQ(1, undeleted_dirs);
}

TEST(GcsFileSystemTest, NoConstraintsEnvironmentVariableTest) {
  unsetenv("GCS_ALLOWED_BUCKET_LOCATIONS");
  // No constraints
  GcsFileSystem fs1;
  EXPECT_EQ(*kAllowedLocationsDefault, fs1.allowed_locations());

  // Cover cache initialization code, any uninitialized cache will cause this to
  // fail
  fs1.FlushCaches(nullptr);
}

TEST(GcsFileSystemTest, BucketLocationConstraintEnvironmentVariableTest) {
  unsetenv("GCS_ALLOWED_BUCKET_LOCATIONS");
  setenv("GCS_ALLOWED_BUCKET_LOCATIONS", "auto", 1);
  GcsFileSystem fs1;
  EXPECT_EQ(*kAllowedLocationsAuto, fs1.allowed_locations());

  setenv("GCS_ALLOWED_BUCKET_LOCATIONS", "CUSTOM,list", 1);
  GcsFileSystem fs2;
  EXPECT_EQ(std::unordered_set<string>({"custom", "list"}),
            fs2.allowed_locations());
}

TEST(GcsFileSystemTest, AdditionalRequestHeaderTest) {
  GcsFileSystem fs1;
  EXPECT_EQ("", fs1.additional_header_name());
  EXPECT_EQ("", fs1.additional_header_value());

  setenv("GCS_ADDITIONAL_REQUEST_HEADER",
         "X-Add-Header:My Additional Header Value", 1);
  GcsFileSystem fs2;
  EXPECT_EQ("X-Add-Header", fs2.additional_header_name());
  EXPECT_EQ("My Additional Header Value", fs2.additional_header_value());

  setenv("GCS_ADDITIONAL_REQUEST_HEADER", "Someinvalidheadervalue", 1);
  GcsFileSystem fs3;
  EXPECT_EQ("", fs3.additional_header_name());
  EXPECT_EQ("", fs3.additional_header_value());

  setenv("GCS_ADDITIONAL_REQUEST_HEADER", ":thisisinvalid", 1);
  GcsFileSystem fs4;
  EXPECT_EQ("", fs4.additional_header_name());
  EXPECT_EQ("", fs4.additional_header_value());

  setenv("GCS_ADDITIONAL_REQUEST_HEADER", "soisthis:", 1);
  GcsFileSystem fs5;
  EXPECT_EQ("", fs5.additional_header_name());
  EXPECT_EQ("", fs5.additional_header_value());

  setenv("GCS_ADDITIONAL_REQUEST_HEADER", "a:b", 1);
  GcsFileSystem fs6;
  EXPECT_EQ("a", fs6.additional_header_name());
  EXPECT_EQ("b", fs6.additional_header_value());

  auto* add_header = new std::pair<const string, const string>(
      "mynewheader", "newheadercontents");

  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest("Uri: https://www.googleapis.com/fake\n"
                           "Auth Token: fake_token\n"
                           "Header mynewheader: newheadercontents\n"
                           "Header Hello: world\n",
                           "{}")});
  GcsFileSystem fs7(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      add_header /* gcs additional header */, false /* compose append */);

  std::unique_ptr<HttpRequest> request;
  TF_EXPECT_OK(fs7.CreateHttpRequest(&request));
  request->SetUri("https://www.googleapis.com/fake");
  request->AddHeader("Hello", "world");
  TF_EXPECT_OK(request->Send());
}

TEST(GcsFileSystemTest, OverrideCacheParameters) {
  // Verify defaults are propagated correctly.
  setenv("GCS_READ_CACHE_BLOCK_SIZE_MB", "16", 1);
  setenv("GCS_READ_CACHE_MAX_SIZE_MB", "128", 1);
  GcsFileSystem fs1;
  EXPECT_EQ(16 * 1024 * 1024, fs1.block_size());
  EXPECT_EQ(128 * 1024 * 1024, fs1.max_bytes());
  EXPECT_EQ(0, fs1.max_staleness());
  EXPECT_EQ(120, fs1.timeouts().connect);
  EXPECT_EQ(60, fs1.timeouts().idle);
  EXPECT_EQ(3600, fs1.timeouts().metadata);
  EXPECT_EQ(3600, fs1.timeouts().read);
  EXPECT_EQ(3600, fs1.timeouts().write);

  // Verify legacy readahead buffer override sets block size.
  unsetenv("GCS_READ_CACHE_BLOCK_SIZE_MB");
  setenv("GCS_READAHEAD_BUFFER_SIZE_BYTES", "123456789", 1);
  GcsFileSystem fs2;
  EXPECT_EQ(123456789L, fs2.block_size());

  // Verify block size, max size, and max staleness overrides.
  setenv("GCS_READ_CACHE_BLOCK_SIZE_MB", "1", 1);
  setenv("GCS_READ_CACHE_MAX_SIZE_MB", "16", 1);
  setenv("GCS_READ_CACHE_MAX_STALENESS", "60", 1);
  GcsFileSystem fs3;
  EXPECT_EQ(1048576L, fs3.block_size());
  EXPECT_EQ(16 * 1024 * 1024, fs3.max_bytes());
  EXPECT_EQ(60, fs3.max_staleness());

  // Verify StatCache and MatchingPathsCache overrides.
  setenv("GCS_STAT_CACHE_MAX_AGE", "60", 1);
  setenv("GCS_STAT_CACHE_MAX_ENTRIES", "32", 1);
  setenv("GCS_MATCHING_PATHS_CACHE_MAX_AGE", "30", 1);
  setenv("GCS_MATCHING_PATHS_CACHE_MAX_ENTRIES", "64", 1);
  GcsFileSystem fs4;
  EXPECT_EQ(60, fs4.stat_cache_max_age());
  EXPECT_EQ(32, fs4.stat_cache_max_entries());
  EXPECT_EQ(30, fs4.matching_paths_cache_max_age());
  EXPECT_EQ(64, fs4.matching_paths_cache_max_entries());

  // Verify timeout overrides.
  setenv("GCS_REQUEST_CONNECTION_TIMEOUT_SECS", "10", 1);
  setenv("GCS_REQUEST_IDLE_TIMEOUT_SECS", "5", 1);
  setenv("GCS_METADATA_REQUEST_TIMEOUT_SECS", "20", 1);
  setenv("GCS_READ_REQUEST_TIMEOUT_SECS", "30", 1);
  setenv("GCS_WRITE_REQUEST_TIMEOUT_SECS", "40", 1);
  GcsFileSystem fs5;
  EXPECT_EQ(10, fs5.timeouts().connect);
  EXPECT_EQ(5, fs5.timeouts().idle);
  EXPECT_EQ(20, fs5.timeouts().metadata);
  EXPECT_EQ(30, fs5.timeouts().read);
  EXPECT_EQ(40, fs5.timeouts().write);
}

TEST(GcsFileSystemTest, CreateHttpRequest) {
  std::vector<HttpRequest*> requests(
      {// IsDirectory is checking whether there are children objects.
       new FakeHttpRequest("Uri: https://www.googleapis.com/fake\n"
                           "Auth Token: fake_token\n"
                           "Header Hello: world\n",
                           "{}")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  std::unique_ptr<HttpRequest> request;
  TF_EXPECT_OK(fs.CreateHttpRequest(&request));
  request->SetUri("https://www.googleapis.com/fake");
  request->AddHeader("Hello", "world");
  TF_EXPECT_OK(request->Send());
}

class TestGcsStats : public GcsStatsInterface {
 public:
  void Configure(GcsFileSystem* fs, GcsThrottle* throttle,
                 const FileBlockCache* block_cache) override {
    CHECK(fs_ == nullptr);
    CHECK(throttle_ == nullptr);
    CHECK(block_cache_ == nullptr);

    fs_ = fs;
    throttle_ = throttle;
    block_cache_ = block_cache;
  }

  void RecordBlockLoadRequest(const string& file, size_t offset) override {
    block_load_request_file_ = file;
  }

  void RecordBlockRetrieved(const string& file, size_t offset,
                            size_t bytes_transferred) override {
    block_retrieved_file_ = file;
    block_retrieved_bytes_transferred_ = bytes_transferred;
  }

  void RecordStatObjectRequest() override { stat_object_request_count_++; }

  HttpRequest::RequestStats* HttpStats() override { return nullptr; }

  GcsFileSystem* fs_ = nullptr;
  GcsThrottle* throttle_ = nullptr;
  const FileBlockCache* block_cache_ = nullptr;

  string block_load_request_file_;
  string block_retrieved_file_;
  size_t block_retrieved_bytes_transferred_ = 0;
  int stat_object_request_count_ = 0;
};

TEST(GcsFileSystemTest, Stat_StatsRecording) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
      "file.txt?fields=size%2Cgeneration%2Cupdated\n"
      "Auth Token: fake_token\n"
      "Timeouts: 5 1 10\n",
      strings::StrCat("{\"size\": \"1010\",\"generation\": \"1\","
                      "\"updated\": \"2016-04-29T23:15:24.896Z\"}"))});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TestGcsStats stats;
  fs.SetStats(&stats);
  EXPECT_EQ(stats.fs_, &fs);

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("gs://bucket/file.txt", nullptr, &stat));
  EXPECT_EQ(1, stats.stat_object_request_count_);
}

TEST(GcsFileSystemTest, NewRandomAccessFile_StatsRecording) {
  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://storage.googleapis.com/bucket/random_access.txt\n"
      "Auth Token: fake_token\n"
      "Range: 0-5\n"
      "Timeouts: 5 1 20\n",
      "012345")});
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 0 /* block size */,
      0 /* max bytes */, 0 /* max staleness */, 0 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  TestGcsStats stats;
  fs.SetStats(&stats);
  EXPECT_EQ(stats.fs_, &fs);

  std::unique_ptr<RandomAccessFile> file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("gs://bucket/random_access.txt", nullptr, &file));

  char scratch[6];
  StringPiece result;

  TF_EXPECT_OK(file->Read(0, sizeof(scratch), &result, scratch));
  EXPECT_EQ("012345", result);

  EXPECT_EQ("gs://bucket/random_access.txt", stats.block_load_request_file_);
  EXPECT_EQ("gs://bucket/random_access.txt", stats.block_retrieved_file_);
  EXPECT_EQ(6, stats.block_retrieved_bytes_transferred_);
}

TEST(GcsFileSystemTest, NewAppendableFile_MultipleFlushesWithCompose) {
  std::vector<string> contents(
      {"content0,", "content1,", "content2,", "content3,"});
  std::vector<HttpRequest*> requests({
      // Fetch the file (stats and then content)
      new FakeHttpRequest(
          "Uri: "
          "https://www.googleapis.com/storage/v1/b/bucket/o/"
          "some%2Fpath%2Fappendable?fields=size%2Cgeneration%2Cupdated\n"
          "Auth Token: fake_token\n"
          "Timeouts: 5 1 10\n",
          strings::StrCat("{\"size\": \"8\",\"generation\": \"1\","
                          "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
      new FakeHttpRequest(
          "Uri: "
          "https://storage.googleapis.com/bucket/some%2Fpath%2Fappendable\n"
          "Auth Token: fake_token\n"
          "Range: 0-1048575\n"
          "Timeouts: 5 1 20\n",
          contents[0]),
      // Upload entire file
      new FakeHttpRequest(
          "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
          "uploadType=resumable&name=some%2Fpath%2Fappendable\n"
          "Auth Token: fake_token\n"
          "Header X-Upload-Content-Length: 18\n"
          "Post: yes\n"
          "Timeouts: 5 1 10\n",
          "", {{"Location", "https://custom/upload/location"}}),
      new FakeHttpRequest(
          strings::StrCat("Uri: https://custom/upload/location\n"
                          "Auth Token: fake_token\n"
                          "Header Content-Range: bytes 0-17/18\n"
                          "Timeouts: 5 1 30\n"
                          "Put body: ",
                          contents[0], contents[1], "\n"),
          ""),
      // Upload new part to a temporary object
      new FakeHttpRequest(
          "Uri: "
          "https://www.googleapis.com/upload/storage/v1/b/bucket/"
          "o?uploadType=resumable&name=some%2Fpath%2F.tmpcompose%2Fappendable."
          "18\n"
          "Auth Token: fake_token\n"
          "Header X-Upload-Content-Length: 9\n"
          "Post: yes\n"
          "Timeouts: 5 1 10\n",
          "",
          {{"Location",
            "https://custom/upload/"
            "location"}}),
      new FakeHttpRequest(
          strings::StrCat("Uri: https://custom/upload/location\n"
                          "Auth Token: fake_token\n"
                          "Header Content-Range: bytes 0-8/9\n"
                          "Timeouts: 5 1 30\n"
                          "Put body: ",
                          contents[2], "\n"),
          ""),
      // Fetch generation
      new FakeHttpRequest(
          "Uri: "
          "https://www.googleapis.com/storage/v1/b/bucket/o/"
          "some%2Fpath%2Fappendable?fields=size%2Cgeneration%2Cupdated\n"
          "Auth Token: fake_token\n"
          "Timeouts: 5 1 10\n",
          strings::StrCat("{\"size\": \"8\",\"generation\": \"1234\","
                          "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
      // Compose the new part at the end of the original object.
      new FakeHttpRequest("Uri: "
                          "https://www.googleapis.com/storage/v1/b/bucket/o/"
                          "some%2Fpath%2Fappendable/compose\n"
                          "Auth Token: fake_token\n"
                          "Timeouts: 5 1 10\n"
                          "Header content-type: application/json\n"
                          "Post body: {'sourceObjects': [{'name': "
                          "'some/path/"
                          "appendable','objectPrecondition':{'"
                          "ifGenerationMatch':1234}},{'name': "
                          "'some/path/.tmpcompose/appendable.18'}]}\n",
                          ""),
      // Delete the temporary object.
      new FakeHttpRequest("Uri: "
                          "https://www.googleapis.com/storage/v1/b/bucket/o/"
                          "some%2Fpath%2F.tmpcompose%2Fappendable.18\n"
                          "Auth Token: fake_token\n"
                          "Timeouts: 5 1 10\n"
                          "Delete: yes\n",
                          ""),
      new FakeHttpRequest(
          "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
          "uploadType=resumable&name=some%2Fpath%2F.tmpcompose%2Fappendable."
          "27\n"
          "Auth Token: fake_token\n"
          "Header X-Upload-Content-Length: 9\n"
          "Post: yes\n"
          "Timeouts: 5 1 10\n",
          "", {{"Location", "https://custom/upload/location"}}),
      new FakeHttpRequest(
          strings::StrCat("Uri: https://custom/upload/location\n"
                          "Auth Token: fake_token\n"
                          "Header Content-Range: bytes 0-8/9\n"
                          "Timeouts: 5 1 30\n"
                          "Put body: ",
                          contents[3], "\n"),
          ""),
      // Fetch generation
      new FakeHttpRequest(
          "Uri: "
          "https://www.googleapis.com/storage/v1/b/bucket/o/"
          "some%2Fpath%2Fappendable?fields=size%2Cgeneration%2Cupdated\n"
          "Auth Token: fake_token\n"
          "Timeouts: 5 1 10\n",
          strings::StrCat("{\"size\": \"8\",\"generation\": \"4567\","
                          "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
      new FakeHttpRequest("Uri: "
                          "https://www.googleapis.com/storage/v1/b/bucket/o/"
                          "some%2Fpath%2Fappendable/compose\n"
                          "Auth Token: fake_token\n"
                          "Timeouts: 5 1 10\n"
                          "Header content-type: application/json\n"
                          "Post body: {'sourceObjects': [{'name': "
                          "'some/path/"
                          "appendable','objectPrecondition':{'"
                          "ifGenerationMatch':4567}},{'name': "
                          "'some/path/.tmpcompose/appendable.27'}]}\n",
                          ""),
      new FakeHttpRequest("Uri: "
                          "https://www.googleapis.com/storage/v1/b/bucket/o/"
                          "some%2Fpath%2F.tmpcompose%2Fappendable."
                          "27\n"
                          "Auth Token: fake_token\n"
                          "Timeouts: 5 1 10\n"
                          "Delete: yes\n",
                          ""),
  });
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 32 /* block size */,
      32 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, true /* compose append */);

  // Create an appendable file. This should read the file from GCS, and pull its
  // contents into the block cache.
  std::unique_ptr<WritableFile> wfile;
  TF_EXPECT_OK(fs.NewAppendableFile("gs://bucket/some/path/appendable", nullptr,
                                    &wfile));
  TF_EXPECT_OK(wfile->Append(contents[1]));
  TF_EXPECT_OK(wfile->Flush());
  TF_EXPECT_OK(wfile->Append(contents[2]));
  TF_EXPECT_OK(wfile->Flush());
  TF_EXPECT_OK(wfile->Append(contents[3]));
  TF_EXPECT_OK(wfile->Close());
}

TEST(GcsFileSystemTest, NewAppendableFile_MultipleFlushesWithoutCompose) {
  std::vector<string> contents(
      {"content0,", "content1,", "content2,", "content3,"});
  std::vector<HttpRequest*> requests({
      new FakeHttpRequest(
          "Uri: https://www.googleapis.com/storage/v1/b/bucket/o/"
          "path%2Fappendable?fields=size%2Cgeneration%2Cupdated\n"
          "Auth Token: fake_token\n"
          "Timeouts: 5 1 10\n",
          strings::StrCat("{\"size\": \"8\",\"generation\": \"1\","
                          "\"updated\": \"2016-04-29T23:15:24.896Z\"}")),
      new FakeHttpRequest(
          "Uri: https://storage.googleapis.com/bucket/path%2Fappendable\n"
          "Auth Token: fake_token\n"
          "Range: 0-1048575\n"
          "Timeouts: 5 1 20\n",
          contents[0]),
      new FakeHttpRequest(
          "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
          "uploadType=resumable&name=path%2Fappendable\n"
          "Auth Token: fake_token\n"
          "Header X-Upload-Content-Length: 18\n"
          "Post: yes\n"
          "Timeouts: 5 1 10\n",
          "", {{"Location", "https://custom/upload/location"}}),
      // Uploads entire file.
      new FakeHttpRequest(
          strings::StrCat("Uri: https://custom/upload/location\n"
                          "Auth Token: fake_token\n"
                          "Header Content-Range: bytes 0-17/18\n"
                          "Timeouts: 5 1 30\n"
                          "Put body: ",
                          contents[0], contents[1], "\n"),
          ""),
      new FakeHttpRequest("Uri: "
                          "https://www.googleapis.com/upload/storage/v1/b/"
                          "bucket/o?"
                          "uploadType=resumable&name=path%2Fappendable\n"
                          "Auth Token: fake_token\n"
                          "Header X-Upload-Content-Length: 27\n"
                          "Post: yes\n"
                          "Timeouts: 5 1 10\n",
                          "",
                          {{"Location",
                            "https://custom/upload/"
                            "location"}}),
      // Uploads entire file again.
      new FakeHttpRequest(
          strings::StrCat("Uri: https://custom/upload/location\n"
                          "Auth Token: fake_token\n"
                          "Header Content-Range: bytes 0-26/27\n"
                          "Timeouts: 5 1 30\n"
                          "Put body: ",
                          contents[0], contents[1], contents[2], "\n"),
          ""),
      new FakeHttpRequest(
          "Uri: https://www.googleapis.com/upload/storage/v1/b/bucket/o?"
          "uploadType=resumable&name=path%2Fappendable\n"
          "Auth Token: fake_token\n"
          "Header X-Upload-Content-Length: 36\n"
          "Post: yes\n"
          "Timeouts: 5 1 10\n",
          "", {{"Location", "https://custom/upload/location"}}),
      // Uploads entire file again.
      new FakeHttpRequest(
          strings::StrCat("Uri: https://custom/upload/location\n"
                          "Auth Token: fake_token\n"
                          "Header Content-Range: bytes 0-35/36\n"
                          "Timeouts: 5 1 30\n"
                          "Put body: ",
                          contents[0], contents[1], contents[2], contents[3],
                          "\n"),
          ""),
  });
  GcsFileSystem fs(
      std::unique_ptr<AuthProvider>(new FakeAuthProvider),
      std::unique_ptr<HttpRequest::Factory>(
          new FakeHttpRequestFactory(&requests)),
      std::unique_ptr<ZoneProvider>(new FakeZoneProvider), 32 /* block size */,
      32 /* max bytes */, 0 /* max staleness */, 3600 /* stat cache max age */,
      0 /* stat cache max entries */, 0 /* matching paths cache max age */,
      0 /* matching paths cache max entries */, kTestRetryConfig,
      kTestTimeoutConfig, *kAllowedLocationsDefault,
      nullptr /* gcs additional header */, false /* compose append */);

  // Create an appendable file. This should read the file from GCS, and pull its
  // contents into the block cache.
  std::unique_ptr<WritableFile> wfile;
  TF_EXPECT_OK(
      fs.NewAppendableFile("gs://bucket/path/appendable", nullptr, &wfile));
  TF_EXPECT_OK(wfile->Append(contents[1]));
  TF_EXPECT_OK(wfile->Flush());
  TF_EXPECT_OK(wfile->Append(contents[2]));
  TF_EXPECT_OK(wfile->Flush());
  TF_EXPECT_OK(wfile->Append(contents[3]));
  TF_EXPECT_OK(wfile->Close());
}

TEST(GcsFileSystemTest, AppendModeCompose) {
  unsetenv("GCS_APPEND_MODE");
  setenv("GCS_APPEND_MODE", "compose", 1);
  GcsFileSystem fs1;
  EXPECT_EQ(true, fs1.compose_append());
}

TEST(GcsFileSystemTest, AppendModeDefault) {
  unsetenv("GCS_APPEND_MODE");
  GcsFileSystem fs1;
  EXPECT_EQ(false, fs1.compose_append());
}

}  // namespace
}  // namespace tensorflow
