/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/http_request.h"
#include <fstream>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// A fake proxy that pretends to be libcurl.
class FakeLibCurl : public LibCurl {
 public:
  FakeLibCurl(const string& response_content, uint64 response_code)
      : response_content(response_content), response_code(response_code) {}
  Status MaybeLoadDll() override { return Status::OK(); }
  CURL* curl_easy_init() override {
    is_initialized = true;
    // The reuslt just needs to be non-null.
    return reinterpret_cast<CURL*>(this);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            uint64 param) override {
    switch (option) {
      case CURLOPT_POST:
        is_post = param;
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            const char* param) override {
    return curl_easy_setopt(curl, option,
                            reinterpret_cast<void*>(const_cast<char*>(param)));
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            void* param) override {
    switch (option) {
      case CURLOPT_URL:
        url = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_RANGE:
        range = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_CUSTOMREQUEST:
        custom_request = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_HTTPHEADER:
        headers = reinterpret_cast<std::vector<string>*>(param);
        break;
      case CURLOPT_ERRORBUFFER:
        error_buffer = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_WRITEDATA:
        write_data = reinterpret_cast<FILE*>(param);
        break;
      case CURLOPT_READDATA:
        read_data = reinterpret_cast<FILE*>(param);
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(void*, size_t, size_t,
                                            FILE*)) override {
    EXPECT_EQ(param, &fread) << "Expected the standard fread() function.";
    return CURLE_OK;
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(const void*, size_t, size_t,
                                            void*)) override {
    switch (option) {
      case CURLOPT_WRITEFUNCTION:
        write_callback = param;
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_perform(CURL* curl) override {
    if (read_data) {
      char buffer[100];
      int bytes_read;
      posted_content = "";
      do {
        bytes_read = fread(buffer, 1, 100, read_data);
        posted_content =
            strings::StrCat(posted_content, StringPiece(buffer, bytes_read));
      } while (bytes_read > 0);
    }
    if (write_data) {
      write_callback(response_content.c_str(), 1, response_content.size(),
                     write_data);
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             uint64* value) override {
    switch (info) {
      case CURLINFO_RESPONSE_CODE:
        *value = response_code;
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             double* value) override {
    switch (info) {
      case CURLINFO_SIZE_DOWNLOAD:
        *value = response_content.size();
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  void curl_easy_cleanup(CURL* curl) override { is_cleaned_up = true; }
  curl_slist* curl_slist_append(curl_slist* list, const char* str) override {
    std::vector<string>* v = list ? reinterpret_cast<std::vector<string>*>(list)
                                  : new std::vector<string>();
    v->push_back(str);
    return reinterpret_cast<curl_slist*>(v);
  }
  void curl_slist_free_all(curl_slist* list) override {
    delete reinterpret_cast<std::vector<string>*>(list);
  }

  // Variables defining the behavior of this fake.
  string response_content;
  uint64 response_code;

  // Internal variables to store the libcurl state.
  string url;
  string range;
  string custom_request;
  char* error_buffer = nullptr;
  bool is_initialized = false;
  bool is_cleaned_up = false;
  std::vector<string>* headers = nullptr;
  FILE* read_data = nullptr;
  bool is_post = false;
  void* write_data = nullptr;
  size_t (*write_callback)(const void* ptr, size_t size, size_t nmemb,
                           void* userdata) = nullptr;
  // Outcome of performing the request.
  string posted_content;
};

TEST(HttpRequestTest, GetRequest) {
  FakeLibCurl* libcurl = new FakeLibCurl("get response", 200);
  HttpRequest http_request((std::unique_ptr<LibCurl>(libcurl)));
  TF_EXPECT_OK(http_request.Init());

  char scratch[100] = "random original scratch content";
  StringPiece result = "random original string piece";

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetRange(100, 199));
  TF_EXPECT_OK(http_request.SetResultBuffer(scratch, 100, &result));
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("get response", result);

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl->is_initialized);
  EXPECT_EQ("http://www.testuri.com", libcurl->url);
  EXPECT_EQ("100-199", libcurl->range);
  EXPECT_EQ("", libcurl->custom_request);
  EXPECT_EQ(1, libcurl->headers->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl->headers)[0]);
  EXPECT_FALSE(libcurl->is_post);
}

TEST(HttpRequestTest, PostRequest_WithBody) {
  FakeLibCurl* libcurl = new FakeLibCurl("", 200);
  HttpRequest http_request((std::unique_ptr<LibCurl>(libcurl)));
  TF_EXPECT_OK(http_request.Init());

  auto content_filename = io::JoinPath(testing::TmpDir(), "content");
  std::ofstream content(content_filename, std::ofstream::binary);
  content << "post body content";
  content.close();

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetPostRequest(content_filename));
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl->is_initialized);
  EXPECT_EQ("http://www.testuri.com", libcurl->url);
  EXPECT_EQ("", libcurl->custom_request);
  EXPECT_EQ(2, libcurl->headers->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl->headers)[0]);
  EXPECT_EQ("Content-Length: 17", (*libcurl->headers)[1]);
  EXPECT_TRUE(libcurl->is_post);
  EXPECT_EQ("post body content", libcurl->posted_content);

  std::remove(content_filename.c_str());
}

TEST(HttpRequestTest, PostRequest_WithoutBody) {
  FakeLibCurl* libcurl = new FakeLibCurl("", 200);
  HttpRequest http_request((std::unique_ptr<LibCurl>(libcurl)));
  TF_EXPECT_OK(http_request.Init());

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetPostRequest());
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl->is_initialized);
  EXPECT_EQ("http://www.testuri.com", libcurl->url);
  EXPECT_EQ("", libcurl->custom_request);
  EXPECT_EQ(2, libcurl->headers->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl->headers)[0]);
  EXPECT_EQ("Content-Length: 0", (*libcurl->headers)[1]);
  EXPECT_TRUE(libcurl->is_post);
  EXPECT_EQ("", libcurl->posted_content);
}

TEST(HttpRequestTest, DeleteRequest) {
  FakeLibCurl* libcurl = new FakeLibCurl("", 200);
  HttpRequest http_request((std::unique_ptr<LibCurl>(libcurl)));
  TF_EXPECT_OK(http_request.Init());

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetDeleteRequest());
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl->is_initialized);
  EXPECT_EQ("http://www.testuri.com", libcurl->url);
  EXPECT_EQ("DELETE", libcurl->custom_request);
  EXPECT_EQ(1, libcurl->headers->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl->headers)[0]);
  EXPECT_FALSE(libcurl->is_post);
}

TEST(HttpRequestTest, WrongSequenceOfCalls_NoUri) {
  FakeLibCurl* libcurl = new FakeLibCurl("", 200);
  HttpRequest http_request((std::unique_ptr<LibCurl>(libcurl)));
  TF_EXPECT_OK(http_request.Init());

  auto s = http_request.Send();
  ASSERT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(StringPiece(s.error_message()).contains("URI has not been set"));
}

TEST(HttpRequestTest, WrongSequenceOfCalls_TwoSends) {
  FakeLibCurl* libcurl = new FakeLibCurl("", 200);
  HttpRequest http_request((std::unique_ptr<LibCurl>(libcurl)));
  TF_EXPECT_OK(http_request.Init());

  http_request.SetUri("http://www.google.com");
  http_request.Send();
  auto s = http_request.Send();
  ASSERT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(StringPiece(s.error_message())
                  .contains("The request has already been sent"));
}

TEST(HttpRequestTest, WrongSequenceOfCalls_ReusingAfterSend) {
  FakeLibCurl* libcurl = new FakeLibCurl("", 200);
  HttpRequest http_request((std::unique_ptr<LibCurl>(libcurl)));
  TF_EXPECT_OK(http_request.Init());

  http_request.SetUri("http://www.google.com");
  http_request.Send();
  auto s = http_request.SetUri("http://mail.google.com");
  ASSERT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(StringPiece(s.error_message())
                  .contains("The request has already been sent"));
}

TEST(HttpRequestTest, WrongSequenceOfCalls_SettingMethodTwice) {
  FakeLibCurl* libcurl = new FakeLibCurl("", 200);
  HttpRequest http_request((std::unique_ptr<LibCurl>(libcurl)));
  TF_EXPECT_OK(http_request.Init());

  http_request.SetDeleteRequest();
  auto s = http_request.SetPostRequest();
  ASSERT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(StringPiece(s.error_message())
                  .contains("HTTP method has been already set"));
}

TEST(HttpRequestTest, WrongSequenceOfCalls_NotInitialized) {
  FakeLibCurl* libcurl = new FakeLibCurl("", 200);
  HttpRequest http_request((std::unique_ptr<LibCurl>(libcurl)));

  auto s = http_request.SetPostRequest();
  ASSERT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(StringPiece(s.error_message())
                  .contains("The object has not been initialized"));
}

}  // namespace
}  // namespace tensorflow
