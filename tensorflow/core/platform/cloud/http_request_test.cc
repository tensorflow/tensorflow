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

#include "tensorflow/core/platform/cloud/http_request.h"
#include <fstream>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

const string kTestContent = "random original scratch content";

// A fake proxy that pretends to be libcurl.
class FakeLibCurl : public LibCurl {
 public:
  FakeLibCurl(const string& response_content, uint64 response_code)
      : response_content_(response_content), response_code_(response_code) {}
  FakeLibCurl(const string& response_content, uint64 response_code,
              const std::vector<string>& response_headers)
      : response_content_(response_content),
        response_code_(response_code),
        response_headers_(response_headers) {}
  CURL* curl_easy_init() override {
    is_initialized_ = true;
    // The reuslt just needs to be non-null.
    return reinterpret_cast<CURL*>(this);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            uint64 param) override {
    switch (option) {
      case CURLOPT_POST:
        is_post_ = param;
        break;
      case CURLOPT_PUT:
        is_put_ = param;
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
        url_ = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_RANGE:
        range_ = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_CUSTOMREQUEST:
        custom_request_ = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_HTTPHEADER:
        headers_ = reinterpret_cast<std::vector<string>*>(param);
        break;
      case CURLOPT_ERRORBUFFER:
        error_buffer_ = reinterpret_cast<char*>(param);
        break;
      case CURLOPT_WRITEDATA:
        write_data_ = reinterpret_cast<FILE*>(param);
        break;
      case CURLOPT_HEADERDATA:
        header_data_ = reinterpret_cast<FILE*>(param);
        break;
      case CURLOPT_READDATA:
        read_data_ = reinterpret_cast<FILE*>(param);
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(void*, size_t, size_t,
                                            FILE*)) override {
    read_callback_ = param;
    return CURLE_OK;
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(const void*, size_t, size_t,
                                            void*)) override {
    switch (option) {
      case CURLOPT_WRITEFUNCTION:
        write_callback_ = param;
        break;
      case CURLOPT_HEADERFUNCTION:
        header_callback_ = param;
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  CURLcode curl_easy_perform(CURL* curl) override {
    if (read_data_) {
      char buffer[3];
      int bytes_read;
      posted_content_ = "";
      do {
        bytes_read = read_callback_(buffer, 1, sizeof(buffer), read_data_);
        posted_content_ =
            strings::StrCat(posted_content_, StringPiece(buffer, bytes_read));
      } while (bytes_read > 0);
    }
    if (write_data_) {
      write_callback_(response_content_.c_str(), 1, response_content_.size(),
                      write_data_);
    }
    for (const auto& header : response_headers_) {
      header_callback_(header.c_str(), 1, header.size(), header_data_);
    }
    if (error_buffer_) {
      strncpy(error_buffer_, curl_easy_perform_error_message_.c_str(),
              curl_easy_perform_error_message_.size() + 1);
    }
    return curl_easy_perform_result_;
  }
  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             uint64* value) override {
    switch (info) {
      case CURLINFO_RESPONSE_CODE:
        *value = response_code_;
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
        *value = response_content_.size();
        break;
      default:
        break;
    }
    return CURLE_OK;
  }
  void curl_easy_cleanup(CURL* curl) override { is_cleaned_up_ = true; }
  curl_slist* curl_slist_append(curl_slist* list, const char* str) override {
    std::vector<string>* v = list ? reinterpret_cast<std::vector<string>*>(list)
                                  : new std::vector<string>();
    v->push_back(str);
    return reinterpret_cast<curl_slist*>(v);
  }
  char* curl_easy_escape(CURL* curl, const char* str, int length) override {
    // This function just does a simple replacing of "/" with "%2F" instead of
    // full url encoding.
    const string victim = "/";
    const string encoded = "%2F";

    string temp_str = str;
    std::string::size_type n = 0;
    while ((n = temp_str.find(victim, n)) != std::string::npos) {
      temp_str.replace(n, victim.size(), encoded);
      n += encoded.size();
    }
    char* out_char_str = reinterpret_cast<char*>(
        port::Malloc(sizeof(char) * temp_str.size() + 1));
    std::copy(temp_str.begin(), temp_str.end(), out_char_str);
    out_char_str[temp_str.size()] = '\0';
    return out_char_str;
  }
  void curl_slist_free_all(curl_slist* list) override {
    delete reinterpret_cast<std::vector<string>*>(list);
  }
  void curl_free(void* p) override { port::Free(p); }

  // Variables defining the behavior of this fake.
  string response_content_;
  uint64 response_code_;
  std::vector<string> response_headers_;

  // Internal variables to store the libcurl state.
  string url_;
  string range_;
  string custom_request_;
  char* error_buffer_ = nullptr;
  bool is_initialized_ = false;
  bool is_cleaned_up_ = false;
  std::vector<string>* headers_ = nullptr;
  bool is_post_ = false;
  bool is_put_ = false;
  void* write_data_ = nullptr;
  size_t (*write_callback_)(const void* ptr, size_t size, size_t nmemb,
                            void* userdata) = nullptr;
  void* header_data_ = nullptr;
  size_t (*header_callback_)(const void* ptr, size_t size, size_t nmemb,
                             void* userdata) = nullptr;
  FILE* read_data_ = nullptr;
  size_t (*read_callback_)(void* ptr, size_t size, size_t nmemb,
                           FILE* userdata) = &fread;
  // Outcome of performing the request.
  string posted_content_;
  CURLcode curl_easy_perform_result_ = CURLE_OK;
  string curl_easy_perform_error_message_;
};

TEST(HttpRequestTest, GetRequest) {
  FakeLibCurl libcurl("get response", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  StringPiece result;
  scratch.reserve(100);

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetRange(100, 199));
  TF_EXPECT_OK(http_request.SetResultBuffer(&scratch));
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("get response", string(scratch.begin(), scratch.end()));

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("100-199", libcurl.range_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
  EXPECT_EQ(200, http_request.GetResponseCode());
}

TEST(HttpRequestTest, GetRequest_Empty) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  std::vector<char> scratch;
  scratch.resize(0);

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetRange(100, 199));
  TF_EXPECT_OK(http_request.SetResultBuffer(&scratch));
  TF_EXPECT_OK(http_request.Send());

  EXPECT_TRUE(scratch.empty());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("100-199", libcurl.range_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
  EXPECT_EQ(200, http_request.GetResponseCode());
}

TEST(HttpRequestTest, GetRequest_RangeOutOfBound) {
  FakeLibCurl libcurl("get response", 416);
  libcurl.curl_easy_perform_result_ = CURLE_WRITE_ERROR;
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetRange(100, 199));
  TF_EXPECT_OK(http_request.SetResultBuffer(&scratch));
  TF_EXPECT_OK(http_request.Send());

  EXPECT_TRUE(scratch.empty());
  EXPECT_EQ(416, http_request.GetResponseCode());
}

TEST(HttpRequestTest, GetRequest_503) {
  FakeLibCurl libcurl("get response", 503);
  libcurl.curl_easy_perform_result_ = CURLE_WRITE_ERROR;
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetRange(100, 199));
  TF_EXPECT_OK(http_request.SetResultBuffer(&scratch));
  const auto& status = http_request.Send();
  EXPECT_EQ(error::UNAVAILABLE, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request (HTTP response code 503, "
      "error code 23, error message '')",
      status.error_message());
  EXPECT_EQ(503, http_request.GetResponseCode());
}

TEST(HttpRequestTest, GetRequest_HttpCode0) {
  FakeLibCurl libcurl("get response", 0);
  libcurl.curl_easy_perform_result_ = CURLE_OPERATION_TIMEDOUT;
  libcurl.curl_easy_perform_error_message_ = "Operation timed out";
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  const auto& status = http_request.Send();
  EXPECT_EQ(error::UNAVAILABLE, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request (HTTP response code 0, "
      "error code 28, error message 'Operation timed out')",
      status.error_message());
  EXPECT_EQ(0, http_request.GetResponseCode());
}

TEST(HttpRequestTest, ResponseHeaders) {
  FakeLibCurl libcurl(
      "get response", 200,
      {"Location: abcd", "Content-Type: text", "unparsable header"});
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("abcd", http_request.GetResponseHeader("Location"));
  EXPECT_EQ("text", http_request.GetResponseHeader("Content-Type"));
  EXPECT_EQ("", http_request.GetResponseHeader("Not-Seen-Header"));
}

TEST(HttpRequestTest, PutRequest_WithBody_FromFile) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  auto content_filename = io::JoinPath(testing::TmpDir(), "content");
  std::ofstream content(content_filename, std::ofstream::binary);
  content << "post body content";
  content.close();

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetPutFromFile(content_filename, 0));
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(2, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_EQ("Content-Length: 17", (*libcurl.headers_)[1]);
  EXPECT_TRUE(libcurl.is_put_);
  EXPECT_EQ("post body content", libcurl.posted_content_);

  std::remove(content_filename.c_str());
}

TEST(HttpRequestTest, PutRequest_WithBody_FromFile_NonZeroOffset) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  auto content_filename = io::JoinPath(testing::TmpDir(), "content");
  std::ofstream content(content_filename, std::ofstream::binary);
  content << "post body content";
  content.close();

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetPutFromFile(content_filename, 7));
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_EQ("dy content", libcurl.posted_content_);

  std::remove(content_filename.c_str());
}

TEST(HttpRequestTest, PostRequest_WithBody_FromMemory) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  string content = "post body content";

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetPostFromBuffer(content.c_str(), content.size()));
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(2, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_EQ("Content-Length: 17", (*libcurl.headers_)[1]);
  EXPECT_TRUE(libcurl.is_post_);
  EXPECT_EQ("post body content", libcurl.posted_content_);
}

TEST(HttpRequestTest, PostRequest_WithoutBody) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetPostEmptyBody());
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(2, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_EQ("Content-Length: 0", (*libcurl.headers_)[1]);
  EXPECT_TRUE(libcurl.is_post_);
  EXPECT_EQ("", libcurl.posted_content_);
}

TEST(HttpRequestTest, DeleteRequest) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  TF_EXPECT_OK(http_request.SetUri("http://www.testuri.com"));
  TF_EXPECT_OK(http_request.AddAuthBearerHeader("fake-bearer"));
  TF_EXPECT_OK(http_request.SetDeleteRequest());
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("DELETE", libcurl.custom_request_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
}

TEST(HttpRequestTest, WrongSequenceOfCalls_NoUri) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  auto s = http_request.Send();
  ASSERT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(StringPiece(s.error_message()).contains("URI has not been set"));
}

TEST(HttpRequestTest, WrongSequenceOfCalls_TwoSends) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  TF_EXPECT_OK(http_request.SetUri("http://www.google.com"));
  TF_EXPECT_OK(http_request.Send());
  auto s = http_request.Send();
  ASSERT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(StringPiece(s.error_message())
                  .contains("The request has already been sent"));
}

TEST(HttpRequestTest, WrongSequenceOfCalls_ReusingAfterSend) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  TF_EXPECT_OK(http_request.SetUri("http://www.google.com"));
  TF_EXPECT_OK(http_request.Send());
  auto s = http_request.SetUri("http://mail.google.com");
  ASSERT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(StringPiece(s.error_message())
                  .contains("The request has already been sent"));
}

TEST(HttpRequestTest, WrongSequenceOfCalls_SettingMethodTwice) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());

  TF_EXPECT_OK(http_request.SetDeleteRequest());
  auto s = http_request.SetPostEmptyBody();
  ASSERT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(StringPiece(s.error_message())
                  .contains("HTTP method has been already set"));
}

TEST(HttpRequestTest, WrongSequenceOfCalls_NotInitialized) {
  FakeLibCurl libcurl("", 200);
  HttpRequest http_request(&libcurl);

  auto s = http_request.SetPostEmptyBody();
  ASSERT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(StringPiece(s.error_message())
                  .contains("The object has not been initialized"));
}

TEST(HttpRequestTest, EscapeString) {
  FakeLibCurl libcurl("get response", 200);
  HttpRequest http_request(&libcurl);
  TF_EXPECT_OK(http_request.Init());
  const string test_string = "a/b/c";
  EXPECT_EQ("a%2Fb%2Fc", http_request.EscapeString(test_string));
}

}  // namespace
}  // namespace tensorflow
