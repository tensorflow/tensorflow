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

#include "tensorflow/tsl/platform/cloud/curl_http_request.h"

#include <fstream>
#include <string>

#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/mem.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {
namespace {

const string kTestContent = "random original scratch content";

class FakeEnv : public EnvWrapper {
 public:
  FakeEnv() : EnvWrapper(Env::Default()) {}

  uint64 NowSeconds() const override { return now_; }
  uint64 now_ = 10000;
};

// A fake proxy that pretends to be libcurl.
class FakeLibCurl : public LibCurl {
 public:
  FakeLibCurl(const string& response_content, uint64 response_code)
      : response_content_(response_content), response_code_(response_code) {}
  FakeLibCurl(const string& response_content, uint64 response_code,
              std::vector<std::tuple<uint64, curl_off_t>> progress_ticks,
              FakeEnv* env)
      : response_content_(response_content),
        response_code_(response_code),
        progress_ticks_(std::move(progress_ticks)),
        env_(env) {}
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
      case CURLOPT_CAINFO:
        ca_info_ = reinterpret_cast<char*>(param);
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
      case CURLOPT_XFERINFODATA:
        progress_data_ = param;
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
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            int (*param)(void* clientp, curl_off_t dltotal,
                                         curl_off_t dlnow, curl_off_t ultotal,
                                         curl_off_t ulnow)) override {
    progress_callback_ = param;
    return CURLE_OK;
  }
  CURLcode curl_easy_perform(CURL* curl) override {
    if (is_post_ || is_put_) {
      char buffer[3];
      int bytes_read;
      posted_content_ = "";
      do {
        bytes_read = read_callback_(buffer, 1, sizeof(buffer), read_data_);
        posted_content_ =
            strings::StrCat(posted_content_, StringPiece(buffer, bytes_read));
      } while (bytes_read > 0);
    }
    if (write_data_ || write_callback_) {
      size_t bytes_handled = write_callback_(
          response_content_.c_str(), 1, response_content_.size(), write_data_);
      // Mimic real libcurl behavior by checking write callback return value.
      if (bytes_handled != response_content_.size()) {
        curl_easy_perform_result_ = CURLE_WRITE_ERROR;
      }
    }
    for (const auto& header : response_headers_) {
      header_callback_(header.c_str(), 1, header.size(), header_data_);
    }
    if (error_buffer_) {
      strncpy(error_buffer_, curl_easy_perform_error_message_.c_str(),
              curl_easy_perform_error_message_.size() + 1);
    }
    for (const auto& tick : progress_ticks_) {
      env_->now_ = std::get<0>(tick);
      if (progress_callback_(progress_data_, 0, std::get<1>(tick), 0, 0)) {
        return CURLE_ABORTED_BY_CALLBACK;
      }
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
  string ca_info_;
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
  int (*progress_callback_)(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                            curl_off_t ultotal, curl_off_t ulnow) = nullptr;
  void* progress_data_ = nullptr;
  // Outcome of performing the request.
  string posted_content_;
  CURLcode curl_easy_perform_result_ = CURLE_OK;
  string curl_easy_perform_error_message_;
  // A vector of <timestamp, progress in bytes> pairs that represent the
  // progress of a transmission.
  std::vector<std::tuple<uint64, curl_off_t>> progress_ticks_;
  FakeEnv* env_ = nullptr;
};

TEST(CurlHttpRequestTest, GetRequest) {
  FakeLibCurl libcurl("get response", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  scratch.reserve(100);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("get response", string(scratch.begin(), scratch.end()));

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("100-199", libcurl.range_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ("", libcurl.ca_info_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
  EXPECT_EQ(200, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_Direct) {
  FakeLibCurl libcurl("get response", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch(100, 0);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBufferDirect(scratch.data(), scratch.capacity());
  TF_EXPECT_OK(http_request.Send());

  string expected_response = "get response";
  size_t response_bytes_transferred =
      http_request.GetResultBufferDirectBytesTransferred();
  EXPECT_EQ(expected_response.size(), response_bytes_transferred);
  EXPECT_EQ(
      "get response",
      string(scratch.begin(), scratch.begin() + response_bytes_transferred));

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("100-199", libcurl.range_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ("", libcurl.ca_info_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
  EXPECT_EQ(200, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_CustomCaInfoFlag) {
  static char set_var[] = "CURL_CA_BUNDLE=test";
  putenv(set_var);
  FakeLibCurl libcurl("get response", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  scratch.reserve(100);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("get response", string(scratch.begin(), scratch.end()));

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("100-199", libcurl.range_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ("test", libcurl.ca_info_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
  EXPECT_EQ(200, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_Direct_ResponseTooLarge) {
  FakeLibCurl libcurl("get response", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch(5, 0);

  http_request.SetUri("http://www.testuri.com");
  http_request.SetResultBufferDirect(scratch.data(), scratch.size());
  const Status& status = http_request.Send();
  EXPECT_EQ(error::FAILED_PRECONDITION, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: libcurl code 23 meaning "
      "'Failed writing received data to disk/application', error details: "
      "Received 12 response bytes for a 5-byte buffer",
      status.message());

  // As long as the request clearly fails, ok to leave truncated response here.
  EXPECT_EQ(5, http_request.GetResultBufferDirectBytesTransferred());
  EXPECT_EQ("get r", string(scratch.begin(), scratch.begin() + 5));
}

TEST(CurlHttpRequestTest, GetRequest_Direct_RangeOutOfBound) {
  FakeLibCurl libcurl("get response", 416);
  CurlHttpRequest http_request(&libcurl);

  const string initialScratch = "abcde";
  std::vector<char> scratch;
  scratch.insert(scratch.end(), initialScratch.begin(), initialScratch.end());

  http_request.SetUri("http://www.testuri.com");
  http_request.SetRange(0, 4);
  http_request.SetResultBufferDirect(scratch.data(), scratch.size());
  TF_EXPECT_OK(http_request.Send());
  EXPECT_EQ(416, http_request.GetResponseCode());

  // Some servers (in particular, GCS) return an error message payload with a
  // 416 Range Not Satisfiable response. We should pretend it's not there when
  // reporting bytes transferred, but it's ok if it writes to scratch.
  EXPECT_EQ(0, http_request.GetResultBufferDirectBytesTransferred());
  EXPECT_EQ("get r", string(scratch.begin(), scratch.end()));
}

TEST(CurlHttpRequestTest, GetRequest_Empty) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.resize(0);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
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

TEST(CurlHttpRequestTest, GetRequest_RangeOutOfBound) {
  FakeLibCurl libcurl("get response", 416);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  TF_EXPECT_OK(http_request.Send());

  // Some servers (in particular, GCS) return an error message payload with a
  // 416 Range Not Satisfiable response. We should pretend it's not there.
  EXPECT_TRUE(scratch.empty());
  EXPECT_EQ(416, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_503) {
  FakeLibCurl libcurl("get response", 503);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  http_request.SetUri("http://www.testuri.com");
  http_request.SetResultBuffer(&scratch);
  const auto& status = http_request.Send();
  EXPECT_EQ(error::UNAVAILABLE, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: HTTP response code 503 with body "
      "'get response'",
      status.message());
}

TEST(CurlHttpRequestTest, GetRequest_HttpCode0) {
  FakeLibCurl libcurl("get response", 0);
  libcurl.curl_easy_perform_result_ = CURLE_OPERATION_TIMEDOUT;
  libcurl.curl_easy_perform_error_message_ = "Operation timed out";
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  http_request.SetUri("http://www.testuri.com");
  const auto& status = http_request.Send();
  EXPECT_EQ(error::UNAVAILABLE, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: libcurl code 28 meaning "
      "'Timeout was reached', error details: Operation timed out",
      status.message());
  EXPECT_EQ(0, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_CouldntResolveHost) {
  FakeLibCurl libcurl("get response", 0);
  libcurl.curl_easy_perform_result_ = CURLE_COULDNT_RESOLVE_HOST;
  libcurl.curl_easy_perform_error_message_ =
      "Could not resolve host 'metadata'";
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  http_request.SetUri("http://metadata");
  const auto& status = http_request.Send();
  EXPECT_EQ(error::FAILED_PRECONDITION, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: libcurl code 6 meaning "
      "'Couldn't resolve host name', error details: Could not resolve host "
      "'metadata'",
      status.message());
  EXPECT_EQ(0, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, GetRequest_SslBadCertfile) {
  FakeLibCurl libcurl("get response", 0);
  libcurl.curl_easy_perform_result_ = CURLE_SSL_CACERT_BADFILE;
  libcurl.curl_easy_perform_error_message_ =
      "error setting certificate verify locations:";
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.end(), kTestContent.begin(), kTestContent.end());

  http_request.SetUri("http://metadata");
  const auto& status = http_request.Send();
  EXPECT_EQ(error::FAILED_PRECONDITION, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: libcurl code 77 meaning "
      "'Problem with the SSL CA cert (path? access rights?)', error details: "
      "error setting certificate verify locations:",
      status.message());
  EXPECT_EQ(0, http_request.GetResponseCode());
}

TEST(CurlHttpRequestTest, ResponseHeaders) {
  FakeLibCurl libcurl(
      "get response", 200,
      {"Location: abcd", "Content-Type: text", "unparsable header"});
  CurlHttpRequest http_request(&libcurl);

  http_request.SetUri("http://www.testuri.com");
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("abcd", http_request.GetResponseHeader("Location"));
  EXPECT_EQ("text", http_request.GetResponseHeader("Content-Type"));
  EXPECT_EQ("", http_request.GetResponseHeader("Not-Seen-Header"));
}

TEST(CurlHttpRequestTest, PutRequest_WithBody_FromFile) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  auto content_filename = io::JoinPath(testing::TmpDir(), "content");
  std::ofstream content(content_filename, std::ofstream::binary);
  content << "post body content";
  content.close();

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
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

TEST(CurlHttpRequestTest, PutRequest_WithBody_FromFile_NonZeroOffset) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  auto content_filename = io::JoinPath(testing::TmpDir(), "content");
  std::ofstream content(content_filename, std::ofstream::binary);
  content << "post body content";
  content.close();

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  TF_EXPECT_OK(http_request.SetPutFromFile(content_filename, 7));
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_EQ("dy content", libcurl.posted_content_);

  std::remove(content_filename.c_str());
}

TEST(CurlHttpRequestTest, PutRequest_WithoutBody) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetPutEmptyBody();
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(3, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_EQ("Content-Length: 0", (*libcurl.headers_)[1]);
  EXPECT_EQ("Transfer-Encoding: identity", (*libcurl.headers_)[2]);
  EXPECT_TRUE(libcurl.is_put_);
  EXPECT_EQ("", libcurl.posted_content_);
}

TEST(CurlHttpRequestTest, PostRequest_WithBody_FromMemory) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  string content = "post body content";

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetPostFromBuffer(content.c_str(), content.size());
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

TEST(CurlHttpRequestTest, PostRequest_WithoutBody) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetPostEmptyBody();
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("", libcurl.custom_request_);
  EXPECT_EQ(3, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_EQ("Content-Length: 0", (*libcurl.headers_)[1]);
  EXPECT_EQ("Transfer-Encoding: identity", (*libcurl.headers_)[2]);
  EXPECT_TRUE(libcurl.is_post_);
  EXPECT_EQ("", libcurl.posted_content_);
}

TEST(CurlHttpRequestTest, DeleteRequest) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetDeleteRequest();
  TF_EXPECT_OK(http_request.Send());

  // Check interactions with libcurl.
  EXPECT_TRUE(libcurl.is_initialized_);
  EXPECT_EQ("http://www.testuri.com", libcurl.url_);
  EXPECT_EQ("DELETE", libcurl.custom_request_);
  EXPECT_EQ(1, libcurl.headers_->size());
  EXPECT_EQ("Authorization: Bearer fake-bearer", (*libcurl.headers_)[0]);
  EXPECT_FALSE(libcurl.is_post_);
}

TEST(CurlHttpRequestTest, WrongSequenceOfCalls_NoUri) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  ASSERT_DEATH((void)http_request.Send(), "URI has not been set");
}

TEST(CurlHttpRequestTest, WrongSequenceOfCalls_TwoSends) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetUri("http://www.google.com");
  TF_EXPECT_OK(http_request.Send());
  ASSERT_DEATH((void)http_request.Send(), "The request has already been sent");
}

TEST(CurlHttpRequestTest, WrongSequenceOfCalls_ReusingAfterSend) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetUri("http://www.google.com");
  TF_EXPECT_OK(http_request.Send());
  ASSERT_DEATH(http_request.SetUri("http://mail.google.com"),
               "The request has already been sent");
}

TEST(CurlHttpRequestTest, WrongSequenceOfCalls_SettingMethodTwice) {
  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetDeleteRequest();
  ASSERT_DEATH(http_request.SetPostEmptyBody(),
               "HTTP method has been already set");
}

TEST(CurlHttpRequestTest, EscapeString) {
  FakeLibCurl libcurl("get response", 200);
  CurlHttpRequest http_request(&libcurl);
  const string test_string = "a/b/c";
  EXPECT_EQ("a%2Fb%2Fc", http_request.EscapeString(test_string));
}

TEST(CurlHttpRequestTest, ErrorReturnsNoResponse) {
  FakeLibCurl libcurl("get response", 500);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  scratch.reserve(100);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  EXPECT_EQ(error::UNAVAILABLE, http_request.Send().code());

  EXPECT_EQ("", string(scratch.begin(), scratch.end()));
}

TEST(CurlHttpRequestTest, ProgressIsOk) {
  // Imitate a steady progress.
  FakeEnv env;
  FakeLibCurl libcurl(
      "test", 200,
      {
          std::make_tuple(100, 0) /* timestamp 100, 0 bytes */,
          std::make_tuple(110, 0) /* timestamp 110, 0 bytes */,
          std::make_tuple(200, 100) /* timestamp 200, 100 bytes */
      },
      &env);
  CurlHttpRequest http_request(&libcurl, &env);
  http_request.SetUri("http://www.testuri.com");
  TF_EXPECT_OK(http_request.Send());
}

TEST(CurlHttpRequestTest, ProgressIsStuck) {
  // Imitate a transmission that got stuck for more than a minute.
  FakeEnv env;
  FakeLibCurl libcurl(
      "test", 200,
      {
          std::make_tuple(100, 10) /* timestamp 100, 10 bytes */,
          std::make_tuple(130, 10) /* timestamp 130, 10 bytes */,
          std::make_tuple(170, 10) /* timestamp 170, 10 bytes */
      },
      &env);
  CurlHttpRequest http_request(&libcurl, &env);
  http_request.SetUri("http://www.testuri.com");
  auto status = http_request.Send();
  EXPECT_EQ(error::UNAVAILABLE, status.code());
  EXPECT_EQ(
      "Error executing an HTTP request: libcurl code 42 meaning 'Operation "
      "was aborted by an application callback', error details: (none)",
      status.message());
}

class TestStats : public HttpRequest::RequestStats {
 public:
  ~TestStats() override = default;

  void RecordRequest(const HttpRequest* request, const string& uri,
                     HttpRequest::RequestMethod method) override {
    has_recorded_request_ = true;
    record_request_request_ = request;
    record_request_uri_ = uri;
    record_request_method_ = method;
  }

  void RecordResponse(const HttpRequest* request, const string& uri,
                      HttpRequest::RequestMethod method,
                      const Status& result) override {
    has_recorded_response_ = true;
    record_response_request_ = request;
    record_response_uri_ = uri;
    record_response_method_ = method;
    record_response_result_ = result;
  }

  const HttpRequest* record_request_request_ = nullptr;
  string record_request_uri_ = "http://www.testuri.com";
  HttpRequest::RequestMethod record_request_method_ =
      HttpRequest::RequestMethod::kGet;

  const HttpRequest* record_response_request_ = nullptr;
  string record_response_uri_ = "http://www.testuri.com";
  HttpRequest::RequestMethod record_response_method_ =
      HttpRequest::RequestMethod::kGet;
  Status record_response_result_;

  bool has_recorded_request_ = false;
  bool has_recorded_response_ = false;
};

class StatsTestFakeLibCurl : public FakeLibCurl {
 public:
  StatsTestFakeLibCurl(TestStats* stats, const string& response_content,
                       uint64 response_code)
      : FakeLibCurl(response_content, response_code), stats_(stats) {}
  CURLcode curl_easy_perform(CURL* curl) override {
    CHECK(!performed_request_);
    performed_request_ = true;
    stats_had_recorded_request_ = stats_->has_recorded_request_;
    stats_had_recorded_response_ = stats_->has_recorded_response_;
    return FakeLibCurl::curl_easy_perform(curl);
  };

  TestStats* stats_;
  bool performed_request_ = false;
  bool stats_had_recorded_request_;
  bool stats_had_recorded_response_;
};

TEST(CurlHttpRequestTest, StatsGetSuccessful) {
  TestStats stats;
  StatsTestFakeLibCurl libcurl(&stats, "get response", 200);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  scratch.reserve(100);

  http_request.SetRequestStats(&stats);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  TF_EXPECT_OK(http_request.Send());

  EXPECT_EQ("get response", string(scratch.begin(), scratch.end()));

  // Check interaction with stats.
  ASSERT_TRUE(stats.has_recorded_request_);
  EXPECT_EQ(&http_request, stats.record_request_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_request_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kGet, stats.record_request_method_);

  ASSERT_TRUE(stats.has_recorded_response_);
  EXPECT_EQ(&http_request, stats.record_response_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_response_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kGet, stats.record_response_method_);
  TF_EXPECT_OK(stats.record_response_result_);

  // Check interaction with libcurl.
  EXPECT_TRUE(libcurl.performed_request_);
  EXPECT_TRUE(libcurl.stats_had_recorded_request_);
  EXPECT_FALSE(libcurl.stats_had_recorded_response_);
}

TEST(CurlHttpRequestTest, StatsGetNotFound) {
  TestStats stats;
  StatsTestFakeLibCurl libcurl(&stats, "get other response", 404);
  CurlHttpRequest http_request(&libcurl);

  std::vector<char> scratch;
  scratch.insert(scratch.begin(), kTestContent.begin(), kTestContent.end());
  scratch.reserve(100);

  http_request.SetRequestStats(&stats);

  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetRange(100, 199);
  http_request.SetResultBuffer(&scratch);
  Status s = http_request.Send();

  // Check interaction with stats.
  ASSERT_TRUE(stats.has_recorded_request_);
  EXPECT_EQ(&http_request, stats.record_request_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_request_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kGet, stats.record_request_method_);

  ASSERT_TRUE(stats.has_recorded_response_);
  EXPECT_EQ(&http_request, stats.record_response_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_response_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kGet, stats.record_response_method_);
  EXPECT_TRUE(errors::IsNotFound(stats.record_response_result_));
  EXPECT_EQ(s, stats.record_response_result_);

  // Check interaction with libcurl.
  EXPECT_TRUE(libcurl.performed_request_);
  EXPECT_TRUE(libcurl.stats_had_recorded_request_);
  EXPECT_FALSE(libcurl.stats_had_recorded_response_);
}

TEST(CurlHttpRequestTest, StatsPost) {
  TestStats stats;

  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);

  http_request.SetRequestStats(&stats);

  string content = "post body content";

  http_request.SetUri("http://www.testuri.com");
  http_request.SetPostFromBuffer(content.c_str(), content.size());
  TF_EXPECT_OK(http_request.Send());

  // Check interaction with stats.
  ASSERT_TRUE(stats.has_recorded_request_);
  EXPECT_EQ(&http_request, stats.record_request_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_request_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kPost, stats.record_request_method_);

  ASSERT_TRUE(stats.has_recorded_response_);
  EXPECT_EQ(&http_request, stats.record_response_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_response_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kPost, stats.record_response_method_);
  TF_EXPECT_OK(stats.record_response_result_);
}

TEST(CurlHttpRequestTest, StatsDelete) {
  TestStats stats;

  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetRequestStats(&stats);
  http_request.SetUri("http://www.testuri.com");
  http_request.SetDeleteRequest();
  TF_EXPECT_OK(http_request.Send());

  // Check interaction with stats.
  ASSERT_TRUE(stats.has_recorded_request_);
  EXPECT_EQ(&http_request, stats.record_request_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_request_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kDelete, stats.record_request_method_);

  ASSERT_TRUE(stats.has_recorded_response_);
  EXPECT_EQ(&http_request, stats.record_response_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_response_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kDelete, stats.record_response_method_);
  TF_EXPECT_OK(stats.record_response_result_);
}

TEST(CurlHttpRequestTest, StatsPut) {
  TestStats stats;

  FakeLibCurl libcurl("", 200);
  CurlHttpRequest http_request(&libcurl);
  http_request.SetRequestStats(&stats);
  http_request.SetUri("http://www.testuri.com");
  http_request.AddAuthBearerHeader("fake-bearer");
  http_request.SetPutEmptyBody();
  TF_EXPECT_OK(http_request.Send());

  // Check interaction with stats.
  ASSERT_TRUE(stats.has_recorded_request_);
  EXPECT_EQ(&http_request, stats.record_request_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_request_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kPut, stats.record_request_method_);

  ASSERT_TRUE(stats.has_recorded_response_);
  EXPECT_EQ(&http_request, stats.record_response_request_);
  EXPECT_EQ("http://www.testuri.com", stats.record_response_uri_);
  EXPECT_EQ(HttpRequest::RequestMethod::kPut, stats.record_response_method_);
  TF_EXPECT_OK(stats.record_response_result_);
}

}  // namespace
}  // namespace tsl
