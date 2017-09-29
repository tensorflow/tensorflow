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

#ifndef TENSORFLOW_CORE_PLATFORM_HTTP_REQUEST_FAKE_H_
#define TENSORFLOW_CORE_PLATFORM_HTTP_REQUEST_FAKE_H_

#include <fstream>
#include <string>
#include <vector>
#include <curl/curl.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/cloud/curl_http_request.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// Fake HttpRequest for testing.
class FakeHttpRequest : public CurlHttpRequest {
 public:
  /// Return the response for the given request.
  FakeHttpRequest(const string& request, const string& response)
      : FakeHttpRequest(request, response, Status::OK(), nullptr, {}, 200) {}

  /// Return the response with headers for the given request.
  FakeHttpRequest(const string& request, const string& response,
                  const std::map<string, string>& response_headers)
      : FakeHttpRequest(request, response, Status::OK(), nullptr,
                        response_headers, 200) {}

  /// \brief Return the response for the request and capture the POST body.
  ///
  /// Post body is not expected to be a part of the 'request' parameter.
  FakeHttpRequest(const string& request, const string& response,
                  string* captured_post_body)
      : FakeHttpRequest(request, response, Status::OK(), captured_post_body, {},
                        200) {}

  /// \brief Return the response and the status for the given request.
  FakeHttpRequest(const string& request, const string& response,
                  Status response_status, uint64 response_code)
      : FakeHttpRequest(request, response, response_status, nullptr, {},
                        response_code) {}

  /// \brief Return the response and the status for the given request
  ///  and capture the POST body.
  ///
  /// Post body is not expected to be a part of the 'request' parameter.
  FakeHttpRequest(const string& request, const string& response,
                  Status response_status, string* captured_post_body,
                  const std::map<string, string>& response_headers,
                  uint64 response_code)
      : expected_request_(request),
        response_(response),
        response_status_(response_status),
        captured_post_body_(captured_post_body),
        response_headers_(response_headers),
        response_code_(response_code) {}

  Status Init() override { return Status::OK(); }
  Status SetUri(const string& uri) override {
    actual_request_ += "Uri: " + uri + "\n";
    return Status::OK();
  }
  Status SetRange(uint64 start, uint64 end) override {
    actual_request_ += strings::StrCat("Range: ", start, "-", end, "\n");
    return Status::OK();
  }
  Status AddHeader(const string& name, const string& value) override {
    actual_request_ += "Header " + name + ": " + value + "\n";
    return Status::OK();
  }
  Status AddAuthBearerHeader(const string& auth_token) override {
    actual_request_ += "Auth Token: " + auth_token + "\n";
    return Status::OK();
  }
  Status SetDeleteRequest() override {
    actual_request_ += "Delete: yes\n";
    return Status::OK();
  }
  Status SetPutFromFile(const string& body_filepath, size_t offset) override {
    std::ifstream stream(body_filepath);
    const string& content = string(std::istreambuf_iterator<char>(stream),
                                   std::istreambuf_iterator<char>())
                                .substr(offset);
    actual_request_ += "Put body: " + content + "\n";
    return Status::OK();
  }
  Status SetPostFromBuffer(const char* buffer, size_t size) override {
    if (captured_post_body_) {
      *captured_post_body_ = string(buffer, size);
    } else {
      actual_request_ +=
          strings::StrCat("Post body: ", StringPiece(buffer, size), "\n");
    }
    return Status::OK();
  }
  Status SetPutEmptyBody() override {
    actual_request_ += "Put: yes\n";
    return Status::OK();
  }
  Status SetPostEmptyBody() override {
    if (captured_post_body_) {
      *captured_post_body_ = "<empty>";
    } else {
      actual_request_ += "Post: yes\n";
    }
    return Status::OK();
  }
  Status SetResultBuffer(std::vector<char>* buffer) override {
    buffer->clear();
    buffer_ = buffer;
    return Status::OK();
  }
  Status Send() override {
    EXPECT_EQ(expected_request_, actual_request_) << "Unexpected HTTP request.";
    if (buffer_) {
      buffer_->insert(buffer_->begin(), response_.c_str(),
                      response_.c_str() + response_.size());
    }
    return response_status_;
  }

  // This function just does a simple replacing of "/" with "%2F" instead of
  // full url encoding.
  string EscapeString(const string& str) override {
    const string victim = "/";
    const string encoded = "%2F";

    string copy_str = str;
    std::string::size_type n = 0;
    while ((n = copy_str.find(victim, n)) != std::string::npos) {
      copy_str.replace(n, victim.size(), encoded);
      n += encoded.size();
    }
    return copy_str;
  }

  string GetResponseHeader(const string& name) const override {
    const auto header = response_headers_.find(name);
    return header != response_headers_.end() ? header->second : "";
  }

  virtual uint64 GetResponseCode() const override { return response_code_; }

 private:
  std::vector<char>* buffer_ = nullptr;
  string expected_request_;
  string actual_request_;
  string response_;
  Status response_status_;
  string* captured_post_body_ = nullptr;
  std::map<string, string> response_headers_;
  uint64 response_code_ = 0;
};

/// Fake HttpRequest factory for testing.
class FakeHttpRequestFactory : public HttpRequest::Factory {
 public:
  FakeHttpRequestFactory(const std::vector<HttpRequest*>* requests)
      : requests_(requests) {}

  ~FakeHttpRequestFactory() {
    EXPECT_EQ(current_index_, requests_->size())
        << "Not all expected requests were made.";
  }

  HttpRequest* Create() override {
    EXPECT_LT(current_index_, requests_->size())
        << "Too many calls of HttpRequest factory.";
    return (*requests_)[current_index_++];
  }

 private:
  const std::vector<HttpRequest*>* requests_;
  int current_index_ = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_HTTP_REQUEST_FAKE_H_
