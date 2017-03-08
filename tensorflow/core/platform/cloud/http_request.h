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

#ifndef TENSORFLOW_CORE_PLATFORM_HTTP_REQUEST_H_
#define TENSORFLOW_CORE_PLATFORM_HTTP_REQUEST_H_

#include <string>
#include <vector>
#include <curl/curl.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class LibCurl;  // libcurl interface as a class, for dependency injection.

/// \brief A basic HTTP client based on the libcurl library.
///
/// The usage pattern for the class reflects the one of the libcurl library:
/// create a request object, set request parameters and call Send().
///
/// For example:
///   HttpRequest request;
///   request.SetUri("http://www.google.com");
///   request.SetResultsBuffer(scratch, 1000, &result);
///   request.Send();
class HttpRequest {
 public:
  class Factory {
   public:
    virtual ~Factory() {}
    virtual HttpRequest* Create() { return new HttpRequest(); }
  };

  HttpRequest();
  explicit HttpRequest(std::unique_ptr<LibCurl> libcurl);
  virtual ~HttpRequest();

  virtual Status Init();

  /// Sets the request URI.
  virtual Status SetUri(const string& uri);

  /// \brief Sets the Range header.
  ///
  /// Used for random seeks, for example "0-999" returns the first 1000 bytes
  /// (note that the right border is included).
  virtual Status SetRange(uint64 start, uint64 end);

  /// Sets a request header.
  virtual Status AddHeader(const string& name, const string& value);

  /// Sets the 'Authorization' header to the value of 'Bearer ' + auth_token.
  virtual Status AddAuthBearerHeader(const string& auth_token);

  /// Makes the request a DELETE request.
  virtual Status SetDeleteRequest();

  /// \brief Makes the request a POST request.
  ///
  /// The request body will be taken from the specified file.
  virtual Status SetPostRequest(const string& body_filepath);

  /// \brief Makes the request a POST request.
  ///
  /// The request body will be taken from the specified buffer.
  virtual Status SetPostRequest(const char* buffer, size_t size);

  /// Makes the request a POST request.
  virtual Status SetPostRequest();

  /// \brief Specifies the buffer for receiving the response body.
  ///
  /// The interface is made similar to RandomAccessFile::Read.
  virtual Status SetResultBuffer(char* scratch, size_t size,
                                 StringPiece* result);

  /// \brief Sends the formed request.
  ///
  /// If the result buffer was defined, the response will be written there.
  /// The object is not designed to be re-used after Send() is executed.
  virtual Status Send();

  // Url encodes str and returns a new string.
  virtual string EscapeString(const string& str);

 private:
  /// A write callback in the form which can be accepted by libcurl.
  static size_t WriteCallback(const void* ptr, size_t size, size_t nmemb,
                              void* userdata);
  /// A read callback in the form which can be accepted by libcurl.
  static size_t ReadCallback(void* ptr, size_t size, size_t nmemb,
                             FILE* userdata);
  Status CheckInitialized() const;
  Status CheckMethodNotSet() const;
  Status CheckNotSent() const;

  std::unique_ptr<LibCurl> libcurl_;

  FILE* post_body_ = nullptr;

  StringPiece post_body_buffer_;
  size_t post_body_read_ = 0;

  char* response_buffer_ = nullptr;
  size_t response_buffer_size_ = 0;
  size_t response_buffer_written_ = 0;
  StringPiece* response_string_piece_ = nullptr;
  CURL* curl_ = nullptr;
  curl_slist* curl_headers_ = nullptr;

  std::unique_ptr<char[]> default_response_buffer_;
  StringPiece default_response_string_piece_;

  // Members to enforce the usage flow.
  bool is_initialized_ = false;
  bool is_uri_set_ = false;
  bool is_method_set_ = false;
  bool is_sent_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(HttpRequest);
};

/// \brief A proxy to the libcurl C interface as a dependency injection measure.
///
/// This class is meant as a very thin wrapper for the libcurl C library.
class LibCurl {
 public:
  virtual ~LibCurl() {}
  /// Lazy initialization of the dynamic libcurl library.
  virtual Status MaybeLoadDll() = 0;

  virtual CURL* curl_easy_init() = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    uint64 param) = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    const char* param) = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    void* param) = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    size_t (*param)(void*, size_t, size_t,
                                                    FILE*)) = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    size_t (*param)(const void*, size_t, size_t,
                                                    void*)) = 0;
  virtual CURLcode curl_easy_perform(CURL* curl) = 0;
  virtual CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                                     uint64* value) = 0;
  virtual CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                                     double* value) = 0;
  virtual void curl_easy_cleanup(CURL* curl) = 0;
  virtual curl_slist* curl_slist_append(curl_slist* list, const char* str) = 0;
  virtual void curl_slist_free_all(curl_slist* list) = 0;
  virtual char* curl_easy_escape(CURL* curl, const char* str, int length) = 0;
  virtual void curl_free(void* p) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_HTTP_REQUEST_H_
