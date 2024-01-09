/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_CLOUD_CURL_HTTP_REQUEST_H_
#define TENSORFLOW_TSL_PLATFORM_CLOUD_CURL_HTTP_REQUEST_H_

#include <string>
#include <unordered_map>
#include <vector>

#include <curl/curl.h>
#include "tsl/platform/cloud/http_request.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/platform/stringpiece.h"
#include "tsl/platform/types.h"

namespace tsl {

class LibCurl;  // libcurl interface as a class, for dependency injection.

/// \brief A basic HTTP client based on the libcurl library.
///
/// The usage pattern for the class reflects the one of the libcurl library:
/// create a request object, set request parameters and call Send().
///
/// For example:
///   std::unique_ptr<HttpRequest> request(http_request_factory->Create());
///   request->SetUri("http://www.google.com");
///   request->SetResultsBuffer(out_buffer);
///   request->Send();
class CurlHttpRequest : public HttpRequest {
 public:
  class Factory : public HttpRequest::Factory {
   public:
    virtual ~Factory() {}
    virtual HttpRequest* Create() { return new CurlHttpRequest(); }
  };

  CurlHttpRequest();
  explicit CurlHttpRequest(LibCurl* libcurl)
      : CurlHttpRequest(libcurl, Env::Default()) {}
  CurlHttpRequest(LibCurl* libcurl, Env* env);
  ~CurlHttpRequest() override;

  /// Sets the request URI.
  void SetUri(const string& uri) override;

  /// \brief Sets the Range header.
  ///
  /// Used for random seeks, for example "0-999" returns the first 1000 bytes
  /// (note that the right border is included).
  void SetRange(uint64 start, uint64 end) override;

  /// Sets a request header.
  void AddHeader(const string& name, const string& value) override;

  void AddResolveOverride(const string& hostname, int64_t port,
                          const string& ip_addr) override;

  /// Sets the 'Authorization' header to the value of 'Bearer ' + auth_token.
  void AddAuthBearerHeader(const string& auth_token) override;

  void SetRequestStats(RequestStats* stats) override;

  /// Makes the request a DELETE request.
  void SetDeleteRequest() override;

  /// \brief Makes the request a PUT request.
  ///
  /// The request body will be taken from the specified file starting from
  /// the given offset.
  Status SetPutFromFile(const string& body_filepath, size_t offset) override;

  /// Makes the request a PUT request with an empty body.
  void SetPutEmptyBody() override;

  /// \brief Makes the request a POST request.
  ///
  /// The request body will be taken from the specified buffer.
  void SetPostFromBuffer(const char* buffer, size_t size) override;

  /// Makes the request a POST request with an empty body.
  void SetPostEmptyBody() override;

  /// \brief Specifies the buffer for receiving the response body.
  ///
  /// Size of out_buffer after an access will be exactly the number of bytes
  /// read. Existing content of the vector will be cleared.
  void SetResultBuffer(std::vector<char>* out_buffer) override;

  /// \brief Specifies the buffer for receiving the response body, when the
  /// caller knows the maximum size of the response body.
  ///
  /// This method allows the caller to receive the response body without an
  /// additional intermediate buffer allocation and copy.  This method should
  /// be called before calling Send(). After Send() has succeeded, the caller
  /// should use the GetResultBufferDirectBytesTransferred() method in order
  /// to learn how many bytes were transferred.
  ///
  /// Using this method is mutually exclusive with using SetResultBuffer().
  void SetResultBufferDirect(char* buffer, size_t size) override;

  /// \brief Distinguish response type (direct vs. implicit).
  bool IsDirectResponse() const;

  /// \brief Returns the number of bytes (of the response body) that were
  /// transferred, when using the SetResultBufferDirect() method. The returned
  /// value will always be less than or equal to the 'size' parameter that
  /// was passed to SetResultBufferDirect(). If the actual HTTP response body
  /// was greater than 'size' bytes, then this transfer method will only copy
  /// the first 'size' bytes, and the rest will be ignored.
  size_t GetResultBufferDirectBytesTransferred() override;

  /// \brief Returns the response headers of a completed request.
  ///
  /// If the header is not found, returns an empty string.
  string GetResponseHeader(const string& name) const override;

  /// Returns the response code of a completed request.
  uint64 GetResponseCode() const override;

  /// \brief Sends the formed request.
  ///
  /// If the result buffer was defined, the response will be written there.
  /// The object is not designed to be re-used after Send() is executed.
  Status Send() override;

  // Url encodes str and returns a new string.
  string EscapeString(const string& str) override;

  void SetTimeouts(uint32 connection, uint32 inactivity, uint32 total) override;

 private:
  /// A write callback in the form which can be accepted by libcurl.
  static size_t WriteCallback(const void* ptr, size_t size, size_t nmemb,
                              void* userdata);

  /// Processes response body content received when using SetResultBufferDirect.
  static size_t WriteCallbackDirect(const void* ptr, size_t size, size_t nmemb,
                                    void* userdata);
  /// A read callback in the form which can be accepted by libcurl.
  static size_t ReadCallback(void* ptr, size_t size, size_t nmemb,
                             FILE* userdata);
  /// A header callback in the form which can be accepted by libcurl.
  static size_t HeaderCallback(const void* ptr, size_t size, size_t nmemb,
                               void* this_object);
  /// A progress meter callback in the form which can be accepted by libcurl.
  static int ProgressCallback(void* this_object, curl_off_t dltotal,
                              curl_off_t dlnow, curl_off_t ultotal,
                              curl_off_t ulnow);
  void CheckMethodNotSet() const;
  void CheckNotSent() const;
  StringPiece GetResponse() const;

  /// Helper to convert the given CURLcode and error buffer, representing the
  /// result of performing a transfer, into a Status with an error message.
  Status CURLcodeToStatus(CURLcode code, const char* error_buffer);

  LibCurl* libcurl_;
  Env* env_;

  FILE* put_body_ = nullptr;

  StringPiece post_body_buffer_;
  size_t post_body_read_ = 0;

  std::vector<char>* response_buffer_ = nullptr;

  struct DirectResponseState {
    char* buffer_;
    size_t buffer_size_;
    size_t bytes_transferred_;
    size_t bytes_received_;
  };
  DirectResponseState direct_response_ = {};

  CURL* curl_ = nullptr;
  curl_slist* curl_headers_ = nullptr;
  curl_slist* resolve_list_ = nullptr;

  RequestStats* stats_ = nullptr;

  std::vector<char> default_response_buffer_;

  std::unordered_map<string, string> response_headers_;
  uint64 response_code_ = 0;

  // The timestamp of the last activity related to the request execution, in
  // seconds since epoch.
  uint64 last_progress_timestamp_ = 0;
  // The last progress in terms of bytes transmitted.
  curl_off_t last_progress_bytes_ = 0;

  // The maximum period of request inactivity.
  uint32 inactivity_timeout_secs_ = 60;  // 1 minute

  // Timeout for the connection phase.
  uint32 connect_timeout_secs_ = 120;  // 2 minutes

  // Timeout for the whole request. Set only to prevent hanging indefinitely.
  uint32 request_timeout_secs_ = 3600;  // 1 hour

  // Members to enforce the usage flow.
  bool is_uri_set_ = false;
  bool is_method_set_ = false;
  bool is_sent_ = false;

  // Store the URI to help disambiguate requests when errors occur.
  string uri_;
  RequestMethod method_ = RequestMethod::kGet;

  // Limit the size of an http response that is copied into an error message.
  const size_t response_to_error_limit_ = 500;

  CurlHttpRequest(const CurlHttpRequest&) = delete;
  void operator=(const CurlHttpRequest&) = delete;
};

/// \brief A proxy to the libcurl C interface as a dependency injection measure.
///
/// This class is meant as a very thin wrapper for the libcurl C library.
class LibCurl {
 public:
  virtual ~LibCurl() {}

  virtual CURL* curl_easy_init() = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    uint64 param) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    const char* param) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    void* param) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_setopt(
      CURL* curl, CURLoption option,
      size_t (*param)(void*, size_t, size_t, FILE*)) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                                    size_t (*param)(const void*, size_t, size_t,
                                                    void*))
      TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_setopt(
      CURL* curl, CURLoption option,
      int (*param)(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                   curl_off_t ultotal,
                   curl_off_t ulnow)) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_perform(CURL* curl) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                                     uint64* value) TF_MUST_USE_RESULT = 0;
  virtual CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                                     double* value) TF_MUST_USE_RESULT = 0;
  virtual void curl_easy_cleanup(CURL* curl) = 0;
  virtual curl_slist* curl_slist_append(curl_slist* list, const char* str) = 0;
  virtual void curl_slist_free_all(curl_slist* list) = 0;
  virtual char* curl_easy_escape(CURL* curl, const char* str, int length) = 0;
  virtual void curl_free(void* p) = 0;
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_CLOUD_CURL_HTTP_REQUEST_H_
