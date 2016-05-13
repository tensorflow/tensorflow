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
#include <dlfcn.h>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

// Windows is not currently supported.
constexpr char kCurlLibLinux[] = "libcurl.so.3";
constexpr char kCurlLibMac[] = "/usr/lib/libcurl.3.dylib";

constexpr char kCertsPath[] = "/etc/ssl/certs";

// Set to 1 to enable verbose debug output from curl.
constexpr uint64 kVerboseOutput = 0;

/// An implementation that dynamically loads libcurl and forwards calls to it.
class LibCurlProxy : public LibCurl {
 public:
  ~LibCurlProxy() {
    if (dll_handle_) {
      dlclose(dll_handle_);
    }
  }

  Status MaybeLoadDll() override {
    if (dll_handle_) {
      return Status::OK();
    }
    // This may have been linked statically; if curl_easy_init is in the
    // current binary, no need to search for a dynamic version.
    dll_handle_ = load_dll(nullptr);
    if (!dll_handle_) {
      dll_handle_ = load_dll(kCurlLibLinux);
    }
    if (!dll_handle_) {
      dll_handle_ = load_dll(kCurlLibMac);
    }
    if (!dll_handle_) {
      return errors::FailedPrecondition(strings::StrCat(
          "Could not initialize the libcurl library. Please make sure that "
          "libcurl is installed in the OS or statically linked to the "
          "TensorFlow binary."));
    }
    curl_global_init_(CURL_GLOBAL_ALL);
    return Status::OK();
  }

  CURL* curl_easy_init() override {
    CHECK(dll_handle_);
    return curl_easy_init_();
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            uint64 param) override {
    CHECK(dll_handle_);
    return curl_easy_setopt_(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            const char* param) override {
    CHECK(dll_handle_);
    return curl_easy_setopt_(curl, option, param);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            void* param) override {
    CHECK(dll_handle_);
    return curl_easy_setopt_(curl, option, param);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(void*, size_t, size_t,
                                            FILE*)) override {
    CHECK(dll_handle_);
    return curl_easy_setopt_(curl, option, param);
  }
  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(const void*, size_t, size_t,
                                            void*)) override {
    CHECK(dll_handle_);
    return curl_easy_setopt_(curl, option, param);
  }

  CURLcode curl_easy_perform(CURL* curl) override {
    CHECK(dll_handle_);
    return curl_easy_perform_(curl);
  }

  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             uint64* value) override {
    CHECK(dll_handle_);
    return curl_easy_getinfo_(curl, info, value);
  }
  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             double* value) override {
    CHECK(dll_handle_);
    return curl_easy_getinfo_(curl, info, value);
  }
  void curl_easy_cleanup(CURL* curl) override {
    CHECK(dll_handle_);
    return curl_easy_cleanup_(curl);
  }

  curl_slist* curl_slist_append(curl_slist* list, const char* str) override {
    CHECK(dll_handle_);
    return curl_slist_append_(list, str);
  }

  void curl_slist_free_all(curl_slist* list) override {
    CHECK(dll_handle_);
    return curl_slist_free_all_(list);
  }

 private:
  // Loads the dynamic library and binds the required methods.
  // Returns the library handle in case of success or nullptr otherwise.
  // 'name' can be nullptr.
  void* load_dll(const char* name) {
    void* handle = nullptr;
    handle = dlopen(name, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (!handle) {
      return nullptr;
    }

#define BIND_CURL_FUNC(function) \
  *reinterpret_cast<void**>(&(function##_)) = dlsym(handle, #function)

    BIND_CURL_FUNC(curl_global_init);
    BIND_CURL_FUNC(curl_easy_init);
    BIND_CURL_FUNC(curl_easy_setopt);
    BIND_CURL_FUNC(curl_easy_perform);
    BIND_CURL_FUNC(curl_easy_getinfo);
    BIND_CURL_FUNC(curl_slist_append);
    BIND_CURL_FUNC(curl_slist_free_all);
    BIND_CURL_FUNC(curl_easy_cleanup);

#undef BIND_CURL_FUNC

    if (curl_global_init_ == nullptr) {
      dlerror();  // Clear dlerror before attempting to open libraries.
      dlclose(handle);
      return nullptr;
    }
    return handle;
  }

  void* dll_handle_ = nullptr;
  CURLcode (*curl_global_init_)(int64) = nullptr;
  CURL* (*curl_easy_init_)(void) = nullptr;
  CURLcode (*curl_easy_setopt_)(CURL*, CURLoption, ...) = nullptr;
  CURLcode (*curl_easy_perform_)(CURL* curl) = nullptr;
  CURLcode (*curl_easy_getinfo_)(CURL* curl, CURLINFO info, ...) = nullptr;
  void (*curl_easy_cleanup_)(CURL* curl) = nullptr;
  curl_slist* (*curl_slist_append_)(curl_slist* list,
                                    const char* str) = nullptr;
  void (*curl_slist_free_all_)(curl_slist* list) = nullptr;
};
}  // namespace

HttpRequest::HttpRequest()
    : HttpRequest(std::unique_ptr<LibCurl>(new LibCurlProxy)) {}

HttpRequest::HttpRequest(std::unique_ptr<LibCurl> libcurl)
    : libcurl_(std::move(libcurl)),
      default_response_buffer_(new char[CURL_MAX_WRITE_SIZE]) {}

HttpRequest::~HttpRequest() {
  if (curl_headers_) {
    libcurl_->curl_slist_free_all(curl_headers_);
  }
  if (post_body_) {
    fclose(post_body_);
  }
  if (curl_) {
    libcurl_->curl_easy_cleanup(curl_);
  }
}

Status HttpRequest::Init() {
  if (!libcurl_) {
    return errors::Internal("libcurl proxy cannot be nullptr.");
  }
  TF_RETURN_IF_ERROR(libcurl_->MaybeLoadDll());
  curl_ = libcurl_->curl_easy_init();
  if (!curl_) {
    return errors::Internal("Couldn't initialize a curl session.");
  }

  libcurl_->curl_easy_setopt(curl_, CURLOPT_VERBOSE, kVerboseOutput);
  libcurl_->curl_easy_setopt(curl_, CURLOPT_CAPATH, kCertsPath);

  // If response buffer is not set, libcurl will print results to stdout,
  // so we always set it.
  is_initialized_ = true;
  auto s = SetResultBuffer(default_response_buffer_.get(), CURL_MAX_WRITE_SIZE,
                           &default_response_string_piece_);
  if (!s.ok()) {
    is_initialized_ = false;
    return s;
  }
  return Status::OK();
}

Status HttpRequest::SetUri(const string& uri) {
  TF_RETURN_IF_ERROR(CheckInitialized());
  TF_RETURN_IF_ERROR(CheckNotSent());
  is_uri_set_ = true;
  libcurl_->curl_easy_setopt(curl_, CURLOPT_URL, uri.c_str());
  return Status::OK();
}

Status HttpRequest::SetRange(uint64 start, uint64 end) {
  TF_RETURN_IF_ERROR(CheckInitialized());
  TF_RETURN_IF_ERROR(CheckNotSent());
  libcurl_->curl_easy_setopt(curl_, CURLOPT_RANGE,
                             strings::StrCat(start, "-", end).c_str());
  return Status::OK();
}

Status HttpRequest::AddAuthBearerHeader(const string& auth_token) {
  TF_RETURN_IF_ERROR(CheckInitialized());
  TF_RETURN_IF_ERROR(CheckNotSent());
  if (!auth_token.empty()) {
    curl_headers_ = libcurl_->curl_slist_append(
        curl_headers_,
        strings::StrCat("Authorization: Bearer ", auth_token).c_str());
  }
  return Status::OK();
}

Status HttpRequest::SetDeleteRequest() {
  TF_RETURN_IF_ERROR(CheckInitialized());
  TF_RETURN_IF_ERROR(CheckNotSent());
  TF_RETURN_IF_ERROR(CheckMethodNotSet());
  is_method_set_ = true;
  libcurl_->curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE");
  return Status::OK();
}

Status HttpRequest::SetPostRequest(const string& body_filepath) {
  TF_RETURN_IF_ERROR(CheckInitialized());
  TF_RETURN_IF_ERROR(CheckNotSent());
  TF_RETURN_IF_ERROR(CheckMethodNotSet());
  is_method_set_ = true;
  if (post_body_) {
    fclose(post_body_);
  }
  post_body_ = fopen(body_filepath.c_str(), "r");
  if (!post_body_) {
    return errors::InvalidArgument("Couldnt' open the specified file: " +
                                   body_filepath);
  }
  fseek(post_body_, 0, SEEK_END);
  const auto size = ftell(post_body_);
  fseek(post_body_, 0, SEEK_SET);

  curl_headers_ = libcurl_->curl_slist_append(
      curl_headers_, strings::StrCat("Content-Length: ", size).c_str());
  libcurl_->curl_easy_setopt(curl_, CURLOPT_POST, 1);
  libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                             reinterpret_cast<void*>(post_body_));
  return Status::OK();
}

Status HttpRequest::SetPostRequest() {
  TF_RETURN_IF_ERROR(CheckInitialized());
  TF_RETURN_IF_ERROR(CheckNotSent());
  TF_RETURN_IF_ERROR(CheckMethodNotSet());
  is_method_set_ = true;
  libcurl_->curl_easy_setopt(curl_, CURLOPT_POST, 1);
  curl_headers_ =
      libcurl_->curl_slist_append(curl_headers_, "Content-Length: 0");
  return Status::OK();
}

Status HttpRequest::SetResultBuffer(char* scratch, size_t size,
                                    StringPiece* result) {
  TF_RETURN_IF_ERROR(CheckInitialized());
  TF_RETURN_IF_ERROR(CheckNotSent());
  if (!scratch) {
    return errors::InvalidArgument("scratch cannot be null");
  }
  if (!result) {
    return errors::InvalidArgument("result cannot be null");
  }
  if (size <= 0) {
    return errors::InvalidArgument("buffer size should be positive");
  }

  response_buffer_ = scratch;
  response_buffer_size_ = size;
  response_string_piece_ = result;
  response_buffer_written_ = 0;

  libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEDATA,
                             reinterpret_cast<void*>(this));
  libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION,
                             &HttpRequest::WriteCallback);
  return Status::OK();
}

size_t HttpRequest::WriteCallback(const void* ptr, size_t size, size_t nmemb,
                                  void* this_object) {
  CHECK(ptr);
  auto that = reinterpret_cast<HttpRequest*>(this_object);
  CHECK(that->response_buffer_);
  CHECK(that->response_buffer_size_ >= that->response_buffer_written_);
  const size_t bytes_to_copy =
      std::min(size * nmemb,
               that->response_buffer_size_ - that->response_buffer_written_);
  memcpy(that->response_buffer_ + that->response_buffer_written_, ptr,
         bytes_to_copy);
  that->response_buffer_written_ += bytes_to_copy;
  return bytes_to_copy;
}

Status HttpRequest::Send() {
  TF_RETURN_IF_ERROR(CheckInitialized());
  TF_RETURN_IF_ERROR(CheckNotSent());
  is_sent_ = true;
  if (!is_uri_set_) {
    return errors::FailedPrecondition("URI has not been set.");
  }
  if (curl_headers_) {
    libcurl_->curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, curl_headers_);
  }

  char error_buffer[CURL_ERROR_SIZE];
  libcurl_->curl_easy_setopt(curl_, CURLOPT_ERRORBUFFER, error_buffer);

  const auto curl_result = libcurl_->curl_easy_perform(curl_);

  double written_size = 0;
  libcurl_->curl_easy_getinfo(curl_, CURLINFO_SIZE_DOWNLOAD, &written_size);

  uint64 response_code;
  libcurl_->curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response_code);

  if (curl_result != CURLE_OK) {
    return errors::Internal(string("curl error: ") + error_buffer);
  }
  switch (response_code) {
    case 200:  // OK
    case 204:  // No Content
    case 206:  // Partial Content
      if (response_buffer_ && response_string_piece_) {
        *response_string_piece_ = StringPiece(response_buffer_, written_size);
      }
      return Status::OK();
    case 401:
      return errors::PermissionDenied(
          "Not authorized to access the given HTTP resource.");
    case 404:
      return errors::NotFound("The requested URL was not found.");
    case 416:  // Requested Range Not Satisfiable
      if (response_string_piece_) {
        *response_string_piece_ = StringPiece();
      }
      return Status::OK();
    default:
      return errors::Internal(
          strings::StrCat("Unexpected HTTP response code ", response_code));
  }
}

Status HttpRequest::CheckInitialized() const {
  if (!is_initialized_) {
    return errors::FailedPrecondition("The object has not been initialized.");
  }
  return Status::OK();
}

Status HttpRequest::CheckMethodNotSet() const {
  if (is_method_set_) {
    return errors::FailedPrecondition("HTTP method has been already set.");
  }
  return Status::OK();
}

Status HttpRequest::CheckNotSent() const {
  if (is_sent_) {
    return errors::FailedPrecondition("The request has already been sent.");
  }
  return Status::OK();
}

}  // namespace tensorflow
