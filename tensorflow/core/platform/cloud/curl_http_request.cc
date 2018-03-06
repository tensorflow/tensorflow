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

#include <algorithm>

#include "tensorflow/core/platform/cloud/curl_http_request.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {

// Set to 1 to enable verbose debug output from curl.
constexpr uint64 kVerboseOutput = 0;

// Proxy to the real libcurl implementation.
class LibCurlProxy : public LibCurl {
 public:
  static LibCurlProxy* Load() {
    static LibCurlProxy* libcurl = []() -> LibCurlProxy* {
      curl_global_init(CURL_GLOBAL_ALL);
      return new LibCurlProxy;
    }();
    return libcurl;
  }

  CURL* curl_easy_init() override { return ::curl_easy_init(); }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            uint64 param) override {
    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            const char* param) override {
    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            void* param) override {
    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(void*, size_t, size_t,
                                            FILE*)) override {
    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            size_t (*param)(const void*, size_t, size_t,
                                            void*)) override {
    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_setopt(CURL* curl, CURLoption option,
                            int (*param)(void* clientp, curl_off_t dltotal,
                                         curl_off_t dlnow, curl_off_t ultotal,
                                         curl_off_t ulnow)) override {
    return ::curl_easy_setopt(curl, option, param);
  }

  CURLcode curl_easy_perform(CURL* curl) override {
    return ::curl_easy_perform(curl);
  }

  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             uint64* value) override {
    return ::curl_easy_getinfo(curl, info, value);
  }

  CURLcode curl_easy_getinfo(CURL* curl, CURLINFO info,
                             double* value) override {
    return ::curl_easy_getinfo(curl, info, value);
  }

  void curl_easy_cleanup(CURL* curl) override {
    return ::curl_easy_cleanup(curl);
  }

  char* curl_easy_escape(CURL* curl, const char* str, int length) override {
    return ::curl_easy_escape(curl, str, length);
  }

  curl_slist* curl_slist_append(curl_slist* list, const char* str) override {
    return ::curl_slist_append(list, str);
  }

  void curl_slist_free_all(curl_slist* list) override {
    return ::curl_slist_free_all(list);
  }

  void curl_free(void* p) override { ::curl_free(p); }

  const char* curl_easy_strerror(CURLcode errornum) override {
    return ::curl_easy_strerror(errornum);
  }
};
}  // namespace

CurlHttpRequest::CurlHttpRequest() : CurlHttpRequest(LibCurlProxy::Load()) {}

CurlHttpRequest::CurlHttpRequest(LibCurl* libcurl, Env* env)
    : libcurl_(libcurl), env_(env) {
  default_response_buffer_.reserve(CURL_MAX_WRITE_SIZE);

  curl_ = libcurl_->curl_easy_init();
  CHECK(curl_ != nullptr) << "Couldn't initialize a curl session.";

  // NOTE: CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt is configured by
  //       default in //third_party:curl.BUILD and can be customized via an
  //       environment variable.

  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_VERBOSE, kVerboseOutput),
      "Setting verbose output");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(
          curl_, CURLOPT_USERAGENT,
          strings::StrCat("TensorFlow/", TF_VERSION_STRING).c_str()),
      "Setting user agent");
  // Do not use signals for timeouts - does not work in multi-threaded programs.
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L),
      "Disabling signals");
  // We don't log an error here because HTTP/2 support may not be built into
  // cURL, and we'd spam the logs.
  //
  // TODO(jhseu): Enable HTTP/2.
  CURLcodeToStatus(libcurl_->curl_easy_setopt(curl_, CURLOPT_HTTP_VERSION,
                                              CURL_HTTP_VERSION_2_0))
      .IgnoreError();

  // Set up the progress meter.
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_NOPROGRESS, 0ULL),
      "Disabling progress meter");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_XFERINFODATA, this),
      "Setting custom pointer to the progress callback");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_XFERINFOFUNCTION,
                                 &CurlHttpRequest::ProgressCallback),
      "Setting the progress callback");

  // If response buffer is not set, libcurl will print results to stdout,
  // so we always set it.
  SetResultBuffer(&default_response_buffer_);
}

CurlHttpRequest::~CurlHttpRequest() {
  if (curl_headers_) {
    libcurl_->curl_slist_free_all(curl_headers_);
  }
  if (resolve_list_) {
    libcurl_->curl_slist_free_all(resolve_list_);
  }
  if (put_body_) {
    fclose(put_body_);
  }
  if (curl_) {
    libcurl_->curl_easy_cleanup(curl_);
  }
}

string CurlHttpRequest::EscapeString(const string& str) {
  char* out_char_str = libcurl_->curl_easy_escape(curl_, str.c_str(), 0);
  string out_str(out_char_str);
  libcurl_->curl_free(out_char_str);
  return out_str;
}

void CurlHttpRequest::SetUri(const string& uri) {
  CheckNotSent();
  is_uri_set_ = true;
  uri_ = uri;
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_URL, uri.c_str()),
      "Setting URL");
}

void CurlHttpRequest::SetRange(uint64 start, uint64 end) {
  CheckNotSent();
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_RANGE,
                                 strings::StrCat(start, "-", end).c_str()),
      "Setting range");
}

void CurlHttpRequest::AddHeader(const string& name, const string& value) {
  CheckNotSent();
  curl_headers_ = libcurl_->curl_slist_append(
      curl_headers_, strings::StrCat(name, ": ", value).c_str());
}

void CurlHttpRequest::AddResolveOverride(const string& hostname, int64 port,
                                         const string& ip_addr) {
  CheckNotSent();
  // Resolve values are hostname:port:IP.add.ress
  resolve_list_ = libcurl_->curl_slist_append(
      resolve_list_,
      strings::StrCat(hostname, ":", port, ":", ip_addr).c_str());
}

void CurlHttpRequest::AddAuthBearerHeader(const string& auth_token) {
  CheckNotSent();
  if (!auth_token.empty()) {
    AddHeader("Authorization", strings::StrCat("Bearer ", auth_token));
  }
}

void CurlHttpRequest::SetDeleteRequest() {
  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE"),
      "Setting delete request");
}

Status CurlHttpRequest::SetPutFromFile(const string& body_filepath,
                                       size_t offset) {
  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  if (put_body_) {
    fclose(put_body_);
  }
  put_body_ = fopen(body_filepath.c_str(), "r");
  if (!put_body_) {
    return errors::InvalidArgument("Couldn't open the specified file: " +
                                   body_filepath);
  }
  fseek(put_body_, 0, SEEK_END);
  const auto size = ftell(put_body_) - offset;
  fseek(put_body_, offset, SEEK_SET);

  curl_headers_ = libcurl_->curl_slist_append(
      curl_headers_, strings::StrCat("Content-Length: ", size).c_str());
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_PUT, 1), "Setting PUT request");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                 reinterpret_cast<void*>(put_body_)),
      "Setting read data");
  // Using the default CURLOPT_READFUNCTION, which is doing an fread() on the
  // FILE * userdata set with CURLOPT_READDATA.
  return Status::OK();
}

void CurlHttpRequest::SetPutEmptyBody() {
  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_PUT, 1), "Setting put request");
  curl_headers_ =
      libcurl_->curl_slist_append(curl_headers_, "Content-Length: 0");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                 reinterpret_cast<void*>(this)),
      "Setting read data");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_READFUNCTION,
                                 &CurlHttpRequest::ReadCallback),
      "Setting read callback");
}

void CurlHttpRequest::SetPostFromBuffer(const char* buffer, size_t size) {
  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  curl_headers_ = libcurl_->curl_slist_append(
      curl_headers_, strings::StrCat("Content-Length: ", size).c_str());
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_POST, 1),
      "Setting POST request");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                 reinterpret_cast<void*>(this)),
      "Setting read data");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_READFUNCTION,
                                 &CurlHttpRequest::ReadCallback),
      "Setting read callback");
  post_body_buffer_ = StringPiece(buffer, size);
}

void CurlHttpRequest::SetPostEmptyBody() {
  CheckNotSent();
  CheckMethodNotSet();
  is_method_set_ = true;
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_POST, 1),
      "Setting POST request");
  curl_headers_ =
      libcurl_->curl_slist_append(curl_headers_, "Content-Length: 0");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_READDATA,
                                 reinterpret_cast<void*>(this)),
      "Setting read data");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_READFUNCTION,
                                 &CurlHttpRequest::ReadCallback),
      "Setting read callback");
}

void CurlHttpRequest::SetResultBuffer(std::vector<char>* out_buffer) {
  CheckNotSent();
  CHECK(out_buffer != nullptr);

  out_buffer->clear();
  response_buffer_ = out_buffer;

  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEDATA,
                                 reinterpret_cast<void*>(this)),
      "Setting write data");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION,
                                 &CurlHttpRequest::WriteCallback),
      "Setting write callback");
}

void CurlHttpRequest::SetResultBufferDirect(char* buffer, size_t size) {
  CHECK(buffer != nullptr);
  CheckNotSent();

  direct_response_ = DirectResponseState{buffer, size, 0};

  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEDATA,
                                 reinterpret_cast<void*>(this)),
      "Setting write data");
  TF_CURL_LOG_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION,
                                 &CurlHttpRequest::WriteCallbackDirect),
      "Setting write callback");
}

bool CurlHttpRequest::IsDirectResponse() const {
  return direct_response_.buffer_ != nullptr;
}

size_t CurlHttpRequest::WriteCallbackDirect(const void* ptr, size_t size,
                                            size_t nmemb, void* userdata) {
  CHECK(ptr != nullptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(userdata);
  DirectResponseState* state = &that->direct_response_;
  CHECK(state->buffer_ != nullptr);
  CHECK(state->bytes_transferred_ <= state->buffer_size_);

  size_t curl_bytes_received = size * nmemb;
  size_t user_buffer_bytes_available =
      state->buffer_size_ - state->bytes_transferred_;

  // The HTTP server may send a response body that is longer than what we
  // expected. We must not use CHECK() for this situation, because that would
  // imply a code bug (in this client code) where none exists; the violation of
  // expectations would have been caused by the server, not the client. So we
  // report a log warning, if an HTTP server is misbehaving.
  if (curl_bytes_received > user_buffer_bytes_available) {
    LOG(WARNING) << "The HTTP response body that we received is longer than we "
                    "requested or expected. "
                 << "Total bytes requested: " << state->buffer_size_
                 << " Bytes received (so far) in HTTP response body: "
                 << (state->bytes_transferred_ + curl_bytes_received);
  }

  size_t bytes_to_copy =
      std::min<size_t>(curl_bytes_received, user_buffer_bytes_available);
  memcpy(&state->buffer_[state->bytes_transferred_], ptr, bytes_to_copy);
  state->bytes_transferred_ += bytes_to_copy;
  return bytes_to_copy;
}

size_t CurlHttpRequest::GetResultBufferDirectBytesTransferred() {
  CHECK(direct_response_.buffer_ != nullptr);
  return direct_response_.bytes_transferred_;
}

void CurlHttpRequest::SetTimeouts(uint32 connection, uint32 inactivity,
                                  uint32 total) {
  CheckNotSent();
  connect_timeout_secs_ = connection;
  inactivity_timeout_secs_ = inactivity;
  request_timeout_secs_ = total;
}

size_t CurlHttpRequest::WriteCallback(const void* ptr, size_t size,
                                      size_t nmemb, void* this_object) {
  CHECK(ptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  CHECK(that->response_buffer_);
  const size_t bytes_to_copy = size * nmemb;
  that->response_buffer_->insert(
      that->response_buffer_->end(), reinterpret_cast<const char*>(ptr),
      reinterpret_cast<const char*>(ptr) + bytes_to_copy);

  return bytes_to_copy;
}

size_t CurlHttpRequest::ReadCallback(void* ptr, size_t size, size_t nmemb,
                                     FILE* this_object) {
  CHECK(ptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  CHECK(that->post_body_read_ <= that->post_body_buffer_.size());
  const size_t bytes_to_copy = std::min(
      size * nmemb, that->post_body_buffer_.size() - that->post_body_read_);
  memcpy(ptr, that->post_body_buffer_.data() + that->post_body_read_,
         bytes_to_copy);
  that->post_body_read_ += bytes_to_copy;
  return bytes_to_copy;
}

size_t CurlHttpRequest::HeaderCallback(const void* ptr, size_t size,
                                       size_t nmemb, void* this_object) {
  CHECK(ptr);
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  StringPiece header(reinterpret_cast<const char*>(ptr), size * nmemb);
  StringPiece name, value;
  // The supplied header has the form "<name>: <value>", parse it.
  if (strings::Scanner(header)
          .ScanEscapedUntil(':')
          .StopCapture()
          .OneLiteral(": ")
          .GetResult(&value, &name)) {
    string str_value = value.ToString();
    str_util::StripTrailingWhitespace(&str_value);
    that->response_headers_[name.ToString()] = str_value;
  }
  return size * nmemb;
}

// This is pulled out as a separate function so that it's only computed when
// an error occurs.
string response_to_error_message(uint64 response_code, StringPiece response,
                                 size_t response_to_error_limit,
                                 CURLcode curl_result,
                                 StringPiece error_buffer) {
  string error_message = strings::StrCat(
      "Error executing an HTTP request (HTTP response code ", response_code,
      ", error code ", curl_result, ", error message '", error_buffer, "')");
  if (!response.empty()) {
    return strings::StrCat(
        error_message, ", response '",
        response.substr(0, std::min(response.size(), response_to_error_limit)),
        "'");
  }
  return error_message;
}

Status CurlHttpRequest::Send() {
  CheckNotSent();
  CHECK(is_uri_set_) << "URI has not been set.";

  is_sent_ = true;

  if (curl_headers_) {
    TF_CURL_RETURN_WITH_CONTEXT_IF_ERROR(
        libcurl_->curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, curl_headers_),
        "Setting HTTP header");
  }
  if (resolve_list_) {
    TF_CURL_RETURN_WITH_CONTEXT_IF_ERROR(
        libcurl_->curl_easy_setopt(curl_, CURLOPT_RESOLVE, resolve_list_),
        "Setting custom resolves");
  }
  TF_CURL_RETURN_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_HEADERDATA,
                                 reinterpret_cast<void*>(this)),
      "Setting header data");
  TF_CURL_RETURN_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION,
                                 &CurlHttpRequest::HeaderCallback),
      "Setting header function");

  TF_CURL_RETURN_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_TIMEOUT, request_timeout_secs_),
      "Setting request timeout");
  TF_CURL_RETURN_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT,
                                 connect_timeout_secs_),
      "Setting connection timeout");

  char error_buffer[CURL_ERROR_SIZE] = {0};
  TF_CURL_RETURN_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_setopt(curl_, CURLOPT_ERRORBUFFER, error_buffer),
      "Setting error buffer");

  const CURLcode curl_result = libcurl_->curl_easy_perform(curl_);
  TF_CURL_RETURN_WITH_CONTEXT_IF_ERROR(
      curl_result, "Performing request. Detailed error: ", error_buffer);

  double written_size = 0;
  TF_CURL_RETURN_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_getinfo(curl_, CURLINFO_SIZE_DOWNLOAD, &written_size),
      "Fetching written size");

  TF_CURL_RETURN_WITH_CONTEXT_IF_ERROR(
      libcurl_->curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE,
                                  &response_code_),
      "Fetching response code");

  Status result;
  switch (response_code_) {
    // The group of response codes indicating that the request achieved
    // the expected goal.
    case 200:  // OK
    case 201:  // Created
    case 204:  // No Content
    case 206:  // Partial Content
      if (curl_result != CURLE_OK) {
        // This means the server executed the request successfully, but then
        // something went wrong during the transmission of the response.
        result = errors::Unavailable(response_to_error_message(
            response_code_, GetResponse(), response_to_error_limit_,
            curl_result, error_buffer));
      } else {
        result = Status::OK();
      }
      break;
    case 416:  // Requested Range Not Satisfiable
      // The requested range had no overlap with the available range.
      // This doesn't indicate an error, but this does mean an empty response
      // body.
      response_buffer_->clear();
      result = Status::OK();
      break;

    // INVALID_ARGUMENT indicates a problem with how the request is constructed.
    case 400:  // Bad Request
    case 411:  // Length Required
      result = errors::InvalidArgument(response_to_error_message(
          response_code_, GetResponse(), response_to_error_limit_, curl_result,
          error_buffer));
      break;

    // PERMISSION_DENIED indicates an authentication or an authorization issue.
    case 401:  // Unauthorized
    case 403:  // Forbidden
      result = errors::PermissionDenied(response_to_error_message(
          response_code_, GetResponse(), response_to_error_limit_, curl_result,
          error_buffer));
      break;

    // NOT_FOUND indicates that the requested resource does not exist.
    case 404:  // Not found
    case 410:  // Gone
      result = errors::NotFound(response_to_error_message(
          response_code_, GetResponse(), response_to_error_limit_, curl_result,
          error_buffer));
      break;

    // FAILED_PRECONDITION indicates that the request failed because some
    // of the underlying assumptions were not satisfied. The request
    // shouldn't be retried unless the external context has changed.
    case 302:  // Found
    case 303:  // See Other
    case 304:  // Not Modified
    case 307:  // Temporary Redirect
    case 412:  // Precondition Failed
    case 413:  // Payload Too Large
      result = errors::FailedPrecondition(response_to_error_message(
          response_code_, GetResponse(), response_to_error_limit_, curl_result,
          error_buffer));
      break;

    // UNAVAILABLE indicates a problem that can go away if the request
    // is just retried without any modification. 308 return codes are intended
    // for write requests that can be retried. See the documentation and the
    // official library:
    // https://cloud.google.com/storage/docs/json_api/v1/how-tos/resumable-upload
    // https://github.com/google/apitools/blob/master/apitools/base/py/transfer.py
    case 308:  // Resume Incomplete
    case 409:  // Conflict
    case 429:  // Too Many Requests
    case 500:  // Internal Server Error
    case 502:  // Bad Gateway
    case 503:  // Service Unavailable
    default:   // All other HTTP response codes also should be retried.
      result = errors::Unavailable(response_to_error_message(
          response_code_, GetResponse(), response_to_error_limit_, curl_result,
          error_buffer));
      break;
  }
  if (!result.ok()) {
    response_buffer_->clear();
  }
  return result;
}

void CurlHttpRequest::CheckMethodNotSet() const {
  CHECK(!is_method_set_) << "HTTP method has been already set.";
}

void CurlHttpRequest::CheckNotSent() const {
  CHECK(!is_sent_) << "The request has already been sent.";
}

StringPiece CurlHttpRequest::GetResponse() const {
  StringPiece response;
  if (IsDirectResponse()) {
    response = StringPiece(direct_response_.buffer_,
                           direct_response_.bytes_transferred_);
  } else {
    response = StringPiece(response_buffer_->data(), response_buffer_->size());
  }
  return response;
}

string CurlHttpRequest::GetResponseHeader(const string& name) const {
  const auto& header = response_headers_.find(name);
  return header != response_headers_.end() ? header->second : "";
}

uint64 CurlHttpRequest::GetResponseCode() const { return response_code_; }

// Cancels the transmission if no progress has been made for too long.
int CurlHttpRequest::ProgressCallback(void* this_object, curl_off_t dltotal,
                                      curl_off_t dlnow, curl_off_t ultotal,
                                      curl_off_t ulnow) {
  auto that = reinterpret_cast<CurlHttpRequest*>(this_object);
  const auto now = that->env_->NowSeconds();
  const auto current_progress = dlnow + ulnow;
  if (that->last_progress_timestamp_ == 0 ||
      current_progress > that->last_progress_bytes_) {
    // This is the first time the callback is called or some progress
    // was made since the last tick.
    that->last_progress_timestamp_ = now;
    that->last_progress_bytes_ = current_progress;
    return 0;
  }

  if (now - that->last_progress_timestamp_ > that->inactivity_timeout_secs_) {
    double lookup_time = -1;
    const auto lookup_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_NAMELOOKUP_TIME, &lookup_time);

    double connect_time = -1;
    const auto connect_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_CONNECT_TIME, &connect_time);

    double pretransfer_time = -1;
    const auto pretransfer_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_PRETRANSFER_TIME, &pretransfer_time);

    double starttransfer_time = -1;
    const auto starttransfer_time_status = that->libcurl_->curl_easy_getinfo(
        that->curl_, CURLINFO_PRETRANSFER_TIME, &starttransfer_time);

    LOG(ERROR) << "The transmission  of request " << this_object
               << " (URI: " << that->uri_ << ") has been stuck at "
               << current_progress << " of " << dltotal + ultotal
               << " bytes for " << now - that->last_progress_timestamp_
               << " seconds and will be aborted. CURL timing information: "
               << "lookup time: " << lookup_time << " ("
               << that->libcurl_->curl_easy_strerror(lookup_time_status)
               << "), connect time: " << connect_time << " ("
               << that->libcurl_->curl_easy_strerror(connect_time_status)
               << "), pre-transfer time: " << pretransfer_time << " ("
               << that->libcurl_->curl_easy_strerror(pretransfer_time_status)
               << "), start-transfer time: " << starttransfer_time << " ("
               << that->libcurl_->curl_easy_strerror(starttransfer_time_status)
               << ")";
    return 1;  // Will abort the request.
  }

  // No progress was made since the last call, but we should wait a bit longer.
  return 0;
}

Status CURLcodeToStatus(CURLcode code) {
  // Return Unavailable to retry by default. We probably should distinguish
  // between permanent or temporary failures.
  return errors::Unavailable("Error executing an HTTP request (error code ",
                             code, ", error message '",
                             curl_easy_strerror(code), "')");
}

}  // namespace tensorflow
