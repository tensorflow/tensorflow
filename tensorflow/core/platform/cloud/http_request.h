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
#include <unordered_map>
#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// \brief An abstract basic HTTP client.
///
/// The usage pattern for the class is based on the libcurl library:
/// create a request object, set request parameters and call Send().
///
/// For example:
///   HttpRequest request;
///   request.SetUri("http://www.google.com");
///   request.SetResultsBuffer(out_buffer);
///   request.Send();
class HttpRequest {
 public:
  class Factory {
   public:
    virtual ~Factory() {}
    virtual HttpRequest* Create() = 0;
  };

  HttpRequest() {}
  virtual ~HttpRequest() {}

  /// Sets the request URI.
  virtual void SetUri(const string& uri) = 0;

  /// \brief Sets the Range header.
  ///
  /// Used for random seeks, for example "0-999" returns the first 1000 bytes
  /// (note that the right border is included).
  virtual void SetRange(uint64 start, uint64 end) = 0;

  /// Sets a request header.
  virtual void AddHeader(const string& name, const string& value) = 0;

  /// Sets a DNS resolve mapping (to skip DNS resolution).
  ///
  /// Note: because GCS is available over HTTPS, we cannot replace the hostname
  /// in the URI with an IP address, as that will cause the certificate check
  /// to fail.
  virtual void AddResolveOverride(const string& hostname, int64 port,
                                  const string& ip_addr) = 0;

  /// Sets the 'Authorization' header to the value of 'Bearer ' + auth_token.
  virtual void AddAuthBearerHeader(const string& auth_token) = 0;

  /// Makes the request a DELETE request.
  virtual void SetDeleteRequest() = 0;

  /// \brief Makes the request a PUT request.
  ///
  /// The request body will be taken from the specified file starting from
  /// the given offset.
  virtual Status SetPutFromFile(const string& body_filepath, size_t offset) = 0;

  /// Makes the request a PUT request with an empty body.
  virtual void SetPutEmptyBody() = 0;

  /// \brief Makes the request a POST request.
  ///
  /// The request body will be taken from the specified buffer.
  virtual void SetPostFromBuffer(const char* buffer, size_t size) = 0;

  /// Makes the request a POST request with an empty body.
  virtual void SetPostEmptyBody() = 0;

  /// \brief Specifies the buffer for receiving the response body.
  ///
  /// Size of out_buffer after an access will be exactly the number of bytes
  /// read. Existing content of the vector will be cleared.
  virtual void SetResultBuffer(std::vector<char>* out_buffer) = 0;

  /// \brief Specifies the buffer for receiving the response body.
  ///
  /// This method should be used when a caller knows the upper bound of the
  /// size of the response data.  The caller provides a pre-allocated buffer
  /// and its size. After the Send() method is called, the
  /// GetResultBufferDirectBytesTransferred() method may be used to learn to the
  /// number of bytes that were transferred using this method.
  virtual void SetResultBufferDirect(char* buffer, size_t size) = 0;

  /// \brief Returns the number of bytes transferred, when using
  /// SetResultBufferDirect(). This method may only be used when using
  /// SetResultBufferDirect().
  virtual size_t GetResultBufferDirectBytesTransferred() = 0;

  /// \brief Returns the response headers of a completed request.
  ///
  /// If the header is not found, returns an empty string.
  virtual string GetResponseHeader(const string& name) const = 0;

  /// Returns the response code of a completed request.
  virtual uint64 GetResponseCode() const = 0;

  /// \brief Sends the formed request.
  ///
  /// If the result buffer was defined, the response will be written there.
  /// The object is not designed to be re-used after Send() is executed.
  virtual Status Send() = 0;

  // Url encodes str and returns a new string.
  virtual string EscapeString(const string& str) = 0;

  /// \brief Set timeouts for this request.
  ///
  /// The connection parameter controls how long we should wait for the
  /// connection to be established. The inactivity parameter controls how long
  /// we should wait between additional responses from the server. Finally the
  /// total parameter controls the maximum total connection time to prevent
  /// hanging indefinitely.
  virtual void SetTimeouts(uint32 connection, uint32 inactivity,
                           uint32 total) = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(HttpRequest);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_HTTP_REQUEST_H_
