/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_URL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_URL_H_

#include <string>

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace data {

// Parses URLs of form host[:port] and provides methods to retrieve its
// components. The port can be a number, named port, or dynamic port
// (i.e.: %port_name%). For example:
//
//   URL url("/worker/task/0:worker");
//   url.has_protocol() == false;
//   url.host() == "/worker/task/0";
//   url.has_port() == true;
//   url.port() == "worker";
class URL {
 public:
  explicit URL(absl::string_view url);

  absl::string_view host() const { return host_; }
  bool has_port() const { return !port_.empty(); }
  absl::string_view port() const { return port_; }

 private:
  void Parse(absl::string_view url);

  std::string host_;
  std::string port_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_URL_H_
