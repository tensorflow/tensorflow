/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_PROXY_COMMON_PROF_UTIL_H_
#define XLA_PYTHON_IFRT_PROXY_COMMON_PROF_UTIL_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/platform/random.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla {
namespace ifrt {
namespace proxy {

// XFlowHelper makes it easier to create trace spans with a flow between them.
// Typical usage:
//
// XFlowHelper flow("my_request");
// ...
//
// auto response_handler = [flow](ResponseMsg msg) {
//   flow.InstantActivity<kRecv>();
//   LOG(INFO) << "Received response: " << msg;
// }
//
// {
//   auto request_span = flow.Span<kSend>();
//   auto request_protobuf = CreateRequestProtobuf();
//   transport.Send(request_protobuf, response_handler);
// }
//
class XFlowHelper {
 public:
  explicit XFlowHelper(absl::string_view name)
      : xflow_id_(tsl::random::New64() >> 8 /*XFlow IDs are 56 bits*/),
        name_(name) {}

  typedef enum { kSend, kRecv, kRecvSend } Direction;

  template <Direction D>
  tsl::profiler::TraceMe Span() const {
    return tsl::profiler::TraceMe([xflow_id = xflow_id_, name = name_] {
      return Encode<D>(xflow_id, name);
    });
  }

  template <Direction D>
  void InstantActivity() const {
    return tsl::profiler::TraceMe::InstantActivity(
        [xflow_id = xflow_id_, name = name_] {
          return Encode<D>(xflow_id, name);
        });
  }

 private:
  template <Direction D>
  static std::string Encode(uint64_t xflow_id, absl::string_view name) {
    using XFlow = ::tsl::profiler::XFlow;
    switch (D) {
      case kSend:
        return tsl::profiler::TraceMeEncode(
            name, {{"dir", "send"},
                   {"flow", XFlow(xflow_id, XFlow::kFlowOut).ToStatValue()}});
      case kRecv:
        return tsl::profiler::TraceMeEncode(
            name, {{"dir", "recv"},
                   {"flow", XFlow(xflow_id, XFlow::kFlowIn).ToStatValue()}});
      case kRecvSend:
        return tsl::profiler::TraceMeEncode(
            name, {{"dir", "recv_send"},
                   {"flow", XFlow(xflow_id, XFlow::kFlowInOut).ToStatValue()}});
    }
  };

  const uint64_t xflow_id_;
  const absl::string_view name_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_COMMON_PROF_UTIL_H_
