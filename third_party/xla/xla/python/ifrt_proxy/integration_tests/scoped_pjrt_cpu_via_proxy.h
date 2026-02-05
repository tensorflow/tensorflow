/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_PROXY_INTEGRATION_TESTS_SCOPED_PJRT_CPU_VIA_PROXY_H_
#define XLA_PYTHON_IFRT_PROXY_INTEGRATION_TESTS_SCOPED_PJRT_CPU_VIA_PROXY_H_

#include <memory>

#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace test_util {

// Test helper that instantiates and registers an IFRT proxy client and server
// that use a PjRt CPU backend, and destroys them when it goes out of scope.
class ScopedPjRtCpuViaProxy {
 public:
  ScopedPjRtCpuViaProxy();
  ~ScopedPjRtCpuViaProxy();

 private:
  std::unique_ptr<GrpcServer> server_;
  std::shared_ptr<xla::ifrt::Client> client_;
};

}  // namespace test_util
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_INTEGRATION_TESTS_SCOPED_PJRT_CPU_VIA_PROXY_H_
