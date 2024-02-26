// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/functional.h"  // from @pybind11  // NOLINT  // IWYU pragma: keep
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil  // NOLINT  // IWYU pragma: keep
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt_proxy/client/registry.h"
#include "xla/python/py_client.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

struct PyClientConnectionOptions {
  std::function<void(absl::Status)> on_disconnect;
  std::function<void(std::string)> on_connection_update;
};

absl::StatusOr<std::shared_ptr<xla::PyClient>> GetClient(
    std::string proxy_server_address,
    const PyClientConnectionOptions& py_options) {
  DCHECK(PyGILState_Check());
  std::unique_ptr<xla::ifrt::Client> client;

  ClientConnectionOptions options;
  if (py_options.on_disconnect) {
    // While it is possible to pass around `py_options.on_disconnect` without
    // wrapping it via a shared_ptr, copying the `py_options.on_disconnect`
    // object can internally attempt to acquire the GIL [1], and can thus block
    // or even deadlock. A unique_ptr or `absl::AnyInvocable` is not sufficient
    // because downstream code can make copies. Reference:
    // https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
    auto py_on_disconnect = std::make_shared<std::function<void(absl::Status)>>(
        std::move(py_options.on_disconnect));

    options.on_disconnect =
        [on_disconnect = std::move(py_on_disconnect)](absl::Status s) mutable {
          LOG(WARNING) << "Connection to server failed, calling supplied "
                       << "`on_disconnect` function: " << s;
          tsl::Env::Default()->SchedClosure([s, on_disconnect]() mutable {
            pybind11::gil_scoped_acquire gil_acquire;
            (*on_disconnect)(s);
            on_disconnect = nullptr;
          });
        };
  }

  if (py_options.on_connection_update) {
    auto fn = std::make_shared<std::function<void(std::string)>>(
        std::move(py_options.on_connection_update));
    options.on_connection_update = [fn](absl::string_view log_line) -> void {
      tsl::Env::Default()->SchedClosure([fn, str = std::string(log_line)] {
        pybind11::gil_scoped_acquire gil_acquire;
        (*fn)(std::string(str));
      });
    };
  }

  {
    pybind11::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(client, CreateClient(proxy_server_address, options));
  }

  // Constructing `xla::PyClient` requires GIL as it may dec-ref Python objects.
  return std::make_shared<xla::PyClient>(std::move(client));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

PYBIND11_MODULE(py_module, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  using ::xla::ifrt::proxy::PyClientConnectionOptions;
  pybind11::class_<PyClientConnectionOptions>(m, "ClientConnectionOptions")
      .def(pybind11::init<>())
      .def_readwrite("on_disconnect", &PyClientConnectionOptions::on_disconnect)
      .def_readwrite("on_connection_update",
                     &PyClientConnectionOptions::on_connection_update);

  m.def("get_client", xla::ValueOrThrowWrapper(xla::ifrt::proxy::GetClient),
        pybind11::arg("proxy_server_address"), pybind11::arg("options"));
}
