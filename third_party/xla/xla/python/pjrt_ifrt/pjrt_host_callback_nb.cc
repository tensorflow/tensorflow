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

#include <Python.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/function.h"  // IWYU pragma: keep
#include "third_party/py/jax/jaxlib/py_client.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/types.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace {

namespace nb = nanobind;

nb::capsule CreateHloOutputCallbackNB(
    jax::PyClient* client, int64_t hlo_id, int64_t num_operands,
    std::function<void(int64_t, int64_t, nb::object)> cb) {
  auto xla_cb = std::make_unique<xla::HloOutputCallback>();
  xla_cb->hlo_id = hlo_id;
  xla_cb->num_operands = num_operands;
  xla_cb->callback = [cb = std::move(cb)](int64_t replica_id,
                                          int64_t partition_id,
                                          absl::Span<Literal const*> literals) {
    nb::gil_scoped_acquire acquire;
    nb::list py_list;
    for (const auto* lit : literals) {
      if (lit != nullptr) {
        auto nbobj =
            xla::LiteralToPython(std::make_shared<xla::Literal>(lit->Clone()));
        if (!nbobj.ok()) {
          LOG(ERROR) << "LiteralToPython failed: " << nbobj.status();
          return;
        }
        py_list.append(*nbobj);
      } else {
        py_list.append(nb::none());
      }
    }
    cb(replica_id, partition_id, py_list);
  };
  auto loaded_host_callback =
      tsl::MakeRef<ifrt::PjRtHloOutputLoadedHostCallback>(client->ifrt_client(),
                                                          std::move(xla_cb));
  return nb::capsule(loaded_host_callback.release(), [](void* ptr) noexcept {
    static_cast<ifrt::LoadedHostCallback*>(ptr)->DropRef();
  });
}

}  // namespace

NB_MODULE(pjrt_host_callback_nb, m) {
  m.def("create_hlo_output_callback", &CreateHloOutputCallbackNB,
        nb::arg("client"), nb::arg("hlo_id"), nb::arg("num_operands"),
        nb::arg("callback"));
}

}  // namespace xla
