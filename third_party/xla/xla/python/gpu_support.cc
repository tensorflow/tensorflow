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

#include <Python.h>

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/set.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/py_client.h"

namespace xla {
namespace {

namespace nb = nanobind;

}  // namespace

void RegisterGpuClientAndDefineGpuAllocatorConfig(nanobind::module_& m_nb) {
  nb::class_<GpuAllocatorConfig> alloc_config(m_nb, "GpuAllocatorConfig");
  alloc_config.def(nb::init<>())
      .def_rw("kind", &GpuAllocatorConfig::kind)
      .def_rw("memory_fraction", &GpuAllocatorConfig::memory_fraction)
      .def_rw("preallocate", &GpuAllocatorConfig::preallocate)
      .def_rw("collective_memory_size",
              &GpuAllocatorConfig::collective_memory_size);
  nb::enum_<GpuAllocatorConfig::Kind>(alloc_config, "Kind")
      .value("DEFAULT", GpuAllocatorConfig::Kind::kDefault)
      .value("PLATFORM", GpuAllocatorConfig::Kind::kPlatform)
      .value("BFC", GpuAllocatorConfig::Kind::kBFC)
      .value("CUDA_ASYNC", GpuAllocatorConfig::Kind::kCudaAsync);

  m_nb.def(
      "get_gpu_client",
      [](bool asynchronous, const GpuAllocatorConfig& allocator_config,
         std::shared_ptr<DistributedRuntimeClient> distributed_client,
         int node_id, int num_nodes,
         std::optional<std::set<int>> allowed_devices,
         std::optional<std::string> platform_name,
         std::optional<bool> mock = false,
         std::optional<std::string> mock_gpu_topology =
             "") -> nb_class_ptr<PyClient> {
        std::unique_ptr<ifrt::PjRtClient> ifrt_client;
        {
          nb::gil_scoped_release gil_release;
          std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr;
          if (distributed_client != nullptr) {
            kv_store = GetDistributedKeyValueStore(distributed_client,
                                                   /*key_prefix=*/"gpu:");
          }
          GpuClientOptions options;
          options.allocator_config = allocator_config;
          options.node_id = node_id;
          options.num_nodes = num_nodes;
          options.allowed_devices = allowed_devices;
          options.platform_name = platform_name;
          options.kv_store = kv_store;
          options.enable_mock_nccl = mock.value_or(false);
          options.mock_gpu_topology = mock_gpu_topology;
          std::unique_ptr<PjRtClient> pjrt_client =
              xla::ValueOrThrow(GetStreamExecutorGpuClient(options));
          ifrt_client = ifrt::PjRtClient::Create(std::move(pjrt_client));
        }
        return PyClient::Make(std::move(ifrt_client));
      },
      nb::arg("asynchronous") = true,
      nb::arg("allocator_config") = GpuAllocatorConfig(),
      nb::arg("distributed_client") = nullptr, nb::arg("node_id") = 0,
      nb::arg("num_nodes") = 1,
      nb::arg("allowed_devices").none() = std::nullopt,
      nb::arg("platform_name").none() = std::nullopt,
      nb::arg("mock").none() = std::nullopt,
      nb::arg("mock_gpu_topology").none() = std::nullopt);
}

}  // namespace xla
