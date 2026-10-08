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

#include <cstdint>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/pjrt/exceptions.h"

namespace xla {

namespace nb = nanobind;

namespace {

class GpuTargetConfigWrapper {
 public:
  explicit GpuTargetConfigWrapper(gpu::GpuTargetConfig config)
      : config_(std::move(config)) {}

  const std::string& platform_name() const { return config_.platform_name; }
  const std::string& device_description_str() const {
    return config_.device_description_str;
  }

  std::string arch_name() const {
    const auto& cc = config_.device_description.cuda_compute_capability();
    if (cc.major > 0) {
      return cc.ToString();
    }
    const auto& rcc = config_.device_description.rocm_compute_capability();
    const auto& arch_name = rcc.ToString();
    if (arch_name != stream_executor::RocmComputeCapability::kInvalidGfx) {
      return arch_name;
    }
    return "";
  }

  int64_t compute_capability() const {
    const auto& cc = config_.device_description.cuda_compute_capability();
    if (cc.major > 0) {
      return cc.major * 10 + cc.minor;
    }
    return 0;
  }

  int64_t core_count() const { return config_.device_description.core_count(); }

  int64_t smem_capacity_bytes() const {
    return config_.device_description.shared_memory_per_core();
  }

 private:
  gpu::GpuTargetConfig config_;
};

}  // namespace

NB_MODULE(_gpu_spec, m) {
  nb::class_<GpuTargetConfigWrapper> gpu_target_config_class(m,
                                                             "GpuTargetConfig");
  gpu_target_config_class
      .def_prop_ro("platform_name", &GpuTargetConfigWrapper::platform_name)
      .def_prop_ro("device_description_str",
                   &GpuTargetConfigWrapper::device_description_str)
      .def_prop_ro("arch_name", &GpuTargetConfigWrapper::arch_name)
      .def_prop_ro("compute_capability",
                   &GpuTargetConfigWrapper::compute_capability)
      .def_prop_ro("core_count", &GpuTargetConfigWrapper::core_count)
      .def_prop_ro("smem_capacity_bytes",
                   &GpuTargetConfigWrapper::smem_capacity_bytes);

  nb::enum_<gpu::GpuModel>(m, "GpuModel")
      .value("A100_PCIE_80", gpu::GpuModel::A100_PCIE_80)
      .value("A100_SXM_40", gpu::GpuModel::A100_SXM_40)
      .value("A100_SXM_80", gpu::GpuModel::A100_SXM_80)
      .value("A6000", gpu::GpuModel::A6000)
      .value("B200", gpu::GpuModel::B200)
      .value("B300", gpu::GpuModel::B300)
      .value("BMG_G21", gpu::GpuModel::BMG_G21)
      .value("H100_PCIE", gpu::GpuModel::H100_PCIE)
      .value("H100_SXM", gpu::GpuModel::H100_SXM)
      .value("H200", gpu::GpuModel::H200)
      .value("MI200", gpu::GpuModel::MI200)
      .value("P100", gpu::GpuModel::P100)
      .value("PVC", gpu::GpuModel::PVC)
      .value("VR_NVL72", gpu::GpuModel::VR_NVL72)
      .value("V100", gpu::GpuModel::V100)
      .value("GB200", gpu::GpuModel::GB200)
      .value("GB300", gpu::GpuModel::GB300)
      .value("RTX6000PRO", gpu::GpuModel::RTX6000PRO)
      .value("GFX1250", gpu::GpuModel::GFX1250)
      .value("MI350", gpu::GpuModel::MI350);

  m.def(
      "get_gpu_spec",
      [](const gpu::GpuModel& gpu_model) -> GpuTargetConfigWrapper {
        auto maybe_proto = gpu::GetGpuTargetConfig(gpu_model);
        if (!maybe_proto.ok()) {
          throw xla::XlaRuntimeError(maybe_proto.status().message().data());
        }
        auto maybe_target_config =
            gpu::GpuTargetConfig::FromProto(*maybe_proto);
        if (!maybe_target_config.ok()) {
          throw xla::XlaRuntimeError(
              maybe_target_config.status().message().data());
        }
        return GpuTargetConfigWrapper(std::move(*maybe_target_config));
      },
      nb::arg("gpu_model"));
}

}  // namespace xla
