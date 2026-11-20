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
#include <memory>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // IWYU pragma: keep
#include "mlir/Dialect/Complex/IR/Complex.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/SCF/IR/SCF.h"  // IWYU pragma: keep
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // IWYU pragma: keep
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/codegen/emitters/transforms/lower_tensors_patterns.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"  // IWYU pragma: keep
#include "xla/codegen/emitters/transforms/lower_tensors_common.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_LOWERTENSORSPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace se = ::stream_executor;

class LowerTensorsPass : public impl::LowerTensorsPassBase<LowerTensorsPass> {
 public:
  LowerTensorsPass() = default;

  explicit LowerTensorsPass(const LowerTensorsPassOptions& options)
      : LowerTensorsPassBase(options) {}

  explicit LowerTensorsPass(const se::DeviceDescription& device_description)
      : device_spec_(device_description) {}

  void runOnOperation() override {
    if (target_type_ == "gpu" && !gpu_device_info_.empty()) {
      se::GpuDeviceInfoProto device_info;
      CHECK(tsl::protobuf::TextFormat::ParseFromString(gpu_device_info_,
                                                       &device_info));
      absl::StatusOr<se::DeviceDescription> device_description =
          se::DeviceDescription::FromProto(device_info);
      CHECK_OK(device_description.status());
      *device_spec_.mutable_type() = *device_description;
    } else if (target_type_ == "cpu") {
      CHECK(gpu_device_info_.empty());
      *device_spec_.mutable_type() = CpuDeviceSpec{};
    }

    if (device_spec_.IsCpu()) {
      if (mlir::failed(
              LowerTensors(getOperation(), GetCpuLowerTensorsConfig()))) {
        signalPassFailure();
      }
      return;
    }

    if (mlir::failed(
            gpu::LowerGpuTensors(getOperation(), device_spec_.gpu()))) {
      signalPassFailure();
    }
  }

 private:
  DeviceSpec device_spec_;
};

}  // namespace

std::unique_ptr<::mlir::Pass> createLowerTensorsPass(
    const se::DeviceDescription& device_description) {
  return std::make_unique<LowerTensorsPass>(device_description);
}

}  // namespace emitters
}  // namespace xla
