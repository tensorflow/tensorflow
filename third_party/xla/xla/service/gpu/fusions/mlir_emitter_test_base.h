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

#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_EMITTER_TEST_BASE_H_
#define XLA_SERVICE_GPU_FUSIONS_MLIR_EMITTER_TEST_BASE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/mlir_fusion_emitter.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

class MlirEmitterTestBaseImpl : public HloTestBase {
 public:
  MlirEmitterTestBaseImpl();

  virtual std::unique_ptr<MlirFusionEmitterBase> GetEmitter(
      const HloFusionAnalysis& analysis) = 0;

  DebugOptions GetDebugOptionsForTest() override;

  absl::StatusOr<std::string> EmitIR(std::string_view hlo_string);
  absl::Status EmitAndCheckIR(std::string_view hlo_string,
                              std::string_view pattern);

  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  mlir::MLIRContext mlir_context_;
  AffineMapPrinter thread_id_printer_;
};

template <typename EmitterType>
class MlirEmitterTestBase : public MlirEmitterTestBaseImpl {
 public:
  std::unique_ptr<MlirFusionEmitterBase> GetEmitter(
      const HloFusionAnalysis& analysis) override {
    return std::make_unique<EmitterType>(analysis);
  }
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_MLIR_EMITTER_TEST_BASE_H_
