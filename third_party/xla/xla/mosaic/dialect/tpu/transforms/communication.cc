/* Copyright 2023 The JAX Authors.

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
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

namespace {

struct CommsAnalysisState {
  bool has_communication = false;
  bool has_custom_barrier = false;

  explicit operator bool() { return has_communication && has_custom_barrier; }
};

void analyzeCrossChipCommunication(mlir::Operation *op,
                                   CommsAnalysisState *state) {
  if (auto dma = dyn_cast<tpu::EnqueueDMAOp>(op)) {
    state->has_communication |= dma.getDeviceId() != nullptr;
  } else if (auto signal = dyn_cast<tpu::SemaphoreSignalOp>(op)) {
    state->has_communication |= signal.getDeviceId() != nullptr;
  } else if (auto barrier = dyn_cast<tpu::GetBarrierSemaphoreOp>(op)) {
    state->has_custom_barrier = true;
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        analyzeCrossChipCommunication(&op, state);
        if (*state) {
          return;
        }
      }
    }
  }
}

}  // namespace

std::pair<bool, bool> mightCommunicateBetweenChips(mlir::Operation *op) {
  CommsAnalysisState state;
  analyzeCrossChipCommunication(op, &state);
  return std::make_pair(state.has_communication, state.has_custom_barrier);
}

#define GEN_PASS_DECL_LOGICALTOPHYSICALDEVICEIDPASS
#define GEN_PASS_DEF_LOGICALTOPHYSICALDEVICEIDPASS
#include "xla/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

template <typename Op>
void logicalToPhysicalDeviceIds(Op op, Value device_assignment) {
  auto device_id = op.getDeviceIdMutable();
  if (device_id.empty()) {
    return;
  }
  CHECK_EQ(device_id.size(), 1);
  mlir::OpBuilder builder(op);
  auto logical_id = builder.create<arith::IndexCastOp>(
      op.getLoc(), builder.getIndexType(), op.getDeviceId());
  auto physical_id = builder.create<memref::LoadOp>(
      op.getLoc(), device_assignment, ValueRange{logical_id});
  device_id.assign(physical_id);
}

}  // namespace

struct LogicalToPhysicalDeviceIdPass
    : public impl::LogicalToPhysicalDeviceIdPassBase<
          LogicalToPhysicalDeviceIdPass> {
  explicit LogicalToPhysicalDeviceIdPass(int64_t total_devices_) {
    total_devices = total_devices_;
  }

  void runOnOperation() override {
    if (total_devices <= 0) {
      signalPassFailure();
      return;
    }
    func::FuncOp func = getOperation();
    if (func.getName() == "main") {
      auto device_assignment_type = MemRefType::get(
          {total_devices}, IntegerType::get(func.getContext(), 32),
          TiledLayoutAttr::get(func.getContext(), {xla::Tile({128})}, {1}),
          MemorySpaceAttr::get(func.getContext(), MemorySpace::kSmem));

      if (failed(func.insertArgument(func.getNumArguments(),
                                     device_assignment_type, nullptr,
                                     UnknownLoc::get(func.getContext())))) {
        return signalPassFailure();
      }
      auto device_assignment_arg = func.getArgument(func.getNumArguments() - 1);
      func.walk([device_assignment_arg](Operation *some_op) {
        if (auto op = dyn_cast<tpu::EnqueueDMAOp>(some_op)) {
          logicalToPhysicalDeviceIds(op, device_assignment_arg);
        } else if (auto op = dyn_cast<tpu::SemaphoreSignalOp>(some_op)) {
          logicalToPhysicalDeviceIds(op, device_assignment_arg);
        }
      });
    } else {
      auto result = func.walk([](Operation *some_op) {
        auto fail = [some_op]() {
          some_op->emitOpError(
              "Communication ops are only allowed in the main function.");
          return WalkResult::interrupt();
        };
        if (auto op = dyn_cast<tpu::EnqueueDMAOp>(some_op)) {
          return fail();
        }
        if (auto op = dyn_cast<tpu::SemaphoreSignalOp>(some_op)) {
          return fail();
        }
        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        signalPassFailure();
      }
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>>
createLogicalToPhysicalDeviceIdPass(int64_t total_devices) {
  return std::make_unique<LogicalToPhysicalDeviceIdPass>(total_devices);
}

}  // namespace mlir::tpu
