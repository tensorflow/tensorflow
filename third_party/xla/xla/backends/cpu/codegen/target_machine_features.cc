/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/target_machine_features.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/codegen/device_spec.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

CpuDeviceSpec CpuDeviceSpecFromTargetMachine(
    const llvm::TargetMachine& target_machine) {
  const llvm::Triple& triple = target_machine.getTargetTriple();
  const llvm::MCSubtargetInfo& subtarget = target_machine.getMCSubtargetInfo();

  if (triple.isX86_64() && triple.getEnvironment() != llvm::Triple::GNUX32) {
    // X86 sets MaxAtomicSizeInBitsSupported to 128 only on 64 bit targets with
    // +cx16 (lock cmpxchg16b); without +cx16, LLVM emits __atomic_*_16 libcalls
    // that the XLA:CPU JIT does not provide. With both +cx16 and +avx (and
    // without soft float), LLVM lowers 16 byte aligned monotonic 128 bit atomic
    // loads and stores directly to single vector moves (vmovdqa).
    if (!subtarget.checkFeatures("+64bit-mode,+cx16")) {
      return {};
    }
    return CpuDeviceSpec{
        /*has_atomic_cmpxchg_128=*/true,
        /*has_atomic_load_store_128=*/
        subtarget.checkFeatures("+avx,-soft-float"),
    };
  }

  if (triple.getArch() == llvm::Triple::aarch64 &&
      triple.isAArch64(/*PointerWidth=*/64)) {
    // AArch64 lowers 128 bit cmpxchg natively via casp when +lse is enabled or
    // via ldxp/stxp when +outline-atomics is disabled. When +outline-atomics is
    // enabled without +lse, LLVM emits __aarch64_cas16_* calls that the XLA:CPU
    // JIT does not resolve. With +lse2, 16 byte aligned monotonic 128 bit
    // atomic loads and stores lower directly to single copy atomic ldp/stp.
    const bool has_lse = subtarget.checkFeatures("+lse");
    const bool outline_atomics = subtarget.checkFeatures("+outline-atomics");
    return CpuDeviceSpec{
        /*has_atomic_cmpxchg_128=*/has_lse || !outline_atomics,
        /*has_atomic_load_store_128=*/subtarget.checkFeatures("+lse2"),
    };
  }

  return {};
}

TargetMachineFeatures::TargetMachineFeatures(
    llvm::TargetMachine* target_machine)
    : target_machine_(target_machine) {
  // Calling virtual methods in the constructor is discouraged, so we don't
  // call `get_target_feature_string` here.
  if (target_machine_) {
    const std::string feature_string =
        target_machine_->getTargetFeatureString().str();
    has_avx512bf16_ = absl::StrContains(feature_string, "+avx512bf16");
    has_avx512fp16_ = absl::StrContains(feature_string, "+avx512fp16");
    has_amx_fp16_ = absl::StrContains(feature_string, "+amx-fp16");
    has_amx_fp8_ = absl::StrContains(feature_string, "+amx-fp8");
    cpu_device_spec_ = CpuDeviceSpecFromTargetMachine(*target_machine_);
  }
}

const llvm::TargetMachine* TargetMachineFeatures::target_machine() const {
  return target_machine_;
}

int32_t TargetMachineFeatures::vectorization_factor_in_bytes() const {
  // Ideally this should be a function of the cache line size (which we can
  // get from llvm::TargetTransformInfo::getCacheLineSize) of the target
  // machine.  Guess a value of 128 bytes for now.
  return 128;
}

int32_t TargetMachineFeatures::vector_register_byte_size(
    const llvm::Function& fn) const {
  llvm::TargetTransformInfo* tti = GetTargetTransformInfoFor(fn);
  return tti->getRegisterBitWidth(
             llvm::TargetTransformInfo::RGK_FixedWidthVector) /
         8;
}

int32_t TargetMachineFeatures::vector_register_num_elements(
    const llvm::Function& fn, PrimitiveType type) const {
  return vector_register_byte_size(fn) / (primitive_util::BitWidth(type) / 8);
}

int32_t TargetMachineFeatures::vector_register_count(
    const llvm::Function& fn) const {
  llvm::TargetTransformInfo* tti = GetTargetTransformInfoFor(fn);
  return static_cast<int32_t>(
      tti->getNumberOfRegisters(tti->getRegisterClassForType(/*Vector=*/true)));
}

llvm::TargetTransformInfo* TargetMachineFeatures::GetTargetTransformInfoFor(
    const llvm::Function& fn) const {
  auto it = target_transform_info_.find(&fn);
  if (it == target_transform_info_.end()) {
    auto emplace_result = target_transform_info_.emplace(
        &fn, target_machine_->getTargetTransformInfo(fn));
    CHECK(emplace_result.second);
    it = emplace_result.first;
  }

  return &it->second;
}

int64_t TargetMachineFeatures::minimum_alignment_for_allocation(
    int64_t size_bytes) const {
  // Assume that all pointers are aligned to at least
  // xla::cpu_function_runtime::kMinAlign.
  if (size_bytes == 0) {
    // No need to align empty buffers.
    return 1;
  }

  // Allow small buffers to be underaligned, there is no vectorization benefit
  // anyways.
  return std::min<int64_t>(llvm::PowerOf2Ceil(size_bytes), MinAlign());
}

std::string TargetMachineFeatures::get_target_feature_string() const {
  return target_machine_ == nullptr
             ? ""
             : target_machine_->getTargetFeatureString().str();
}

}  // namespace xla::cpu
