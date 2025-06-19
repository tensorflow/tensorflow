/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/hlo/hlo_program.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "highwayhash/arch_specific.h"
#include "highwayhash/hh_types.h"
#include "highwayhash/highwayhash.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OperationSupport.h"

namespace xla::ifrt {

char HloProgram::ID = 0;

namespace {

// Calculates a HighwayHash fingerprint in a streaming manner.
class HighwayHashStream final : public llvm::raw_ostream {
 public:
  HighwayHashStream() : hash_(kHighwayHashKey), pos_(0) { SetUnbuffered(); }

  // Destructively calculates the fingerprint of the data consumed so far.
  uint64_t fingerprint() && {
    highwayhash::HHResult64 result;
    hash_.Finalize(&result);
    return result;
  }

 private:
  // Arbitrarily chosen, forever-unchanging hash key required by HighwayHash.
  static constexpr highwayhash::HHKey kHighwayHashKey = {
      0x4451e30f87db9609ULL,
      0xca7358a1fd2737f8ULL,
      0x4b2c991fcee4fdeaULL,
      0x0b2658e18326f6baULL,
  };

  void write_impl(const char* Ptr, size_t Size) final {
    hash_.Append(Ptr, Size);
    pos_ += Size;
  }

  uint64_t current_pos() const final { return pos_; }

  highwayhash::HighwayHashCatT<HH_TARGET> hash_;
  uint64_t pos_;
};

}  // namespace

uint64_t HloProgram::Fingerprint() const {
  HighwayHashStream os;
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(false);
  mlir_module->print(os, flags);
  return std::move(os).fingerprint();
}

}  // namespace xla::ifrt
