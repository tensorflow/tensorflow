/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_FUNCTION_H_
#define TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_FUNCTION_H_

#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/kernel.h"

namespace mlrt {
namespace bc {

class Function {
 public:
  struct StorageType {
    using Self = StorageType;
    DEFINE_BYTECODE_FIELD(String, name);
    DEFINE_BYTECODE_FIELD(uint32_t, num_regs);
    DEFINE_BYTECODE_FIELD(Vector<uint32_t>, input_regs);
    DEFINE_BYTECODE_FIELD(Vector<uint32_t>, output_regs);
    DEFINE_BYTECODE_FIELD(Vector<uint8_t>, output_last_uses);
    DEFINE_BYTECODE_FIELD(Vector<Kernel>, kernels);
  };

  class Constructor {
   public:
    Constructor(Allocator* allocator, BcAddr_t address)
        : allocator_(allocator), address_(address) {}

    template <typename... Args>
    auto construct_name(Args&&... args) {
      return StorageType::construct_name(allocator_, address_,
                                         std::forward<Args>(args)...);
    }

    void set_num_regs(uint32_t num_regs) {
      StorageType::construct_num_regs(allocator_, address_, num_regs);
    }

    template <typename... Args>
    auto construct_input_regs(Args&&... args) {
      return StorageType::construct_input_regs(allocator_, address_,
                                               std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto construct_output_regs(Args&&... args) {
      return StorageType::construct_output_regs(allocator_, address_,
                                                std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto construct_output_last_uses(Args&&... args) {
      return StorageType::construct_output_last_uses(
          allocator_, address_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto construct_kernels(Args&&... args) {
      return StorageType::construct_kernels(allocator_, address_,
                                            std::forward<Args>(args)...);
    }

    BcAddr_t address() const { return address_; }

   private:
    Allocator* allocator_;
    BcAddr_t address_;
  };
  using NonTrivialConstructorType = Constructor;

  Function() = default;
  // NOLINTNEXTLINE(google-explicit-constructor)
  Function(std::nullptr_t) : p_(nullptr) {}
  explicit Function(const char* p) : p_(p) {}

  String name() const { return StorageType::read_name(p_); }
  uint32_t num_regs() const { return StorageType::read_num_regs(p_); }
  Vector<uint32_t> input_regs() const {
    return StorageType::read_input_regs(p_);
  }
  Vector<uint32_t> output_regs() const {
    return StorageType::read_output_regs(p_);
  }
  Vector<uint8_t> output_last_uses() const {
    return StorageType::read_output_last_uses(p_);
  }
  Vector<Kernel> kernels() const { return StorageType::read_kernels(p_); }

  explicit operator bool() const { return p_ != nullptr; }

 private:
  const char* p_ = nullptr;
};

}  // namespace bc
}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_FUNCTION_H_
