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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_KERNEL_H_
#define TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_KERNEL_H_

#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"

namespace mlrt {
namespace bc {

class Kernel {
 public:
  struct StorageType {
    using Self = StorageType;
    DEFINE_BYTECODE_FIELD(uint32_t, code);
    DEFINE_BYTECODE_FIELD(bc::Vector<uint32_t>, arguments);
    DEFINE_BYTECODE_FIELD(bc::Vector<uint32_t>, results);
    DEFINE_BYTECODE_FIELD(bc::Vector<uint32_t>, attributes);
    DEFINE_BYTECODE_FIELD(bc::Vector<uint8_t>, last_uses);
  };

  class Constructor {
   public:
    Constructor(Allocator* allocator, BcAddr_t address)
        : allocator_(allocator), address_(address) {}

    void set_code(uint32_t code) {
      StorageType::construct_code(allocator_, address_, code);
    }

    template <typename... Args>
    auto construct_arguments(Args&&... args) {
      return StorageType::construct_arguments(allocator_, address_,
                                              std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto construct_results(Args&&... args) {
      return StorageType::construct_results(allocator_, address_,
                                            std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto construct_attributes(Args&&... args) {
      return StorageType::construct_attributes(allocator_, address_,
                                               std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto construct_last_uses(Args&&... args) {
      return StorageType::construct_last_uses(allocator_, address_,
                                              std::forward<Args>(args)...);
    }

    BcAddr_t address() const { return address_; }

   private:
    Allocator* allocator_;
    BcAddr_t address_;
  };
  using NonTrivialConstructorType = Constructor;

  explicit Kernel(const char* p) : p_(p) {}
  Kernel() : p_(nullptr) {}

  uint32_t code() const { return StorageType::read_code(p_); }
  Vector<uint32_t> arguments() const { return StorageType::read_arguments(p_); }
  Vector<uint32_t> results() const { return StorageType::read_results(p_); }
  Vector<uint32_t> attributes() const {
    return StorageType::read_attributes(p_);
  }
  Vector<uint8_t> last_uses() const { return StorageType::read_last_uses(p_); }

 private:
  const char* p_;
};

}  // namespace bc
}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_KERNEL_H_
