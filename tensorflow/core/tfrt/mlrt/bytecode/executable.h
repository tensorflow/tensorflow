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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_EXECUTABLE_H_
#define TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_EXECUTABLE_H_

#include "tensorflow/core/tfrt/mlrt/bytecode/function.h"

namespace mlrt {
namespace bc {

// Defines the bytecode format for the executable, which contains the following
// section:
//  1) kernel_names: an ordered list of strings for kernel names that appear in
//  this file. The `code` fields of kernels in `functions` will be indices to
//  this list.
//
//  2) attributes: an ordered list of strings that are raw bytes. It is kernel
//  implementations' resposiblity to decode the bytes properly. The `attributes`
//  field of kernels in `functions` will be indices to this list.
//
//  3) functions: an order list of functions, which contains kernels and other
//  metadata. Please refer to function.h for its detailed format.
class Executable {
 public:
  struct StorageType {
    using Self = StorageType;
    DEFINE_BYTECODE_FIELD(Vector<String>, kernel_names);
    DEFINE_BYTECODE_FIELD(Vector<Function>, functions);
    DEFINE_BYTECODE_FIELD(Vector<String>, attributes);
  };

  class Constructor {
   public:
    Constructor(Allocator* allocator, BcAddr_t address)
        : allocator_(allocator), address_(address) {}

    template <typename... Args>
    auto construct_kernel_names(Args&&... args) {
      return StorageType::construct_kernel_names(allocator_, address_,
                                                 std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto construct_attributes(Args&&... args) {
      return StorageType::construct_attributes(allocator_, address_,
                                               std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto construct_functions(Args&&... args) {
      return StorageType::construct_functions(allocator_, address_,
                                              std::forward<Args>(args)...);
    }

    BcAddr_t address() const { return address_; }

   private:
    Allocator* allocator_;
    BcAddr_t address_;
  };
  using NonTrivialConstructorType = Constructor;

  explicit Executable(const char* p) : p_(p) {}

  Vector<String> kernel_names() const {
    return StorageType::read_kernel_names(p_);
  }
  Vector<Function> functions() const { return StorageType::read_functions(p_); }
  Vector<String> attributes() const { return StorageType::read_attributes(p_); }

 private:
  const char* p_;
};

}  // namespace bc
}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_EXECUTABLE_H_
