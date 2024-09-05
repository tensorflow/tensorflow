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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_ATTRIBUTE_ATTRIBUTE_H_
#define TENSORFLOW_CORE_TFRT_MLRT_ATTRIBUTE_ATTRIBUTE_H_

#include <string>

#include "absl/status/statusor.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/translate/mlrt/mlir_to_bytecode.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"

namespace tensorflow {
namespace tf_mlrt {

class ShapeAttr {
 public:
  struct StorageType {
    using Self = StorageType;
    DEFINE_BYTECODE_FIELD(uint8_t, unranked);
    DEFINE_BYTECODE_FIELD(mlrt::bc::Vector<int64_t>, dims);
  };

  class Constructor {
   public:
    Constructor(mlrt::bc::Allocator* allocator, mlrt::bc::BcAddr_t address)
        : allocator_(allocator), address_(address) {}

    void set_unranked(bool unranked) {
      StorageType::construct_unranked(allocator_, address_, unranked);
    }

    template <typename... Args>
    auto construct_shape(Args&&... args) {
      return StorageType::construct_dims(allocator_, address_,
                                         std::forward<Args>(args)...);
    }

    mlrt::bc::BcAddr_t address() const { return address_; }

   private:
    mlrt::bc::Allocator* allocator_;
    mlrt::bc::BcAddr_t address_;
  };
  using NonTrivialConstructorType = Constructor;

  explicit ShapeAttr(const char* p) : p_(p) {}

  bool unranked() const { return StorageType::read_unranked(p_); }
  mlrt::bc::Vector<int64_t> dims() const { return StorageType::read_dims(p_); }

 private:
  const char* p_ = nullptr;
};

class TensorAttr {
 public:
  struct StorageType {
    using Self = StorageType;
    DEFINE_BYTECODE_FIELD(tensorflow::DataType, dtype);
    DEFINE_BYTECODE_FIELD(uint64_t, num_elements);
    DEFINE_BYTECODE_FIELD(mlrt::bc::Vector<int64_t>, shape);
    DEFINE_BYTECODE_FIELD(mlrt::bc::Vector<char>, data);
  };

  class Constructor {
   public:
    Constructor(mlrt::bc::Allocator* allocator, mlrt::bc::BcAddr_t address,
                tensorflow::DataType dtype)
        : allocator_(allocator), address_(address) {
      StorageType::construct_dtype(allocator_, address_, dtype);
    }

    void set_num_elements(size_t num) {
      StorageType::construct_num_elements(allocator_, address_, num);
    }

    template <typename... Args>
    auto construct_shape(Args&&... args) {
      return StorageType::construct_shape(allocator_, address_,
                                          std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto construct_data(Args&&... args) {
      return StorageType::construct_data(allocator_, address_,
                                         std::forward<Args>(args)...);
    }

    mlrt::bc::BcAddr_t address() const { return address_; }

   private:
    mlrt::bc::Allocator* allocator_;
    mlrt::bc::BcAddr_t address_;
  };
  using NonTrivialConstructorType = Constructor;

  explicit TensorAttr(const char* p) : p_(p) {}

  tensorflow::DataType dtype() const { return StorageType::read_dtype(p_); }
  mlrt::bc::Vector<int64_t> shape() const {
    return StorageType::read_shape(p_);
  }
  mlrt::bc::Vector<char> data() const { return StorageType::read_data(p_); }

 private:
  const char* p_ = nullptr;
};

absl::StatusOr<std::string> EncodeTensorflowAttribute(
    const mlrt::ModuleEmitterContext& module_context, mlir::Attribute attr);

}  // namespace tf_mlrt
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_MLRT_ATTRIBUTE_ATTRIBUTE_H_
