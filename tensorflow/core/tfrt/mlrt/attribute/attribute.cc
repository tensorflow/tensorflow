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
#include "tensorflow/core/tfrt/mlrt/attribute/attribute.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tfrt/translate/mlrt/mlir_to_bytecode.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"

namespace tensorflow {
namespace tf_mlrt {

absl::StatusOr<std::string> EncodeTensorflowAttribute(
    const mlrt::ModuleEmitterContext& module_context, mlir::Attribute attr) {
  if (auto result = mlrt::EncodeSimpleAttribute(module_context, attr)) {
    return std::move(*result);
  }

  if (auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    auto element_type = dense_attr.getElementType();

    tensorflow::DataType dtype;
    TF_RETURN_IF_ERROR(tensorflow::ConvertToDataType(element_type, &dtype));

    if (dtype == tensorflow::DT_STRING) {
      return absl::InvalidArgumentError(
          "String tensor attribute is not yet supported");
    }

    mlrt::bc::Buffer buffer;
    mlrt::bc::Allocator allocator(&buffer);
    auto tensor_ctor = mlrt::bc::New<TensorAttr>(&allocator, dtype);

    auto shaped_type = dense_attr.getType();
    size_t num_elements = shaped_type.getNumElements();

    tensor_ctor.set_num_elements(num_elements);

    std::vector<int64_t> shape(shaped_type.getShape().begin(),
                               shaped_type.getShape().end());
    tensor_ctor.construct_shape(shape);

    if (dtype == tensorflow::DT_BOOL) {
      // bool values has special encoding in MLIR. It occupies one bit in MLIR
      // but in bytecode it is one byte.
      std::vector<uint8_t> data(num_elements);
      int i = 0;
      for (auto v : dense_attr.getValues<bool>()) {
        data[i++] = static_cast<uint8_t>(v);
      }
      tensor_ctor.construct_data(data.size())
          .Place(reinterpret_cast<const char*>(data.data()), data.size());
    } else {
      auto raw_data = dense_attr.getRawData();
      if (dense_attr.isSplat()) {
        std::vector<char> data(raw_data.size() * num_elements);
        char* p = data.data();
        for (int i = 0; i < num_elements; ++i, p += raw_data.size()) {
          std::memcpy(p, raw_data.data(), raw_data.size());
        }
        tensor_ctor.construct_data(data.size()).Place(data.data(), data.size());
      } else {
        tensor_ctor.construct_data(raw_data.size())
            .Place(raw_data.data(), raw_data.size());
      }
    }

    return std::string(buffer.data(), buffer.size());
  }

  // Handle dtype attrs
  if (auto type_attr = mlir::dyn_cast<mlir::TypeAttr>(attr)) {
    tensorflow::DataType dtype;
    TF_RETURN_IF_ERROR(
        tensorflow::ConvertToDataType(type_attr.getValue(), &dtype));
    std::string data(sizeof(dtype), '\0');
    std::memcpy(data.data(), &dtype, sizeof(dtype));
    return data;
  }

  // Handle shape attrs
  if (auto shape_attr = mlir::dyn_cast<mlir::TF::ShapeAttr>(attr)) {
    llvm::ArrayRef<int64_t> shape;
    if (!shape_attr.getUnranked()) {
      auto shape_or = shape_attr.getValue();
      if (!shape_or.has_value()) {
        std::string attr_str;
        llvm::raw_string_ostream os(attr_str);
        attr.print(os);

        return absl::InvalidArgumentError(
            absl::StrCat("Failed to get shape from shape attr: ", attr_str));
      }
      shape = *shape_or;
    }

    mlrt::bc::Buffer buffer;
    mlrt::bc::Allocator allocator(&buffer);
    auto shape_attr_ctor = mlrt::bc::New<ShapeAttr>(&allocator);
    shape_attr_ctor.set_unranked(shape_attr.getUnranked());

    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    shape_attr_ctor.construct_shape(shape_vec);
    return std::string(buffer.data(), buffer.size());
  }

  // Handle attribute arrays.
  if (auto array_attr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    mlrt::bc::Buffer buffer;
    mlrt::bc::Allocator allocator(&buffer);
    auto ctor = mlrt::bc::New<mlrt::bc::Vector<tensorflow::DataType>>(
        &allocator, array_attr.size());

    int i;
    for (i = 0; i < array_attr.size(); ++i) {
      if (auto type_attr = mlir::dyn_cast<mlir::TypeAttr>(array_attr[i])) {
        tensorflow::DataType dtype;
        TF_RETURN_IF_ERROR(
            tensorflow::ConvertToDataType(type_attr.getValue(), &dtype));
        ctor.ConstructAt(i, dtype);
      } else {
        break;
      }
    }

    if (i == array_attr.size()) {
      return std::string(buffer.data(), buffer.size());
    }
  }

  std::string attr_str;
  llvm::raw_string_ostream os(attr_str);
  attr.print(os);

  return absl::InvalidArgumentError(
      absl::StrCat("Try to encode unsupported attribute: ", attr_str));
}

}  // namespace tf_mlrt
}  // namespace tensorflow
