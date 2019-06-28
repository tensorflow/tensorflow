/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

namespace {
using stream_executor::port::StatusOr;

// Sets type list attribute with the given `name` to the given `types`. If the
// attribute already exists with a different value, returns an error.
template <typename ContainerT,
          typename = typename std::enable_if<
              std::is_same<mlir::Type, decltype(*std::declval<ContainerT>()
                                                     .begin())>::value>::type>
Status SetAttribute(absl::string_view name, ContainerT types,
                    AttrValueMap* values) {
  AttrValue value;
  auto& type_list = *value.mutable_list();
  for (auto type : types) {
    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, &dtype));
    type_list.add_type(dtype);
  }

  auto result = values->insert({string(name), value});
  if (!result.second) {
    const auto& prev_dtypes = result.first->second.list();
    int count = prev_dtypes.type_size();
    if (count != type_list.type_size()) {
      return errors::InvalidArgument("Type list count mismatch");
    }

    for (int i = 0; i < count; ++i) {
      if (prev_dtypes.type(i) != type_list.type(i))
        return errors::InvalidArgument("Type list mismatch");
    }
  }

  return Status::OK();
}

// Include the auto generated derived attribute populator function taking
// TensorFlow dialect operation as an argument. This file contains the function
// definitions and isn't a header file.
#include "tensorflow/compiler/mlir/tensorflow/translate/derived_attr_populator.inc"

static StatusOr<string> getTensorFlowOpName(llvm::StringRef op_name) {
  if (!op_name.consume_front("tf.")) {
    return errors::FailedPrecondition("op name not prefixed with 'tf.': " +
                                      op_name.str());
  }
  return op_name.str();
}

}  // namespace

StatusOr<std::unique_ptr<NodeDef>> ConvertTFDialectOpToNodeDef(
    mlir::Operation* inst, llvm::StringRef name) {
  TF_ASSIGN_OR_RETURN(auto node_def,
                      GetOperationNodeDef(inst, name, getTensorFlowOpName));

  // Use auto generated function to populate derived attribute.
  //
  // Note: This only populates derived attributes for TensorFlow ops that are
  // generated using the TableGen. Manually defined ops and TF ops with control
  // edges (i.e TF op names with leading '_' in names) should have all the
  // attributes present as native MLIR op attributes.
  //
  // TODO(hinsu): Handle TF ops with control edges that has auto-generated
  // TensorFlow ops available. Such ops can drop all the derived attributes
  // during import to MLIR and rely on exporter to reintroduce them.
  auto status = PopulateDerivedAttrs(inst, node_def->mutable_attr());
  if (!status.ok()) {
    return errors::Internal("Falied to populate derived attrs: " +
                            status.ToString());
  }
  return node_def;
}

}  // namespace tensorflow
