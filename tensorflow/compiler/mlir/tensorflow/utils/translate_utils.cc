/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"

#include "mlir/IR/OpDefinition.h"  // TF:llvm-project
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

void PopulateTfVersions(mlir::ModuleOp module, const VersionDef& versions) {
  mlir::Builder b(module.getContext());
  auto producer =
      b.getNamedAttr("producer", b.getI32IntegerAttr(versions.producer()));
  auto min_consumer = b.getNamedAttr(
      "min_consumer", b.getI32IntegerAttr(versions.min_consumer()));
  auto bad_consumers = b.getNamedAttr(
      "bad_consumers",
      b.getI32ArrayAttr(llvm::ArrayRef<int32_t>(
          versions.bad_consumers().begin(), versions.bad_consumers().end())));
  module.setAttr("tf.versions",
                 b.getDictionaryAttr(llvm::ArrayRef<mlir::NamedAttribute>(
                     {producer, min_consumer, bad_consumers})));
}

mlir::LogicalResult ExtractTfVersions(mlir::ModuleOp module,
                                      VersionDef* versions) {
  versions->Clear();
  auto version_attr = module.getAttrOfType<mlir::DictionaryAttr>("tf.versions");
  if (!version_attr) return mlir::failure();

  auto producer =
      version_attr.get("producer").dyn_cast_or_null<mlir::IntegerAttr>();
  if (!producer) return mlir::failure();
  versions->set_producer(producer.getInt());

  auto min_consumer =
      version_attr.get("min_consumer").dyn_cast_or_null<mlir::IntegerAttr>();
  if (min_consumer) versions->set_min_consumer(min_consumer.getInt());

  auto bad_consumers =
      version_attr.get("bad_consumers").dyn_cast_or_null<mlir::ArrayAttr>();
  if (!bad_consumers) return mlir::success();

  for (auto bad_consumer : bad_consumers) {
    auto bad_consumer_int_attr =
        bad_consumer.dyn_cast_or_null<mlir::IntegerAttr>();
    if (!bad_consumer_int_attr) return mlir::failure();

    versions->mutable_bad_consumers()->Add(bad_consumer_int_attr.getInt());
  }
  return mlir::success();
}

::stream_executor::port::StatusOr<int64_t> GetTfGraphProducerVersion(
    mlir::ModuleOp module) {
  auto versions = module.getAttrOfType<::mlir::DictionaryAttr>("tf.versions");
  if (!versions) {
    return errors::Internal(
        "Missing 'tf.versions' attribute on the module, abort.\n");
  }
  auto producer = versions.get("producer").dyn_cast<mlir::IntegerAttr>();
  if (!producer) {
    return errors::Internal(
        "Missing 'producer' attribute on the module, abort.\n");
  }
  return producer.getInt();
}

}  // namespace tensorflow
