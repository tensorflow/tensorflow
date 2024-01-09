/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/manipulate_model_attr.h"

#include <string>
#include <utility>

#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project

namespace mlir {
namespace quant {

constexpr StringRef kTfEntryFunctionAttr = "tf.entry_function";

void AddEntryFunctionInput(StringRef input_name, func::FuncOp func_op) {
  auto entry_func_attr =
      func_op->getAttrOfType<DictionaryAttr>(kTfEntryFunctionAttr);
  if (!entry_func_attr) return;

  auto entry_func_attrs = SmallVector<NamedAttribute>(entry_func_attr.begin(),
                                                      entry_func_attr.end());

  MLIRContext* ctx = func_op.getContext();
  for (auto& named_attr : entry_func_attrs) {
    if (named_attr.getName() != "inputs") continue;

    // Splits the "inputs" field to retrieve individual input names. Ignores
    // empty strings.
    SmallVector<StringRef> inputs_attrs{};
    cast<StringAttr>(named_attr.getValue())
        .strref()
        .split(inputs_attrs, /*Separator=*/',', /*MaxSplit=*/-1,
               /*KeepEmpty=*/false);

    inputs_attrs.emplace_back(input_name);

    const std::string new_inputs_attr_str =
        llvm::join(std::move(inputs_attrs), /*Separator=*/",");

    named_attr.setValue(StringAttr::get(ctx, new_inputs_attr_str));
  }

  func_op->setAttr(kTfEntryFunctionAttr,
                   DictionaryAttr::get(ctx, entry_func_attrs));
}
}  // namespace quant
}  // namespace mlir
