/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/python/slim_model_importer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"

namespace tensorflow {
namespace {

using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::mlir::func::FuncOp;

absl::Status InjectMappedArg(
    FuncOp func, int arg_idx, size_t offset,
    const std::shared_ptr<llvm::MemoryBuffer>& weights_buffer,
    llvm::DenseMap<uint64_t, mlir::Attribute>& cache) {
  if (arg_idx >= (int)func.getNumArguments()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Argument index out of bounds: arg_idx=", arg_idx,
                     ", func.getNumArguments()=", func.getNumArguments(),
                     ", func=", func.getSymName().str()));
  }
  mlir::BlockArgument arg = func.getArgument(arg_idx);
  auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(arg.getType());
  if (!shaped_type) return absl::OkStatus();

  mlir::Attribute attr;
  auto it = cache.find(offset);
  if (it != cache.end()) {
    attr = it->second;
  } else {
    size_t num_elements = shaped_type.getNumElements();
    size_t bit_width = shaped_type.getElementTypeBitWidth();
    size_t num_bytes = num_elements * (bit_width / 8);
    if (offset + num_bytes > weights_buffer->getBufferSize()) {
      return absl::InvalidArgumentError("Weight offset out of bounds.");
    }

    if (num_elements > 64) {
      auto deleter = [weights_buffer](void*, size_t, size_t) {};
      auto blob = mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
          llvm::ArrayRef<char>(weights_buffer->getBufferStart() + offset,
                               num_bytes),
          /*align=*/64, std::move(deleter), /*dataIsMutable=*/false);
      std::string blob_name = absl::StrCat("dense_resource_off_", offset);
      attr = mlir::DenseResourceElementsAttr::get(shaped_type, blob_name,
                                                  std::move(blob));
    } else {
      attr = mlir::DenseElementsAttr::getFromRawBuffer(
          shaped_type,
          llvm::ArrayRef<char>(weights_buffer->getBufferStart() + offset,
                               num_bytes));
    }
    cache[offset] = attr;
  }

  mlir::OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(&func.getBody().front());
  auto constant =
      builder.create<mlir::stablehlo::ConstantOp>(func.getLoc(), attr);
  arg.replaceAllUsesWith(constant);
  return absl::OkStatus();
}

absl::Status InjectWeights(
    ModuleOp module, const std::shared_ptr<llvm::MemoryBuffer>& weights_buffer,
    const llvm::json::Object& metadata) {
  llvm::DenseMap<uint64_t, mlir::Attribute> cache;

  const auto* signatures = metadata.getObject("signatures");
  if (!signatures) {
    return absl::InvalidArgumentError("Metadata JSON missing 'signatures'.");
  }

  for (const auto& sig : *signatures) {
    std::string sig_name = sig.first.str();
    const auto* args = sig.second.getAsArray();
    if (!args) continue;

    auto func = module.lookupSymbol<FuncOp>(sig_name);
    if (!func) {
      LOG(WARNING) << "Function " << sig_name << " not found in module.";
      continue;
    }

    std::vector<int> args_to_erase;
    for (const auto& arg_val : *args) {
      const auto* arg_obj = arg_val.getAsObject();
      if (!arg_obj) continue;

      std::optional<int> arg_index = arg_obj->getInteger("arg_index");
      std::optional<int64_t> offset = arg_obj->getInteger("offset");

      if (arg_index && offset) {
        auto status =
            InjectMappedArg(func, *arg_index, static_cast<size_t>(*offset),
                            weights_buffer, cache);
        if (!status.ok()) return status;
        args_to_erase.push_back(*arg_index);
      }
    }

    std::sort(args_to_erase.rbegin(), args_to_erase.rend());
    for (int idx : args_to_erase) {
      (void)func.eraseArgument(idx);
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<OwningOpRef<ModuleOp>> LoadSlimModel(
    absl::string_view model_dir, mlir::MLIRContext* context) {
  llvm::SmallString<128> metadata_path_buf(
      llvm::StringRef(model_dir.data(), model_dir.size()));
  llvm::sys::path::append(metadata_path_buf, "weights_metadata.json");
  std::string metadata_path = std::string(metadata_path_buf.str());
  auto buffer_or_err = llvm::MemoryBuffer::getFile(metadata_path);
  if (!buffer_or_err) {
    return absl::NotFoundError(
        absl::StrCat("Failed to read metadata file: ", metadata_path));
  }
  std::string json_data = buffer_or_err.get()->getBuffer().str();
  auto json_val = llvm::json::parse(json_data);
  if (!json_val) {
    return absl::InvalidArgumentError("Failed to parse metadata JSON.");
  }
  const auto* metadata_obj = json_val->getAsObject();
  if (!metadata_obj) {
    return absl::InvalidArgumentError("Metadata root must be a JSON object.");
  }

  OwningOpRef<ModuleOp> combined_module =
      ModuleOp::create(mlir::UnknownLoc::get(context));
  mlir::OpBuilder combined_builder(combined_module->getBodyRegion());
  combined_module.get()->setAttr("tf_saved_model.semantics",
                                 mlir::UnitAttr::get(context));

  const auto* signatures = metadata_obj->getObject("signatures");
  if (!signatures) {
    return absl::InvalidArgumentError("Metadata JSON missing 'signatures'.");
  }

  for (const auto& sig : *signatures) {
    std::string sig_name = sig.first.str();
    llvm::SmallString<128> mlirbc_path_buf(
        llvm::StringRef(model_dir.data(), model_dir.size()));
    llvm::sys::path::append(mlirbc_path_buf, absl::StrCat(sig_name, ".mlirbc"));
    std::string mlirbc_path = std::string(mlirbc_path_buf.str());

    mlir::ParserConfig config(context);
    auto module = mlir::parseSourceFile<ModuleOp>(mlirbc_path, config);
    if (!module) {
      return absl::InternalError(absl::StrCat("Failed to parse ", mlirbc_path));
    }

    mlir::PassManager pm(context);
    pm.addPass(mlir::odml::CreateDropShapeAssertionsPass());
    pm.addPass(mlir::stablehlo::createVhloLegalizeToStablehloPass());
    if (mlir::failed(pm.run(*module))) {
      return absl::InternalError("Failed to legalize VHLO to StableHLO.");
    }

    FuncOp main_func = module->lookupSymbol<FuncOp>("main");
    if (!main_func) {
      module->walk([&](FuncOp func) {
        if (func.getSymName() == "main") {
          main_func = func;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
    }

    if (main_func) {
      mlir::SymbolTable sym_table(*module);
      llvm::SmallVector<FuncOp, 4> funcs_to_move;
      for (auto func : module->getOps<FuncOp>()) {
        funcs_to_move.push_back(func);
      }
      for (auto func : funcs_to_move) {
        if (func == main_func) {
          if (func.getSymName() != sig_name) {
            if (mlir::failed(sym_table.rename(func, sig_name))) {
              return absl::InternalError("Failed to rename main function.");
            }
          }
          func->setAttr("sym_visibility",
                        combined_builder.getStringAttr("public"));
        } else {
          std::string new_name =
              absl::StrCat(sig_name, "_", func.getSymName().str());
          if (func.getSymName() != new_name) {
            if (mlir::failed(sym_table.rename(func, new_name))) {
              return absl::InternalError("Failed to rename helper function.");
            }
          }
          func->setAttr("sym_visibility",
                        combined_builder.getStringAttr("private"));
        }
      }
      for (auto func : funcs_to_move) {
        func->remove();
        combined_module->getBody()->push_back(func);
      }
    }
  }

  llvm::SmallString<128> weights_path_buf(
      llvm::StringRef(model_dir.data(), model_dir.size()));
  llvm::sys::path::append(weights_path_buf, "params.bin");
  std::string weights_path = std::string(weights_path_buf.str());
  auto weights_buffer_or_err = llvm::MemoryBuffer::getFile(weights_path);
  if (weights_buffer_or_err) {
    std::shared_ptr<llvm::MemoryBuffer> weights_buffer =
        std::move(weights_buffer_or_err.get());
    auto status =
        InjectWeights(*combined_module, weights_buffer, *metadata_obj);
    if (!status.ok()) return status;
  }

  const auto* sig_inputs = metadata_obj->getObject("signature_inputs");
  const auto* sig_outputs = metadata_obj->getObject("signature_outputs");

  for (auto func : combined_module.get().getOps<FuncOp>()) {
    std::string sig_name = func.getName().str();
    if (!signatures || !signatures->get(sig_name)) continue;
    func->setAttr("tf_saved_model.exported_names",
                  combined_builder.getArrayAttr(
                      {combined_builder.getStringAttr(sig_name)}));

    std::vector<std::string> arg_names;
    arg_names.reserve(func.getNumArguments());
    const llvm::json::Array* custom_arg_names = nullptr;
    if (sig_inputs) custom_arg_names = sig_inputs->getArray(sig_name);

    for (int i = 0; i < func.getNumArguments(); ++i) {
      std::string arg_name;
      if (custom_arg_names && i < custom_arg_names->size()) {
        if (auto str_val = (*custom_arg_names)[i].getAsString()) {
          arg_name = str_val->str();
        }
      }
      if (arg_name.empty()) {
        arg_name = absl::StrCat("args_", i);
      }
      arg_names.push_back(arg_name);
      func.setArgAttr(i, "tf_saved_model.index_path",
                      combined_builder.getArrayAttr(
                          {combined_builder.getStringAttr(arg_name)}));
    }

    std::vector<std::string> res_names;
    res_names.reserve(func.getNumResults());
    const llvm::json::Array* custom_res_names = nullptr;
    if (sig_outputs) custom_res_names = sig_outputs->getArray(sig_name);

    for (int i = 0; i < func.getNumResults(); ++i) {
      std::string res_name;
      if (custom_res_names && i < custom_res_names->size()) {
        if (auto str_val = (*custom_res_names)[i].getAsString()) {
          res_name = str_val->str();
        }
      }
      if (res_name.empty()) {
        res_name = absl::StrCat("output_", i);
      }
      res_names.push_back(res_name);
      func.setResultAttr(i, "tf_saved_model.index_path",
                         combined_builder.getArrayAttr(
                             {combined_builder.getStringAttr(res_name)}));
    }

    std::string inputs_str = absl::StrJoin(arg_names, ",");
    std::string outputs_str = absl::StrJoin(res_names, ",");
    llvm::SmallVector<mlir::NamedAttribute, 2> entry_function_attrs;
    entry_function_attrs.push_back(combined_builder.getNamedAttr(
        "inputs", combined_builder.getStringAttr(inputs_str)));
    entry_function_attrs.push_back(combined_builder.getNamedAttr(
        "outputs", combined_builder.getStringAttr(outputs_str)));
    func->setAttr("tf.entry_function",
                  combined_builder.getDictionaryAttr(entry_function_attrs));
  }

  return combined_module;
}

}  // namespace tensorflow
