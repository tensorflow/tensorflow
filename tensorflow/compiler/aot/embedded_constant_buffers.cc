/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/aot/embedded_constant_buffers.h"

#include <sys/types.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace tensorflow {
namespace tfcompile {

using xla::llvm_ir::AsStringRef;

void ConstantToEmbed::SerializeIntoBuffer(absl::Span<const uint8_t> buffer) {
  // Allocate memory for the size of the buffer and the buffer itself.
  const uint64_t buffer_size = buffer.size();
  data_buffer.resize(sizeof(uint64_t) + buffer_size);
  std::memcpy(data_buffer.data(), &buffer_size, sizeof(uint64_t));
  std::memcpy(data_buffer.data() + sizeof(uint64_t), buffer.data(),
              buffer.size());
}

static absl::Status AddBufferToLlvmModule(
    llvm::Module* module, const ConstantToEmbed& constant_to_embed,
    absl::string_view unique_identifier,
    std::string& constant_array_symbol_name) {
  if (constant_to_embed.data().empty()) {
    return xla::Internal(
        "Constant buffer shouldn't be empty, it should at least contain the "
        "size of the buffer.");
  }

  absl::Span<const uint8_t> buffer_contents = constant_to_embed.data();

  llvm::Constant* buffer_initializer = llvm::ConstantDataVector::get(
      module->getContext(),
      llvm::ArrayRef<uint8_t>(buffer_contents.data(), buffer_contents.size()));

  constant_array_symbol_name =
      absl::StrCat(unique_identifier, "_constant_buffer_contents");
  new llvm::GlobalVariable(
      *module, buffer_initializer->getType(),
      /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage,
      buffer_initializer, AsStringRef(constant_array_symbol_name));

  return absl::OkStatus();
}

static absl::StatusOr<std::string> CodegenModule(
    llvm::TargetMachine* target_machine, std::unique_ptr<llvm::Module> module) {
  llvm::SmallVector<char, 0> stream_buffer;
  llvm::raw_svector_ostream ostream(stream_buffer);
  llvm::legacy::PassManager codegen_passes;

  if (target_machine->addPassesToEmitFile(codegen_passes, ostream, nullptr,
                                          llvm::CodeGenFileType::ObjectFile)) {
    return xla::Internal(
        "Could not create pass pipeline to generate object file");
  }

  codegen_passes.run(*module);

  return std::string(stream_buffer.begin(), stream_buffer.end());
}

static absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
GetTargetMachineFromTriple(absl::string_view target_triple) {
  std::string error;
  std::string normalized_triple =
      llvm::Triple::normalize(AsStringRef(absl::string_view(target_triple)));
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(normalized_triple, error);
  if (target == nullptr) {
    return xla::Internal("TargetRegistry::lookupTarget failed: %s",
                         error.c_str());
  }

  return absl::WrapUnique(target->createTargetMachine(
      normalized_triple, /*CPU=*/"",
      /*Features=*/"", llvm::TargetOptions(), std::nullopt));
}

absl::StatusOr<EmbeddedConstantBuffers> CreateEmbeddedConstantBuffers(
    absl::string_view target_triple,
    absl::Span<ConstantToEmbed> constants_to_embed) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::TargetMachine> target_machine,
                      GetTargetMachineFromTriple(target_triple));

  llvm::LLVMContext llvm_context;
  auto module_with_serialized_proto = std::make_unique<llvm::Module>(
      "embedded_constant_data_module", llvm_context);

  EmbeddedConstantBuffers result;

  for (const ConstantToEmbed& constant_to_embed : constants_to_embed) {
    std::string constant_array_symbol_name;

    TF_RETURN_IF_ERROR(AddBufferToLlvmModule(
        module_with_serialized_proto.get(), constant_to_embed,
        constant_to_embed.symbol_prefix, constant_array_symbol_name));

    std::string cpp_variable_decl =
        absl::StrCat("extern \"C\" char ", constant_array_symbol_name, "[];");

    std::string cpp_access_shim = absl::StrFormat(R"(
    [](char* buffer) -> std::pair<uint64_t, char*> {
      uint64_t buffer_size;
      std::memcpy(&buffer_size, buffer, sizeof(uint64_t));
      return {buffer_size, buffer + sizeof(uint64_t)};
    }(%s)
    )",
                                                  constant_array_symbol_name);
    result.variable_decls.push_back(
        {constant_array_symbol_name, cpp_variable_decl, cpp_access_shim});
  }

  TF_ASSIGN_OR_RETURN(result.object_file_data,
                      CodegenModule(target_machine.get(),
                                    std::move(module_with_serialized_proto)));
  return result;
}

}  // namespace tfcompile
}  // namespace tensorflow
