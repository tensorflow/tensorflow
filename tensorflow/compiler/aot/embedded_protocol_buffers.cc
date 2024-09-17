/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/aot/embedded_protocol_buffers.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_replace.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tfcompile {

using xla::llvm_ir::AsStringRef;

static void AddEmbeddedProtocolBufferToLlvmModule(
    llvm::Module* module, const ::tensorflow::protobuf::MessageLite& proto,
    absl::string_view unique_identifier, string* protobuf_array_symbol_name,
    int64_t* protobuf_array_size) {
  string protobuf_array_contents = proto.SerializeAsString();
  *protobuf_array_symbol_name =
      absl::StrCat(unique_identifier, "_protobuf_array_contents");
  *protobuf_array_size = protobuf_array_contents.size();

  llvm::Constant* protobuf_array_initializer =
      llvm::ConstantDataArray::getString(module->getContext(),
                                         AsStringRef(protobuf_array_contents),
                                         /*AddNull=*/false);
  new llvm::GlobalVariable(
      *module, protobuf_array_initializer->getType(),
      /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage,
      protobuf_array_initializer, AsStringRef(*protobuf_array_symbol_name));
}

static string CreateCPPShimExpression(
    absl::string_view qualified_cpp_protobuf_name,
    absl::string_view protobuf_array_symbol_name, int64_t protobuf_array_size) {
  string code =
      "[]() {\n"
      "    {{PROTOBUF_NAME}}* proto = new {{PROTOBUF_NAME}};\n"
      "    proto->ParseFromArray(&{{ARRAY_SYMBOL}}[0], {{ARRAY_SIZE}});\n"
      "    return proto;\n"
      "  }()";

  return absl::StrReplaceAll(
      code,
      {
          {"{{ARRAY_SYMBOL}}", absl::StrCat(protobuf_array_symbol_name)},
          {"{{ARRAY_SIZE}}", absl::StrCat(protobuf_array_size)},
          {"{{PROTOBUF_NAME}}", absl::StrCat(qualified_cpp_protobuf_name)},
      });
}

static absl::StatusOr<string> CodegenModule(
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

  return string(stream_buffer.begin(), stream_buffer.end());
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

absl::StatusOr<EmbeddedProtocolBuffers> CreateEmbeddedProtocolBuffers(
    absl::string_view target_triple,
    absl::Span<const ProtobufToEmbed> protobufs_to_embed) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::TargetMachine> target_machine,
                      GetTargetMachineFromTriple(target_triple));

  llvm::LLVMContext llvm_context;
  auto module_with_serialized_proto =
      std::make_unique<llvm::Module>("embedded_data_module", llvm_context);

  EmbeddedProtocolBuffers result;

  for (const ProtobufToEmbed& protobuf_to_embed : protobufs_to_embed) {
    string cpp_shim, cpp_variable_decl;
    if (protobuf_to_embed.message) {
      string protobuf_array_symbol_name;
      int64_t protobuf_array_size;

      AddEmbeddedProtocolBufferToLlvmModule(
          module_with_serialized_proto.get(), *protobuf_to_embed.message,
          protobuf_to_embed.symbol_prefix, &protobuf_array_symbol_name,
          &protobuf_array_size);
      cpp_shim = CreateCPPShimExpression(
          protobuf_to_embed.qualified_cpp_protobuf_name,
          protobuf_array_symbol_name, protobuf_array_size);

      cpp_variable_decl =
          absl::StrCat("extern \"C\" char ", protobuf_array_symbol_name, "[];");
    } else {
      cpp_shim = "nullptr";
    }
    result.cpp_shims.push_back({cpp_shim, cpp_variable_decl});
  }

  TF_ASSIGN_OR_RETURN(result.object_file_data,
                      CodegenModule(target_machine.get(),
                                    std::move(module_with_serialized_proto)));
  return result;
}

}  // namespace tfcompile
}  // namespace tensorflow
