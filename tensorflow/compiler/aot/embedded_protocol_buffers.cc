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
#include <string>

#include "llvm/ADT/Triple.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "tensorflow/compiler/tf2xla/str_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace tensorflow {
namespace tfcompile {

using xla::llvm_ir::AsStringRef;

static std::unique_ptr<llvm::Module> CreateModuleWithEmbeddedProtocolBuffer(
    llvm::LLVMContext* llvm_context, llvm::TargetMachine* target_machine,
    const ::tensorflow::protobuf::MessageLite& proto,
    StringPiece unique_identifier, string* protobuf_array_symbol_name,
    int64* protobuf_array_size) {
  string protobuf_array_contents = proto.SerializeAsString();
  *protobuf_array_symbol_name =
      strings::StrCat(unique_identifier, "_protobuf_array_contents");
  *protobuf_array_size = protobuf_array_contents.size();

  std::unique_ptr<llvm::Module> module =
      MakeUnique<llvm::Module>("embedded_data_module", *llvm_context);

  llvm::Constant* protobuf_array_initializer =
      llvm::ConstantDataArray::getString(*llvm_context,
                                         AsStringRef(protobuf_array_contents),
                                         /*AddNull=*/false);
  new llvm::GlobalVariable(
      *module, protobuf_array_initializer->getType(),
      /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage,
      protobuf_array_initializer, AsStringRef(*protobuf_array_symbol_name));

  return module;
}

static string CreateCPPShimExpression(StringPiece qualified_cpp_protobuf_name,
                                      StringPiece protobuf_array_symbol_name,
                                      int64 protobuf_array_size) {
  string code =
      "[]() {\n"
      "    {{PROTOBUF_NAME}}* proto = new {{PROTOBUF_NAME}};\n"
      "    proto->ParseFromArray(&{{ARRAY_SYMBOL}}[0], {{ARRAY_SIZE}});\n"
      "    return proto;\n"
      "  }()";

  str_util::ReplaceAllPairs(
      &code,
      {
          {"{{ARRAY_SYMBOL}}", strings::StrCat(protobuf_array_symbol_name)},
          {"{{ARRAY_SIZE}}", strings::StrCat(protobuf_array_size)},
          {"{{PROTOBUF_NAME}}", strings::StrCat(qualified_cpp_protobuf_name)},
      });
  return code;
}

static StatusOr<string> CodegenModule(llvm::TargetMachine* target_machine,
                                      std::unique_ptr<llvm::Module> module) {
  llvm::SmallVector<char, 0> stream_buffer;
  llvm::raw_svector_ostream ostream(stream_buffer);
  llvm::legacy::PassManager codegen_passes;

  if (target_machine->addPassesToEmitFile(
          codegen_passes, ostream, llvm::TargetMachine::CGFT_ObjectFile)) {
    return xla::InternalError(
        "Could not create pass pipeline to generate object file");
  }

  codegen_passes.run(*module);

  return string(stream_buffer.begin(), stream_buffer.end());
}

static StatusOr<std::unique_ptr<llvm::TargetMachine>>
GetTargetMachineFromTriple(StringPiece target_triple) {
  std::string error;
  std::string normalized_triple =
      llvm::Triple::normalize(AsStringRef(target_triple));
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(normalized_triple, error);
  if (target == nullptr) {
    return xla::InternalError("TargetRegistry::lookupTarget failed: %s",
                              error.c_str());
  }

  return WrapUnique(target->createTargetMachine(
      normalized_triple, /*CPU=*/"",
      /*Features=*/"", llvm::TargetOptions(), llvm::None));
}

StatusOr<EmbeddedProtocolBuffer> CreateEmbeddedProtocolBuffer(
    StringPiece target_triple, StringPiece symbol_prefix,
    StringPiece qualified_cpp_protobuf_name,
    const ::tensorflow::protobuf::MessageLite* proto) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::TargetMachine> target_machine,
                      GetTargetMachineFromTriple(target_triple));

  llvm::LLVMContext llvm_context;
  string object_file, cpp_shim, cpp_variable_decl;

  if (proto) {
    string protobuf_array_symbol_name;
    int64 protobuf_array_size;

    std::unique_ptr<llvm::Module> module_with_serialized_proto =
        CreateModuleWithEmbeddedProtocolBuffer(
            &llvm_context, target_machine.get(), *proto, symbol_prefix,
            &protobuf_array_symbol_name, &protobuf_array_size);
    TF_ASSIGN_OR_RETURN(object_file,
                        CodegenModule(target_machine.get(),
                                      std::move(module_with_serialized_proto)));
    cpp_shim = CreateCPPShimExpression(qualified_cpp_protobuf_name,
                                       protobuf_array_symbol_name,
                                       protobuf_array_size);

    cpp_variable_decl = strings::StrCat("extern \"C\" char ",
                                        protobuf_array_symbol_name, "[];");
  } else {
    TF_ASSIGN_OR_RETURN(
        object_file,
        CodegenModule(target_machine.get(),
                      MakeUnique<llvm::Module>("empty_module", llvm_context)));
    cpp_shim = "nullptr";
  }

  return {{cpp_shim, cpp_variable_decl, object_file}};
}

}  // namespace tfcompile
}  // namespace tensorflow
