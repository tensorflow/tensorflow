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

// This program compiles an XLA program which computes 123 and writes the
// resulting object file to stdout.

#include <iostream>
#include <vector>

#include "llvm/ADT/Triple.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

using xla::string;

xla::Computation Doubler(xla::Client* client) {
  xla::ComputationBuilder builder(client, "doubler");
  auto r0f32 = xla::ShapeUtil::MakeShape(xla::F32, {});
  auto x = builder.Parameter(0, r0f32, "x");
  builder.Mul(x, builder.ConstantR0<float>(2.0));
  return std::move(builder.Build().ValueOrDie());
}

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  auto client = xla::ClientLibrary::GetOrCreateCompileOnlyClient().ValueOrDie();

  xla::ComputationBuilder builder(client, "aot_test_helper");
  auto opaque_shape = xla::ShapeUtil::MakeOpaqueShape();
  auto opaque_param = builder.Parameter(0, opaque_shape, "x");
  auto r0f32 = xla::ShapeUtil::MakeShape(xla::F32, {});
  auto sum = builder.CustomCall("SumStructElements", {opaque_param}, r0f32);
  builder.Call(Doubler(client), {sum});

  if (argc != 2) {
    LOG(FATAL) << "local_client_aot_test_helper TARGET_CPU";
  }

  string triple_string;
  string target_cpu = argv[1];
  if (target_cpu == "k8") {
    triple_string = "x86_64-none-linux-gnu";
  } else if (target_cpu == "darwin") {
    triple_string = "x86_64-apple-macosx";
  } else if (target_cpu == "arm") {
    triple_string = "aarch64-none-linux-gnu";
  } else if (target_cpu == "ppc") {
    triple_string = "powerpc64le-unknown-linux-gnu";
  } else if (target_cpu == "local") {
    triple_string = xla::llvm_ir::AsString(llvm::sys::getDefaultTargetTriple());
  } else {
    LOG(FATAL) << "unsupported TARGET_CPU: " << target_cpu;
  }

  llvm::Triple triple(xla::llvm_ir::AsStringRef(triple_string));

  xla::Computation computation = builder.Build().ConsumeValueOrDie();
  xla::CompileOnlyClient::AotComputationInstance instance{
      &computation, /*argument_layouts=*/{&opaque_shape}, &r0f32};

  xla::cpu::CpuAotCompilationOptions options(
      triple_string,
      /*cpu_name=*/"", /*features=*/"", "SumAndDouble",
      xla::cpu::CpuAotCompilationOptions::RelocationModel::Static);

  auto results =
      client->CompileAheadOfTime({instance}, options).ConsumeValueOrDie();
  auto result = xla::unique_ptr_static_cast<xla::cpu::CpuAotCompilationResult>(
      std::move(results.front()));
  // It's lame to hard-code the buffer assignments, but we need
  // local_client_aot_test.cc to be able to easily invoke the function.
  CHECK_EQ(result->result_buffer_index(), 1);
  CHECK_EQ(result->buffer_sizes().size(), 3);
  CHECK_EQ(result->buffer_sizes()[0], -1);             // param buffer
  CHECK_EQ(result->buffer_sizes()[1], sizeof(float));  // result buffer
  CHECK_EQ(result->buffer_sizes()[2], sizeof(float));  // temp buffer
  if (triple.isOSBinFormatELF()) {
    // Check the ELF magic.
    CHECK_EQ(result->object_file_data()[0], 0x7F);
    CHECK_EQ(result->object_file_data()[1], 'E');
    CHECK_EQ(result->object_file_data()[2], 'L');
    CHECK_EQ(result->object_file_data()[3], 'F');
    // Check the ELF class.
    CHECK_EQ(result->object_file_data()[4], triple.isArch32Bit() ? 1 : 2);
    // Check the ELF endianness: it should be little.
    CHECK_EQ(result->object_file_data()[5], triple.isLittleEndian() ? 1 : 2);
    // Check the ELF version: it should be 1.
    CHECK_EQ(result->object_file_data()[6], 1);
  }

  const std::vector<char>& object_file_data = result->object_file_data();
  std::cout.write(object_file_data.data(), object_file_data.size());

  return 0;
}
