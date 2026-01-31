/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/tf_graph_to_hlo_compiler.h"

#include <cstdlib>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/tf_graph_to_hlo_compiler.pb.h"
#include "tensorflow/compiler/tf2xla/xla_argument.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/env.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"
#include "tsl/platform/path.h"

namespace tensorflow {

TfGraphToHloCompiler::TfGraphToHloCompiler(const XlaCompiler::Options& options)
    : xla_compiler_(options) {
  absl::string_view dump_base_dir =
      absl::NullSafeStringView(getenv("TF_GRAPH_TO_HLO_COMPILER_DUMP_DIR"));
  if (dump_base_dir.empty()) {
    return;
  }

  TfGraphToHloCompilerOptions options_proto;
  options_proto.set_device_type(options.device_type.type_string());
  options_proto.set_graph_def_version(options.graph_def_version);
  *options_proto.mutable_flib_def() = options.flib_def->ToProto();

  std::string dump_dir = tsl::io::JoinPath(
      dump_base_dir,
      absl::StrCat(tsl::DeterministicProtoHash64(options_proto)));
  if (!Env::Default()->RecursivelyCreateDir(dump_dir).ok()) {
    LOG(WARNING) << "[TfGraphToHloCompiler] Failed to create dump directory: "
                 << dump_dir;
    return;
  }
  dump_dir_ = dump_dir;

  std::string options_path = tsl::io::JoinPath(dump_dir, "options.pb");
  if (!tsl::WriteBinaryProto(Env::Default(), options_path, options_proto)
           .ok()) {
    LOG(WARNING) << "[TfGraphToHloCompiler] Failed to dump XLA compiler "
                    "options proto to "
                 << options_path;
  }
}

absl::Status TfGraphToHloCompiler::Compile(
    const XlaCompiler::CompileOptions& options, const NameAttrList& function,
    absl::Span<const XlaArgument> args, XlaCompilationResult* result) {
  if (!dump_dir_.empty()) {
    TfGraphToHloCompilerCompileCallArgs call_args;
    call_args.set_compile_options(options.DebugString());
    *call_args.mutable_function() = function;
    for (const XlaArgument& arg : args) {
      call_args.add_xla_arguments(arg.HumanString());
    }

    std::string dump_path = tsl::io::JoinPath(
        dump_dir_,
        absl::StrCat(tsl::DeterministicProtoHash64(call_args), ".pb"));
    if (!tsl::WriteBinaryProto(Env::Default(), dump_path, call_args).ok()) {
      LOG(WARNING) << "[TfGraphToHloCompiler] Failed to dump "
                      "TfGraphToHloCompilerCompileCallArgs proto to "
                   << dump_path;
    }
  }

  return ADD_SOURCE_LOCATION(
      xla_compiler_.CompileFunction(options, function, args, result));
}

absl::Status TfGraphToHloCompiler::CompileSingleOp(
    const XlaCompiler::CompileOptions& options, const OpKernelContext* ctx,
    absl::Span<const XlaArgument> args, XlaCompilationResult* result) {
  return ADD_SOURCE_LOCATION(xla_compiler_.CompileSingleOp(
      options, XlaCompiler::SingleOpCompileArgument(*ctx), args, result));
}

}  // namespace tensorflow
