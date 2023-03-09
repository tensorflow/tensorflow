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

#include "tensorflow/compiler/xla/runtime/runner/runner.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/compiler/xla/runtime/results.h"
#include "tensorflow/compiler/xla/runtime/runner/runner.pb.h"
#include "tensorflow/compiler/xla/runtime/types.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/init_main.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/protobuf.h"

namespace xla {
namespace runtime {

using absl::InternalError;
using absl::InvalidArgumentError;
using absl::StrFormat;

using tsl::Env;
using tsl::ReadBinaryProto;
using tsl::ReadFileToString;
using tsl::ReadTextProto;
using tsl::WriteBinaryProto;
using tsl::WriteTextProto;

using RunnerArgs = Arguments<ScalarArg, MemrefDesc>;

void AppendRunnerFlags(std::vector<tsl::Flag>* flag_list, RunnerFlags* flags) {
  flag_list->emplace_back("function", &flags->function, "Test function name.");

  flag_list->emplace_back("module", &flags->module_path, "Path to MLIR input.");

  flag_list->emplace_back(
      "arguments", &flags->arguments_path,
      "Path to arguments file. If the file ends in '.pbtxt' it is expected to "
      "be in the human-readable proto text format, otherwise it is expected "
      "to be in the proto binary format.");

  flag_list->emplace_back(
      "results", &flags->results_path,
      "Path to results file. The runner tool will serialize results into a "
      " proto message and write it to this file path.");
}
//===----------------------------------------------------------------------===//

AsyncTaskRunner* NoAsyncTaskRunner() {
  return reinterpret_cast<AsyncTaskRunner*>(0xDEADBEEF);
}

//===----------------------------------------------------------------------===//
// Helper functions to Read/Write protobuf messages.
//===----------------------------------------------------------------------===//

template <typename T>
static absl::Status ReadProtoFile(Env* env, const std::string& fname,
                                  T* proto) {
  if (absl::EndsWith(fname, ".pbtxt")) {
    return ToAbslStatus(ReadTextProto(env, fname, proto));
  } else {
    return ToAbslStatus(ReadBinaryProto(env, fname, proto));
  }
}

template <typename T>
static absl::Status WriteProtoFile(Env* env, const std::string& fname,
                                   T& proto) {
  if (absl::EndsWith(fname, ".pbtxt")) {
    return ToAbslStatus(WriteTextProto(env, fname, proto));
  } else {
    return ToAbslStatus(WriteBinaryProto(env, fname, proto));
  }
}

//===----------------------------------------------------------------------===//
// Convert ArgumentsProto message to Xla runtime arguments.
//===----------------------------------------------------------------------===//

static absl::Status ConvertScalar(const ScalarProto& scalar, RunnerArgs& args) {
  switch (scalar.value_case()) {
    case ScalarProto::ValueCase::kI32:
      args.emplace_back<ScalarArg>(scalar.i32());
      break;
    case ScalarProto::ValueCase::kI64:
      args.emplace_back<ScalarArg>(scalar.i64());
      break;
    default:
      return InvalidArgumentError(
          StrFormat("unsupported scalar argument: %s", scalar.DebugString()));
  }
  return absl::OkStatus();
}

static absl::Status ConvertTensor(const TensorProto& tensor, RunnerArgs& args) {
  args.emplace_back<MemrefDesc>(
      tensor.dtype(),
      static_cast<void*>(const_cast<std::string*>(&tensor.contents())),
      /*offset=*/0, tensor.sizes(), tensor.strides());
  return absl::OkStatus();
}

// Converts arguments protobuf message into Xla runtime arguments.
static absl::Status ConvertArgs(ArgumentsProto& proto, RunnerArgs& args) {
  for (auto& arg : proto.arguments()) {
    switch (arg.argument_case()) {
      // Convert `ScalarProto` -> `ScalarArg`.
      case ArgumentProto::ArgumentCase::kScalar:
        if (auto st = ConvertScalar(arg.scalar(), args); !st.ok()) return st;
        break;
      // Convert `TensorProto` -> `MemrefDesc`.
      case ArgumentProto::ArgumentCase::kTensor:
        if (auto st = ConvertTensor(arg.tensor(), args); !st.ok()) return st;
        break;
      // Unsupported argument type.
      default:
        return InvalidArgumentError(
            StrFormat("unsupported argument: %s", arg.DebugString()));
    }
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Convert returned results to ResultsProto message.
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Implement error propagation through the results proto.
static void CheckNoError(const absl::Status& status) {
  LOG(FATAL) << "Unexpected call to `ReturnError`";
}

// Converts results returned from compiled Xla executable to results proto.
struct ReturnResults {
  LogicalResult operator()(unsigned result_index, const Type* type,
                           const Type* runtime_type, void* ret) const {
    // We rely on the fact that result converter handles results from left to
    // right and we can push new results to the back of the list.
    auto* result = proto->add_results();

    // Return scalar result as `ScalarProto`.
    auto* scalar = llvm::dyn_cast<ScalarType>(type);
    switch (scalar ? scalar->type() : PrimitiveType::PRIMITIVE_TYPE_INVALID) {
      case PrimitiveType::S32:
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(ret, sizeof(int32_t));
        result->mutable_scalar()->set_i32(*reinterpret_cast<int32_t*>(ret));
        return success();
      default:
        break;
    }

    // Assuming result cannot be processed as Scalar, try `TensorProto`
    auto* memref = llvm::dyn_cast<MemrefType>(runtime_type);
    if (memref) {
      auto desc = ConvertReturnedMemref<MemrefDesc>(*this, memref, ret);
      if (failed(desc)) return failure();

      char* data = static_cast<char*>(desc->data());
      int64_t size_in_bytes = primitive_util::ByteWidth(desc->dtype());

      TensorProto* tensor_proto = result->mutable_tensor();
      for (int64_t size : desc->sizes()) {
        size_in_bytes *= size;
        tensor_proto->add_sizes(size);
      }

      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(data, size_in_bytes);
      tensor_proto->set_contents(std::string(data, size_in_bytes));
      tensor_proto->set_dtype(desc->dtype());

      std::free(desc->data());
      return success();
    }

    return failure();
  }

  MemrefDesc operator()(PrimitiveType element_type, void* base_ptr,
                        void* data_ptr, int64_t offset,
                        absl::Span<const int64_t> sizes,
                        absl::Span<const int64_t> strides) const {
    return MemrefDesc(element_type, base_ptr, offset, sizes, strides);
  }

  ResultsProto* proto = nullptr;
};

// Converts arguments protobuf message into Xla runtime arguments.
static absl::Status WriteInoutResults(ArgumentsProto& proto, RunnerArgs& args,
                                      ResultsProto* results) {
  for (int i = 0; i < proto.arguments().size(); ++i) {
    ArgumentProto arg = proto.arguments().Get(i);
    switch (arg.argument_case()) {
      case ArgumentProto::ArgumentCase::kScalar:
        continue;
      case ArgumentProto::ArgumentCase::kTensor:
        if (arg.tensor().inout()) {
          auto* result = results->add_results();
          TensorProto* tensor_proto = result->mutable_tensor();

          auto* memref = llvm::cast<MemrefDesc>(&args[i]);

          char* sv = static_cast<char*>(memref->data());
          int64_t size_in_bytes = primitive_util::ByteWidth(memref->dtype());

          for (int64_t size : memref->sizes()) {
            size_in_bytes *= size;
            tensor_proto->add_sizes(size);
          }

          tensor_proto->set_contents(std::string(sv, size_in_bytes));
          tensor_proto->set_dtype(memref->dtype());
        }
        break;
      // Unsupported argument type.
      default:
        return InvalidArgumentError(
            StrFormat("unsupported argument: %s", arg.DebugString()));
    }
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//

absl::Status Execute(RunnerFlags flags,
                     const JitExecutable::Options& compile_opts,
                     const Executable::ExecuteOpts& execute_opts) {
  LOG(INFO) << "Executing runner tool:\n"
            << " - module: " << flags.module_path << "\n"
            << " - arguments: " << flags.arguments_path << "\n"
            << " - results: " << flags.results_path;

  Env* env = Env::Default();

  // Read MLIR module from the input file.
  std::string module;
  if (auto st = ReadFileToString(env, flags.module_path, &module); !st.ok()) {
    return InternalError(
        StrFormat("failed to read module input from %s, error: %s",
                  flags.module_path, st.error_message()));
  }

  // Read arguments from the input file.
  ArgumentsProto args_proto;
  if (auto read = ReadProtoFile(env, flags.arguments_path, &args_proto);
      !read.ok()) {
    return InternalError(
        StrFormat("failed to read arguments input from %s, error %s",
                  flags.arguments_path, read.message()));
  }

  // Convert arguments proto message to the Xla runtime arguments.
  RunnerArgs args(args_proto.arguments_size());
  if (auto converted = ConvertArgs(args_proto, args); !converted.ok())
    return converted;

  // Instantiate JitExecutable from the input module.
  absl::StatusOr<JitExecutable> jit_executable =
      JitExecutable::Instantiate(module, compile_opts, {flags.function});
  if (!jit_executable.ok()) return jit_executable.status();

  // TODO(ezhulenev): Add support for specializing to arguments shapes/values.
  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  if (executable.IsError()) return executable.GetError();

  // Convert returned results to results proto.
  ResultsProto results_proto;
  ResultConverterSet converter(CheckNoError, ReturnResults{&results_proto});

  // Execute and convert results to proto message.
  if (auto executed = executable->Execute(args, converter, execute_opts);
      !executed.ok())
    return executed.status();

  if (auto inout = WriteInoutResults(args_proto, args, &results_proto);
      !inout.ok())
    return inout;

  // Write results proto to the requested file location.
  if (auto wrote = WriteProtoFile(env, flags.results_path, results_proto);
      !wrote.ok())
    return InternalError(
        StrFormat("failed to write results proto to %s, error %s",
                  flags.results_path, wrote.message()));

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Compose Xla Runtime Runner into `main` function.
//===----------------------------------------------------------------------===//

int Main(int argc, char** argv, const JitExecutable::Options& compile_opts,
         const Executable::ExecuteOpts& execute_opts) {
  xla::runtime::RunnerFlags flags;

  std::vector<tsl::Flag> flag_list;
  xla::runtime::AppendRunnerFlags(&flag_list, &flags);

  if (auto parsed = tsl::Flags::Parse(&argc, argv, flag_list); !parsed) {
    std::cerr << "Failed to parse runner flags";
    return 1;
  }

  tsl::port::InitMain(argv[0], &argc, &argv);

  if (auto executed = Execute(flags, compile_opts, execute_opts);
      !executed.ok()) {
    std::cerr << "Failed to execute runner tool: " << executed.message();
    return 1;
  }

  return 0;
}

}  // namespace runtime
}  // namespace xla
