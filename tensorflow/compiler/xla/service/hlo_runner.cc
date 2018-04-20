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
#define EIGEN_USE_THREADS

#include "tensorflow/compiler/xla/service/hlo_runner.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunner::CreateModuleFromString(const tensorflow::StringPiece hlo_string,
                                  const DebugOptions& debug_options) {
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return tools::Parse(hlo_string, config);
}

namespace {

// Creates an HloModule from the given proto.
StatusOr<std::unique_ptr<HloModule>> HloProtoToModule(
    const HloProto& proto, const DebugOptions& debug_options) {
  TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                      HloModule::CreateModuleConfigFromProto(proto.hlo_module(),
                                                             debug_options));
  TF_ASSIGN_OR_RETURN(auto module,
                      HloModule::CreateFromProto(proto.hlo_module(), config));
  return std::move(module);
}

}  // namespace

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunner::ReadModuleFromBinaryProtoFile(const std::string& filename,
                                         const DebugOptions& debug_options) {
  HloProto proto;
  TF_RETURN_IF_ERROR(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                                 filename, &proto));
  return HloProtoToModule(proto, debug_options);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunner::ReadModuleFromTextProtoFile(const std::string& filename,
                                       const DebugOptions& debug_options) {
  HloProto proto;
  TF_RETURN_IF_ERROR(
      tensorflow::ReadTextProto(tensorflow::Env::Default(), filename, &proto));
  return HloProtoToModule(proto, debug_options);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunner::ReadModuleFromHloTextFile(const std::string& filename,
                                     const DebugOptions& debug_options) {
  string hlo_string;
  TF_RETURN_IF_ERROR(tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                                  filename, &hlo_string));
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return tools::Parse(hlo_string, config);
}

HloRunner::HloRunner(se::Platform* platform) {
  BackendOptions backend_options;
  backend_options.set_platform(platform);
  backend_ = Backend::CreateBackend(backend_options).ConsumeValueOrDie();
  VLOG(1) << "Created HloRunner for platform: " << platform->Name();
}

HloRunner::~HloRunner() {}

StatusOr<std::unique_ptr<Literal>> HloRunner::Execute(
    std::unique_ptr<HloModule> module,
    const tensorflow::gtl::ArraySlice<Literal*> arguments,
    bool run_hlo_passes) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      CreateExecutable(std::move(module), run_hlo_passes));
  se::Stream stream(backend().default_stream_executor());
  stream.Init();

  ServiceExecutableRunOptions service_run_options(GetServiceRunOptionsForDevice(
      backend().default_device_ordinal(), &stream, nullptr));
  const ExecutableRunOptions& run_options = service_run_options.run_options();

  // Copy arguments to device.
  std::vector<std::unique_ptr<ScopedShapedBuffer>> argument_buffers;
  std::vector<ShapedBuffer*> argument_buffer_ptrs;
  for (Literal* argument : arguments) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<ScopedShapedBuffer> argument_buffer,
        backend().transfer_manager()->AllocateScopedShapedBuffer(
            argument->shape(), run_options.allocator(),
            run_options.device_ordinal()));
    TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
        stream.parent(), *argument, *argument_buffer));
    argument_buffers.push_back(std::move(argument_buffer));
    argument_buffer_ptrs.push_back(argument_buffers.back().get());
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ShapedBuffer> result,
      executable->ExecuteOnStreamWrapper(
          &service_run_options, /*profile=*/nullptr, argument_buffer_ptrs));

  // Create a ScopedShapedBuffer of the result to manage deallocation. This will
  // deallocate all the device memory when it goes out of scope.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ScopedShapedBuffer> scoped_result,
      ScopedShapedBuffer::MakeScoped(result.get(), run_options.allocator()));

  auto result_literal = backend().transfer_manager()->TransferLiteralFromDevice(
      stream.parent(), *scoped_result);
  if (result_literal.ok()) {
    VLOG(4) << "Executed binary and got result: "
            << result_literal.ValueOrDie()->ToString();
  } else {
    VLOG(4) << "Executed binary and got status: "
            << result_literal.status().ToString();
  }
  return result_literal;
}

StatusOr<std::vector<std::unique_ptr<Literal>>> HloRunner::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const ReplicatedExecuteOptions& options) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      CreateExecutable(std::move(module), options.run_hlo_passes));
  TF_ASSIGN_OR_RETURN(
      DeviceAssignment device_assignment,
      backend().computation_placer()->AssignDevices(options.num_replicas, 1));
  std::vector<std::unique_ptr<se::Stream>> streams;
  std::vector<ServiceExecutableRunOptions> service_run_options;
  std::vector<std::unique_ptr<ScopedShapedBuffer>> argument_buffers;
  // Plus one so we can safely get &argument_buffer_ptrs[0] in case there are
  // no arguments.
  std::vector<const ShapedBuffer*> argument_buffer_ptrs(
      options.num_replicas * options.arguments.size() + 1);
  std::vector<tensorflow::gtl::ArraySlice<const ShapedBuffer*>>
      argument_buffer_slices;
  int64 index = 0;
  for (int64 i = 0; i < options.num_replicas; ++i) {
    int64 device = device_assignment(i, 0);
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                        backend().stream_executor(device));
    streams.push_back(absl::make_unique<se::Stream>(executor));
    streams.back()->Init();
    service_run_options.emplace_back(GetServiceRunOptionsForDevice(
        device, streams.back().get(), &device_assignment));

    // Copy arguments to device.
    for (const Literal* argument : options.arguments) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<ScopedShapedBuffer> argument_buffer,
          backend().transfer_manager()->AllocateScopedShapedBuffer(
              argument->shape(), backend().memory_allocator(), device));
      TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
          executor, *argument, *argument_buffer));
      argument_buffers.push_back(std::move(argument_buffer));
      argument_buffer_ptrs[index++] = argument_buffers.back().get();
    }
    argument_buffer_slices.emplace_back(
        &argument_buffer_ptrs[index - options.arguments.size()],
        options.arguments.size());
  }

  std::unique_ptr<tensorflow::thread::ThreadPool> pool;
  int64 num_threads = (options.infeed != nullptr) ? options.num_replicas : 0;
  if (ShapeUtil::IsInitialized(options.outfeed_shape)) {
    num_threads += options.num_replicas;
  }
  if (num_threads > 0) {
    pool = absl::make_unique<tensorflow::thread::ThreadPool>(
        tensorflow::Env::Default(), "infeed_outfeed",
        /*num_threads=*/num_threads);
  }
  if (options.infeed != nullptr) {
    for (int64 i = 0; i < options.num_replicas; ++i) {
      int64 device = device_assignment(i, 0);
      pool->Schedule([this, device, &options]() {
        se::StreamExecutor* executor =
            backend().stream_executor(device).ValueOrDie();
        VLOG(1) << "Starting infeed on device " << device;
        for (int64 step = 1;
             options.infeed_steps < 0 || step <= options.infeed_steps; ++step) {
          TF_CHECK_OK(backend().transfer_manager()->TransferLiteralToInfeed(
              executor, *options.infeed));
          if (step % 100 == 0) {
            VLOG(1) << "Infeed step " << step;
          }
        }
      });
    }
  }
  if (ShapeUtil::IsInitialized(options.outfeed_shape)) {
    for (int64 i = 0; i < options.num_replicas; ++i) {
      int64 device = device_assignment(i, 0);
      pool->Schedule([this, device, &options]() {
        se::StreamExecutor* executor =
            backend().stream_executor(device).ValueOrDie();
        VLOG(1) << "Starting outfeed on device " << device;
        for (int64 step = 1;
             options.infeed_steps < 0 || step <= options.infeed_steps; ++step) {
          auto literal = absl::make_unique<Literal>();
          TF_CHECK_OK(backend().transfer_manager()->TransferLiteralFromOutfeed(
              executor, options.outfeed_shape, literal.get()));
          if (options.outfeed_values != nullptr) {
            options.outfeed_values->push_back(std::move(literal));
          }
          if (step % 100 == 0) {
            VLOG(1) << "Outfeed step " << step;
          }
        }
      });
    }
  }

  LOG(INFO) << "Replicated execution started";
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<ShapedBuffer>> results,
                      executable->ExecuteOnStreams(service_run_options,
                                                   argument_buffer_slices));
  LOG(INFO) << "Replicated execution terminated";

  std::vector<std::unique_ptr<Literal>> exec_results;
  for (int64 i = 0; i < options.num_replicas; ++i) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<ScopedShapedBuffer> result,
                        ScopedShapedBuffer::MakeScoped(
                            results[i].get(), backend().memory_allocator()));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Literal> literal,
                        backend().transfer_manager()->TransferLiteralFromDevice(
                            streams[i]->parent(), *result));
    exec_results.push_back(std::move(literal));
  }
  return std::move(exec_results);
}

StatusOr<std::unique_ptr<Executable>> HloRunner::CreateExecutable(
    std::unique_ptr<HloModule> module, bool run_hlo_passes) {
  if (run_hlo_passes) {
    TF_ASSIGN_OR_RETURN(
        module, backend().compiler()->RunHloPasses(
                    std::move(module), backend().default_stream_executor(),
                    backend().memory_allocator()));
  }
  return backend().compiler()->RunBackend(std::move(module),
                                          backend().default_stream_executor(),
                                          backend().memory_allocator());
}

ServiceExecutableRunOptions HloRunner::GetServiceRunOptionsForDevice(
    int64 device, se::Stream* stream, DeviceAssignment* device_assignment) {
  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(device);
  run_options.set_stream(stream);
  run_options.set_allocator(backend().memory_allocator());
  run_options.set_inter_op_thread_pool(backend().inter_op_thread_pool());
  run_options.set_intra_op_thread_pool(
      backend().eigen_intra_op_thread_pool_device());
  if (device_assignment != nullptr) {
    run_options.set_device_assignment(device_assignment);
  }
  return ServiceExecutableRunOptions(run_options, backend().StreamBorrower(),
                                     backend().inter_op_thread_pool());
}

Backend& HloRunner::backend() {
  if (!backend_) {
    backend_ = Backend::CreateDefaultBackend().ConsumeValueOrDie();
    VLOG(1) << "Executing on platform " << backend().platform()->Name();
  }
  return *backend_;
}

}  // namespace xla
