/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_engine_instance.pb.h"  // NOLINT
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
using ::nvinfer1::IRuntime;

class CreateTRTEngineCache : public OpKernel {
 public:
  explicit CreateTRTEngineCache(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("resource_name", &resource_name_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("max_cached_engines_count", &max_cached_engines_));
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "Creating TRT engine cache resource in container " << container_
            << " for op " << resource_name_ << " on device "
            << ctx->device()->name();
    OP_REQUIRES_OK(ctx,
                   ctx->resource_manager()->Create(
                       container_, resource_name_,
                       new TRTEngineCacheResource(ctx, max_cached_engines_)));

    Tensor* handle;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    handle->scalar<ResourceHandle>()() =
        MakeResourceHandle<TRTEngineCacheResource>(ctx, container_,
                                                   resource_name_);
  }

 private:
  string container_;
  string resource_name_;

  // Maximum number of cached engines
  int max_cached_engines_;

  TF_DISALLOW_COPY_AND_ASSIGN(CreateTRTEngineCache);
};

REGISTER_KERNEL_BUILDER(Name("CreateTRTEngineCache")
                            .Device(DEVICE_GPU)
                            .HostMemory("engine_cache_handle"),
                        CreateTRTEngineCache);

class PopulateTRTEngineCache : public OpKernel {
 public:
  explicit PopulateTRTEngineCache(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    ResourceHandle handle = HandleFromInput(ctx, 0);
    core::RefCountPtr<TRTEngineCacheResource> resource;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, handle, &resource));

    auto allocator = resource->allocator_.get();
    OP_REQUIRES(ctx, allocator != nullptr,
                errors::Internal("Not able to initialize TRT engine cache when "
                                 "GPU allocator is empty."));
    OP_REQUIRES(ctx, resource->cache_.size() == 0,
                errors::Internal("Expect engine cache to be empty, but got ",
                                 resource->cache_.size(), " entries."));

    // Get the file name.
    const string& filename = ctx->input(1).scalar<string>()();
    OP_REQUIRES(ctx, !filename.empty(),
                errors::InvalidArgument("filename cannot be empty."));

    // Parse the serialized engines and add them to the cache.
    std::unique_ptr<RandomAccessFile> file;
    OP_REQUIRES_OK(ctx, ctx->env()->NewRandomAccessFile(filename, &file));
    auto reader = absl::make_unique<io::RecordReader>(file.get());

    uint64 offset = 0;
    int num_loaded_engine = 0;
    do {
      string record;
      Status status = reader->ReadRecord(&offset, &record);
      if (errors::IsOutOfRange(status)) break;

      TRTEngineInstance engine_instance;
      engine_instance.ParseFromString(record);
      std::vector<TensorShape> engine_input_shapes;
      for (const TensorShapeProto& shape : engine_instance.input_shapes()) {
        engine_input_shapes.emplace_back(shape);
      }

      TrtUniquePtrType<IRuntime> infer(
          nvinfer1::createInferRuntime(TRTEngineCacheResource::GetLogger()));
      infer->setGpuAllocator(allocator);
      TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
          infer->deserializeCudaEngine(
              engine_instance.serialized_engine().c_str(),
              engine_instance.serialized_engine().size(), nullptr));
      auto raw_engine = engine.get();
      resource->cache_.emplace(
          engine_input_shapes,
          absl::make_unique<EngineContext>(
              std::move(engine), TrtUniquePtrType<nvinfer1::IExecutionContext>(
                                     raw_engine->createExecutionContext())));
      ++num_loaded_engine;
    } while (1);
    VLOG(1) << "Loaded " << num_loaded_engine << " TRT engines to container "
            << handle.container() << " for op " << handle.name()
            << " on device " << ctx->device()->name() << " from file "
            << filename;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopulateTRTEngineCache);
};

REGISTER_KERNEL_BUILDER(Name("PopulateTRTEngineCache")
                            .Device(DEVICE_GPU)
                            .HostMemory("engine_cache_handle"),
                        PopulateTRTEngineCache);

class DumpTRTEngineCache : public OpKernel {
 public:
  explicit DumpTRTEngineCache(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("delete_cache_after_dump",
                                     &delete_cache_after_dump_));
  }

  void Compute(OpKernelContext* ctx) override {
    const string& container = ctx->input(0).scalar<string>()();
    const string& resource_name = ctx->input(1).scalar<string>()();
    const string& filename = ctx->input(2).scalar<string>()();
    OP_REQUIRES(ctx, !filename.empty(),
                errors::InvalidArgument("filename cannot be empty."));

    TRTEngineCacheResource* resource = nullptr;
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(
                            container, resource_name, &resource));
    core::ScopedUnref unref_me(resource);

    // Serialize the engines and write them to file.
    std::unique_ptr<WritableFile> file;
    OP_REQUIRES_OK(ctx, ctx->env()->NewWritableFile(filename, &file));
    auto writer = absl::make_unique<io::RecordWriter>(file.get());

    for (const auto& pair : resource->cache_) {
      TRTEngineInstance engine_instance;
      // Add input shapes.
      const std::vector<TensorShape>& engine_input_shapes = pair.first;
      for (const TensorShape& shape : engine_input_shapes) {
        shape.AsProto(engine_instance.add_input_shapes());
      }
      // Add the serialized engine.
      const std::unique_ptr<EngineContext>& engine = pair.second;
      TrtUniquePtrType<nvinfer1::IHostMemory> engine_data(
          engine->cuda_engine->serialize());
      engine_instance.set_serialized_engine(engine_data->data(),
                                            engine_data->size());

      OP_REQUIRES_OK(ctx,
                     writer->WriteRecord(engine_instance.SerializeAsString()));
    }
    VLOG(1) << "Serialized " << resource->cache_.size()
            << " TRT engines in container " << container << " for op "
            << resource_name << " on device " << ctx->device()->name()
            << " to file " << filename;

    if (delete_cache_after_dump_) {
      VLOG(1) << "Destroying TRT engine cache resource in container "
              << container << " for op " << resource_name << " on device "
              << ctx->device()->name();
      OP_REQUIRES_OK(ctx,
                     ctx->resource_manager()->Delete<TRTEngineCacheResource>(
                         container, resource_name));
    }
  }

 private:
  bool delete_cache_after_dump_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(DumpTRTEngineCache);
};

REGISTER_KERNEL_BUILDER(Name("DumpTRTEngineCache").Device(DEVICE_GPU),
                        DumpTRTEngineCache);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
