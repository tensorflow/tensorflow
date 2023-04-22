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
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
using ::nvinfer1::IRuntime;

class CreateTRTResourceHandle : public OpKernel {
 public:
  explicit CreateTRTResourceHandle(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("resource_name", &resource_name_));
  }

  void Compute(OpKernelContext* ctx) override {
    {
      mutex_lock l(mutex_);
      if (!initialized_) {
        AllocatorAttributes attr;
        attr.set_on_host(true);
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                               &handle_, attr));

        VLOG(1) << "Creating TRT engine cache resource handle for op "
                << resource_name_ << " on device " << ctx->device()->name();
        handle_.scalar<ResourceHandle>()() =
            MakeResourceHandle<TRTEngineCacheResource>(
                ctx, std::string(kTfTrtContainerName), resource_name_);
        initialized_ = true;
      }
    }
    ctx->set_output(0, handle_);
  }

 private:
  string resource_name_;
  Tensor handle_;
  mutex mutex_;
  bool initialized_ TF_GUARDED_BY(mutex_) = false;

  TF_DISALLOW_COPY_AND_ASSIGN(CreateTRTResourceHandle);
};

REGISTER_KERNEL_BUILDER(Name("CreateTRTResourceHandle")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource_handle"),
                        CreateTRTResourceHandle);

class InitializeTRTResource : public OpKernel {
 public:
  explicit InitializeTRTResource(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("max_cached_engines_count", &max_cached_engines_));
  }

  void Compute(OpKernelContext* ctx) override {
    ResourceHandle handle = HandleFromInput(ctx, 0);
    core::RefCountPtr<TRTEngineCacheResource> resource;
    OP_REQUIRES_OK(
        ctx, LookupOrCreateResource<TRTEngineCacheResource>(
                 ctx, handle, &resource,
                 [this, ctx](TRTEngineCacheResource** resource) -> Status {
                   *resource = new TRTEngineCacheResource(
                       ctx, this->max_cached_engines_);
                   return Status::OK();
                 }));

    auto allocator = resource->allocator_.get();
    OP_REQUIRES(ctx, allocator != nullptr,
                errors::Internal("Not able to initialize TRT engine cache when "
                                 "GPU allocator is empty."));
    OP_REQUIRES(ctx, resource->cache_.size() == 0,
                errors::Internal("Expect engine cache to be empty, but got ",
                                 resource->cache_.size(), " entries."));

    // Get the file name.
    const string& filename = ctx->input(1).scalar<tstring>()();
    OP_REQUIRES(ctx, !filename.empty(),
                errors::InvalidArgument("filename cannot be empty."));

    // Parse the serialized engines and add them to the cache.
    std::unique_ptr<RandomAccessFile> file;
    OP_REQUIRES_OK(ctx, ctx->env()->NewRandomAccessFile(filename, &file));
    auto reader = absl::make_unique<io::RecordReader>(file.get());

    uint64 offset = 0;
    int num_loaded_engine = 0;
    do {
      tstring record;
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
      std::vector<ExecutionContext> ctx_vec;
      if (num_loaded_engine == 0) {
        // Restore profiles if there are any. Currently only 1 engine is allowed
        // in dynamic mode therefore we call this only for the 0th engine.
        // it is a no-op in implicit batch mode.
        OP_REQUIRES_OK(ctx, resource->profiles_.RestoreProfiles(raw_engine));
        OP_REQUIRES_OK(ctx, resource->profiles_.CreateExecutionContexts(
                                raw_engine, &ctx_vec));
      } else {
        // Multiple engines are only available in static mode. For each engine
        // we have only a single execution context.
        ctx_vec.push_back(ExecutionContext::Create(raw_engine));
      }
      resource->cache_.emplace(engine_input_shapes,
                               absl::make_unique<EngineContext>(
                                   std::move(engine), std::move(ctx_vec)));
      ++num_loaded_engine;
    } while (1);
    VLOG(1) << "Loaded " << num_loaded_engine << " TRT engines for op "
            << handle.name() << " on device " << ctx->device()->name()
            << " from file " << filename;
  }

 private:
  // Maximum number of cached engines
  int max_cached_engines_;

  TF_DISALLOW_COPY_AND_ASSIGN(InitializeTRTResource);
};

REGISTER_KERNEL_BUILDER(Name("InitializeTRTResource")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource_handle"),
                        InitializeTRTResource);

class SerializeTRTResource : public OpKernel {
 public:
  explicit SerializeTRTResource(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("delete_resource", &delete_resource_));
  }

  void Compute(OpKernelContext* ctx) override {
    const string& resource_name = ctx->input(0).scalar<tstring>()();
    const string& filename = ctx->input(1).scalar<tstring>()();
    OP_REQUIRES(ctx, !filename.empty(),
                errors::InvalidArgument("filename cannot be empty."));

    // Lookup engine cache resource.
    TRTEngineCacheResource* resource = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->Lookup(std::string(kTfTrtContainerName),
                                             resource_name, &resource));
    core::ScopedUnref unref_me(resource);

    // Terminate the calibration if any.
    if (resource->calib_ctx_) resource->calib_ctx_->TerminateCalibration();

    // Serialize the engines and write them to file.
    std::unique_ptr<WritableFile> file;
    OP_REQUIRES_OK(ctx, ctx->env()->NewWritableFile(filename, &file));
    auto writer = absl::make_unique<io::RecordWriter>(file.get());

    int num_serialized_engines = 0;
    for (const auto& pair : resource->cache_) {
      // Ignore engines that failed to build.
      const std::unique_ptr<EngineContext>& engine = pair.second;
      if (!engine || !engine->cuda_engine) continue;

      TRTEngineInstance engine_instance;
      // Add input shapes.
      const std::vector<TensorShape>& engine_input_shapes = pair.first;
      for (const TensorShape& shape : engine_input_shapes) {
        shape.AsProto(engine_instance.add_input_shapes());
      }
      // Add the serialized engine.
      TrtUniquePtrType<nvinfer1::IHostMemory> engine_data(
          engine->cuda_engine->serialize());
      engine_instance.set_serialized_engine(engine_data->data(),
                                            engine_data->size());

      OP_REQUIRES_OK(ctx,
                     writer->WriteRecord(engine_instance.SerializeAsString()));
      ++num_serialized_engines;
    }
    VLOG(1) << "Serialized " << num_serialized_engines << " TRT engines for op "
            << resource_name << " on device " << ctx->device()->name()
            << " to file " << filename;

    if (delete_resource_) {
      VLOG(1) << "Destroying TRT engine cache resource for op " << resource_name
              << " on device " << ctx->device()->name();
      OP_REQUIRES_OK(ctx,
                     ctx->resource_manager()->Delete<TRTEngineCacheResource>(
                         std::string(kTfTrtContainerName), resource_name));
    }
  }

 private:
  bool delete_resource_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(SerializeTRTResource);
};

REGISTER_KERNEL_BUILDER(Name("SerializeTRTResource").Device(DEVICE_GPU),
                        SerializeTRTResource);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
