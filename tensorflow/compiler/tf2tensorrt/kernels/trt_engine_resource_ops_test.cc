/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <dirent.h>
#include <string.h>

#include <fstream>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_engine_instance.pb.h"  // NOLINT
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

class TRTEngineResourceOpsTest : public OpsTestBase {
 protected:
  void Reset() {
    for (auto& temp : tensors_) {
      delete temp;
    }
    for (auto& temp : managed_outputs_) {
      delete temp;
    }
    tensors_.clear();
    managed_outputs_.clear();
    inputs_.clear();
  }

  TrtUniquePtrType<nvinfer1::ICudaEngine> CreateTRTEngine(bool dynamic=false) {
    TrtUniquePtrType<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger));
    TrtUniquePtrType<nvinfer1::INetworkDefinition> network;
    if (!dynamic) {
      network = TrtUniquePtrType<nvinfer1::INetworkDefinition>(
          builder->createNetwork());
    } else {
      network = TrtUniquePtrType<nvinfer1::INetworkDefinition>(
          builder->createNetworkV2( 1U << static_cast<int>(
              nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    }
    // Add the input.
    nvinfer1::Dims dims;
    dims.nbDims = 1;
    dims.d[0] = dynamic ? -1 : 1;
    const char *inName = "input";
    nvinfer1::ITensor* input =
        network->addInput(inName, nvinfer1::DataType::kFLOAT, dims);
    EXPECT_NE(nullptr, input);

    // Add a unary layer.
    nvinfer1::IUnaryLayer* layer =
        network->addUnary(*input, nvinfer1::UnaryOperation::kEXP);
    EXPECT_NE(nullptr, layer);

    // Mark the output.
    nvinfer1::ITensor* output = layer->getOutput(0);
    output->setName("output");
    network->markOutput(*output);

    // Build the engine
    TrtUniquePtrType<nvinfer1::IBuilderConfig> builder_config(
          builder->createBuilderConfig());

    builder->setMaxBatchSize(1);
    builder_config->setMaxWorkspaceSize(1 << 10);

    if (dynamic) {
      // Create three profiles
      for (int i=1; i<=3; i++) {
        auto* profile = builder->createOptimizationProfile();
        nvinfer1::Dims dim;
        dim.nbDims = 1;
        dim.d[0] = i;
        profile->setDimensions(inName, nvinfer1::OptProfileSelector::kMIN, dim);
        profile->setDimensions(inName, nvinfer1::OptProfileSelector::kOPT, dim);
        profile->setDimensions(inName, nvinfer1::OptProfileSelector::kMAX, dim);
        int idx = builder_config->addOptimizationProfile(profile);
        EXPECT_NE(-1, idx);
      }
    }
    TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
        builder->buildEngineWithConfig(*network, *builder_config));

    EXPECT_NE(nullptr, engine);
    return engine;
  }
  Logger logger;
};

TEST_F(TRTEngineResourceOpsTest, Basic) {
  // Create the GPU device.
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:worker/replica:0/task:0"));
  ResourceMgr* rm = device->resource_manager();
  SetDevice(DEVICE_GPU, std::move(device));

  // Create a resource handle.
  const string container(kTfTrtContainerName);
  const string resource_name = "myresource";
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "CreateTRTResourceHandle")
                   .Attr("resource_name", resource_name)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  TF_ASSERT_OK(RunOpKernel());
  ResourceHandle handle =
      context_->mutable_output(0)->scalar<ResourceHandle>()();

  // Check that a resource hasn't been created yet.
  TRTEngineCacheResource* resource = nullptr;
  EXPECT_TRUE(
      errors::IsNotFound(rm->Lookup(container, resource_name, &resource)));

  // Create a resource and use an empty file to initialize the resource.
  Reset();
  Env* env = Env::Default();
  const string filename = io::JoinPath(testing::TmpDir(), "trt_engine_file");
  {
    std::unique_ptr<WritableFile> file;
    TF_ASSERT_OK(env->NewWritableFile(filename, &file));
  }
  TF_ASSERT_OK(NodeDefBuilder("op", "InitializeTRTResource")
                   .Input(FakeInput(DT_RESOURCE))
                   .Input(FakeInput(DT_STRING))
                   .Attr("max_cached_engines_count", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  AddInputFromArray<tstring>(TensorShape({}), {filename});
  TF_ASSERT_OK(RunOpKernel());

  // Check that the resource is registered with the resource manager and the
  // cache of the resource is empty.
  EXPECT_TRUE(rm->Lookup(container, resource_name, &resource).ok());
  EXPECT_EQ(0, resource->cache_.size());

  // Create an engine and add it to the cache of the resource.
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine = CreateTRTEngine();
  TrtUniquePtrType<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  resource->cache_.emplace(
      std::vector<TensorShape>{TensorShape({1, 1})},
      absl::make_unique<EngineContext>(std::move(engine), std::move(context)));
  // Check that the resource has multiple references before it is unregistered
  // from the resource manager.
  EXPECT_FALSE(resource->RefCountIsOne());

  // Serialize the engine to a file and unregistered the resource from the
  // resource manager.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "SerializeTRTResource")
                   .Attr("delete_resource", true)
                   .Input(FakeInput(DT_STRING))
                   .Input(FakeInput(DT_STRING))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<tstring>(TensorShape({}), {resource_name});
  AddInputFromArray<tstring>(TensorShape({}), {filename});
  TF_ASSERT_OK(RunOpKernel());
  // Check that the resource now has only one reference. Detach the reference
  // to the resource to destroy the resource.
  EXPECT_TRUE(resource->RefCountIsOne());
  resource->Unref();

  // Check that unregistering the resource from the resource manager returns an
  // error as the resource has already been unregistered.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "DestroyResourceOp")
                   .Attr("ignore_lookup_error", false)
                   .Input(FakeInput(DT_RESOURCE))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  EXPECT_TRUE(errors::IsNotFound(RunOpKernel()));

  // Verify the file for the serialized engine.
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(filename, &file));
  auto reader = absl::make_unique<io::RecordReader>(file.get());
  uint64 offset = 0;
  tstring record;
  TF_ASSERT_OK(reader->ReadRecord(&offset, &record));
  TRTEngineInstance engine_instance;
  engine_instance.ParseFromString(record);
  EXPECT_EQ(1, engine_instance.input_shapes_size());
  EXPECT_EQ(2, engine_instance.input_shapes(0).dim_size());
  EXPECT_EQ(1, engine_instance.input_shapes(0).dim(0).size());
  EXPECT_EQ(1, engine_instance.input_shapes(0).dim(1).size());
  EXPECT_TRUE(errors::IsOutOfRange(reader->ReadRecord(&offset, &record)));

  // Recreate the resource and use the file with the serialized engine to
  // initialize the resource.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "InitializeTRTResource")
                   .Input(FakeInput(DT_RESOURCE))
                   .Input(FakeInput(DT_STRING))
                   .Attr("max_cached_engines_count", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  AddInputFromArray<tstring>(TensorShape({}), {filename});
  TF_ASSERT_OK(RunOpKernel());

  // Check that the resource is registered with the resource manager again and
  // the cache of the resource is not empty.
  EXPECT_TRUE(rm->Lookup(container, resource_name, &resource).ok());
  EXPECT_EQ(1, resource->cache_.size());
  // Check that the resource has multiple references before it is unregistered
  // from the resource manager.
  EXPECT_FALSE(resource->RefCountIsOne());

  // Unregister the resource from the resource manager two times, expect that
  // the second time produces an error.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "DestroyResourceOp")
                   .Attr("ignore_lookup_error", false)
                   .Input(FakeInput(DT_RESOURCE))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_TRUE(errors::IsNotFound(RunOpKernel()));

  // Check that the resource now has only one reference. Detach the reference
  // to the resource to destroy resource.
  EXPECT_TRUE(resource->RefCountIsOne());
  resource->Unref();
}

TEST_F(TRTEngineResourceOpsTest, Profiles) {
  // Create the GPU device.
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:worker/replica:0/task:0"));
  ResourceMgr* rm = device->resource_manager();
  SetDevice(DEVICE_GPU, std::move(device));

  // Create the resource handle.
  const string container(kTfTrtContainerName);
  const string resource_name = "myresource";
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "CreateTRTResourceHandle")
                   .Attr("resource_name", resource_name)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  TF_ASSERT_OK(RunOpKernel());
  ResourceHandle handle =
      context_->mutable_output(0)->scalar<ResourceHandle>()();

  TRTEngineCacheResource* resource = nullptr;
  EXPECT_TRUE(
      errors::IsNotFound(rm->Lookup(container, resource_name, &resource)));

  // Create the resource using an empty file with InitializeTRTResource.
  Reset();
  Env* env = Env::Default();
  const string filename = io::JoinPath(testing::TmpDir(), "trt_engine_file");
  {
    std::unique_ptr<WritableFile> file;
    TF_ASSERT_OK(env->NewWritableFile(filename, &file));
  }
  TF_ASSERT_OK(NodeDefBuilder("op", "InitializeTRTResource")
                   .Input(FakeInput(DT_RESOURCE))
                   .Input(FakeInput(DT_STRING))
                   .Attr("max_cached_engines_count", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  AddInputFromArray<tstring>(TensorShape({}), {filename});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_TRUE(rm->Lookup(container, resource_name, &resource).ok());
  EXPECT_EQ(0, resource->cache_.size());

  // Create a serialized TRT engine file.
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine = CreateTRTEngine(true);
  TrtUniquePtrType<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  resource->cache_.emplace(
      std::vector<TensorShape>{TensorShape({1, 1})},
      absl::make_unique<EngineContext>(std::move(engine), std::move(context)));
  resource->Unref();

  // Serialize the engine using SerializeTRTResource op.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "SerializeTRTResource")
                   .Attr("delete_resource", true)
                   .Input(FakeInput(DT_STRING))
                   .Input(FakeInput(DT_STRING))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<tstring>(TensorShape({}), {resource_name});
  AddInputFromArray<tstring>(TensorShape({}), {filename});
  TF_ASSERT_OK(RunOpKernel());

  // Make sure the cache is deleted.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "DestroyResourceOp")
                   .Attr("ignore_lookup_error", false)
                   .Input(FakeInput(DT_RESOURCE))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  EXPECT_TRUE(errors::IsNotFound(RunOpKernel()));

  // Verify the serialized engine file.
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(filename, &file));
  auto reader = absl::make_unique<io::RecordReader>(file.get());
  uint64 offset = 0;
  tstring record;
  TF_ASSERT_OK(reader->ReadRecord(&offset, &record));
  TRTEngineInstance engine_instance;
  engine_instance.ParseFromString(record);
  EXPECT_EQ(1, engine_instance.input_shapes_size());
  EXPECT_EQ(2, engine_instance.input_shapes(0).dim_size());
  EXPECT_EQ(1, engine_instance.input_shapes(0).dim(0).size());
  EXPECT_EQ(1, engine_instance.input_shapes(0).dim(1).size());
  EXPECT_TRUE(errors::IsOutOfRange(reader->ReadRecord(&offset, &record)));

  // Recreate the cache resource.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "InitializeTRTResource")
                   .Input(FakeInput(DT_RESOURCE))
                   .Input(FakeInput(DT_STRING))
                   .Attr("max_cached_engines_count", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  AddInputFromArray<tstring>(TensorShape({}), {filename});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_TRUE(rm->Lookup(container, resource_name, &resource).ok());
  EXPECT_EQ(1, resource->cache_.size());
  EXPECT_EQ(3, resource->profiles_.GetNumProfiles());
  EXPECT_EQ(3, resource->cache_.begin()->second->execution_context.size());
  resource->Unref();

  // Destroy the engine cache again.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "DestroyResourceOp")
                   .Attr("ignore_lookup_error", false)
                   .Input(FakeInput(DT_RESOURCE))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_TRUE(errors::IsNotFound(RunOpKernel()));
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
