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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2tensorrt/common/datavec.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_engine_instance.pb.h"  // NOLINT
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

struct TestParam {
  nvinfer1::Dims dims;
  bool dynamic_shape;
  int n_inputs;
};

class TRTEngineResourceOpsTest
    : public OpsTestBase,
      public ::testing::WithParamInterface<TestParam> {
 public:
  TRTEngineResourceOpsTest() : param_(GetParam()) {}

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

  nvinfer1::ITensor* NetworkWith1Input(nvinfer1::INetworkDefinition* network,
                                       nvinfer1::ITensor* input) {
    // Add a unary layer.
    nvinfer1::IUnaryLayer* layer =
        network->addUnary(*input, nvinfer1::UnaryOperation::kEXP);
    EXPECT_NE(nullptr, layer);
    return layer->getOutput(0);
  }

  // Constructs a network with two inputs, where the second input is a shape
  // tensor. We take a slice of the first input with the size of the slice
  // specified by the second input, assuming the first input is a 2D tensor.
  // We then add the slice to itself to produce the output of the network.
  nvinfer1::ITensor* NetworkWith2Inputs(nvinfer1::INetworkDefinition* network,
                                        nvinfer1::ITensor* input) {
    nvinfer1::Dims dims2{1, {2}};
    nvinfer1::ITensor* input2 =
        network->addInput("input2", nvinfer1::DataType::kINT32, dims2);
    EXPECT_NE(nullptr, input2);

    nvinfer1::Dims start{2, {0, 0}};
    nvinfer1::Dims stride{2, {1, 1}};
    auto slice_layer = network->addSlice(*input, start, stride, stride);
    EXPECT_NE(nullptr, slice_layer);

    slice_layer->setInput(2, *input2);
    nvinfer1::ITensor* sliced_input = slice_layer->getOutput(0);
    EXPECT_NE(nullptr, sliced_input);

    auto layer = network->addElementWise(*sliced_input, *sliced_input,
                                         nvinfer1::ElementWiseOperation::kSUM);
    EXPECT_NE(nullptr, layer);
    return layer->getOutput(0);
  }

  TrtUniquePtrType<nvinfer1::ICudaEngine> CreateTRTEngine() {
    TrtUniquePtrType<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger_));
    TrtUniquePtrType<nvinfer1::INetworkDefinition> network;
    if (!this->param_.dynamic_shape || !IS_TRT_VERSION_GE(6, 0, 0, 0)) {
      network = TrtUniquePtrType<nvinfer1::INetworkDefinition>(
          builder->createNetwork());
    } else {
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
      network = TrtUniquePtrType<nvinfer1::INetworkDefinition>(
          builder->createNetworkV2(
              1U << static_cast<int>(
                  nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
#endif
    }

    // Add the input.
    nvinfer1::Dims dims = this->param_.dims;
    if (this->param_.dynamic_shape) {
      std::fill(dims.d, dims.d + dims.nbDims, -1);
    }
    const char* in_name = "input";
    nvinfer1::ITensor* input =
        network->addInput(in_name, nvinfer1::DataType::kFLOAT, dims);
    EXPECT_NE(nullptr, input);
    // Mark the output.
    nvinfer1::ITensor* output =
        this->param_.n_inputs == 1
            ? this->NetworkWith1Input(network.get(), input)
            : this->NetworkWith2Inputs(network.get(), input);
    output->setName("output");
    network->markOutput(*output);

    // Build the engine
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    TrtUniquePtrType<nvinfer1::IBuilderConfig> builder_config(
        builder->createBuilderConfig());
    builder_config->setMaxWorkspaceSize(1 << 10);
#else
    builder->setMaxWorkspaceSize(1 << 10);
#endif
    builder->setMaxBatchSize(1);

    if (this->param_.dynamic_shape) {
      TrtShapeOptimizationProfile profile;
      profile.SetShapeTensorMask(network.get());
      // The for loop defines three optimization profiles for the network.
      for (int i = 1; i <= 3; i++) {
        const int n_input = param_.n_inputs;
        std::vector<TensorShape> shape_vec(n_input);
        // Define a shape with all dimensions set to 3*i.
        std::vector<int> dimvec(this->param_.dims.nbDims, 3 * i);
        TensorShape shape;
        TF_CHECK_OK(
            TensorShapeUtils::MakeShape(dimvec.data(), dimvec.size(), &shape));

        const nvinfer1::ITensor* input = network->getInput(0);
        const char* name = input->getName();
        VLOG(2) << "Defining profile for input " << name;
        shape_vec[0] = shape;
        if (this->param_.n_inputs == 2) {
          // The shape of the shape tensor.
          TF_CHECK_OK(TensorShapeUtils::MakeShape(
              std::vector<int32>{param_.dims.nbDims}, &shape));
          shape_vec[1] = shape;
          // Values of the shape tensor
          Tensor shape_tensor(DT_INT32, shape);
          // Define shape values {1, i}, where 1 is the value of the first dim,
          // and i is the value of the second dimension.
          std::vector<int32> vals{1, i};
          std::copy_n(vals.data(), vals.size(),
                      shape_tensor.flat<int32_t>().data());
          DataVec shape_values{{"one", {}}, {"two", shape_tensor}};
          TF_CHECK_OK(profile.CollectShapeValues(shape_values));
        } else {
          TF_CHECK_OK(profile.CollectShapeValues({{"one", {}}}));
        }
        profile.AddShape(shape_vec);
      }
      std::vector<PartialTensorShape> input_partial_shapes;
      TF_CHECK_OK(GetNetworkInputShapes(network.get(), &input_partial_shapes));
      profile.InitProfiles(input_partial_shapes);
      // Configure and build engine
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
      TF_CHECK_OK(profile.ConfigureBuilder(builder.get(), builder_config.get(),
                                           network.get()));
#endif
    }
    VLOG(2) << "ConfigureBuilder Finished";
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
        builder->buildEngineWithConfig(*network, *builder_config));
#else
    TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
        builder->buildCudaEngine(*network));
#endif
    VLOG(2) << "Engine constructed";
    EXPECT_NE(nullptr, engine);
    return engine;
  }
  Logger logger_;
  TestParam param_;
};

#if IS_TRT_VERSION_GE(7, 1, 3, 0)
constexpr std::array<TestParam, 3> TestParameters = {
    TestParam{nvinfer1::Dims{1, {1}}, false, 1},
    TestParam{nvinfer1::Dims{1, {1}}, true, 1},
    TestParam{nvinfer1::Dims{2, {3, 3}}, true, 2}};
#elif IS_TRT_VERSION_GE(6, 0, 0, 0)
constexpr std::array<TestParam, 2> TestParameters = {
    TestParam{nvinfer1::Dims{1, {1}}, false, 1},
    TestParam{nvinfer1::Dims{1, {1}}, true, 1}};
#else
constexpr std::array<TestParam, 1> TestParameters = {
    TestParam{nvinfer1::Dims{1, {1}}, false, 1}};
#endif

INSTANTIATE_TEST_CASE_P(EngineResourceOpsTestInstantiation,
                        TRTEngineResourceOpsTest,
                        ::testing::ValuesIn(TestParameters));

TEST_P(TRTEngineResourceOpsTest, Basic) {
  // Create the GPU device.
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:worker/replica:0/task:0"));
  ResourceMgr* rm = device->resource_manager();
  SetDevice(DEVICE_GPU, std::move(device));

  VLOG(2) << "Is TRT64 ? " << IS_TRT_VERSION_GE(6, 0, 0, 0);

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
  auto context_status =
      ExecutionContext::Create(engine.get(), resource->allocator_.get());
  TF_ASSERT_OK(context_status.status());

  resource->cache_.emplace(
      std::vector<TensorShape>{TensorShape({1, 1})},
      absl::make_unique<EngineContext>(std::move(engine),
                                       std::move(context_status.ValueOrDie())));
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

  // Check that unregistering the resource from the resource manager returns
  // an error as the resource has already been unregistered.
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
  if (this->param_.dynamic_shape) {
    EXPECT_EQ(3, resource->profiles_.GetNumProfiles());
    EXPECT_EQ(3, resource->cache_.begin()->second->GetNumContexts());

    if (this->param_.n_inputs == 1) {
      // Check if profiles are restored correctly.
      std::vector<TensorShape> shapes(1);
      // We create a shape vector that matches only profile 1.
      TF_CHECK_OK(
          TensorShapeUtils::MakeShape(std::vector<int32>{6}, &shapes[0]));
      EXPECT_EQ(1, resource->profiles_.GetProfileNumber(shapes));
    } else {
      // Check if shape values are restored corretly.
      std::vector<TensorShape> shapes(2);
      // We create a shape vector that matches only profile 2.
      TF_CHECK_OK(
          TensorShapeUtils::MakeShape(std::vector<int32>{9, 9}, &shapes[0]));
      TF_CHECK_OK(
          TensorShapeUtils::MakeShape(std::vector<int32>{2}, &shapes[1]));
      Tensor shape_tensor(DT_INT32, shapes[1]);
      std::vector<int32> vals{1, 3};
      std::copy_n(vals.data(), vals.size(),
                  shape_tensor.flat<int32_t>().data());
      // DataVec names are not in used CollectShapeValues, only the order
      // matters.
      DataVec shape_values{{"one", {}}, {"two", shape_tensor}};
      TF_CHECK_OK(resource->profiles_.CollectShapeValues(shape_values));
      EXPECT_EQ(2, resource->profiles_.GetProfileNumber(shapes));
    }
  }
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

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
