/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_OPS_TESTUTIL_H_
#define TENSORFLOW_CORE_KERNELS_OPS_TESTUTIL_H_

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace test {

inline void SetOutputAttrs(OpKernelContext::Params* params,
                           std::vector<AllocatorAttributes>* attrs) {
  attrs->clear();
  for (int index = 0; index < params->op_kernel->num_outputs(); index++) {
    AllocatorAttributes attr;
    const bool on_host =
        (params->op_kernel->output_memory_types()[index] == HOST_MEMORY);
    attr.set_on_host(on_host);
    attrs->push_back(attr);
  }
  params->output_attr_array = attrs->data();
}

}  // namespace test

// Helpful functions to test operators.
//
// This class will eventually be replaced / heavily modified
// to use the BrainClient interface.
class OpsTestBase : public ::testing::Test {
 public:
  OpsTestBase() : device_type_(DEVICE_CPU) {
    auto device =
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0");
    CHECK(device) << "Could not create CPU device";

    device_ = device.get();
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(device));

    allocator_ = device_->GetAllocator(AllocatorAttributes());

    flib_def_ = absl::make_unique<FunctionLibraryDefinition>(
        OpRegistry::Global(), FunctionDefLibrary{});
    pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, flib_def_.get(), OptimizerOptions());
  }

  ~OpsTestBase() override {
    for (auto& temp : tensors_) {
      delete temp;
    }
    for (auto& temp : managed_outputs_) {
      delete temp;
    }
    tensors_.clear();
    managed_outputs_.clear();
    context_.reset(nullptr);
    params_.reset(nullptr);
  }

  // Allow kernel unit tests to run on GPU
  void SetDevice(const DeviceType& device_type, std::unique_ptr<Device> device);

  void set_node_def(const NodeDef& node_def) { node_def_.CopyFrom(node_def); }

  // Clients can manipulate the underlying NodeDef via this accessor.
  NodeDef* node_def() { return &node_def_; }

  // Initializes an operator that takes in 'input_types' as input
  // and output types as output.
  //
  // Returns the status of initialization.
  Status InitOp() { return InitOpWithGraphVersion(TF_GRAPH_DEF_VERSION); }

  // Only use this directly if you have a deprecated op that you need to test.
  Status InitOpWithGraphVersion(int graph_def_version) {
    Status status;
    kernel_ = CreateOpKernel(device_type_, device_, allocator(), node_def_,
                             graph_def_version, &status);
    if (kernel_ != nullptr) input_types_ = kernel_->input_types();
    return status;
  }

  // Adds an input for every element described by the shape.
  // 'input_mapping' maps an index (0...NumElements(shape)) to a
  // value.
  //
  // TODO(vrv): Replace with something like a BrainClient Feed.
  template <typename T>
  void AddInput(const TensorShape& shape, std::function<T(int)> input_mapping) {
    test::FillFn(AddInput(DataTypeToEnum<T>::v(), shape), input_mapping);
  }

  // Like AddInput but takes in an explicit arrayslice of data.
  template <typename T>
  void AddInputFromArray(const TensorShape& shape,
                         const gtl::ArraySlice<T>& data) {
    test::FillValues<T>(AddInput(DataTypeToEnum<T>::v(), shape), data);
  }

  // Convenience function to add an input and populate it with the elements from
  // an initializer list converting the types as needed.
  template <typename T, typename SrcType>
  void AddInputFromList(const TensorShape& shape,
                        std::initializer_list<SrcType> data) {
    test::FillValues<T>(AddInput(DataTypeToEnum<T>::v(), shape), data);
  }

  // Adds a Resource type as input. If <container> is empty, uses the default
  // container name.
  template <typename T>
  void AddResourceInput(const string& container, const string& name,
                        T* resource) {
    CHECK_GT(input_types_.size(), inputs_.size())
        << "Adding more inputs than types; perhaps you need to call MakeOp";
    ResourceMgr* rm = device_->resource_manager();
    std::string container_name =
        container == "" ? rm->default_container() : container;
    EXPECT_TRUE(rm->Create(container_name, name, resource).ok());
    TypeIndex type_index = MakeTypeIndex<T>();
    ResourceHandle handle;
    handle.set_device(device_->name());
    handle.set_container(container_name);
    handle.set_name(name);
    handle.set_hash_code(type_index.hash_code());
    handle.set_maybe_type_name(type_index.name());
    Tensor* input = new Tensor(allocator(), DT_RESOURCE, TensorShape({}));
    input->scalar<ResourceHandle>()() = handle;
    tensors_.push_back(input);
    inputs_.push_back({nullptr, input});
  }

  // Runs an operation producing 'num_outputs' outputs.
  //
  // Returns the context's status after running the operation.
  Status RunOpKernel() {
    // Make sure the old OpKernelContext is deleted before the Params
    // it was using.
    context_.reset(nullptr);

    // Delete the output copies from previous runs.
    for (auto& temp : managed_outputs_) {
      delete temp;
    }
    managed_outputs_.clear();
    managed_outputs_.resize(0);

    params_.reset(new OpKernelContext::Params);
    params_->device = device_;
    params_->frame_iter = FrameAndIter(0, 0);
    params_->inputs = &inputs_;
    params_->op_kernel = kernel_.get();
    step_container_.reset(new ScopedStepContainer(0, [](const string&) {}));
    params_->step_container = step_container_.get();
    std::vector<AllocatorAttributes> attrs;
    test::SetOutputAttrs(params_.get(), &attrs);
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params_->slice_reader_cache = &slice_reader_cache_wrapper;
    params_->resource_manager = device_->resource_manager();
    params_->function_library = pflr_->GetFLR(device_->name());

    context_.reset(new OpKernelContext(params_.get()));
    device_->Compute(kernel_.get(), context_.get());
    return context_->status();
  }

  // Returns the tensor input for 'input_index'.
  //
  // REQUIRES: 0 <= input_index < context_->num_inputs()
  const Tensor& GetInput(int input_index) const {
    CHECK_LT(input_index, context_->num_inputs());
    CHECK(!IsRefType(context_->input_dtype(input_index)));
    return context_->input(input_index);
  }

  TensorValue mutable_input(int input_index) {
    CHECK_LT(input_index, inputs_.size());
    return inputs_[input_index];
  }
  // Returns the tensor output for 'output_index'.
  //
  // REQUIRES: 0 <= output_index < context_->num_outputs()
  Tensor* GetOutput(int output_index);

  Allocator* allocator() { return allocator_; }

  const DataTypeVector& output_types() const { return kernel_->output_types(); }

 protected:
  Tensor* AddInput(DataType dtype, const TensorShape& shape) {
    CHECK_GT(input_types_.size(), inputs_.size())
        << "Adding more inputs than types; perhaps you need to call MakeOp";
    bool is_ref = IsRefType(input_types_[inputs_.size()]);
    Tensor* input = new Tensor(allocator(), dtype, shape);
    tensors_.push_back(input);
    if (is_ref) {
      CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]), dtype);
      inputs_.push_back({&lock_for_refs_, input});
    } else {
      CHECK_EQ(input_types_[inputs_.size()], dtype);
      inputs_.push_back({nullptr, input});
    }
    return input;
  }

  // device_mgr_ owns device_.
  std::unique_ptr<DeviceMgr> device_mgr_;
  Device* device_;

  // The device allocator, or the managed_allocator_ below if running on GPU.
  Allocator* allocator_;

  std::unique_ptr<OpKernel> kernel_;
  std::unique_ptr<ScopedStepContainer> step_container_;
  NodeDef node_def_;
  DataTypeVector input_types_;
  DeviceType device_type_;

  mutex lock_for_refs_;  // Used as the Mutex for inputs added as refs

  gtl::InlinedVector<TensorValue, 4> inputs_;
  // Owns Tensors.
  std::vector<Tensor*> tensors_;
  // Copies of the outputs in unified memory (host and device accessible).
  std::vector<Tensor*> managed_outputs_;

  std::unique_ptr<OpKernelContext::Params> params_;
  std::unique_ptr<OpKernelContext> context_;
  // Unified memory allocator, only used when running on GPU.
  std::unique_ptr<Allocator> managed_allocator_;

  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(OpsTestBase);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_OPS_TESTUTIL_H_
