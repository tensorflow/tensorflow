/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_OPS_TESTUTIL_H_
#define TENSORFLOW_KERNELS_OPS_TESTUTIL_H_

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
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

// Return a NodeDef with the specified name/op/inputs.
NodeDef Node(const string& name, const string& op,
             const std::vector<string>& inputs);

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
  params->output_attr_array = gtl::vector_as_array(attrs);
}

}  // namespace test

// Helpful functions to test operators.
//
// This class will eventually be replaced / heavily modified
// to use the BrainClient interface.
class OpsTestBase : public ::testing::Test {
 public:
  OpsTestBase() : device_type_(DEVICE_CPU) {
    device_.reset(
        DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));
    CHECK(device_.get()) << "Could not create CPU device";
  }

  ~OpsTestBase() override {
    gtl::STLDeleteElements(&tensors_);
    context_.reset(nullptr);
    params_.reset(nullptr);
  }

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
    kernel_ = CreateOpKernel(device_type_, device_.get(), allocator(),
                             node_def_, graph_def_version, &status);
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
    CHECK_GT(input_types_.size(), inputs_.size())
        << "Adding more inputs than types; perhaps you need to call MakeOp";
    bool is_ref = IsRefType(input_types_[inputs_.size()]);
    Tensor* input = new Tensor(device_->GetAllocator(AllocatorAttributes()),
                               DataTypeToEnum<T>::v(), shape);
    test::FillFn(input, input_mapping);
    tensors_.push_back(input);
    if (is_ref) {
      CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]),
               DataTypeToEnum<T>::v());
      inputs_.push_back({&lock_for_refs_, input});
    } else {
      CHECK_EQ(input_types_[inputs_.size()], DataTypeToEnum<T>::v());
      inputs_.push_back({nullptr, input});
    }
  }

  // Like AddInput but takes in an explicit arrayslice of data.
  template <typename T>
  void AddInputFromArray(const TensorShape& shape,
                         const gtl::ArraySlice<T>& data) {
    CHECK_GT(input_types_.size(), inputs_.size())
        << "Adding more inputs than types; perhaps you need to call MakeOp";
    bool is_ref = IsRefType(input_types_[inputs_.size()]);
    Tensor* input = new Tensor(device_->GetAllocator(AllocatorAttributes()),
                               DataTypeToEnum<T>::v(), shape);
    test::FillValues<T>(input, data);
    tensors_.push_back(input);
    if (is_ref) {
      CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]),
               DataTypeToEnum<T>::v());
      inputs_.push_back({&lock_for_refs_, input});
    } else {
      CHECK_EQ(input_types_[inputs_.size()], DataTypeToEnum<T>::v());
      inputs_.push_back({nullptr, input});
    }
  }

  // Runs an operation producing 'num_outputs' outputs.
  //
  // Returns the context's status after running the operation.
  Status RunOpKernel() {
    // Make sure the old OpKernelContext is deleted before the Params
    // it was using.
    context_.reset(nullptr);

    params_.reset(new OpKernelContext::Params);
    params_.get()->device = device_.get();
    params_.get()->frame_iter = FrameAndIter(0, 0);
    params_.get()->inputs = &inputs_;
    params_.get()->op_kernel = kernel_.get();
    std::vector<AllocatorAttributes> attrs;
    test::SetOutputAttrs(params_.get(), &attrs);
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params_.get()->slice_reader_cache = &slice_reader_cache_wrapper;

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
  Tensor* GetOutput(int output_index) {
    CHECK_LT(output_index, context_->num_outputs());
    return context_->mutable_output(output_index);
  }

  Allocator* allocator() {
    return device_->GetAllocator(AllocatorAttributes());
  }

  const DataTypeVector& output_types() const { return kernel_->output_types(); }

 protected:
  std::unique_ptr<Device> device_;

  std::unique_ptr<OpKernel> kernel_;
  NodeDef node_def_;
  DataTypeVector input_types_;
  DeviceType device_type_;

  mutex lock_for_refs_;  // Used as the Mutex for inputs added as refs

  gtl::InlinedVector<TensorValue, 4> inputs_;
  // Owns Tensors.
  std::vector<Tensor*> tensors_;

  std::unique_ptr<OpKernelContext::Params> params_;
  std::unique_ptr<OpKernelContext> context_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(OpsTestBase);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_OPS_TESTUTIL_H_
