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

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/type_index.h"
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
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace test {

void SetOutputAttrs(OpKernelContext::Params* params,
                    std::vector<AllocatorAttributes>* attrs);

}  // namespace test

// Helpful functions to test operators.
//
// This class will eventually be replaced / heavily modified
// to use the BrainClient interface.
class OpsTestBase : public ::testing::Test {
 public:
  OpsTestBase();

  ~OpsTestBase() override;

  // Allow kernel unit tests to run on GPU
  void SetDevice(const DeviceType& device_type, std::unique_ptr<Device> device);

  void set_node_def(const NodeDef& node_def);

  // Clients can manipulate the underlying NodeDef via this accessor.
  NodeDef* node_def();

  // Initializes an operator that takes in 'input_types' as input
  // and output types as output.
  //
  // Returns the status of initialization.
  Status InitOp();

  // Only use this directly if you have a deprecated op that you need to test.
  Status InitOpWithGraphVersion(int graph_def_version);

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
                         const gtl::ArraySlice<T> data) {
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
        container.empty() ? rm->default_container() : container;
    EXPECT_TRUE(rm->Create(container_name, name, resource).ok());
    AddResourceInputInternal(container_name, name, TypeIndex::Make<T>());
  }

  // Runs an operation producing 'num_outputs' outputs.
  //
  // Returns the context's status after running the operation.
  Status RunOpKernel();

  // Returns the tensor input for 'input_index'.
  //
  // REQUIRES: 0 <= input_index < context_->num_inputs()
  const Tensor& GetInput(int input_index) const;

  TensorValue mutable_input(int input_index);

  // Returns the tensor output for 'output_index'.
  //
  // REQUIRES: 0 <= output_index < context_->num_outputs()
  Tensor* GetOutput(int output_index);

  Allocator* allocator();

  OpKernel* op_kernel();

  const DataTypeVector& output_types() const;

  void set_session_metadata(SessionMetadata session_metadata) {
    session_metadata_ = std::move(session_metadata);
  }

  const SessionMetadata& session_metadata() const { return session_metadata_; }

 protected:
  void CreateContext();
  Tensor* AddInput(DataType dtype, const TensorShape& shape);
  void AddResourceInputInternal(const std::string& container_name,
                                const std::string& name,
                                const TypeIndex& type_index);

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

  // AllocatorAttributes for the allocators of the outputs.
  std::vector<AllocatorAttributes> out_alloc_attrs_;
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper_;
  CancellationManager default_cancellation_manager_;
  std::unique_ptr<OpKernelContext::Params> params_;
  std::unique_ptr<OpKernelContext> context_;
  // Unified memory allocator, only used when running on GPU.
  std::unique_ptr<Allocator> managed_allocator_;

  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;

  SessionMetadata session_metadata_;

 private:
  OpsTestBase(const OpsTestBase&) = delete;
  void operator=(const OpsTestBase&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_OPS_TESTUTIL_H_
