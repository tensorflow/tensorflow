/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EAGER_CONTEXT_INTERFACE_H_
#define TENSORFLOW_C_EAGER_CONTEXT_INTERFACE_H_

#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/operation_interface.h"
#include "tensorflow/c/eager/tensor_handle_interface.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_api.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {

// Abstract interface to a context.
//
// A context is responsible for creating key objects such as Tensors,
// TensorHandles & Operations.
class AbstractContextInterface {
 public:
  // Release any underlying resources, including the interface object.
  //
  // WARNING: The destructor of this class is marked as protected to disallow
  // clients from directly destroying this object since it may manage it's own
  // lifetime through ref counting. Thus clients MUST call Release() in order to
  // destroy an instance of this class.
  virtual void Release() = 0;

  // Optimized scalar creation functions
  virtual AbstractTensorInterface* CreateInt64Scalar(int64 value) = 0;
  virtual AbstractTensorInterface* CreateUint64Scalar(uint64 value) = 0;
  virtual AbstractTensorInterface* CreateInt32Scalar(int32 value) = 0;
  virtual AbstractTensorInterface* CreateFloatScalar(float value) = 0;
  virtual AbstractTensorInterface* CreateDoubleScalar(double value) = 0;
  virtual AbstractTensorInterface* CreateHalfScalar(Eigen::half value) = 0;
  virtual AbstractTensorInterface* CreateStringScalar(tstring value) = 0;
  virtual AbstractTensorInterface* CreateComplex128Scalar(complex128 value) = 0;
  virtual AbstractTensorInterface* CreateBoolScalar(bool value) = 0;

  // Tensor creation functions
  virtual AbstractTensorInterface* CreateTensor(
      DataType dtype, absl::Span<const int64> dim_sizes) = 0;

  typedef void (*MemoryReleaser)(void* data, size_t len, void* arg);

  // Create a tensor instance from the given data buffer and description.
  // `memory_releaser` will be called on destruction, and it's responsible for
  // cleaning up the underlying buffer. `convert_string` indicates whether it
  // has to handle tstring conversion. Expected to be removed once tstring
  // migration is done.
  virtual AbstractTensorInterface* CreateTensor(DataType dtype,
                                                const int64_t* dims,
                                                int num_dims, void* data,
                                                size_t len, bool convert_string,
                                                MemoryReleaser memory_releaser,
                                                void* memory_releaser_arg) = 0;

  // Create a handle to wrap and manage a Tensor
  virtual AbstractTensorHandleInterface* CreateLocalHandle(
      AbstractTensorInterface* t) = 0;
  // Copy the handle to another device.
  virtual AbstractTensorHandleInterface* CopyTensorHandleToDevice(
      AbstractTensorHandleInterface* handle, const char* device_name,
      Status* status) = 0;

  // Create an operation to perform op execution
  virtual AbstractOperationInterface* CreateOperation() = 0;

  // Load a SavedModelAPI object from the given directory and tags
  virtual std::unique_ptr<SavedModelAPI> LoadSavedModelAPI(
      const std::string& directory,
      const absl::optional<std::unordered_set<std::string>>& tags,
      tensorflow::Status* status) = 0;

  // List attributes of available devices
  virtual void ListDevices(std::vector<DeviceAttributes>* devices) = 0;

  virtual void ClearCachesAndThreadExecutors() = 0;

  // Initialize the step resource container for a training step. This is used
  // in current TF runtime. For tfrt, it is used by fallback op handler.
  virtual void StartStep() = 0;
  // Destroy the step resource container for a training step.
  virtual void EndStep() = 0;

  // Block until all pending nodes are finished.
  virtual Status AsyncWait() = 0;

  // Add a function (serialized FunctionDef protocol buffer) so that it can
  // be executed as an op. Return error if the function with the same name
  // already exists.
  virtual Status AddFunctionDef(const FunctionDef& fdef) = 0;
  // Remove a function. 'func' argument is the name of a previously added
  // FunctionDef. The name is in fdef.signature.name.
  virtual Status RemoveFunction(const string& func) = 0;

 protected:
  virtual ~AbstractContextInterface() {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_CONTEXT_INTERFACE_H_
