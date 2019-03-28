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

// This file contains a C++ client for XRT, that communicates with a remote
// TensorFlow Eager server over gRPC.
//
// This client is a prototype and its API is not stable yet.
//
// TODO(phawkins): add support for multi-host configurations.
// * currently the API names accelerator devices using a flat space of device
//   ordinals, with no particular meaning to the device ordinals. The plan is to
//   instead to use the linearized device topology coordinates as device
//   ordinals.

#ifndef TENSORFLOW_COMPILER_XRT_CLIENT_XRT_CLIENT_H_
#define TENSORFLOW_COMPILER_XRT_CLIENT_XRT_CLIENT_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xrt/client/xrt_tf_client.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"

namespace tensorflow {

class XrtContext;

// RAII class that holds ownership of an XRT buffer.
class XrtBuffer {
 public:
  // Builds a new XrtBuffer from an XLA literal, copying the buffer to the
  // remote host.
  static xla::StatusOr<std::shared_ptr<XrtBuffer>> FromLiteral(
      const std::shared_ptr<XrtContext>& context, int xrt_device_ordinal,
      const xla::LiteralSlice& literal);

  // Converts an XrtBuffer to an XLA literal, copying the buffer from the remote
  // host. Blocks until the buffer is available.
  xla::StatusOr<xla::Literal> ToLiteral() const;

  // Deletes the remote buffer.
  void Delete();

  // Destructures a tuple-shaped buffer into its constituent pieces.
  xla::StatusOr<std::vector<std::shared_ptr<XrtBuffer>>> DestructureTuple();

  // TODO(phawkins): add a static method for building tuples of buffers.

  // TODO(phawkins): add a mechanism for converting XrtBuffers into remote
  // tensors and vice-versa for TF interoperability.

  XrtBuffer() = default;
  XrtBuffer(XrtTensorHandle handle, xla::Shape shape);
  ~XrtBuffer();  // Calls Delete().

  // A buffer reference is moveable but not copyable.
  XrtBuffer(const XrtBuffer&) = delete;
  XrtBuffer(XrtBuffer&&) = default;
  XrtBuffer& operator=(const XrtBuffer&) = delete;
  XrtBuffer& operator=(XrtBuffer&&) = default;

  const XrtTensorHandle& handle() const { return handle_; }

 private:
  // Tensor that contains the XRT allocation ID.
  XrtTensorHandle handle_;
  xla::Shape shape_;
};

// RAII class that holds ownership of an XRT executable.
class XrtExecutable {
 public:
  // Constructs an XrtExecutable by compiling a program.
  // `xrt_device_ordinal` must be the ordinal of a device known to XrtContext
  // on which the compile operator should be placed.
  // `hlo_module_proto` is the serialized HLO program to compile.
  // `argument_shapes` and `result_shape` describe the shapes of the
  // arguments/result and their layout.
  // `device_assignment` is the set of devices to which compilation should be
  // targeted. The device numbers in the device assignment are the XRT device
  // ordinals.
  // TODO(phawkins): device assignments with more than one computation per
  // replica do not work yet, even though the API appears to support them.
  static xla::StatusOr<std::shared_ptr<XrtExecutable>> Compile(
      std::shared_ptr<XrtContext> context,
      const xla::HloModuleProto& hlo_module_proto,
      const std::vector<xla::Shape>& argument_shapes,
      const xla::Shape& result_shape, xla::DeviceAssignment device_assignment);

  explicit XrtExecutable(std::shared_ptr<XrtContext> context,
                         XrtTensorHandle handles, xla::ProgramShape shape,
                         xla::DeviceAssignment device_assignment);
  ~XrtExecutable();  // Calls Delete().

  // Deletes the XrtExecutable.
  void Delete();

  // Runs the executable. Simplified API without replication or model
  // parallelism.
  xla::StatusOr<std::shared_ptr<XrtBuffer>> Execute(
      const std::vector<std::shared_ptr<XrtBuffer>>& args);

  // General API that runs replicated, model-parallel computations.
  //
  // Arguments are indexed by [computation][replica][arg]. Since each
  // computation may have a different arity, we use a Span<Array2D> to represent
  // a possibly ragged array.
  //
  // Return values are indexed by [computation][replica]. XLA computations
  // always have exactly one return value, so there is no possibility of
  // raggedness.
  xla::StatusOr<xla::Array2D<std::shared_ptr<XrtBuffer>>> ExecuteReplicated(
      absl::Span<const xla::Array2D<std::shared_ptr<XrtBuffer>>> args);

  // Moveable but not copyable.
  XrtExecutable(const XrtExecutable&) = delete;
  XrtExecutable(XrtExecutable&&) = default;
  XrtExecutable& operator=(const XrtExecutable&) = delete;
  XrtExecutable& operator=(XrtExecutable&&) = default;

  const xla::DeviceAssignment& device_assignment() const {
    return device_assignment_;
  }

 private:
  std::shared_ptr<XrtContext> context_;

  // A copy of the executable's handle in host memory. If the computation is
  // unreplicated, this lives on the target device. If the computation is
  // replicated, this lives on the CPU device.
  XrtTensorHandle handle_;
  xla::ProgramShape shape_;

  // The TF device ordinal on which this handle was compiled and on which it
  // should be deleted.
  xla::DeviceAssignment device_assignment_;
};

// Manages an XRT session.
//
// The XrtTfClient/XrtTfContext classes wrap the TensorFlow API more directly,
// without any XRT-specific knowledge. The higher level XrtClient
// adds XRT-specific functionality on top.
//
// It is intended that all clients talking to the same XRT session use the same
// XrtContext and that objects such as buffers and executables must not be
// shared between XrtContexts. However, clients may run non-XRT TensorFlow ops
// using the XrtTfContext that underlies an XrtContext.
//
// TODO(phawkins): Currently this code only supports a single remote host; each
// XrtContext communicates via a single XrtTfContext. The plan is to support
// multihost configurations (e.g., TPU pods) in the future, in which case
// XrtContext will be extended to have one XrtTfContext per remote host.
//
// TODO(phawkins): This API is intended to be thread-safe, but this is untested.
class XrtContext {
 public:
  // Creates an XrtContext. Fails if no accelerators of 'device_type' are found.
  static xla::StatusOr<std::shared_ptr<XrtContext>> Create(
      std::shared_ptr<XrtTfContext> tf_context, string device_type);

  // Use Create() instead.
  XrtContext(std::shared_ptr<XrtTfContext> tf_context, string device_type);

  // Returns the number of accelerator devices of 'device_type'.
  int device_count() const;

  const std::shared_ptr<XrtTfContext>& tf_context() const {
    return tf_context_;
  }
  const std::vector<int>& tf_device_ids() const { return tf_device_ids_; }

  const std::vector<
      xrt::DeviceAssignment::ComputationDevice::DeviceMeshCoordinates>&
  device_mesh_coordinates() const {
    return device_mesh_coordinates_;
  }

 private:
  friend class XrtExecutable;

  const std::shared_ptr<XrtTfContext> tf_context_;
  const string device_type_;  // Type of accelerator device to use (e.g., TPU)

  // Initializes TPU devices. Synchronous; called by Create().
  Status InitializeTPU();

  // IDs of devices of type `device_type_` in `tf_context_`.
  std::vector<int> tf_device_ids_;

  // Device coordinates of each device, indexed by XRT device ordinal.
  std::vector<xrt::DeviceAssignment::ComputationDevice::DeviceMeshCoordinates>
      device_mesh_coordinates_;

  // Returns the name of a function that launches a replicated computation
  // with input arity `input_arity` and device assignment `device_assignment`.
  xla::StatusOr<string> GetExecuteReplicatedFunction(
      absl::Span<const int> input_arity,
      const xla::DeviceAssignment& device_assignment);

  struct ExecuteReplicatedKey {
    ExecuteReplicatedKey(absl::Span<const int> input_arity,
                         xla::DeviceAssignment device_assignment);
    std::vector<int> input_arity;
    xla::DeviceAssignment device_assignment;
    bool operator==(const ExecuteReplicatedKey& other) const;
  };
  template <typename H>
  friend H AbslHashValue(H h, const ExecuteReplicatedKey& key);

  absl::Mutex mu_;
  absl::flat_hash_map<ExecuteReplicatedKey, string> replicated_fns_
      GUARDED_BY(mu_);
};

template <typename H>
H AbslHashValue(H h, const XrtContext::ExecuteReplicatedKey& key) {
  h = H::combine_contiguous(std::move(h), key.input_arity.data(),
                            key.input_arity.size());
  return H::combine_contiguous(std::move(h), key.device_assignment.data(),
                               key.device_assignment.num_elements());
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_CLIENT_XRT_CLIENT_H_
