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

// This file defines helper routines for the XLA device.

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_HELPERS_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_HELPERS_H_

#include <string>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/mhlo_to_hlo/layout_util.h"
#include "xla/service/computation_placer.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

using XlaLayoutPreference = mlir::XlaLayoutPreference;

inline std::string GetDeviceToHostChannelName(absl::string_view channel_key,
                                              int index) {
  return absl::StrCat(channel_key, "_dtoh_", index);
}
inline std::string GetHostToDeviceChannelName(absl::string_view channel_key,
                                              int index) {
  return absl::StrCat(channel_key, "_htod_", index);
}

// Helper methods for building XLA computations.
class XlaHelpers {
 public:
  // Returns a handle representing the zero value of a scalar
  // element of data_type.
  static xla::XlaOp Zero(xla::XlaBuilder* b, DataType data_type);

  // Returns a handle representing the one value of a scalar
  // element of data_type.
  static xla::XlaOp One(xla::XlaBuilder* b, DataType data_type);

  // Returns a handle representing the given value of an integer scalar
  // element of data_type.
  // Note that unlike One and Zero, does not work on boolean types.
  static xla::XlaOp IntegerLiteral(xla::XlaBuilder* b, DataType data_type,
                                   int64_t value);

  // Returns a handle representing the given value of a floating-point scalar
  // element of data_type.
  static xla::XlaOp FloatLiteral(xla::XlaBuilder* b, DataType data_type,
                                 double value);

  // Reshapes literal 'input' to have 'shape'. Both the original shape and
  // 'shape' must contain the same number of elements.
  static absl::Status ReshapeLiteral(const xla::Literal& input,
                                     absl::Span<const int64_t> shape,
                                     xla::Literal* output);

  // Converts `indices` into a one-hot representation. `depth` is the size
  // of the new axis to add. `axis` is the position at which to add the new
  // axis. `indices_shape` is the shape of `indices`. `on_value` and
  // `off_value` represent the values to use for the on and off positions,
  // respectively.
  static absl::Status OneHot(xla::XlaBuilder* builder, int64_t depth, int axis,
                             DataType index_type,
                             const TensorShape& indices_shape,
                             xla::XlaOp indices, xla::XlaOp on_value,
                             xla::XlaOp off_value, xla::XlaOp* one_hot);

  // Certain DataTypes should use increased precision DataTypes when performing
  // reductions.  This function remaps a given DataType to a higher precision
  // DataType if needed.
  static DataType SumAccumulationType(const DataType& dtype);

  // A helper for creating a ConvertElementType xla op given a DataType rather
  // than the xla::PrimitiveType.
  static xla::XlaOp ConvertElementType(xla::XlaOp operand,
                                       const DataType new_element_type);

  typedef std::function<absl::StatusOr<xla::Shape>(const TensorShape&, DataType,
                                                   bool, XlaLayoutPreference)>
      ShapeRepresentationFn;
};

// Creates an identity shape representation function.
XlaHelpers::ShapeRepresentationFn IdentityShapeRepresentationFn();

struct XlaOutputDescription {
  // Type and shape of the output. The shape is the unflattened shape.
  // When `type` is DT_RESOURCE, `shape` is the shape of the resource
  // variable's value.
  DataType type;
  TensorShape shape;

  // Constant output value, if known to be constant at JIT compilation time.
  // 'Tensor' is in host memory.
  bool is_constant = false;
  Tensor constant_value;

  // When this output is a resource, i.e. `type == DT_RESOURCE`, this is
  // the index of the input that contains the resource.
  int input_index;

  // Whether this output is a TensorList.
  bool is_tensor_list = false;
};

// Describes a variable write side effect of the computation.
struct XlaResourceUpdate {
  // Index of the input that contains the variable resource to write to.
  int input_index;

  // Type and shape of the tensor to be written back.
  // The `shape` field has the same meaning as the Argument::shape field.
  DataType type;
  TensorShape shape;

  // Was the value of the variable modified by the computation?
  // (Always true, unless `return_updated_values_for_all_resources` is true.)
  bool modified;

  // If the resource is a TensorArray, the set of gradients read or written.
  std::set<string> tensor_array_gradients_accessed;
};

struct XlaCompilationResult {
  // Vector that maps from the parameters of the XLA computation to their
  // original argument positions. To handle compile-time constant inputs, the
  // parameters to the XLA computation may be a subset of the original
  // arguments. The relative ordering of parameters are maintained.
  std::vector<int> input_mapping;

  // Input shapes of the computation. If we are flattening inputs, these are
  // the flattened shapes.
  std::vector<xla::Shape> xla_input_shapes;

  // Output shape in XLA format. The output shape is always a tuple. If we
  // are flattening outputs, these are the flattened shapes.
  xla::Shape xla_output_shape;

  // TensorFlow shapes of outputs, together with the values of any
  // constant arguments. Vector indexed by Tensorflow _Retval number,
  // containing both constant and non-constant results.
  std::vector<XlaOutputDescription> outputs;

  // TensorFlow shapes and types of sends/recvs from HostCompute Ops to their
  // matching RecvAtHost/SendFromHost Ops in the outer graph.
  tf2xla::HostComputeMetadata host_compute_metadata;

  // Resources whose values were updated by the computation, ordered
  // by return value position (which is the same as the order the resources
  // were passed as arguments). Resource updates follow the non-constant
  // results in the outputs of XLA computation.
  std::vector<XlaResourceUpdate> resource_updates;

  // The XLA computation built from the tensorflow subgraph.
  std::shared_ptr<xla::XlaComputation> computation;

  // Meta-info about encountered collective ops.
  struct CollectiveInfo {
    int group_key;
    int group_size;
    int next_id;

    template <typename H>
    friend H AbslHashValue(H h, const CollectiveInfo& info) {
      return H::combine(std::move(h), info.group_key, info.group_size,
                        info.next_id);
    }

    friend bool operator==(const CollectiveInfo& lhs,
                           const CollectiveInfo& rhs) {
      return lhs.group_key == rhs.group_key &&
             lhs.group_size == rhs.group_size && lhs.next_id == rhs.next_id;
    }
  };

  // Information of the collectives encountered during the translation.
  std::optional<CollectiveInfo> collective_info;
};

// Resolves the device assignment based on CollectiveInfo.
// CollectiveInfo records collective ops in the cluster. Note that
// this relies on a rendezvous and blocks until all replicas are there.
//
// Takes several extra configuration objects by reference since
// xla::ExecutableRunOptions does not take ownership; these are configured and
// bundled into `run_options` if applicable.
absl::Status ResolveDeviceAssignment(
    OpKernelContext* ctx,
    const XlaCompilationResult::CollectiveInfo& collective_info,
    xla::ExecutableRunOptions& run_options,
    xla::DeviceAssignment& device_assignment,
    xla::gpu::GpuExecutableRunOptions& gpu_options);

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_HELPERS_H_
