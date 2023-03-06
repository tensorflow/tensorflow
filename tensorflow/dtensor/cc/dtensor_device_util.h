/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_CC_DTENSOR_DEVICE_UTIL_H_
#define TENSORFLOW_DTENSOR_CC_DTENSOR_DEVICE_UTIL_H_

#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/parallel_device/parallel_device_lib.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/small_constant_optimization.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/cc/tensor_with_layout.h"
#include "tensorflow/tsl/platform/fingerprint.h"
#include "tensorflow/tsl/platform/refcount.h"

namespace tensorflow {
namespace dtensor {

#define RETURN_STATUS(status, code, message)   \
  {                                            \
    TF_SetStatus((status), (code), (message)); \
    return;                                    \
  }

#define RETURN_C_STATUS_IF_NOT_OK(cpp_status, c_status)                   \
  {                                                                       \
    auto return_if_not_ok_status = (cpp_status);                          \
    if (!return_if_not_ok_status.ok()) {                                  \
      RETURN_STATUS((c_status),                                           \
                    static_cast<TF_Code>(return_if_not_ok_status.code()), \
                    return_if_not_ok_status.error_message().c_str());     \
    }                                                                     \
  }

// Using a counter to uniquify instead of a new block allows `var` to declare a
// new variable.
#define ASSIGN_OR_RETURN_C_STATUS(var, cpp_status, c_status)               \
  ASSIGN_OR_RETURN_C_STATUS_IMPL(                                          \
      TF_STATUS_MACROS_CONCAT_NAME(_dtensor_status_or_value, __COUNTER__), \
      var, cpp_status, c_status)

#define ASSIGN_OR_RETURN_C_STATUS_IMPL(statusor, var, cpp_status, c_status) \
  auto statusor = (cpp_status);                                             \
  RETURN_C_STATUS_IF_NOT_OK(statusor.status(), (c_status));                 \
  var = std::move(statusor.value());

struct TranslatedFunction {
  // Mesh for which specified function will run.
  Mesh function_mesh;

  // StatefulPartitionedCall op to run the mesh function.
  const Node* node_to_execute = nullptr;

  // Maps i-th local input index to input index in global graph.
  std::vector<int> input_index_map;

  // Maps i-th local output to output index of global graph.
  std::vector<int> output_index_map;

  std::string translated_function_name;
  // For resource ops, layouts of resource handles are inferred lazily
  // during SPMD expansion of resource assign ops. In that case,
  // inferred layouts of resource handles are attached to arg nodes
  // of the returned graph.
  std::map<int, Layout> resource_input_layouts;
  // Record some metadata for output of a shape op. This would help recover
  // local shape on future operations over the Tensor.
  std::map<int, Layout> shape_output_metadata;
  std::vector<Layout> output_layouts;
  // Local shapes inferred for function outputs; these may be partially known.
  std::vector<PartialTensorShape> local_output_shapes;
  // Output data types.
  std::vector<TF_DataType> output_dtypes;
};

struct ExecutionFunctions {
  // Stores information about all functions to execute for provided computation.
  std::vector<TranslatedFunction> function_list;
  // Number of device ids args added to translated functions.
  // During translation, we insert one device id arg node per mesh.
  // For a single mesh function, it equals 1.
  // For a multi-mesh function (e.g. pipelining), it equals the number of
  // meshes.
  int num_device_ids;

  // Mesh fingerprint of function_list. Set only when ExecutionFunctions refers
  // to a function for performance reason, since an eager op doesn't use it.
  uint64 function_mesh_fingerprint = 0;
};

struct DTensorOperation {
  // For all fields: not owned. lifetime covers the whole usage.
  const char* name;
  const FunctionDef* function_def;
  // Default mesh is used when Mesh Propagation does not identify a mesh
  // otherwise.
  const Mesh& default_mesh;
  inline bool is_func() const { return function_def != nullptr; }
};

// Contains a mesh bundled with a parallel device over all of the devices in
// that mesh.
class MeshWithParallelDevice {
 public:
  MeshWithParallelDevice(
      const Mesh& mesh_config,
      std::unique_ptr<parallel_device::ParallelDevice> parallel_device)
      : mesh_config_(mesh_config),
        parallel_device_(std::move(parallel_device)),
        // Device IDs are constructed lazily because we don't have a context
        // until we start executing ops.
        device_ids_tensor_(nullptr) {}

  // A parallel tensor containing scalar integer device IDs for underlying
  // devices, each placed on its corresponding device.
  //
  // TODO(allenl): It would be nice if DeviceID worked as an op inside the
  // function's graph. Then we wouldn't need to feed it as an argument.
  parallel_device::ParallelTensor* DeviceIDs(TFE_Context* context,
                                             TF_Status* status) const;
  const parallel_device::ParallelDevice& parallel_device() const {
    return *parallel_device_;
  }

  const dtensor::Mesh& mesh_config() const { return mesh_config_; }

 private:
  dtensor::Mesh mesh_config_;
  std::unique_ptr<parallel_device::ParallelDevice> parallel_device_;

  // Constructed lazily; contains a parallel tensor with scalar integer device
  // IDs for each device.
  mutable std::unique_ptr<parallel_device::ParallelTensor> device_ids_tensor_;
};

class TensorWithLayoutTf
    : public llvm::RTTIExtends<TensorWithLayoutTf, TensorWithLayout> {
 public:
  // Broadcast a single non-parallel tensor onto `mesh` with a fully replicated
  // sharding spec. Does not take ownership of `tensor`.
  static std::unique_ptr<TensorWithLayoutTf> Broadcast(
      TFE_Context* context, TFE_TensorHandle* tensor,
      const MeshWithParallelDevice& mesh,
      const std::string& dtensor_device_name, TF_Status* status);

  // Given an already-parallel tensor, wraps it with a mesh and a layout.
  static StatusOr<std::unique_ptr<TensorWithLayoutTf>> Wrap(
      std::unique_ptr<parallel_device::ParallelTensor> tensor, const Mesh& mesh,
      const Layout& layout);

  // Creates a dummy TensorWithLayoutTf without holding a ParallelTensor.
  static std::unique_ptr<TensorWithLayoutTf> Dummy(
      const std::vector<int64_t>& local_shape, TF_DataType dtype,
      const Mesh& mesh, const Layout& layout);

  ~TensorWithLayoutTf() override = default;

  const Layout& layout() const override { return layout_; }

  TensorType tensor_type() const override { return TensorType::kDense; }

  TF_DataType dtype() const override {
    return dtype_.has_value() ? dtype_.value() : tensor_->dtype();
  }

  // Encodes the NodeDef via provided builder, if applicable.
  void EncodeAttributes(tensorflow::NodeDefBuilder& builder) const override {}

  tensorflow::Fprint128 CacheKey() const override;

  TFE_TensorHandle* get_tensor(size_t index) const override {
    return tensor_->tensor(index);
  }

  size_t num_tensors() const override { return tensor_->num_tensors(); }

  parallel_device::ParallelTensor* tensor() const { return tensor_.get(); }

  std::string SummarizeValue() const override;

  std::string DebugString() const override;

  const Mesh& mesh() const override { return mesh_; }

  std::vector<int64_t> global_shape() const override {
    return layout_.GlobalShapeFromLocalShape(local_shape_);
  }

  ConstValueNode* const_value_node() const override {
    return const_value_node_.get();
  }

  // llvm::RTTIExtends ID.
  static char ID;  // NOLINT

 protected:
  TensorWithLayoutTf(std::unique_ptr<parallel_device::ParallelTensor> tensor,
                     const Mesh& mesh, const Layout& layout,
                     const std::vector<int64_t>& local_shape,
                     std::optional<TF_DataType> dtype = std::nullopt,
                     std::optional<NodeDef> const_value = std::nullopt)
      : tensor_(std::move(tensor)),
        layout_(layout),
        mesh_(mesh),
        local_shape_(local_shape),
        dtype_(dtype) {
    const_value_node_ = std::make_unique<ConstValueNode>(const_value);
  }

  std::unique_ptr<parallel_device::ParallelTensor> tensor_;

  Layout layout_;

  const Mesh& mesh_;

  // The local shape of tensors placed on each of `tensor_`'s component devices.
  std::vector<int64_t> local_shape_;

  std::optional<TF_DataType> dtype_;

  std::unique_ptr<ConstValueNode> const_value_node_;
};

// Extension of TensorWithLayout which holds resource handle with layout.
//
// The major differences are
// 1. The layout, shape, dtype are lazily set as they are unavailable upon
//    creation.
// 2. Small const optimization should be disabled.
class ResourceHandleWithLayout
    : public llvm::RTTIExtends<ResourceHandleWithLayout, TensorWithLayoutTf> {
 public:
  // Similar to `Wrap` in `TensorWithLayoutTf` but for resource handle.
  static StatusOr<std::unique_ptr<ResourceHandleWithLayout>> Wrap(
      std::unique_ptr<parallel_device::ParallelTensor> tensor, const Mesh& mesh,
      const Layout& layout);

  // Similar to `Dummy` in `TensorWithLayoutTf` but for resource handle.
  static std::unique_ptr<ResourceHandleWithLayout> Dummy(
      const std::vector<int64_t>& local_shape, const Mesh& mesh,
      const Layout& layout);

  // The layout of uninitialized resource tensors, or the layout of the tensor
  // contained in an initialized resource.
  const Layout& layout() const override {
    return dereferenced_layout_.has_value() ? dereferenced_layout_.value()
                                            : layout_;
  }

  TensorType tensor_type() const override { return TensorType::kResource; }

  TF_DataType dtype() const override {
    return dtype_.has_value() ? dtype_.value() : tensor_->dtype();
  }

  void EncodeAttributes(tensorflow::NodeDefBuilder& builder) const override;

  tensorflow::Fprint128 CacheKey() const override;

  // Updates the layout for the tensors.
  tsl::Status UpdateLayout(const Layout& new_layout);

  // Updates the element layouts for the tensors.
  tsl::Status UpdateElementLayouts(const std::vector<Layout>& layouts) {
    dereferenced_element_layouts_.emplace(layouts);
    return tsl::OkStatus();
  }

  // Updates the local shape and dtype of the tensors.
  tsl::Status UpdateShapeAndDType(const TensorShapeProto& shape,
                                  const DataType& dtype) {
    set_dereferenced_shape(shape);
    set_dereferenced_dtype(dtype);
    return tsl::OkStatus();
  }

  // Updates the attributes for the tensors.
  tsl::Status UpdateAttrs(const EmbeddingResourceAttrs& attrs);

  ConstValueNode* const_value_node() const override { return nullptr; }

  void UpdateDirtyness(bool is_dirty, TF_Status* status) {
    if (!attrs_.has_value()) {
      TF_SetStatus(status, TF_INTERNAL,
                   "Attempt to update dirtyness on non embedding resource");
    }
    attrs_.value().is_dirty = is_dirty;
  }

  void set_dereferenced_shape(const TensorShapeProto& shape) {
    dereferenced_shape_.emplace(shape);
  }
  void set_dereferenced_dtype(const DataType& dtype) {
    dereferenced_dtype_.emplace(dtype);
  }

  const std::optional<std::vector<Layout>>& dereferenced_element_layouts()
      const {
    return dereferenced_element_layouts_;
  }

  const std::optional<TensorShapeProto>& dereferenced_shape() const {
    return dereferenced_shape_;
  }
  const std::optional<DataType>& dereferenced_dtype() const {
    return dereferenced_dtype_;
  }

  // Gets the resource input attributes for embedding inputs.
  const std::optional<EmbeddingResourceAttrs>& attrs() const { return attrs_; }

  // llvm::RTTIExtends ID.
  static char ID;  // NOLINT

 private:
  ResourceHandleWithLayout(
      std::unique_ptr<parallel_device::ParallelTensor> tensor, const Mesh& mesh,
      const Layout& layout, const std::vector<int64_t>& local_shape)
      : llvm::RTTIExtends<ResourceHandleWithLayout, TensorWithLayoutTf>(
            std::move(tensor), mesh, layout, local_shape, TF_RESOURCE) {}

  // The layout of the tensor pointed to by this handle, if any.
  std::optional<Layout> dereferenced_layout_;
  // The layouts of the tensors emitted by this resource handle if it is an
  // iterator resource.
  std::optional<std::vector<Layout>> dereferenced_element_layouts_;
  // The shape and dtype of the tensor pointed to by this resource tensor.
  std::optional<TensorShapeProto> dereferenced_shape_;
  std::optional<DataType> dereferenced_dtype_;

  // Resource input attributes for embedding inputs.
  std::optional<EmbeddingResourceAttrs> attrs_;  // NOLINT
};

// TensorWithLayout for SparseTensors.
//
// The main difference between this and TensorWithLayout is this
// contains 3 lists of tensors as opposed to one (values, indices, shapes).
// The shapes of the SparseTensors will always be the dense view of the shapes,
// and thus will have no difference with the TensorWithLayout in terms of
// shapes.
class SparseTensorWithLayout
    : public llvm::RTTIExtends<SparseTensorWithLayout, TensorWithLayoutTf> {
 public:
  static StatusOr<std::unique_ptr<SparseTensorWithLayout>> Wrap(
      std::unique_ptr<parallel_device::ParallelTensor> indices_tensor,
      std::unique_ptr<parallel_device::ParallelTensor> values_tensor,
      std::unique_ptr<parallel_device::ParallelTensor> shapes_tensor,
      const Mesh& mesh, const Layout& layout,
      const std::vector<int64_t>& local_shape);

  // A dummy TensorWithLayout without holding a ParallelTensor.
  static std::unique_ptr<SparseTensorWithLayout> Dummy(
      const std::vector<int64_t>& local_shape, const Mesh& mesh,
      const Layout& layout) {
    return absl::WrapUnique(new SparseTensorWithLayout(
        /*indices=*/nullptr, /*values=*/nullptr, /*dense_shapes=*/nullptr, mesh,
        layout, local_shape));
  }

  // Add attribute '_sparse' to the NodeDefBuilder so that the mlir::Value
  // that originate from SparseTensorWithLayout are marked as '_sparse'.
  void EncodeAttributes(tensorflow::NodeDefBuilder& builder) const override {
    builder.Attr("_sparse", true);
  }

  TensorType tensor_type() const override { return TensorType::kSparse; }

  size_t num_tensors() const override { return 3 * indices_->num_tensors(); }

  TFE_TensorHandle* get_tensor(size_t index) const override;

  std::string SummarizeValue() const override;

  std::string DebugString() const override;

  TF_DataType dtype() const override;

  parallel_device::ParallelTensor* indices() const { return indices_.get(); }

  parallel_device::ParallelTensor* values() const { return values_.get(); }

  parallel_device::ParallelTensor* dense_shapes() const {
    return dense_shapes_.get();
  }

  ConstValueNode* const_value_node() const override { return nullptr; }

  // llvm::RTTIExtends ID.
  static char ID;  // NOLINT

 private:
  SparseTensorWithLayout(
      std::unique_ptr<parallel_device::ParallelTensor> indices,
      std::unique_ptr<parallel_device::ParallelTensor> values,
      std::unique_ptr<parallel_device::ParallelTensor> dense_shapes,
      const Mesh& mesh, const Layout& layout,
      const std::vector<int64_t>& local_shape,
      std::optional<TF_DataType> dtype = std::nullopt,
      std::optional<NodeDef> const_value = std::nullopt)
      : llvm::RTTIExtends<SparseTensorWithLayout, TensorWithLayoutTf>(
            nullptr, mesh, layout, local_shape),
        indices_(std::move(indices)),
        values_(std::move(values)),
        dense_shapes_(std::move(dense_shapes)) {}

  std::unique_ptr<parallel_device::ParallelTensor> indices_;
  std::unique_ptr<parallel_device::ParallelTensor> values_;
  std::unique_ptr<parallel_device::ParallelTensor> dense_shapes_;
};

// TODO(b/256016071): Instead of having the following two functions, create a
// factory which can branch the creation of `TensorWithLayoutTf`,
// `ResourceHandleWithLayout`, `SparseTensorWithLayout` and the incoming
// `TensorWithLayoutPw`.

std::unique_ptr<TensorWithLayoutTf> CreateDummyTensorWithLayout(
    const std::vector<int64_t>& local_shape, TF_DataType dtype,
    const Mesh& mesh, const Layout& layout);

StatusOr<std::unique_ptr<TensorWithLayoutTf>> CreateTensorWithLayout(
    std::unique_ptr<parallel_device::ParallelTensor> tensor, const Mesh& mesh,
    const Layout& layout);

template <typename T>
std::string ShapeToDebugString(const std::vector<T> shape_vector) {
  std::vector<tensorflow::int64> cast_shape(shape_vector.begin(),
                                            shape_vector.end());
  tensorflow::PartialTensorShape shape;
  if (!tensorflow::PartialTensorShape::MakePartialShape(
           cast_shape.data(), cast_shape.size(), &shape)
           .ok()) {
    return "<error displaying shape>";
  } else {
    return shape.DebugString();
  }
}

// Internal class with shared functions for every ExecutableManager<T>.
class ExecutableManagerImpl {
  template <typename T>
  friend class ExecutableManager;

 public:
  absl::flat_hash_map<int, NodeDef> GetConstantFoldableTensors(
      const std::vector<TensorWithLayout*>& inputs);

  // Cache key for dtensor operation name, which includes the op name
  // and the input shapes. This is needed as a higher level cache for constant
  // folding.
  tensorflow::Fprint128 CacheKeyForDTensorOperation(
      const DTensorOperation& doperation) const;

 private:
  ExecutableManagerImpl() = default;
};

struct ExecutionManagerStats {
  int64_t hits;    // number of hits.
  int64_t misses;  // number of misses.
  int64_t size;    // size of cache (number of entries).
};

// Template Class that holds information about DTensor executable ran, including
// cached lowered executable and constant folding input information per
// function.
//
//
// The caching policy for constant folded inputs is the following:
//   In the first call to a function, we assume that all the inputs that
//   are constant foldable are constant folded and save these values. In the
//   next call to the same function call, we compare the values of constant
//   folded inputs to the previous constant folded inputs. We disable constant
//   folding for the changed values, and save these new inputs.
// TODO(b/169348205) Support cache eviction if the cache gets bloated.
template <typename T>
class ExecutableManager : public tsl::core::WeakRefCounted {
 public:
  ExecutableManager() = default;

  // Caches the executable with ParallelExecutable.
  const T* AddCachedExecutable(tensorflow::Fprint128 cache_key, T executable);

  // Removes the executable.
  void Remove(tensorflow::Fprint128 cache_key);

  // Returns the cache key and the cached lowered executable for the function.
  // Returns a nullptr for the lowered executable if there is a cache miss.
  // Upon a cache miss, this will save some metadata about the function
  // and the small inputs to keep track of information for constant folding.
  std::pair<tensorflow::Fprint128, const T*> GetCachedExecutable(
      const DTensorOperation& doperation, const NameAttrList& attributes,
      const std::vector<TensorWithLayout*>& inputs,
      const std::vector<const Layout*>& output_layouts);

  // Returns the cached lowered graph for the function.
  // Returns a nullptr for the lowered graph if there is a cache miss.
  // This Get operation has no side effect.
  const T* GetCachedExecutableSimple(tensorflow::Fprint128 cache_key);

  // Returns whether the input at `input_index` should be constant
  // folded into function `doperation`. An input is not constant folded if we
  // have ran this function at least twice and the small input value changed
  // across separate runs.
  bool ShouldFoldInput(const DTensorOperation& doperation,
                       int input_index) const;

  // Returns the current Stats of the execution manager.
  // The result is a snapshot at the moment of the call.
  ExecutionManagerStats GetStats() const {
    ExecutionManagerStats stats;
    stats.hits = stats_.hits;
    stats.misses = stats_.misses;
    // A reader Lock is probably more suitable, but this code branch is
    // barely executed.
    mutex_lock lock(mu_);
    stats.size = function_cache_.size();
    return stats;
  }

 private:
  // Generates a cache key for the graph, including its attributes,
  // inputs, and outputs.
  tensorflow::Fprint128 CacheKeyForGraph(
      const DTensorOperation& doperation, const NameAttrList& attributes,
      const std::vector<TensorWithLayout*>& inputs,
      const std::vector<const Layout*>& output_layouts);

  // Returns true for a missing entry in the small inputs cache.
  bool UpdateDTensorOpAndSmallInputsCache(
      const DTensorOperation& doperation,
      const std::vector<TensorWithLayout*>& inputs);

  mutable mutex mu_;
  mutable mutex dtensor_op_and_small_inputs_mu_;

  // Maps the hash of a graph with the lowered graph.
  absl::flat_hash_map<tensorflow::Fprint128, T, tensorflow::Fprint128Hasher>
      function_cache_ TF_GUARDED_BY(mu_);

  // Maps the hash of dtensor_operation and its input shapes to a map
  // representing the small constant indices and values to the function. The
  // small constant indices are saved to make faster comparisons for constant
  // folding validation.
  absl::flat_hash_map<tensorflow::Fprint128, absl::flat_hash_map<int, NodeDef>,
                      tensorflow::Fprint128Hasher>
      dtensor_op_and_small_inputs_
          TF_GUARDED_BY(dtensor_op_and_small_inputs_mu_);

  ExecutableManagerImpl executable_manager_impl_;
  struct {
    std::atomic<int64_t> hits = 0;
    std::atomic<int64_t> misses = 0;
  } stats_;
};

// Returns the shape of a given tensor.
std::vector<int64_t> TensorShapeAsVector(TFE_TensorHandle* tensor,
                                         TF_Status* status);

// Creates a Graph with _Arg and _Retval nodes surrounding an
// `operation_name`-type node.
Status PrepareGraphForMlir(
    const ExecutableManager<ExecutionFunctions>& function_manager,
    const std::vector<TensorWithLayout*>& inputs,
    const DTensorOperation& doperation,
    const tensorflow::FunctionLibraryDefinition& flib_def,
    const NameAttrList& attributes, const std::optional<Layout>& default_layout,
    tensorflow::Graph* graph,
    std::vector<PartialTensorShape>* global_output_shapes,
    std::vector<const Layout*>* output_layouts);

// Returns set of functions to run to execute DTensor computation.
StatusOr<ExecutionFunctions> IdentifyAllFunctionsToExecute(
    const tensorflow::Graph& graph,
    const std::vector<PartialTensorShape>& global_output_shapes);

// For functions with control outputs, add identity nodes between
// StatefulPartitionedCall and _Retvals, in order to preserve control output
// dependencies after StatefulPartitionedCall is inlined at runtime.
// Consider calling this in PrepareGraphForMlir, once the identity nodes won't
// be dropped during MLIR lowering.
// TODO(b/171265131): fix the underlying issue to avoid inserting identity
// nodes.
Status MaybeInsertIdentityNodes(const FunctionDef* function_def, Graph* graph);

// Add DTensor specific function attributes to be compatible with eager runtime.
void AddDTensorFunctionAttr(FunctionDef& function_def);

// Prepare inputs of embeddings for checkpoint functions.
StatusOr<std::vector<parallel_device::ParallelTensor*>> PrepareEmbeddingInputs(
    const std::vector<TensorWithLayoutTf*>& inputs);

Status InsertFunctionForTPUEmbeddingCheckpoint(
    TF_Status* status, Graph* graph,
    const std::vector<TensorWithLayout*>& inputs,
    const std::string& checkpoint_fn_name);

////////////////////////////////////////////////////////////////////////////////
// Implementation details for ExecutableManager<T>

// Thread safe method.
// Generates a cache key for the graph, including its attributes,
// inputs, and outputs.
// Cache key computation should consider all features of an op that affects
// the SPMD lowering. The cache keys of two ops must be different if the
// translated functions are different.
// - op name and attr
// - input shapes and layouts
// - default layout of outputs.
// - default mesh.
// - values of constant foldable inputs.
template <typename T>
tensorflow::Fprint128 ExecutableManager<T>::CacheKeyForGraph(
    const DTensorOperation& doperation, const NameAttrList& attributes,
    const std::vector<TensorWithLayout*>& inputs,
    const std::vector<const Layout*>& output_layouts) {
  tensorflow::Fprint128 cache_key = tensorflow::Fingerprint128(doperation.name);
  std::string serialized;
  SerializeToStringDeterministic(attributes, &serialized);
  cache_key =
      FingerprintCat128(cache_key, tensorflow::Fingerprint128(serialized));
  cache_key = FingerprintCat128(
      cache_key,
      tensorflow::Fingerprint128(doperation.default_mesh.ToString()));
  // Higher level cache based on operation name and input shapes.
  for (int i = 0; i < inputs.size(); ++i) {
    if (!ShouldFoldInput(doperation, i) &&
        inputs[i]->const_value_node() != nullptr) {
      inputs[i]->const_value_node()->reset_const_value();
    }
    cache_key = FingerprintCat128(cache_key, inputs[i]->CacheKey());
  }
  for (int output_index = 0; output_index < output_layouts.size();
       ++output_index) {
    if (output_layouts[output_index]) {
      cache_key = FingerprintCat128(cache_key, output_index);
      cache_key = FingerprintCat128(
          cache_key,
          tensorflow::Fingerprint128(output_layouts[output_index]->ToString()));
    }
  }
  return cache_key;
}

// Thread-safe method.
template <typename T>
std::pair<tensorflow::Fprint128, const T*>
ExecutableManager<T>::GetCachedExecutable(
    const DTensorOperation& doperation, const NameAttrList& attributes,
    const std::vector<TensorWithLayout*>& inputs,
    const std::vector<const Layout*>& output_layouts) {
  tensorflow::Fprint128 cache_key =
      CacheKeyForGraph(doperation, attributes, inputs, output_layouts);

  {
    mutex_lock lock(mu_);
    // Early return if we have a cache hit.
    if (auto iter = function_cache_.find(cache_key);
        iter != function_cache_.end()) {
      stats_.hits++;
      return std::pair<Fprint128, T*>(cache_key, &iter->second);
    }
  }
  // For eager ops we early return the cache miss and do not make further
  // optimizations.
  if (!doperation.is_func()) {
    stats_.misses++;
    return std::pair<Fprint128, std::nullptr_t>(cache_key, nullptr);
  }

  bool missed = UpdateDTensorOpAndSmallInputsCache(doperation, inputs);

  if (missed) {
    stats_.misses++;
    return std::pair<Fprint128, std::nullptr_t>(cache_key, nullptr);
  }
  // Generate a new cache key since we updated small const inputs which change
  // the cache key.
  cache_key = CacheKeyForGraph(doperation, attributes, inputs, output_layouts);

  stats_.misses++;
  return std::pair<Fprint128, std::nullptr_t>(cache_key, nullptr);
}

template <typename T>
bool ExecutableManager<T>::UpdateDTensorOpAndSmallInputsCache(
    const DTensorOperation& doperation,
    const std::vector<TensorWithLayout*>& inputs) {
  const tensorflow::Fprint128 doperation_hash =
      executable_manager_impl_.CacheKeyForDTensorOperation(doperation);

  mutex_lock lock(dtensor_op_and_small_inputs_mu_);
  // Save the constant folded inputs to this doperation if we have not seen
  // this before. This is needed so that in the next call to this operation,
  // we can compare these inputs to confirm which one is indeed a constant.
  auto doperation_iter = dtensor_op_and_small_inputs_.find(doperation_hash);
  if (doperation_iter == dtensor_op_and_small_inputs_.end()) {
    dtensor_op_and_small_inputs_.insert(
        {doperation_hash,
         executable_manager_impl_.GetConstantFoldableTensors(inputs)});
    return true;
  }
  // If we are here, then we have ran this function before but constant folded
  // some input(s) when it was not a constant input i.e. one of the small
  // value to this function input changed. So mark those changed values as
  // non-constant.
  absl::flat_hash_map<int, NodeDef>& previous_small_inputs =
      doperation_iter->second;
  std::vector<int> non_constant_indices;

  for (auto const& [index, previous_small_input] : previous_small_inputs) {
    auto* const_value_node = inputs[index]->const_value_node();
    if (const_value_node == nullptr) {
      continue;
    }
    if (const_value_node->const_value().has_value()) {
      if (NodeDefsHaveDifferentTensorProto(
              previous_small_input, const_value_node->const_value().value())) {
        const_value_node->reset_const_value();
        non_constant_indices.push_back(index);
      }
    }
  }
  for (int non_constant_index : non_constant_indices) {
    previous_small_inputs.erase(non_constant_index);
  }
  return false;
}

// Thread-safe method.
template <typename T>
const T* ExecutableManager<T>::GetCachedExecutableSimple(
    tensorflow::Fprint128 cache_key) {
  mutex_lock lock(mu_);
  auto iter = function_cache_.find(cache_key);
  if (iter == function_cache_.end()) {
    stats_.misses++;
    return nullptr;
  }
  stats_.hits++;
  return &iter->second;
}

template <typename T>
const T* ExecutableManager<T>::AddCachedExecutable(
    tensorflow::Fprint128 cache_key, T executable) {
  mutex_lock lock(mu_);
  return &function_cache_.insert({cache_key, std::move(executable)})
              .first->second;
}

template <typename T>
bool ExecutableManager<T>::ShouldFoldInput(const DTensorOperation& doperation,
                                           const int input_index) const {
  // For eager ops, assume the inputs are constant foldable.
  if (!doperation.is_func()) return true;
  const tensorflow::Fprint128 doperation_hash =
      executable_manager_impl_.CacheKeyForDTensorOperation(doperation);

  mutex_lock lock(dtensor_op_and_small_inputs_mu_);
  // If we didn't see this doperation before then optimisticly assume this is
  // foldable. The input at `input_index` is foldable only if it is one of the
  // indices we have saved as the small inputs.
  auto doperation_iter = dtensor_op_and_small_inputs_.find(doperation_hash);
  return doperation_iter == dtensor_op_and_small_inputs_.end() ||
         doperation_iter->second.contains(input_index);
}

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_CC_DTENSOR_DEVICE_UTIL_H_
