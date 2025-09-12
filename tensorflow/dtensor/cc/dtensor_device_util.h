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
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/parallel_device/parallel_device_lib.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/safe_ptr.h"
#include "tensorflow/c/tf_status.h"
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
#include "tensorflow/dtensor/cc/dtensor_operation.h"
#include "tensorflow/dtensor/cc/small_constant_optimization.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/cc/tensor_with_layout.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/refcount.h"

namespace tensorflow {
namespace dtensor {

using TensorHandlePtr = tensorflow::Safe_TFE_TensorHandlePtr;

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
                    absl::StatusMessageAsCStr(return_if_not_ok_status));  \
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

  // Original function name in the graph.
  std::string function_name;
  // Translated function name to be called.
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
  // Number of local outputs for each layout.
  std::vector<std::int64_t> num_local_outputs;
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

class TensorWithLayoutTf
    : public llvm::RTTIExtends<TensorWithLayoutTf, TensorWithLayout> {
 public:
  // Broadcast a single non-parallel tensor onto `mesh` with a fully replicated
  // sharding spec. Does not take ownership of `tensor`. The tensor must not
  // already be on a DTensorDevice.
  static std::unique_ptr<TensorWithLayoutTf> Broadcast(
      TFE_Context* context, TFE_TensorHandle* tensor, const Mesh& target_mesh,
      TF_Status* status);

  // Given an already-parallel tensor, wraps it with a mesh and a layout.
  static StatusOr<std::unique_ptr<TensorWithLayoutTf>> Wrap(
      std::vector<TensorHandlePtr>&& tensors, const Layout& layout,
      std::optional<std::vector<int64_t>>&& shape);

  // Given a single tensor, wraps it with a single device layout.
  static std::unique_ptr<TensorWithLayoutTf> Wrap(TensorHandlePtr single_tensor,
                                                  const Layout& layout,
                                                  TF_Status* status);

  // Creates a dummy TensorWithLayoutTf without holding a ParallelTensor.
  static std::unique_ptr<TensorWithLayoutTf> Dummy(
      const std::vector<int64_t>& local_shape, TF_DataType dtype,
      const Layout& layout);

  ~TensorWithLayoutTf() override = default;

  const Layout& layout() const override { return layout_; }

  TensorType tensor_type() const override { return TensorType::kDense; }

  TF_DataType dtype() const override {
    return dtype_.has_value() ? *dtype_
                              : TFE_TensorHandleDataType(tensors_[0].get());
  }
  // Encodes the NodeDef via provided builder, if applicable.
  void EncodeAttributes(tensorflow::NodeDefBuilder& builder) const override {}

  tensorflow::Fprint128 CacheKey() const override;

  TFE_TensorHandle* get_tensor(size_t index) const override {
    return tensors_[index].get();
  }

  size_t num_tensors() const override {
    return layout_.IsSingleDevice() ? 1 : tensors_.size();
  }

  std::vector<TFE_TensorHandle*> tensors() const {
    std::vector<TFE_TensorHandle*> result;
    result.reserve(tensors_.size());
    for (const TensorHandlePtr& tensor : tensors_) {
      result.emplace_back(tensor.get());
    }
    return result;
  }

  TFE_TensorHandle* single_tensor() const {
    return layout_.IsSingleDevice() ? get_tensor(0) : nullptr;
  }

  std::string SummarizeValue() const override;

  std::string DebugString() const override;

  std::vector<int64_t> global_shape() const override {
    return layout_.GlobalShapeFromLocalShape(local_shape_, &local_shapes_);
  }

  ConstValueNode* const_value_node() const override {
    return const_value_node_.get();
  }

  // llvm::RTTIExtends ID.
  static char ID;  // NOLINT

 protected:
  TensorWithLayoutTf(std::vector<TensorHandlePtr>&& tensors,
                     const Layout& layout,
                     const std::vector<int64_t>& local_shape,
                     const std::vector<std::vector<int64_t>>& local_shapes,
                     std::optional<TF_DataType> dtype = std::nullopt,
                     std::optional<NodeDef> const_value = std::nullopt)
      : tensors_(std::move(tensors)),
        layout_(layout),
        local_shape_(local_shape),
        local_shapes_(std::move(local_shapes)),
        dtype_(dtype) {
    const_value_node_ = std::make_unique<ConstValueNode>(const_value);
  }

  TensorWithLayoutTf(TensorHandlePtr&& single_tensor, const Layout& layout,
                     const std::vector<int64_t>& local_shape,
                     std::optional<TF_DataType> dtype = std::nullopt,
                     std::optional<NodeDef> const_value = std::nullopt)
      : tensors_([&single_tensor] {
          std::vector<TensorHandlePtr> result;
          result.emplace_back(std::move(single_tensor));
          return result;
        }()),
        layout_(layout),
        local_shape_(local_shape),
        dtype_(dtype) {
    const_value_node_ = std::make_unique<ConstValueNode>(const_value);
  }

  std::vector<TensorHandlePtr> tensors_;

  Layout layout_;

  // The local shape of tensors placed on each of `tensor_`'s component devices.
  std::vector<int64_t> local_shape_;
  // The local shape of each individual tensor in `tensors_`.
  // Initialized only when there is dynamic shape.
  std::vector<std::vector<int64_t>> local_shapes_;

  // dtype of tensor_. Empty if the layout is Single Device.
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
      std::vector<TensorHandlePtr>&& tensors, const Layout& layout,
      std::optional<std::vector<int64_t>>&& shape);

  // Similar to `Dummy` in `TensorWithLayoutTf` but for resource handle.
  static std::unique_ptr<ResourceHandleWithLayout> Dummy(
      const std::vector<int64_t>& local_shape, const Layout& layout);

  // The layout of uninitialized resource tensors, or the layout of the tensor
  // contained in an initialized resource.
  const Layout& layout() const override {
    return dereferenced_layout_.has_value() ? dereferenced_layout_.value()
                                            : layout_;
  }

  TensorType tensor_type() const override { return TensorType::kResource; }

  TF_DataType dtype() const override {
    return dtype_.has_value() ? *dtype_
                              : TFE_TensorHandleDataType(tensors_[0].get());
  }

  void EncodeAttributes(tensorflow::NodeDefBuilder& builder) const override;

  tensorflow::Fprint128 CacheKey() const override;

  // Updates the layout for the tensors.
  absl::Status UpdateLayout(const Layout& new_layout);

  // Updates the element layouts for the tensors.
  absl::Status UpdateElementLayouts(const std::vector<Layout>& layouts) {
    dereferenced_element_layouts_.emplace(layouts);
    return absl::OkStatus();
  }

  // Updates the local shape and dtype of the tensors.
  absl::Status UpdateShapeAndDType(const TensorShapeProto& shape,
                                   const DataType& dtype) {
    set_dereferenced_shape(shape);
    set_dereferenced_dtype(dtype);
    return absl::OkStatus();
  }

  ConstValueNode* const_value_node() const override { return nullptr; }

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

  // llvm::RTTIExtends ID.
  static char ID;  // NOLINT

 private:
  ResourceHandleWithLayout(std::vector<TensorHandlePtr>&& tensors,
                           const Layout& layout,
                           const std::vector<int64_t>& local_shape)
      : llvm::RTTIExtends<ResourceHandleWithLayout, TensorWithLayoutTf>(
            std::move(tensors), layout, local_shape, {}, TF_RESOURCE) {}

  // The layout of the tensor pointed to by this handle, if any.
  std::optional<Layout> dereferenced_layout_;
  // The layouts of the tensors emitted by this resource handle if it is an
  // iterator resource.
  std::optional<std::vector<Layout>> dereferenced_element_layouts_;
  // The shape and dtype of the tensor pointed to by this resource tensor.
  std::optional<TensorShapeProto> dereferenced_shape_;
  std::optional<DataType> dereferenced_dtype_;
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
      const Layout& layout, const std::vector<int64_t>& local_shape);

  // A dummy TensorWithLayout without holding a ParallelTensor.
  static std::unique_ptr<SparseTensorWithLayout> Dummy(
      const std::vector<int64_t>& local_shape, const Layout& layout) {
    return absl::WrapUnique(new SparseTensorWithLayout(
        /*indices=*/nullptr, /*values=*/nullptr, /*dense_shapes=*/nullptr,
        layout, local_shape));
  }

  // Add attribute '_sparse' to the NodeDefBuilder so that the mlir::Value
  // that originate from SparseTensorWithLayout are marked as '_sparse'.
  void EncodeAttributes(tensorflow::NodeDefBuilder& builder) const override {
    builder.Attr("_sparse", true);
  }

  TensorType tensor_type() const override { return TensorType::kSparse; }

  size_t num_tensors() const override {
    return kSparseTensorNum * indices_->num_tensors();
  }

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
      const Layout& layout, const std::vector<int64_t>& local_shape,
      std::optional<TF_DataType> dtype = std::nullopt,
      std::optional<NodeDef> const_value = std::nullopt)
      : llvm::RTTIExtends<SparseTensorWithLayout, TensorWithLayoutTf>(
            std::vector<TensorHandlePtr>(), layout, local_shape, {}),
        indices_(std::move(indices)),
        values_(std::move(values)),
        dense_shapes_(std::move(dense_shapes)) {}

  std::unique_ptr<parallel_device::ParallelTensor> indices_;
  std::unique_ptr<parallel_device::ParallelTensor> values_;
  std::unique_ptr<parallel_device::ParallelTensor> dense_shapes_;
};

std::unique_ptr<TensorWithLayoutTf> CreateDummyTensorWithLayout(
    const std::vector<int64_t>& local_shape, TF_DataType dtype,
    const Layout& layout);

// Creates a DTensor from one or more tensor handles and a compatible
// layout. Optionally accepts a `shape` argument that overrides the
// actual shape of the underlying tensors; this argument should be
// provided when there's a possibility of the inferred shape from
// differing from the actual shape (like when it is dynamic).
StatusOr<std::unique_ptr<TensorWithLayoutTf>> CreateTensorWithLayout(
    std::vector<TensorHandlePtr>&& tensor, const Layout& layout,
    std::optional<std::vector<int64_t>>&& shape = std::nullopt);

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
  StatusOr<std::pair<tensorflow::Fprint128, const T*>> GetCachedExecutable(
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
  StatusOr<bool> ShouldFoldInput(const DTensorOperation& doperation,
                                 const std::vector<TensorWithLayout*>& inputs,
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
  StatusOr<tensorflow::Fprint128> CacheKeyForGraph(
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
StatusOr<std::vector<int64_t>> GetTensorShapeAsVector(
    const tensorflow::PartialTensorShape& shape);

// Returns the shape of a given tensor.
StatusOr<std::vector<int64_t>> GetTensorShapeAsVector(TFE_TensorHandle* tensor);

absl::Status InferOutputLayouts(const DTensorOperation& doperation,
                                const NameAttrList& attributes,
                                const std::optional<Layout>& default_layout,
                                tensorflow::Graph* graph,
                                std::vector<const Layout*>* output_layouts);
// Creates a Graph with _Arg and _Retval nodes surrounding an
// `operation_name`-type node.
absl::Status PrepareGraphForMlir(
    const ExecutableManager<mlir::OwningOpRef<mlir::ModuleOp>>& module_manager,
    const std::vector<TensorWithLayout*>& inputs,
    const DTensorOperation& doperation,
    const tensorflow::FunctionLibraryDefinition& flib_def,
    const NameAttrList& attributes,
    const std::vector<const Layout*>& output_layouts, tensorflow::Graph* graph,
    std::vector<PartialTensorShape>* global_output_shapes);

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
absl::Status MaybeInsertIdentityNodes(const FunctionDef* function_def,
                                      Graph* graph);

// Add DTensor specific function attributes to be compatible with eager runtime.
void AddDTensorFunctionAttr(FunctionDef& function_def);

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
StatusOr<tensorflow::Fprint128> ExecutableManager<T>::CacheKeyForGraph(
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
    TF_ASSIGN_OR_RETURN(bool should_fold_input,
                        ShouldFoldInput(doperation, inputs, i));
    if (!should_fold_input && inputs[i]->const_value_node()) {
      inputs[i]->const_value_node()->reset_const_value();
    }
    cache_key = FingerprintCat128(
        cache_key, tensorflow::Fingerprint128(absl::StrFormat("%x", i)));
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
StatusOr<std::pair<tensorflow::Fprint128, const T*>>
ExecutableManager<T>::GetCachedExecutable(
    const DTensorOperation& doperation, const NameAttrList& attributes,
    const std::vector<TensorWithLayout*>& inputs,
    const std::vector<const Layout*>& output_layouts) {
  TF_ASSIGN_OR_RETURN(
      tensorflow::Fprint128 cache_key,
      CacheKeyForGraph(doperation, attributes, inputs, output_layouts));

  {
    mutex_lock lock(mu_);
    // Early return if we have a cache hit.
    if (auto iter = function_cache_.find(cache_key);
        iter != function_cache_.end()) {
      stats_.hits++;
      return {{cache_key, &iter->second}};
    }
  }

  bool missed = UpdateDTensorOpAndSmallInputsCache(doperation, inputs);

  if (missed) {
    stats_.misses++;
    return {{cache_key, nullptr}};
  }
  // Generate a new cache key since we updated small const inputs which change
  // the cache key.
  TF_ASSIGN_OR_RETURN(cache_key, CacheKeyForGraph(doperation, attributes,
                                                  inputs, output_layouts));

  stats_.misses++;
  return {{cache_key, nullptr}};
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
    // Some Ops the number of inputs can vary. We'll just skip updating them.
    if (index >= inputs.size()) continue;
    auto* const_value_node = inputs[index]->const_value_node();
    if (const_value_node == nullptr) {
      continue;
    }
    if (const_value_node->const_value().has_value()) {
      if (NodeDefsHaveDifferentTensorProto(
              previous_small_input, const_value_node->const_value().value())) {
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
void ExecutableManager<T>::Remove(tensorflow::Fprint128 cache_key) {
  mutex_lock lock(mu_);
  auto iter = function_cache_.find(cache_key);
  if (iter != function_cache_.end()) {
    function_cache_.erase(iter);
  }
}

template <typename T>
StatusOr<bool> ExecutableManager<T>::ShouldFoldInput(
    const DTensorOperation& doperation,
    const std::vector<TensorWithLayout*>& inputs, const int input_index) const {
  const auto input = inputs[input_index];
  const bool can_fold = input->const_value_node() &&
                        input->const_value_node()->const_value().has_value();
  // For eager ops, assume the inputs are constant foldable.
  if (!doperation.is_func()) {
    // Fold if we are in a function or if a special eager op.
    // TODO(b/270762002): Think about how to generalize this so it does not
    // depend on operation_name. For example, we can check the max abs value of
    // the tensor value.

    if (doperation.name == absl::string_view("StatelessRandomUniform") ||
        doperation.name == absl::string_view("StatelessRandomUniformFullInt") ||
        doperation.name == absl::string_view("StatelessRandomNormal") ||
        doperation.name == absl::string_view("StatelessTruncatedNormal")) {
      // For all stateless rng ops, we avoid fold seed (input_index==1) in
      // graph. This is an important optimization to avoid unnecessary MLIR SPMD
      // lowering and TPU compilation during model parameters initialization
      // process. which typically have the same shape for rng ops but different
      // seeds.
      return can_fold && (input_index != 1);
    }
    // Certain Ops we shall never fold in their inputs. Enable caching to reduce
    // sizes of the graphs. This list is incomplete.
    // FIXME(b/270762002): We only need constant folding for args that are
    // matched against Constants in MLIR.
    if (doperation.name != absl::string_view("Identity") &&
        doperation.name != absl::string_view("DivNoNan") &&
        doperation.name != absl::string_view("CopyToMesh") &&
        doperation.name != absl::string_view("CopyToMeshGrad") &&
        doperation.name != absl::string_view("Relayout") &&
        doperation.name != absl::string_view("RelayoutGrad")) {
      return can_fold;
    }
  }
  const tensorflow::Fprint128 doperation_hash =
      executable_manager_impl_.CacheKeyForDTensorOperation(doperation);

  mutex_lock lock(dtensor_op_and_small_inputs_mu_);
  // If we didn't see this doperation before then optimisticly assume this is
  // foldable. The input at `input_index` is foldable only if it is one of the
  // indices we have saved as the small inputs.
  auto doperation_iter = dtensor_op_and_small_inputs_.find(doperation_hash);
  return can_fold && (doperation_iter == dtensor_op_and_small_inputs_.end() ||
                      doperation_iter->second.contains(input_index));
}

// ExecutionFunctions manager can not check if the input is foldable.
template <>
StatusOr<bool> ExecutableManager<ExecutionFunctions>::ShouldFoldInput(
    const DTensorOperation& doperation,
    const std::vector<TensorWithLayout*>& inputs, int input_index) const;

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_CC_DTENSOR_DEVICE_UTIL_H_
