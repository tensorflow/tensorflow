/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_H_
#define TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_H_

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/saved_model/saved_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tfrt {

class BEFFile;
class HostContext;

namespace tpu {
class TpuModelResource;
}  // namespace tpu

}  // namespace tfrt

namespace tensorflow {
namespace tfrt_stub {

// TODO(tfrt-dev): Replace tfrt::TensorSpec with tensorflow::TensorSpec once the
// latter is checked in.
struct TensorSpec {
  tensorflow::DataType dtype;
  tensorflow::PartialTensorShape shape;

  explicit TensorSpec(tensorflow::DataType dtype) : dtype(dtype) {}
  TensorSpec(tensorflow::DataType dtype, tensorflow::PartialTensorShape shape)
      : dtype(dtype), shape(std::move(shape)) {}
};

inline bool operator==(const TensorSpec& a, const TensorSpec& b) {
  return a.dtype == b.dtype && a.shape.IsIdenticalTo(b.shape);
}

namespace internal {

struct Signature {
  std::vector<tensorflow::Tensor> captures;

  // The following three fields should have the same size.
  std::vector<std::string> input_names;
  std::vector<TensorSpec> input_specs;
  std::vector<std::string> input_devices;

  // The following two fields should have the same size.
  std::vector<std::string> output_names;
  std::vector<TensorSpec> output_specs;
};

}  // namespace internal

class FunctionMetadata {
 public:
  explicit FunctionMetadata(const internal::Signature* signature)
      : signature_(signature) {
    assert(signature);
  }

  const std::vector<std::string>& GetInputNames() const {
    return signature_->input_names;
  }

  const std::vector<TensorSpec>& GetInputSpecs() const {
    return signature_->input_specs;
  }

  const std::vector<std::string>& GetOutputNames() const {
    return signature_->output_names;
  }

  const std::vector<TensorSpec>& GetOutputSpecs() const {
    return signature_->output_specs;
  }

 private:
  friend class SavedModelImpl;

  const internal::Signature* signature_ = nullptr;
};

// SavedModel represents the in-memory states (graphs and variables) loaded from
// a tensorflow saved model directory.
class SavedModel {
 public:
  struct Options {
    explicit Options(const Runtime* rt) : graph_execution_options(rt) {}

    // If the number of signagures is greater than the threshold, the loading of
    // any signature (or signature combination) will be deferred until the first
    // corresponding invocationof running. Otherwise, the individual signatures
    // will be loaded along with the saved model.
    int32_t lazy_loading_threshold = std::numeric_limits<int32_t>::max();

    // If true, the loading path points to an MLArchive containing a SavedModel.
    bool load_from_mla = false;

    GraphExecutionOptions graph_execution_options;
  };

  // Per-request options.
  using RunOptions = GraphExecutionRunOptions;

  explicit SavedModel(const Runtime* runtime) : runtime_(runtime) {
    DCHECK(runtime_);
  }
  virtual ~SavedModel();

  const Runtime& runtime() const {
    DCHECK(runtime_);
    return *runtime_;
  }
  tfrt::HostContext* GetHostContext() const;

  // Returns meta graph def. Note that the graph_def field in the MetaGraphDef
  // has already been removed.
  //
  // TODO(b/191931702): Change the method to return SignatureDefs instead.
  virtual const tensorflow::MetaGraphDef& GetMetaGraphDef() const = 0;

  // Returns all the function names.
  virtual std::vector<std::string> GetFunctionNames() const = 0;

  // Returns the `FunctionMetadata` for a function. If the function is not
  // found, returns nullopt instead.
  virtual absl::optional<FunctionMetadata> GetFunctionMetadata(
      absl::string_view func_name) const = 0;

  // Runs the signature specified by `name`. Both `inputs` and `outputs`
  // are all host tensors. The `outputs` must be non-null. If the returned
  // status is non-OK, the `outputs` are invalid.
  virtual tensorflow::Status Run(const RunOptions& run_options,
                                 absl::string_view name,
                                 absl::Span<const tensorflow::Tensor> inputs,
                                 std::vector<tensorflow::Tensor>* outputs) = 0;

  // Runs the signatures specified by `names`. Both `inputs` and `outputs` are
  // all host tensors. The `outputs` must be non-null. If the returned status is
  // non-OK, the `outputs` are invalid.
  //
  // NOTE: If the given signatures have overlapping input nodes, the input
  // tensors for these overlapping nodes must be the same. Having different
  // input tensors for overlapping nodes results UNDEFINED BEHAVIOR.
  //
  // NOTE: The input/output tensors can only be dense tensors (as opposed to
  // sparse tensors or composite tensors).
  virtual tensorflow::Status RunMultipleSignatures(
      const RunOptions& run_options, absl::Span<const std::string> names,
      absl::Span<const std::vector<tensorflow::Tensor>> multi_inputs,
      std::vector<std::vector<tensorflow::Tensor>>* multi_outputs) = 0;

  // Runs the graphs specified by the tensor names terminal tensors (eg. feed
  // tensors, fetch tesnors) in the graph.
  virtual tensorflow::Status RunByTensorNames(
      const RunOptions& run_options,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_node_names,
      std::vector<tensorflow::Tensor>* outputs) = 0;

 private:
  const Runtime* runtime_ = nullptr;
};

class SavedModelImpl final : public SavedModel {
 public:
  struct JoinedSignature;

  // Loads all SignatureDefs in a MetaGraphDef that matches the `tags` in the
  // tensorflow saved model from `saved_model_dir`. Refer to
  // http://g3doc/learning/serving/g3doc/saved_model/overview.md
  // for explanations on SavedModel.
  //
  // If `options.load_from_mla` is true, treat `saved_model_dir` as an MLA path
  // that wraps the saved model path.
  static std::unique_ptr<SavedModel> LoadSavedModel(
      Options options, absl::string_view saved_model_dir,
      const std::unordered_set<std::string>& tags, tensorflow::Status* status);

  SavedModelImpl(
      Options options, tensorflow::MetaGraphDef meta_graph_def,
      tfrt::BefBuffer bef, tfrt::RCReference<tfrt::BEFFile> bef_file,
      absl::flat_hash_map<std::string, internal::Signature> signatures,
      std::unique_ptr<FallbackState> fallback_state,
      std::unique_ptr<tfrt::tpu::TpuModelResource> tpu_model_resource,
      std::unique_ptr<tfrt::ResourceContext> resource_context,
      std::unique_ptr<GraphExecutor> graph_executor);

  ~SavedModelImpl() override;

  SavedModelImpl(const SavedModelImpl&) = delete;
  SavedModelImpl& operator=(const SavedModelImpl&) = delete;

  const tensorflow::MetaGraphDef& GetMetaGraphDef() const override;

  std::vector<std::string> GetFunctionNames() const override;

  absl::optional<FunctionMetadata> GetFunctionMetadata(
      absl::string_view func_name) const override;

  tensorflow::Status Run(const RunOptions& run_options, absl::string_view name,
                         absl::Span<const tensorflow::Tensor> inputs,
                         std::vector<tensorflow::Tensor>* outputs) override;

  tensorflow::Status RunMultipleSignatures(
      const RunOptions& run_options, absl::Span<const std::string> names,
      absl::Span<const std::vector<tensorflow::Tensor>> multi_inputs,
      std::vector<std::vector<tensorflow::Tensor>>* multi_outputs) override;

  tensorflow::Status RunByTensorNames(
      const RunOptions& run_options,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_node_names,
      std::vector<tensorflow::Tensor>* outputs) override;

 private:
  // The result of loading signature(s).
  struct LoadingResult {
    std::string name;
    tfrt::BefBuffer bef;
    tfrt::RCReference<tfrt::BEFFile> bef_file;
    std::unique_ptr<tfrt::ResourceContext> resource_context;
  };

  // Imports a subgraph as an MLIR module with the specified `input_nodes`,
  // `output_nodes`.
  tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSubgraph(
      mlir::MLIRContext* context,
      const tensorflow::GraphImportConfig::InputArrays& input_nodes,
      const std::vector<std::string>& output_nodes,
      const std::vector<std::string>& target_nodes);

  // Given the joined signature, loads the subgraph and returns loading result.
  tensorflow::StatusOr<
      std::reference_wrapper<const SavedModelImpl::LoadingResult>>
  LoadJoinedSignature(const JoinedSignature& joined_signature)
      TF_EXCLUSIVE_LOCKS_REQUIRED(loading_result_cache_mu_);

  // Returns the loading result given the signature names.
  tensorflow::StatusOr<
      std::reference_wrapper<const SavedModelImpl::LoadingResult>>
  GetOrCreateLoadingResult(absl::Span<const std::string> names)
      TF_LOCKS_EXCLUDED(loading_result_cache_mu_);

  // Runs `func` with the given inputs, and outputs the result.
  tensorflow::Status RunInternal(const RunOptions& run_options,
                                 absl::string_view signature_name,
                                 const tfrt::Function& func,
                                 absl::Span<const tensorflow::Tensor> inputs,
                                 absl::Span<const tensorflow::Tensor> captures,
                                 std::vector<tensorflow::Tensor>* outputs,
                                 tfrt::ResourceContext* resource_context);

  Options options_;
  // `meta_graph_def_` only contains metadata of the model. The graph_def field
  // is removed.
  //
  // TODO(b/191931702): We should only keep content that are actually used
  // (eg. SignatureDefs), instead of keeping the whole saved model, to avoid
  // unnecessary memory usage.
  tensorflow::MetaGraphDef meta_graph_def_;
  tfrt::BefBuffer bef_;
  tfrt::RCReference<tfrt::BEFFile> bef_file_;
  tfrt::RequestDeadlineTracker req_deadline_tracker_;
  absl::flat_hash_map<std::string, internal::Signature> signatures_;
  std::unique_ptr<FallbackState> fallback_state_;
  // TODO(b/178227859): Change the hardcoding of this specific TPU resource
  // (TpuModelResource) to a general and plugable interface.
  std::unique_ptr<tfrt::tpu::TpuModelResource> tpu_model_resource_;
  std::unique_ptr<tfrt::ResourceContext> resource_context_;
  tensorflow::mutex loading_result_cache_mu_;
  // For pointer stability of values in `absl::flat_hash_map<>`, additional
  // `std::unique_ptr<>` is necessary. (See https://abseil.io/tips/136.)
  absl::flat_hash_map<std::string /*joined_name*/,
                      std::unique_ptr<LoadingResult>>
      loading_result_cache_ TF_GUARDED_BY(loading_result_cache_mu_);
  std::unique_ptr<GraphExecutor> graph_executor_;
  bool lazy_loading_enabled_ = false;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

namespace tfrt {

using SavedModel = ::tensorflow::tfrt_stub::SavedModel;
using SavedModelImpl = ::tensorflow::tfrt_stub::SavedModelImpl;
using TensorSpec = ::tensorflow::tfrt_stub::TensorSpec;
using FunctionMetadata = ::tensorflow::tfrt_stub::FunctionMetadata;

namespace internal {
using Signature = ::tensorflow::tfrt_stub::internal::Signature;
}

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_H_
