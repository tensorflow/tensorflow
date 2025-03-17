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

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_util.h"
#include "tsl/platform/protobuf.h"
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tfrt {
class BEFFile;
class HostContext;
}  // namespace tfrt

namespace tensorflow {
namespace tfrt_stub {

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

  const protobuf::Map<std::string, TensorProto>& GetDefaultInputs() const {
    return signature_->default_inputs;
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

    // If true, the loading of any signature (or signature combination) will be
    // deferred until the first corresponding invocationof running. Otherwise,
    // the individual signatures will be loaded along with the saved model.
    bool enable_lazy_loading = false;

    // If true, we'll attempt to find MLArchive within the given loading path.
    // If not found, will use the path as a normal SavedModel directory.
    //
    // This field is deprecated.
    bool maybe_load_from_mla = false;

    // If true, the lazy loading path will use tfrt_stub::GraphExecutor.
    //
    // TODO(b/216379787): Remove this option once b/279197040 is unblocked.
    bool lazy_loading_use_graph_executor = false;

    // True if and only if SavedModel is being loaded to generate AOT results.
    bool aot_generation = false;

    // Make a best-effort guess at the model type and emit a metric. E.g.
    // detecting JAX models by looking for the `XlaCallModule` op in the
    // MetaGraphDef.
    bool emit_model_type_metric = true;

    GraphExecutionOptions graph_execution_options;

    bool disable_output_filter = false;
  };

  // Per-request options.
  using RunOptions = GraphExecutionRunOptions;

  explicit SavedModel(const Runtime* runtime) : options_(runtime) {
    DCHECK(runtime);
  }
  explicit SavedModel(Options options,
                      std::unique_ptr<GraphExecutor> graph_executor)
      : options_(std::move(options)),
        graph_executor_(std::move(graph_executor)) {}
  virtual ~SavedModel();

  const SessionMetadata& model_metadata() const {
    return options_.graph_execution_options.model_metadata;
  }

  const Runtime& runtime() const {
    DCHECK(options_.graph_execution_options.runtime);
    return *options_.graph_execution_options.runtime;
  }

  bool disable_output_filter() const { return options_.disable_output_filter; }

  tfrt::HostContext* GetHostContext() const;

  GraphExecutor& graph_executor() const { return *graph_executor_; }

  // Returns meta graph def. Note that the graph_def field in the MetaGraphDef
  // has already been removed.
  //
  // TODO(b/191931702): Change the method to return SignatureDefs instead.
  virtual const tensorflow::MetaGraphDef& GetMetaGraphDef() const = 0;

  // Returns all the function names.
  virtual std::vector<std::string> GetFunctionNames() const = 0;

  // Returns the `FunctionMetadata` for a function. If the function is not
  // found, returns nullopt instead.
  virtual std::optional<FunctionMetadata> GetFunctionMetadata(
      absl::string_view func_name) const = 0;

  // Runs the signature specified by `name`. Both `inputs` and `outputs`
  // are all host tensors. The `outputs` must be non-null. If the returned
  // status is non-OK, the `outputs` are invalid.
  virtual absl::Status Run(const RunOptions& run_options,
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
  virtual absl::Status RunMultipleSignatures(
      const RunOptions& run_options, absl::Span<const std::string> names,
      absl::Span<const std::vector<tensorflow::Tensor>> multi_inputs,
      std::vector<std::vector<tensorflow::Tensor>>* multi_outputs) = 0;

  // Runs the graphs specified by the tensor names terminal tensors (eg. feed
  // tensors, fetch tesnors) in the graph.
  virtual absl::Status RunByTensorNames(
      const RunOptions& run_options,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_node_names,
      std::vector<tensorflow::Tensor>* outputs) = 0;

 protected:
  const FallbackState& fallback_state() const {
    return graph_executor_->fallback_state();
  }
  FallbackState& fallback_state() { return graph_executor_->fallback_state(); }

  const Options options_;
  std::unique_ptr<GraphExecutor> graph_executor_;
};

using SignatureMap = absl::flat_hash_map<std::string, internal::Signature>;
using ::tensorflow::StatusOr;

enum class InferredModelType {
  kUncategorized = 0,
  kJax,
};

class SavedModelImpl final : public SavedModel {
 public:
  struct JoinedSignature;

  // Loads all SignatureDefs in a MetaGraphDef that matches the `tags` in the
  // tensorflow saved model from `saved_model_dir`. Refer to
  // http://g3doc/learning/serving/g3doc/saved_model/overview.md
  // for explanations on SavedModel.
  //
  // If `options.maybe_load_from_mla` is true, tries opening `saved_model_dir`
  // as an MLA. If it's not an MLA, uses it as a normal SavedModel directory.
  static absl::StatusOr<std::unique_ptr<SavedModel>> LoadSavedModel(
      Options options, absl::string_view saved_model_dir,
      const std::unordered_set<std::string>& tags);

  // Loads all SignatureDefs in `meta_graph_def`. Refer to
  // http://g3doc/learning/serving/g3doc/saved_model/overview.md
  // for explanations on SavedModel.
  static absl::StatusOr<std::unique_ptr<SavedModel>> LoadSavedModel(
      Options options, tensorflow::MetaGraphDef meta_graph_def,
      absl::string_view saved_model_dir);

  SavedModelImpl(
      Options options, SymbolUids symbol_uids,
      tensorflow::MetaGraphDef meta_graph_def, tfrt::BefBuffer bef,
      tfrt::RCReference<tfrt::BEFFile> bef_file, mlrt::bc::Buffer bytecode,
      std::optional<mlrt::LoadedExecutable> loaded_executable,
      absl::flat_hash_map<std::string, internal::Signature> signatures,
      std::unique_ptr<OpKernelRunnerTable> runner_table,
      std::unique_ptr<tfd::FallbackResourceArray> resource_array,
      std::unique_ptr<GraphExecutor> graph_executor,
      InferredModelType inferred_model_type);

  ~SavedModelImpl() override = default;

  SavedModelImpl(const SavedModelImpl&) = delete;
  SavedModelImpl& operator=(const SavedModelImpl&) = delete;

  const tensorflow::MetaGraphDef& GetMetaGraphDef() const override;

  std::vector<std::string> GetFunctionNames() const override;

  std::optional<FunctionMetadata> GetFunctionMetadata(
      absl::string_view func_name) const override;

  absl::Status Run(const RunOptions& run_options, absl::string_view name,
                   absl::Span<const tensorflow::Tensor> inputs,
                   std::vector<tensorflow::Tensor>* outputs) override;

  absl::Status RunMultipleSignatures(
      const RunOptions& run_options, absl::Span<const std::string> names,
      absl::Span<const std::vector<tensorflow::Tensor>> multi_inputs,
      std::vector<std::vector<tensorflow::Tensor>>* multi_outputs) override;

  absl::Status RunByTensorNames(
      const RunOptions& run_options,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_node_names,
      std::vector<tensorflow::Tensor>* outputs) override;

 private:
  // The result of loading signature(s).
  struct LoadingResult {
    std::string name;
    SymbolUids symbol_uids;

    // For the MLRT path.
    mlrt::bc::Buffer bytecode_buffer;
    std::unique_ptr<mlrt::LoadedExecutable> bytecode_executable;

    // For the TFRT path.
    tfrt::BefBuffer bef;
    tfrt::RCReference<tfrt::BEFFile> bef_file;

    std::unique_ptr<OpKernelRunnerTable> runner_table;
    std::unique_ptr<tfd::FallbackResourceArray> resource_array;

    // There are some resources that need re-creating when the executable is
    // re-created, so a resource context is stored along with the executable.
    // This resource context is meant to be passed to the op kernels for their
    // references. See the comment above `GraphExecutor::resource_context_`
    // about the todo to merge that resource context with this one.
    std::unique_ptr<tfrt::ResourceContext> resource_context;
  };

  // Imports a subgraph as an MLIR module with the specified `input_nodes`,
  // `output_nodes`.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSubgraph(
      mlir::MLIRContext* context, absl::string_view name,
      const tensorflow::GraphImportConfig::InputArrays& input_nodes,
      const std::vector<std::string>& output_nodes,
      const std::vector<std::string>& target_nodes);

  // Given the joined signature, loads the subgraph and returns loading result.
  absl::StatusOr<std::reference_wrapper<const SavedModelImpl::LoadingResult>>
  LoadJoinedSignature(const JoinedSignature& joined_signature)
      TF_EXCLUSIVE_LOCKS_REQUIRED(loading_result_cache_mu_);

  // Returns the loading result given the signature names.
  absl::StatusOr<std::reference_wrapper<const SavedModelImpl::LoadingResult>>
  GetOrCreateLoadingResult(const RunOptions& run_options,
                           absl::Span<const std::string> names)
      TF_LOCKS_EXCLUDED(loading_result_cache_mu_);

  void IncrementInferredModelTypePerRequestCount();

  SymbolUids symbol_uids_;
  // `meta_graph_def_` only contains metadata of the model. The graph_def field
  // is removed.
  //
  // TODO(b/191931702): We should only keep content that are actually used
  // (eg. SignatureDefs), instead of keeping the whole saved model, to avoid
  // unnecessary memory usage.
  tensorflow::MetaGraphDef meta_graph_def_;
  tfrt::BefBuffer bef_;
  tfrt::RCReference<tfrt::BEFFile> bef_file_;

  mlrt::bc::Buffer bytecode_;
  std::optional<mlrt::LoadedExecutable> loaded_executable_;

  tfrt::RequestDeadlineTracker req_deadline_tracker_;
  absl::flat_hash_map<std::string, internal::Signature> signatures_;
  std::unique_ptr<OpKernelRunnerTable> runner_table_;
  std::unique_ptr<tfd::FallbackResourceArray> resource_array_;
  tensorflow::mutex loading_result_cache_mu_;
  // For pointer stability of values in `absl::flat_hash_map<>`, additional
  // `std::unique_ptr<>` is necessary. (See https://abseil.io/tips/136.)
  absl::flat_hash_map<std::string /*joined_name*/,
                      std::unique_ptr<LoadingResult>>
      loading_result_cache_ TF_GUARDED_BY(loading_result_cache_mu_);

  InferredModelType inferred_model_type_ = InferredModelType::kUncategorized;
};

class SavedModelMiraImpl;

}  // namespace tfrt_stub
}  // namespace tensorflow

namespace tfrt {

using SavedModel = ::tensorflow::tfrt_stub::SavedModel;
using SavedModelImpl = ::tensorflow::tfrt_stub::SavedModelImpl;
using SavedModelMiraImpl = ::tensorflow::tfrt_stub::SavedModelMiraImpl;
using TensorSpec = ::tensorflow::tfrt_stub::TensorSpec;
using FunctionMetadata = ::tensorflow::tfrt_stub::FunctionMetadata;

namespace internal {
using Signature = ::tensorflow::tfrt_stub::internal::Signature;
}

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_H_
