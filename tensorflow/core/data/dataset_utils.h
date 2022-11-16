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
#ifndef TENSORFLOW_CORE_DATA_DATASET_UTILS_H_
#define TENSORFLOW_CORE_DATA_DATASET_UTILS_H_

#include <functional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

// Constant used for indicating that the argument of tf.data.Dataset.shard
// should be supplied by the auto-sharding rewrite.
constexpr int kShardHint = -1;

// The initial parallelism value before Autotune has a chance to optimize.
constexpr int kAutotuneDefaultParallelism = 16;

// Creates a resource handle with a unique name for the given resource where
// the resource is managed by the Resource Manager.
template <typename T>
Status CreateWeakHandle(OpKernelContext* ctx, T* resource,
                        const string& container_name, ResourceHandle* handle) {
  static std::atomic<int64_t> resource_id_counter(0);
  string unique_name =
      strings::StrCat(container_name, resource_id_counter.fetch_add(1));
  ResourceMgr* mgr = ctx->resource_manager();
  TF_RETURN_IF_ERROR(mgr->Create<T>(container_name, unique_name, resource));

  *handle = MakeResourceHandle(container_name, unique_name, *ctx->device(),
                               TypeIndex::Make<T>());
  return OkStatus();
}

// Creates a ref-counting resource handle for the given resource, where the
// resource is owned by the handle.
template <typename T>
Status CreateHandle(OpKernelContext* ctx, T* resource, ResourceHandle* handle) {
  ResourceMgr* mgr = ctx->resource_manager();
  *handle =
      ResourceHandle::MakeRefCountingHandle(resource, ctx->device()->name());
  TF_RETURN_IF_ERROR(
      mgr->CreateUnowned<T>(handle->container(), handle->name(), resource));
  return OkStatus();
}

// TODO(b/198162355): Merge this class with ResourceOpKernel.
template <typename T>
class AnonymousResourceOp : public OpKernel {
 public:
  // Creates an AnonymousResourceOp.
  // ref_counting: Determines if the Op returns a ref-counting ResourceHandle.
  // ResourceHandle. See go/tf-resource-handle-ref-count.
  // return_deleter: Determines if the Op outputs a deleter tensor in addition
  // to the resource handle tensor.
  // If the resource handle is ref-counting, a no-op deleter is returned.
  explicit AnonymousResourceOp(OpKernelConstruction* context, bool ref_counting,
                               bool return_deleter)
      : OpKernel(context),
        ref_counting_(ref_counting),
        return_deleter_(return_deleter) {}

  void Compute(OpKernelContext* ctx) override {
    FunctionLibraryRuntime* lib;
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
    OP_REQUIRES_OK(
        ctx, ctx->function_library()->Clone(&flib_def, &pflr, &lib, true));
    T* resource;
    OP_REQUIRES_OK(ctx, CreateResource(ctx, std::move(flib_def),
                                       std::move(pflr), lib, &resource));

    ResourceHandle handle;
    if (ref_counting_) {
      OP_REQUIRES_OK(ctx, CreateHandle(ctx, resource, &handle));
    } else {
      OP_REQUIRES_OK(ctx, CreateWeakHandle(ctx, resource, name(), &handle));
    }
    Tensor* handle_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle_t));
    handle_t->scalar<ResourceHandle>()() = handle;

    if (return_deleter_) {
      Tensor* deleter_t;
      AllocatorAttributes attr;
      attr.set_on_host(true);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(1, TensorShape({}), &deleter_t, attr));
      // TODO(feyu): Consider returning an OptionalVariant.
      if (!ref_counting_) {
        // A deleter output that deletes the resource when destroyed.
        deleter_t->scalar<Variant>()() =
            ResourceDeleter(handle, ctx->resource_manager());
      }
    }
  }

 protected:
  virtual string name() = 0;

  virtual Status CreateResource(
      OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
      FunctionLibraryRuntime* lib, T** resource) = 0;

 private:
  const bool ref_counting_;
  const bool return_deleter_;
};

// Returns OkStatus() if `expected` and `received` types match,
// errors::InvalidArgument otherwise.
Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received);

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const std::vector<Tensor>& received);

// Returns OkStatus() if `expected` and `received` shapes are compatible,
// errors::InvalidArgument otherwise.
Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received);

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<Tensor>& received);

// Dataset op level determinism policy.
class DeterminismPolicy {
 public:
  enum class Type : int {
    // The op must produce elements deterministically.
    kDeterministic,
    // The op may relax determinism to improve performance.
    kNondeterministic,
    // The determinism policy is not specified at the op level. In this case we
    // use the experimental_deterministic dataset option to determine the
    // determinism policy.
    kDefault,
  };
  static constexpr const char* const kDeterministic = "true";
  static constexpr const char* const kNondeterministic = "false";
  static constexpr const char* const kDefault = "default";

  DeterminismPolicy() : determinism_(Type::kDefault) {}
  explicit DeterminismPolicy(Type determinism) : determinism_(determinism) {}
  // Creates a DeterminismPolicy with Type kDeterministic or
  // kNondeterministic, depending on the values of `is_deterministic`.
  explicit DeterminismPolicy(bool is_deterministic);

  static Status FromString(const std::string& s, DeterminismPolicy* out);

  // Returns the string representing the determinism policy. This will be one of
  // the string constants defined above.
  std::string String() const;

  /// Convenience methods for checking the DeterminismPolicy::Type.
  bool IsDeterministic() const { return determinism_ == Type::kDeterministic; }
  bool IsNondeterministic() const {
    return determinism_ == Type::kNondeterministic;
  }
  bool IsDefault() const { return determinism_ == Type::kDefault; }

 private:
  Type determinism_;
};

// Resolves non-deterministic seeds if necessary, returning either the original
// seeds or the resolved seeds.
//
// By TensorFlow convention, if both seeds are 0, they should be replaced with
// non-deterministically chosen seeds.
std::pair<int64_t, int64_t> MaybeOverrideSeeds(
    std::pair<int64_t, int64_t> seeds);

// Adds the functions in `to_add` to `base`. If a function with a matching
// signature already exists in `base`, replaces it with the function from
// `to_add`.
Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionLibraryDefinition& to_add);
Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionDefLibrary& to_add);

// Determines whether the given function is stateful.
Status IsFunctionStateful(const FunctionLibraryDefinition& library,
                          const FunctionDef& function_def);

// Determines whether the given node is stateful.
Status IsNodeStateful(const FunctionLibraryDefinition& library,
                      const NodeDef& node);

// Creates a runner that runs functions with limited parallelism.
std::function<void(std::function<void()>)> RunnerWithMaxParallelism(
    std::function<void(std::function<void()>)> runner, int max_parallelism);

// Op for creating a typed dummy resource.
//
// This op is used to provide a resource "placeholder" for ops such as
// `CacheDatasetV2` or `ShuffleDatasetV2` that expects a resource input.
// Originally, the lifetime of the resources passed into these ops was managed
// externally. After the implementation changed to manage the lifetime of the
// resources (including creation) by the ops themselves, the resource input is
// only needed to pass a resource handle through graph rewrites. When they are
// invoked from user code, the implementation passes in a dummy resource.
template <typename ResourceType>
class DummyResourceOp : public OpKernel {
 public:
  explicit DummyResourceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &tensor));
    tensor->scalar<ResourceHandle>()() = MakeResourceHandle<ResourceType>(
        ctx, /*container=*/"", /*name=*/"dummy_resource");
  }
};

// Given an op prefix and an op to match, returns whether the op to match
// is a match for any version of the op prefix. For example,
// MatchesAnyVersion("BatchDataset", "BatchDataset") == true
// MatchesAnyVersion("BatchDataset", "BatchDatasetV2") == true
// MatchesAnyVersion("BatchDataset", "BatchDatasetV3") == true
// MatchesAnyVersion("PaddedBatchDataset", "BatchDataset") == false
bool MatchesAnyVersion(StringPiece op_prefix, StringPiece op_to_match);

// Returns the index-th slice of a given tensor. If the index-th slice of
// the tensor is not aligned, returns a deep copy of the tensor.
Tensor MaybeCopySubSlice(const Tensor& tensor, int64 index);

// Removes device placements from the ops of all functions in `library`.
void StripDevicePlacement(FunctionDefLibrary* library);

// Copies partial of the batch output.
Status CopyPartialBatch(int64_t num_elements, const Tensor& value,
                        Tensor* output);

// Reads a batch when restoring the iterator.
Status ReadBatch(IteratorContext* ctx, IteratorStateReader* reader,
                 int64_t batch_size, const string& iterator_prefix,
                 const string& batch_prefix, std::vector<Tensor>* batch);

// Writes a batch when saving the iterator.
Status WriteBatch(int64_t batch_size, int64_t num_elements,
                  const string& iterator_prefix, const string& batch_prefix,
                  IteratorStateWriter* writer, std::vector<Tensor>* batch);

// Reads a status when restoring the iterator.
Status ReadStatus(const string& iterator_prefix, const string& prefix,
                  IteratorStateReader* reader, Status* status);

// Writes a status when saving the iterator.
Status WriteStatus(const string& iterator_prefix, const string& prefix,
                   const Status& status, IteratorStateWriter* writer);

// Processes a batch to output. In the case a partial batch is encountered, copy
// only partial of the batch.
Status ProcessBatch(int64_t batch_size, int64_t num_elements,
                    bool drop_remainder, const Status& status,
                    IteratorContext* ctx, std::vector<Tensor>* output,
                    bool* end_of_sequence, std::vector<Tensor>* batch);

// Constructs and stores the parameters for the CopyBatch function.
struct CopyBatchParams {
  Allocator* allocator;
  std::function<void(std::function<void()>)>* runner;
  int64 runner_threadpool_size;

  explicit CopyBatchParams(IteratorContext* ctx) {
    allocator = ctx->allocator({});
    runner = ctx->runner();
    runner_threadpool_size = ctx->runner_threadpool_size();
  }

  explicit CopyBatchParams(OpKernelContext* ctx) {
    allocator = ctx->get_allocator({});
    runner = ctx->runner();
    runner_threadpool_size = GetRunnerThreadpoolSizeFromOpKernelContext(ctx);
  }
};

// Copies the input elements to a batch.
//
// The `batch_elements` argument contains the individual elements to copy into a
// batch. The `parallel_copy` argument indicates whether to parallelize the
// copy. The `allocation_callback` argument can be used to pass a callback to
// invoke upon successful allocation of the memory for the batch. The
// `out_tensors` argument will be used to store the resulting batch (one for
// each component of the input).
Status CopyBatch(CopyBatchParams params,
                 const std::vector<std::vector<Tensor>>& batch_elements,
                 bool parallel_copy,
                 std::function<Status()> allocation_callback,
                 std::vector<Tensor>* out_tensors);

// Computes the set of experiments to apply based on the job name, task id,
// rollout percentage of registered experiments, and the
// TF_DATA_EXPERIMENT_OPT_IN and TF_DATA_EXPERIMENT_OPT_OUT environment
// variables.
absl::flat_hash_set<string> GetExperiments();
absl::flat_hash_set<string> GetExperiments(
    const std::string& job_name, int64_t task_id,
    std::function<uint64_t(const string&)> hash_func);

// Logs and records the experiments that will be applied.
void LogAndRecordExperiments(const absl::flat_hash_set<string>& experiments);

// Computes the set of enabled, disabled, and default optimizations based on the
// given options. An optimization must be a graph optimizer name that has been
// registered with Grappler.
void GetOptimizations(const Options& options,
                      absl::flat_hash_set<tstring>* optimizations_enabled,
                      absl::flat_hash_set<tstring>* optimizations_disabled,
                      absl::flat_hash_set<tstring>* optimizations_default);

// Creates graph rewrite configs based on the given options. The configs will
// only be used if their corresponding optimizers registered with Grappler are
// enabled.
// A config is a string with the following format:
//   <optimizer name>:<attribute name>:<attribute value>
absl::flat_hash_set<tstring> CreateGraphRewriteConfigs(const Options& options);

// Determines whether max intra-op parallelism should be configured.
bool ShouldConfigureMaxIntraOpParallelism(const Options& options);

// Determines whether private threadpool should be used.
bool ShouldUsePrivateThreadPool(const Options& options);

// Determines whether autotuning should be used.
bool ShouldUseAutotuning(const Options& options);

// Determines whether optimizations should be applied.
bool ShouldApplyOptimizations(
    const Options& options,
    const absl::flat_hash_set<tstring>& optimizations_enabled,
    const absl::flat_hash_set<tstring>& optimizations_default);

// Returns the default CPU budget.
inline int GetCpuBudget() {
  static bool in_experiment = GetExperiments().contains("tune_cpu_budget");
  return (in_experiment ? 1.2 : 1.0) * port::NumSchedulableCPUs();
}

// Returns the initial value for parallelism parameter before the first Autotune
// optimization.
int64 GetAutotuneDefaultParallelism(IteratorContext* ctx);

// Registry of tf.data experiments.
class DatasetExperimentRegistry {
 public:
  using JobSelector = std::function<bool(
      std::function<uint64_t(const string&)> hash_func,
      const std::string& experiment_name, const std::string& job_name)>;
  using TaskSelector = std::function<bool(int64_t task_id)>;

  struct ExperimentSelector {
    JobSelector job_selector;
    TaskSelector task_selector;
  };

  // Registers the experiment.
  static void Register(const string& experiment, JobSelector job_selector,
                       TaskSelector task_selector);

  // Returns all registered experiments.
  static absl::flat_hash_map<string, ExperimentSelector> Experiments();
};

// Helper class to register a dataset experiment.
class DatasetExperimentRegistrar {
 public:
  explicit DatasetExperimentRegistrar(
      const string& experiment,
      DatasetExperimentRegistry::JobSelector job_selector,
      DatasetExperimentRegistry::TaskSelector task_selector) {
    DatasetExperimentRegistry::Register(experiment, job_selector,
                                        task_selector);
  }
};

// Macro that can be used to register a dataset experiment.
#define REGISTER_DATASET_EXPERIMENT(experiment, job_selector, task_selector)  \
  REGISTER_DATASET_OP_NAME_UNIQ_HELPER(__COUNTER__, experiment, job_selector, \
                                       task_selector)

#define REGISTER_DATASET_OP_NAME_UNIQ_HELPER(ctr, experiment, job_selector, \
                                             task_selector)                 \
  REGISTER_DATASET_OP_NAME_UNIQ(ctr, experiment, job_selector, task_selector)

#define REGISTER_DATASET_OP_NAME_UNIQ(ctr, experiment, job_selector, \
                                      task_selector)                 \
  static ::tensorflow::data::DatasetExperimentRegistrar              \
      registrar__body__##ctr##__object(experiment, job_selector,     \
                                       task_selector)

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_DATASET_UTILS_H_
