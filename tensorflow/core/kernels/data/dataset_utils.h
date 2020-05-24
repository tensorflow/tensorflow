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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_DATASET_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_DATASET_UTILS_H_

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

// Creates a resource handle with a unique name for the given resource.
template <typename T>
Status CreateHandle(OpKernelContext* ctx, T* resource,
                    const string& container_name, ResourceHandle* handle) {
  static std::atomic<int64> resource_id_counter(0);
  string unique_name =
      strings::StrCat(container_name, resource_id_counter.fetch_add(1));
  ResourceMgr* mgr = ctx->resource_manager();
  TF_RETURN_IF_ERROR(mgr->Create<T>(container_name, unique_name, resource));

  *handle = MakeResourceHandle(container_name, unique_name, *ctx->device(),
                               MakeTypeIndex<T>());
  return Status::OK();
}

template <typename T>
class AnonymousResourceOp : public OpKernel {
 public:
  explicit AnonymousResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

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
    OP_REQUIRES_OK(ctx, CreateHandle(ctx, resource, name(), &handle));
    Tensor* handle_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle_t));
    handle_t->scalar<ResourceHandle>()() = handle;

    if (create_deleter_) {
      Tensor* deleter_t;
      AllocatorAttributes attr;
      attr.set_on_host(true);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(1, TensorShape({}), &deleter_t, attr));
      deleter_t->scalar<Variant>()() =
          ResourceDeleter(handle, ctx->resource_manager());
    }
  }

 protected:
  virtual string name() = 0;

  virtual Status CreateResource(
      OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
      FunctionLibraryRuntime* lib, T** resource) = 0;

  bool create_deleter_ = true;
};

// Registers the given cancellation callback, returning a function that can be
// used to deregister the callback.
Status RegisterCancellationCallback(CancellationManager* cancellation_manager,
                                    std::function<void()> register_fn,
                                    std::function<void()>* deregister_fn);

// Returns Status::OK() if `expected` and `received` types match,
// errors::InvalidArgument otherwise.
Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received);

// Returns Status::OK() if `expected` and `received` shapes are compatible,
// errors::InvalidArgument otherwise.
Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received);

// Returns a stable hash of the subgraph rooted at the given node.
//
// NOTE: There is currently no guarantee that the hash of a subgraph will stay
// the same between TensorFlow builds.
Status HashNode(const GraphDef& graph, const NodeDef& node, uint64* hash);
Status HashNode(const GraphDef& graph, const NodeDef& node,
                const FunctionLibraryDefinition& flib_def, uint64* hash);

// Returns a stable hash of the given tensor.
//
// NOTE: There is currently no guarantee that the hash of a subgraph will stay
// the same between TensorFlow builds.
Status HashTensor(const Tensor& tensor, uint64* hash);

// Returns a stable hash of the given graph.
//
// NOTE: There is currently no guarantee that the hash of a subgraph will stay
// the same between TensorFlow builds.
Status HashGraph(const GraphDef& graph, uint64* hash);

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
std::pair<int64, int64> MaybeOverrideSeeds(std::pair<int64, int64> seeds);

// Helper class for reading data from a vector of VariantTensorData objects.
class VariantTensorDataReader : public IteratorStateReader {
 public:
  explicit VariantTensorDataReader(
      const std::vector<const VariantTensorData*>& data);

  Status ReadScalar(StringPiece key, int64* val) override;
  Status ReadScalar(StringPiece key, tstring* val) override;
  Status ReadTensor(StringPiece key, Tensor* val) override;
  bool Contains(StringPiece key) override;

  Status ReadScalar(StringPiece name, StringPiece key, int64* val) override;
  Status ReadScalar(StringPiece name, StringPiece key, tstring* val) override;
  Status ReadTensor(StringPiece name, StringPiece key, Tensor* val) override;
  bool Contains(StringPiece name, StringPiece key) override;

 private:
  template <typename T>
  Status ReadScalarInternal(StringPiece key, T* val);
  Status ReadTensorInternal(StringPiece key, Tensor* val);

  template <typename T>
  Status ReadScalarInternal(StringPiece name, StringPiece key, T* val);
  Status ReadTensorInternal(StringPiece name, StringPiece key, Tensor* val);

  std::map<string, std::map<string, size_t>> map_;
  std::map<string, const VariantTensorData*> data_;  // Not owned.
};

// Helper class used to build a list of VariantTensorData objects, one for each
// iterator which is determined from the key supplied from the Write* calls.
// Sample usage:
// VariantTensorDataWriter writer;
// writer.WriteScalar(full_name("buffer_size"), buffer_.size());
// writer.WriteScalar(full_name("num_threads"), threadpool_.size());
// ....
// std::vector<std::unique_ptr<VariantTensorData>> variants;
// writer.ReleaseData(&variants);
// Now the VariantTensorData objects can be used to serialize.
class VariantTensorDataWriter : public IteratorStateWriter {
 public:
  Status WriteScalar(StringPiece key, const int64 val) override;
  Status WriteScalar(StringPiece key, const tstring& val) override;
  Status WriteTensor(StringPiece key, const Tensor& val) override;

  Status WriteScalar(StringPiece name, StringPiece key,
                     const int64 val) override;
  Status WriteScalar(StringPiece name, StringPiece key,
                     const tstring& val) override;
  Status WriteTensor(StringPiece name, StringPiece key,
                     const Tensor& val) override;

  // Releases the built VariantTensorData's to `variants`. Clears out all
  // class state.
  void ReleaseData(std::vector<std::unique_ptr<VariantTensorData>>* variants);

  // Obtains a read-only version of the VariantTensorData's built.
  void GetData(std::vector<const VariantTensorData*>* variants);

 private:
  void MaybeFlush();
  void Reset();

  template <typename T>
  Status WriteScalarInternal(StringPiece key, const T& val);
  Status WriteTensorInternal(StringPiece key, const Tensor& val);

  template <typename T>
  Status WriteScalarInternal(StringPiece name, StringPiece key, const T& val);
  Status WriteTensorInternal(StringPiece name, StringPiece key,
                             const Tensor& val);

  bool is_flushed_ = false;
  std::map<string, std::unique_ptr<VariantTensorData>> data_;
  std::map<string, std::vector<string>> keys_;
};

// Adds the functions in `to_add` to `base`. If a function with a matching
// signature already exists in `base`, replaces it with the function from
// `to_add`.
Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionLibraryDefinition& to_add);
Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionDefLibrary& to_add);

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

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_DATASET_UTILS_H_
