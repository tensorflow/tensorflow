// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "tensorflow/contrib/boosted_trees/lib/utils/parallel_for.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/tensor_utils.h"
#include "tensorflow/contrib/boosted_trees/resources/stamped_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace boosted_trees {

namespace {
const char* const kStampTokenName = "stamp_token";
const char* const kNextStampTokenName = "next_stamp_token";

struct PartitionKey {
  PartitionKey() : partition_id(-1), feature_id(-1) {}

  PartitionKey(int32 p, int64 f) : partition_id(p), feature_id(f) {}

  bool operator==(const PartitionKey& other) const {
    return (feature_id == other.feature_id) &&
           (partition_id == other.partition_id);
  }

  // Compare for PartitionKey.
  struct Less {
    bool operator()(const PartitionKey& a, const PartitionKey& b) const {
      if (a.partition_id < b.partition_id) {
        return true;
      }
      if ((a.partition_id == b.partition_id) && (a.feature_id < b.feature_id)) {
        return true;
      }
      return false;
    }
  };

  // Tree partition defined by traversing the tree to the leaf.
  int32 partition_id;

  // Feature Id within the feature column.
  int64 feature_id;
};

template <typename GradientType, typename HessianType>
class StatsAccumulatorResource : public boosted_trees::StampedResource {
  using StatsByPartition =
      std::map<PartitionKey, std::pair<GradientType, HessianType>,
               PartitionKey::Less>;

 public:
  StatsAccumulatorResource(const TensorShape& gradient_shape,
                           const TensorShape& hessian_shape)
      : gradient_shape_(gradient_shape),
        hessian_shape_(hessian_shape),
        num_updates_(0) {
    // If GradientType/HessianType is scalar float then the shapes should be
    // scalar and vice versa.
    CHECK_EQ((std::is_same<GradientType, float>::value),
             TensorShapeUtils::IsScalar(gradient_shape));
    CHECK_EQ((std::is_same<HessianType, float>::value),
             TensorShapeUtils::IsScalar(hessian_shape));
  }

  string DebugString() override {
    return strings::StrCat("StatsAccumulatorResource[size=", values_.size(),
                           "]");
  }

  void Clear() {
    values_.clear();
    num_updates_ = 0;
  }

  tensorflow::mutex* mutex() { return &mu_; }
  StatsByPartition* mutable_values() { return &values_; }
  const StatsByPartition& values() const { return values_; }
  const int64& num_updates() const { return num_updates_; }
  void set_num_updates(int64 val) { num_updates_ = val; }
  const TensorShape& gradient_shape() const { return gradient_shape_; }
  const TensorShape& hessian_shape() const { return hessian_shape_; }

 private:
  // Key into a specific partition to accumulate stats for the specified feature
  // id.
  StatsByPartition values_;
  const TensorShape gradient_shape_;
  const TensorShape hessian_shape_;
  int64 num_updates_;
  tensorflow::mutex mu_;
  TF_DISALLOW_COPY_AND_ASSIGN(StatsAccumulatorResource);
};

using StatsAccumulatorScalarResource = StatsAccumulatorResource<float, float>;
using StatsAccumulatorTensorResource =
    StatsAccumulatorResource<std::vector<float>, std::vector<float>>;

void SerializeScalarAccumulatorToOutput(
    const StatsAccumulatorScalarResource& accumulator_resource,
    OpKernelContext* context) {
  int64 num_slots = accumulator_resource.values().size();
  Tensor* partition_ids_t = nullptr;
  OP_REQUIRES_OK(
      context,
      context->allocate_output("output_partition_ids", TensorShape({num_slots}),
                               &partition_ids_t));
  auto partition_ids = partition_ids_t->vec<int32>();

  Tensor* feature_ids_t = nullptr;
  OP_REQUIRES_OK(
      context,
      context->allocate_output("output_feature_ids", TensorShape({num_slots}),
                               &feature_ids_t));
  auto feature_ids = feature_ids_t->vec<int64>();

  Tensor* gradients_t = nullptr;
  OP_REQUIRES_OK(
      context,
      context->allocate_output("output_gradients", TensorShape({num_slots}),
                               &gradients_t));
  auto gradients = gradients_t->vec<float>();

  Tensor* hessians_t = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(
                     "output_hessians", TensorShape({num_slots}), &hessians_t));
  auto hessians = hessians_t->vec<float>();

  int i = 0;
  for (const auto& iter : accumulator_resource.values()) {
    partition_ids(i) = iter.first.partition_id;
    feature_ids(i) = iter.first.feature_id;
    gradients(i) = iter.second.first;
    hessians(i) = iter.second.second;
    ++i;
  }
}

void SerializeTensorAccumulatorToOutput(
    const StatsAccumulatorTensorResource& accumulator_resource,
    OpKernelContext* context) {
  int64 num_slots = accumulator_resource.values().size();
  Tensor* partition_ids_t = nullptr;
  OP_REQUIRES_OK(
      context,
      context->allocate_output("output_partition_ids", TensorShape({num_slots}),
                               &partition_ids_t));
  auto partition_ids = partition_ids_t->vec<int32>();

  Tensor* feature_ids_t = nullptr;
  OP_REQUIRES_OK(
      context,
      context->allocate_output("output_feature_ids", TensorShape({num_slots}),
                               &feature_ids_t));
  auto feature_ids = feature_ids_t->vec<int64>();

  TensorShape gradient_shape = accumulator_resource.gradient_shape();
  int64 num_gradient_elements = gradient_shape.num_elements();
  gradient_shape.InsertDim(0, num_slots);
  Tensor* gradients_t = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output("output_gradients", gradient_shape,
                                          &gradients_t));
  auto gradients = gradients_t->flat_outer_dims<float>();

  TensorShape hessian_shape = accumulator_resource.hessian_shape();
  int64 num_hessian_elements = hessian_shape.num_elements();
  hessian_shape.InsertDim(0, num_slots);
  Tensor* hessians_t = nullptr;
  OP_REQUIRES_OK(
      context,
      context->allocate_output("output_hessians", hessian_shape, &hessians_t));
  auto hessians = hessians_t->flat_outer_dims<float>();

  int i = 0;
  for (const auto& iter : accumulator_resource.values()) {
    partition_ids(i) = iter.first.partition_id;
    feature_ids(i) = iter.first.feature_id;
    for (int j = 0; j < num_gradient_elements; ++j) {
      gradients(i, j) = iter.second.first[j];
    }
    for (int j = 0; j < num_hessian_elements; ++j) {
      hessians(i, j) = iter.second.second[j];
    }
    ++i;
  }
}

void AddToScalarAccumulator(
    StatsAccumulatorScalarResource* accumulator_resource,
    const Tensor& partition_ids_t, const Tensor& feature_ids_t,
    const Tensor& gradients_t, const Tensor& hessians_t) {
  accumulator_resource->set_num_updates(accumulator_resource->num_updates() +
                                        1);
  const TensorShape& partition_ids_shape = partition_ids_t.shape();
  const auto& partition_ids = partition_ids_t.vec<int32>();
  const auto& feature_ids = feature_ids_t.vec<int64>();
  const auto& gradients = gradients_t.vec<float>();
  const auto& hessians = hessians_t.vec<float>();

  int64 num_updates = partition_ids_shape.dim_size(0);
  auto stats_map = accumulator_resource->mutable_values();
  for (int64 i = 0; i < num_updates; ++i) {
    const auto key = PartitionKey(partition_ids(i), feature_ids(i));
    auto itr = stats_map->find(key);
    if (itr != stats_map->end()) {
      itr->second.first += gradients(i);
      itr->second.second += hessians(i);
    } else {
      (*stats_map)[key] = {gradients(i), hessians(i)};
    }
  }
}

void AddToScalarAccumulator(
    StatsAccumulatorScalarResource* accumulator_resource,
    OpKernelContext* context) {
  const Tensor* partition_ids_t;
  OP_REQUIRES_OK(context, context->input("partition_ids", &partition_ids_t));
  const Tensor* feature_ids_t;
  OP_REQUIRES_OK(context, context->input("feature_ids", &feature_ids_t));
  const Tensor* gradients_t;
  OP_REQUIRES_OK(context, context->input("gradients", &gradients_t));
  const Tensor* hessians_t;
  OP_REQUIRES_OK(context, context->input("hessians", &hessians_t));
  AddToScalarAccumulator(accumulator_resource, *partition_ids_t, *feature_ids_t,
                         *gradients_t, *hessians_t);
}

void AddToTensorAccumulator(
    StatsAccumulatorTensorResource* accumulator_resource,
    const Tensor& partition_ids_t, const Tensor& feature_ids_t,
    const Tensor& gradients_t, const Tensor& hessians_t,
    OpKernelContext* context) {
  accumulator_resource->set_num_updates(accumulator_resource->num_updates() +
                                        1);

  const TensorShape& partition_ids_shape = partition_ids_t.shape();
  const auto& partition_ids = partition_ids_t.vec<int32>();
  const auto& feature_ids = feature_ids_t.vec<int64>();
  TensorShape gradients_shape = gradients_t.shape();
  const auto& gradients = gradients_t.flat_outer_dims<float>();
  TensorShape hessians_shape = hessians_t.shape();
  const auto& hessians = hessians_t.flat_outer_dims<float>();

  gradients_shape.RemoveDim(0);
  hessians_shape.RemoveDim(0);

  // TODO(soroush): Move gradient and hessian shape check to ShapeFn.
  OP_REQUIRES(
      context, gradients_shape == accumulator_resource->gradient_shape(),
      errors::InvalidArgument(strings::StrCat(
          "Gradients dimensions must match: ", gradients_shape.DebugString(),
          ", ", accumulator_resource->gradient_shape().DebugString())));

  OP_REQUIRES(
      context, hessians_shape == accumulator_resource->hessian_shape(),
      errors::InvalidArgument(strings::StrCat(
          "Hessian dimensions must match: ", hessians_shape.DebugString(), ", ",
          accumulator_resource->hessian_shape().DebugString())));

  int64 num_updates = partition_ids_shape.dim_size(0);
  auto stats_map = accumulator_resource->mutable_values();
  for (int64 i = 0; i < num_updates; ++i) {
    const auto key = PartitionKey(partition_ids(i), feature_ids(i));
    auto itr = stats_map->find(key);
    if (itr == stats_map->end()) {
      std::vector<float> new_gradients(gradients_shape.num_elements());
      for (int j = 0; j < gradients_shape.num_elements(); ++j) {
        new_gradients[j] = gradients(i, j);
      }
      std::vector<float> new_hessians(hessians_shape.num_elements());
      for (int j = 0; j < hessians_shape.num_elements(); ++j) {
        new_hessians[j] = hessians(i, j);
      }
      (*stats_map)[key] = {new_gradients, new_hessians};
    } else {
      auto& stored_gradients = itr->second.first;
      for (int j = 0; j < gradients_shape.num_elements(); ++j) {
        stored_gradients[j] += gradients(i, j);
      }
      auto& stored_hessians = itr->second.second;
      for (int j = 0; j < hessians_shape.num_elements(); ++j) {
        stored_hessians[j] += hessians(i, j);
      }
    }
  }
}

void AddToTensorAccumulator(
    StatsAccumulatorTensorResource* accumulator_resource,
    OpKernelContext* context) {
  const Tensor* partition_ids_t;
  OP_REQUIRES_OK(context, context->input("partition_ids", &partition_ids_t));
  const Tensor* feature_ids_t;
  OP_REQUIRES_OK(context, context->input("feature_ids", &feature_ids_t));
  const Tensor* gradients_t;
  OP_REQUIRES_OK(context, context->input("gradients", &gradients_t));
  const Tensor* hessians_t;
  OP_REQUIRES_OK(context, context->input("hessians", &hessians_t));
  AddToTensorAccumulator(accumulator_resource, *partition_ids_t, *feature_ids_t,
                         *gradients_t, *hessians_t, context);
}

}  // namespace

REGISTER_RESOURCE_HANDLE_KERNEL(StatsAccumulatorScalarResource);
REGISTER_RESOURCE_HANDLE_KERNEL(StatsAccumulatorTensorResource);

REGISTER_KERNEL_BUILDER(
    Name("StatsAccumulatorScalarIsInitialized").Device(DEVICE_CPU),
    IsResourceInitialized<StatsAccumulatorScalarResource>);

REGISTER_KERNEL_BUILDER(
    Name("StatsAccumulatorTensorIsInitialized").Device(DEVICE_CPU),
    IsResourceInitialized<StatsAccumulatorTensorResource>);

class CreateStatsAccumulatorScalarOp : public OpKernel {
 public:
  explicit CreateStatsAccumulatorScalarOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));

    TensorShape gradient_shape = TensorShape({});
    TensorShape hessian_shape = TensorShape({});

    auto* result =
        new StatsAccumulatorScalarResource(gradient_shape, hessian_shape);
    result->set_stamp(stamp_token_t->scalar<int64>()());
    // Only create one, if one does not exist already. Report status for all
    // other exceptions. If one already exists, it unrefs the new one.
    auto status = CreateResource(context, HandleFromInput(context, 0), result);
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      OP_REQUIRES(context, false, status);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CreateStatsAccumulatorScalar").Device(DEVICE_CPU),
                        CreateStatsAccumulatorScalarOp);

class CreateStatsAccumulatorTensorOp : public OpKernel {
 public:
  explicit CreateStatsAccumulatorTensorOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input("stamp_token", &stamp_token_t));

    const Tensor* gradient_shape_t;
    OP_REQUIRES_OK(
        context, context->input("per_slot_gradient_shape", &gradient_shape_t));

    const Tensor* hessian_shape_t;
    OP_REQUIRES_OK(context,
                   context->input("per_slot_hessian_shape", &hessian_shape_t));
    TensorShape gradient_shape = TensorShape(gradient_shape_t->vec<int64>());
    TensorShape hessian_shape = TensorShape(hessian_shape_t->vec<int64>());
    auto* result =
        new StatsAccumulatorTensorResource(gradient_shape, hessian_shape);
    result->set_stamp(stamp_token_t->scalar<int64>()());

    // Only create one, if one does not exist already. Report status for all
    // other exceptions. If one already exists, it unrefs the new one.
    auto status = CreateResource(context, HandleFromInput(context, 0), result);
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      OP_REQUIRES(context, false, status);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CreateStatsAccumulatorTensor").Device(DEVICE_CPU),
                        CreateStatsAccumulatorTensorOp);

class StatsAccumulatorScalarAddOp : public OpKernel {
 public:
  explicit StatsAccumulatorScalarAddOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OpInputList resource_handle_list;
    OP_REQUIRES_OK(context, context->input_list("stats_accumulator_handles",
                                                &resource_handle_list));
    OpInputList partition_ids_list;
    OP_REQUIRES_OK(context,
                   context->input_list("partition_ids", &partition_ids_list));

    OpInputList feature_ids_list;
    OP_REQUIRES_OK(context,
                   context->input_list("feature_ids", &feature_ids_list));
    OpInputList gradients_list;
    OP_REQUIRES_OK(context, context->input_list("gradients", &gradients_list));
    OpInputList hessians_list;
    OP_REQUIRES_OK(context, context->input_list("hessians", &hessians_list));

    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    thread::ThreadPool* const worker_threads =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    boosted_trees::utils::ParallelFor(
        resource_handle_list.size(), worker_threads->NumThreads(),
        worker_threads,
        [&context, &resource_handle_list, &partition_ids_list,
         &feature_ids_list, &gradients_list, &hessians_list,
         stamp_token](int64 start, int64 end) {
          for (int resource_handle_idx = start; resource_handle_idx < end;
               ++resource_handle_idx) {
            ResourceHandle handle = resource_handle_list[resource_handle_idx]
                                        .flat<ResourceHandle>()(0);

            StatsAccumulatorScalarResource* accumulator_resource;
            OP_REQUIRES_OK(context, LookupResource(context, handle,
                                                   &accumulator_resource));
            mutex_lock l(*accumulator_resource->mutex());
            core::ScopedUnref unref_me(accumulator_resource);

            // If the stamp is invalid we drop the update.
            if (!accumulator_resource->is_stamp_valid(stamp_token)) {
              VLOG(1) << "Invalid stamp token in StatsAccumulatorScalarAddOp. "
                      << "Passed stamp token: " << stamp_token << " "
                      << "Current token: " << accumulator_resource->stamp();
              return;
            }
            AddToScalarAccumulator(accumulator_resource,
                                   partition_ids_list[resource_handle_idx],
                                   feature_ids_list[resource_handle_idx],
                                   gradients_list[resource_handle_idx],
                                   hessians_list[resource_handle_idx]);
          }
        });
  }
};

REGISTER_KERNEL_BUILDER(Name("StatsAccumulatorScalarAdd").Device(DEVICE_CPU),
                        StatsAccumulatorScalarAddOp);

class StatsAccumulatorTensorAddOp : public OpKernel {
 public:
  explicit StatsAccumulatorTensorAddOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    OpInputList resource_handle_list;
    OP_REQUIRES_OK(context, context->input_list("stats_accumulator_handles",
                                                &resource_handle_list));
    OpInputList partition_ids_list;
    OP_REQUIRES_OK(context,
                   context->input_list("partition_ids", &partition_ids_list));

    OpInputList feature_ids_list;
    OP_REQUIRES_OK(context,
                   context->input_list("feature_ids", &feature_ids_list));
    OpInputList gradients_list;
    OP_REQUIRES_OK(context, context->input_list("gradients", &gradients_list));
    OpInputList hessians_list;
    OP_REQUIRES_OK(context, context->input_list("hessians", &hessians_list));

    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    thread::ThreadPool* const worker_threads =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    boosted_trees::utils::ParallelFor(
        resource_handle_list.size(), worker_threads->NumThreads(),
        worker_threads,
        [&context, &resource_handle_list, &partition_ids_list,
         &feature_ids_list, &gradients_list, &hessians_list,
         stamp_token](int64 start, int64 end) {
          for (int resource_handle_idx = start; resource_handle_idx < end;
               ++resource_handle_idx) {
            ResourceHandle handle = resource_handle_list[resource_handle_idx]
                                        .flat<ResourceHandle>()(0);

            StatsAccumulatorTensorResource* accumulator_resource;
            OP_REQUIRES_OK(context, LookupResource(context, handle,
                                                   &accumulator_resource));
            mutex_lock l(*accumulator_resource->mutex());
            core::ScopedUnref unref_me(accumulator_resource);

            // If the stamp is invalid we drop the update.
            if (!accumulator_resource->is_stamp_valid(stamp_token)) {
              VLOG(1) << "Invalid stamp token in StatsAccumulatorScalarAddOp. "
                      << "Passed stamp token: " << stamp_token << " "
                      << "Current token: " << accumulator_resource->stamp();
              return;
            }
            AddToTensorAccumulator(accumulator_resource,
                                   partition_ids_list[resource_handle_idx],
                                   feature_ids_list[resource_handle_idx],
                                   gradients_list[resource_handle_idx],
                                   hessians_list[resource_handle_idx], context);
          }
        });
  }
};

REGISTER_KERNEL_BUILDER(Name("StatsAccumulatorTensorAdd").Device(DEVICE_CPU),
                        StatsAccumulatorTensorAddOp);

class StatsAccumulatorScalarFlushOp : public OpKernel {
 public:
  explicit StatsAccumulatorScalarFlushOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    StatsAccumulatorScalarResource* accumulator_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &accumulator_resource));
    mutex_lock l(*accumulator_resource->mutex());
    core::ScopedUnref unref_me(accumulator_resource);

    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    // If the stamp is invalid we restart the PS. It shouldn't happen since
    // only Chief should call this function and chief is guaranteed to be in
    // a consistent state.
    CHECK(accumulator_resource->is_stamp_valid(stamp_token));

    const Tensor* next_stamp_token_t;
    OP_REQUIRES_OK(context,
                   context->input(kNextStampTokenName, &next_stamp_token_t));
    int64 next_stamp_token = next_stamp_token_t->scalar<int64>()();
    CHECK(stamp_token != next_stamp_token);

    SerializeScalarAccumulatorToOutput(*accumulator_resource, context);
    Tensor* num_updates_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("num_updates", TensorShape({}),
                                            &num_updates_t));
    num_updates_t->scalar<int64>()() = accumulator_resource->num_updates();

    accumulator_resource->Clear();
    accumulator_resource->set_stamp(next_stamp_token);
  }
};

REGISTER_KERNEL_BUILDER(Name("StatsAccumulatorScalarFlush").Device(DEVICE_CPU),
                        StatsAccumulatorScalarFlushOp);

class StatsAccumulatorTensorFlushOp : public OpKernel {
 public:
  explicit StatsAccumulatorTensorFlushOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    StatsAccumulatorTensorResource* accumulator_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &accumulator_resource));
    mutex_lock l(*accumulator_resource->mutex());
    core::ScopedUnref unref_me(accumulator_resource);

    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();

    const Tensor* next_stamp_token_t;
    OP_REQUIRES_OK(context,
                   context->input(kNextStampTokenName, &next_stamp_token_t));
    int64 next_stamp_token = next_stamp_token_t->scalar<int64>()();

    // If the stamp is invalid we restart the PS. It shouldn't happen since
    // only Chief should call this function and chief is guaranteed to be in
    // a consistent state.
    CHECK(accumulator_resource->is_stamp_valid(stamp_token));
    CHECK(stamp_token != next_stamp_token);
    SerializeTensorAccumulatorToOutput(*accumulator_resource, context);
    Tensor* num_updates_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("num_updates", TensorShape({}),
                                            &num_updates_t));
    num_updates_t->scalar<int64>()() = accumulator_resource->num_updates();
    accumulator_resource->Clear();
    accumulator_resource->set_stamp(next_stamp_token);
  }
};

REGISTER_KERNEL_BUILDER(Name("StatsAccumulatorTensorFlush").Device(DEVICE_CPU),
                        StatsAccumulatorTensorFlushOp);

class StatsAccumulatorScalarDeserializeOp : public OpKernel {
 public:
  explicit StatsAccumulatorScalarDeserializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    StatsAccumulatorScalarResource* accumulator_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &accumulator_resource));
    mutex_lock l(*accumulator_resource->mutex());
    core::ScopedUnref unref_me(accumulator_resource);

    // Check the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();
    accumulator_resource->Clear();
    accumulator_resource->set_stamp(stamp_token);
    AddToScalarAccumulator(accumulator_resource, context);
    const Tensor* num_updates_t;
    OP_REQUIRES_OK(context, context->input("num_updates", &num_updates_t));
    accumulator_resource->set_num_updates(num_updates_t->scalar<int64>()());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("StatsAccumulatorScalarDeserialize").Device(DEVICE_CPU),
    StatsAccumulatorScalarDeserializeOp);

class StatsAccumulatorTensorDeserializeOp : public OpKernel {
 public:
  explicit StatsAccumulatorTensorDeserializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    StatsAccumulatorTensorResource* accumulator_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &accumulator_resource));
    mutex_lock l(*accumulator_resource->mutex());
    core::ScopedUnref unref_me(accumulator_resource);

    // Check the stamp token.
    const Tensor* stamp_token_t;
    OP_REQUIRES_OK(context, context->input(kStampTokenName, &stamp_token_t));
    int64 stamp_token = stamp_token_t->scalar<int64>()();
    accumulator_resource->Clear();
    accumulator_resource->set_stamp(stamp_token);
    AddToTensorAccumulator(accumulator_resource, context);
    const Tensor* num_updates_t;
    OP_REQUIRES_OK(context, context->input("num_updates", &num_updates_t));
    accumulator_resource->set_num_updates(num_updates_t->scalar<int64>()());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("StatsAccumulatorTensorDeserialize").Device(DEVICE_CPU),
    StatsAccumulatorTensorDeserializeOp);

class StatsAccumulatorScalarSerializeOp : public OpKernel {
 public:
  explicit StatsAccumulatorScalarSerializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    StatsAccumulatorScalarResource* accumulator_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &accumulator_resource));
    mutex_lock l(*accumulator_resource->mutex());
    core::ScopedUnref unref_me(accumulator_resource);
    SerializeScalarAccumulatorToOutput(*accumulator_resource, context);
    Tensor* stamp_token_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("stamp_token", TensorShape({}),
                                            &stamp_token_t));
    stamp_token_t->scalar<int64>()() = accumulator_resource->stamp();

    Tensor* num_updates_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("num_updates", TensorShape({}),
                                            &num_updates_t));
    num_updates_t->scalar<int64>()() = accumulator_resource->num_updates();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("StatsAccumulatorScalarSerialize").Device(DEVICE_CPU),
    StatsAccumulatorScalarSerializeOp);

class StatsAccumulatorTensorSerializeOp : public OpKernel {
 public:
  explicit StatsAccumulatorTensorSerializeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    StatsAccumulatorTensorResource* accumulator_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &accumulator_resource));
    mutex_lock l(*accumulator_resource->mutex());
    core::ScopedUnref unref_me(accumulator_resource);
    SerializeTensorAccumulatorToOutput(*accumulator_resource, context);
    Tensor* stamp_token_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("stamp_token", TensorShape({}),
                                            &stamp_token_t));
    stamp_token_t->scalar<int64>()() = accumulator_resource->stamp();

    Tensor* num_updates_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("num_updates", TensorShape({}),
                                            &num_updates_t));
    num_updates_t->scalar<int64>()() = accumulator_resource->num_updates();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("StatsAccumulatorTensorSerialize").Device(DEVICE_CPU),
    StatsAccumulatorTensorSerializeOp);

class StatsAccumulatorScalarMakeSummaryOp : public OpKernel {
 public:
  explicit StatsAccumulatorScalarMakeSummaryOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TensorShape gradient_shape = TensorShape({});
    TensorShape hessian_shape = TensorShape({});
    StatsAccumulatorScalarResource* accumulator_resource =
        new StatsAccumulatorScalarResource(gradient_shape, hessian_shape);
    core::ScopedUnref unref_me(accumulator_resource);
    // Check the stamp token.
    AddToScalarAccumulator(accumulator_resource, context);
    SerializeScalarAccumulatorToOutput(*accumulator_resource, context);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("StatsAccumulatorScalarMakeSummary").Device(DEVICE_CPU),
    StatsAccumulatorScalarMakeSummaryOp);

class StatsAccumulatorTensorMakeSummaryOp : public OpKernel {
 public:
  explicit StatsAccumulatorTensorMakeSummaryOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* gradients_t;
    OP_REQUIRES_OK(context, context->input("gradients", &gradients_t));
    TensorShape gradients_shape = gradients_t->shape();
    gradients_shape.RemoveDim(0);

    const Tensor* hessians_t;
    OP_REQUIRES_OK(context, context->input("hessians", &hessians_t));
    TensorShape hessians_shape = hessians_t->shape();
    hessians_shape.RemoveDim(0);

    StatsAccumulatorTensorResource* accumulator_resource =
        new StatsAccumulatorTensorResource(gradients_shape, hessians_shape);
    core::ScopedUnref unref_me(accumulator_resource);
    // Check the stamp token.
    AddToTensorAccumulator(accumulator_resource, context);
    SerializeTensorAccumulatorToOutput(*accumulator_resource, context);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("StatsAccumulatorTensorMakeSummary").Device(DEVICE_CPU),
    StatsAccumulatorTensorMakeSummaryOp);

}  // namespace boosted_trees
}  // namespace tensorflow
