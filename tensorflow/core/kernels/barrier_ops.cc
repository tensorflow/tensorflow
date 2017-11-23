/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
// See docs in ../ops/data_flow_ops.cc.

#include <limits.h>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/priority_queue.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace barrier {

class Barrier : public ResourceBase {
 public:
  typedef std::vector<Tensor> Tuple;
  typedef std::function<void()> DoneCallback;
  typedef std::function<void(const Tensor&, const Tensor&, const Tuple&)>
      IndicesKeysValuesCallback;

  Barrier(const DataTypeVector& value_component_types,
          const std::vector<TensorShape>& value_component_shapes,
          const string& name)
      : closed_(false),
        queue_closed_(false),
        queue_cancelled_(false),
        cancel_pending_enqueues_(false),
        value_component_types_(value_component_types),
        value_component_shapes_(value_component_shapes),
        name_(name),
        input_index_(std::numeric_limits<int64>::min()) {
    DataTypeVector queue_component_types;
    std::vector<TensorShape> queue_component_shapes;

    // First queue component is for the input index;
    // Second queue component is for the key;
    // remaining queue components are for the value.
    queue_component_types.push_back(DT_INT64);
    queue_component_types.push_back(DT_STRING);
    for (DataType dt : value_component_types) {
      queue_component_types.push_back(dt);
    }

    // NOTE(mrry): PriorityQueue expects all shapes specified because
    // we'll be issuing TakeMany.
    queue_component_shapes.push_back(TensorShape({}));
    queue_component_shapes.push_back(TensorShape({}));
    queue_component_shapes.insert(queue_component_shapes.end(),
                                  value_component_shapes.begin(),
                                  value_component_shapes.end());

    ready_queue_ = new PriorityQueue(
        QueueBase::kUnbounded /* capacity */, queue_component_types,
        queue_component_shapes, strings::StrCat(name_, "_queue"));
  }

  Status Initialize() { return ready_queue_->Initialize(); }

  template <typename T>
  void TryInsertMany(const Tensor& keys, int component_index,
                     const Tensor& values, OpKernelContext* ctx,
                     const DoneCallback& callback) {
    TensorShape element_shape = values.shape();
    OP_REQUIRES_ASYNC(
        ctx, keys.NumElements() == 0 || element_shape.num_elements() > 0,
        errors::InvalidArgument("Tensors with no elements are not supported ",
                                name_, ": received shape ",
                                element_shape.DebugString()),
        callback);
    if (element_shape.dims() > 0) element_shape.RemoveDim(0);
    const std::size_t num_inserted = keys.NumElements();

    // For each key, update the corresponding incomplete tuple with the
    // the corresponding given value at component_index.
    // This will be passed to the final callback at the very end.
    bool new_elements = false;

    // Will be used for the final insert into the queue.
    Tuple insert_tuple;

    {
      mutex_lock lock(mu_);
      if (closed_) {
        OP_REQUIRES_ASYNC(
            ctx, !cancel_pending_enqueues_ &&
                     (num_inserted == 0 || !incomplete_.empty()),
            errors::Cancelled(
                "Barrier ", name_, " is closed.  Pending enqueues cancelled: ",
                cancel_pending_enqueues_, ".  Number of new insertions: ",
                num_inserted, ".  Number of incomplete keys: ",
                incomplete_.size(), "."),
            callback);
      }

      // Step 1: insert into the incomplete map and identify which
      // entries are, in fact, complete and ready for enqueueing.  Store
      // them in a vector
      std::vector<Tuple> ready_tuples;

      for (int i = 0; i < num_inserted; ++i) {
        OP_REQUIRES_OK_ASYNC(
            ctx, InsertOneLocked<T>(ctx, keys, values, element_shape,
                                    component_index, i, &ready_tuples,
                                    &new_elements),
            callback);
      }

      if (new_elements) ++input_index_;

      // This probably won't happen before the heat death of the
      // universe, but who knows?  Moore's law FTW.
      OP_REQUIRES_ASYNC(
          ctx, input_index_ != std::numeric_limits<int64>::max(),
          errors::Internal(
              "Barrier has had ", input_index_,
              " insertions and can no longer keep track of new ones."),
          callback);

      if (ready_tuples.empty()) {
        // Nothing to insert into the queue - so return early.
        callback();
        return;
      }

      // We have something to Enqueue.  Convert the Tuples into a single
      // tuple by slicing entries into new Tensors.  This part is slow
      // but seems the cleanest solution for now.
      insert_tuple.reserve(2 + num_components());  // indices, keys, rest
      int insertion_size = ready_tuples.size();
      for (int i = 0; i < 2 + num_components(); ++i) {
        TensorShape component_shape(ready_tuples[0][i].shape());
        component_shape.InsertDim(0, insertion_size);
        Tensor component(ready_tuples[0][i].dtype(), component_shape);
        for (int b = 0; b < insertion_size; ++b) {
          OP_REQUIRES_OK_ASYNC(
              ctx,
              batch_util::CopyElementToSlice(std::move(ready_tuples[b][i]),
                                             &component, b),
              callback);
        }
        insert_tuple.push_back(component);
      }
    }

    // Update the input index for the next batch.
    ready_queue_->TryEnqueueMany(
        insert_tuple, ctx,
        // To avoid early closing of the queue, only close it if the
        // SQSS is closed, nothing is left in the incomplete set,
        // the queue is not already marked as closed, and (most
        // importantly), the queue has entries in it.
        [this, ctx, callback, component_index]() {
          if (!ctx->status().ok()) {
            callback();
            return;
          }
          {
            mutex_lock lock(mu_);
            int32 ready = ready_size();
            if (closed_ && incomplete_.empty() && queue_closed_ && ready > 0) {
              CloseQueueLocked(ctx, false, callback);
            } else {
              callback();
            }
            return;
          }
        });
  }

  void TryTakeMany(int num_elements, bool allow_small_batch, int64 timeout,
                   OpKernelContext* ctx,
                   const IndicesKeysValuesCallback& callback) {
    int num_elements_to_deliver = num_elements;
    {
      mutex_lock lock(mu_);
      if (closed_) {
        int available_elements = ready_size();
        if (allow_small_batch) {
          // We want to deliver a maximum of num_elements, if there are less
          // elements available, we deliver at most the available_elements. If
          // there are no
          // elements available, a call to TryTakeMany should fail with
          // OutOfRange. We trigger this error by setting the request here to 1.
          num_elements_to_deliver = std::min(num_elements, available_elements);
        } else {
          // We're happy to wait for additional elements to be completed.
          available_elements += incomplete_.size();
        }
        // If there are 0 available elements or less elements than the
        // number we can deliver, then we are done.
        if (available_elements < std::max(num_elements_to_deliver, 1)) {
          ctx->SetStatus(errors::OutOfRange(
              "Barrier '", name_, "' is closed and has ",
              "insufficient elements (requested ", num_elements_to_deliver,
              ", total size ", available_elements, ")"));
          callback(Tensor(DT_INT64), Tensor(DT_STRING), Tuple());
          return;
        }
      }
    }

    ready_queue_->TryDequeueMany(
        num_elements_to_deliver, ctx, allow_small_batch,
        [this, ctx, callback](const Tuple& t) {
          Tensor indices(DT_INT64);
          Tensor keys(DT_STRING);
          Tuple values;

          if (!ctx->status().ok()) {
            callback(indices, keys, values);
            return;
          }

          CHECK_EQ(t.size(), 2 + num_components());
          indices = t[0];
          keys = t[1];
          values.insert(values.begin(), t.begin() + 2, t.end());
          callback(indices, keys, values);
          return;
        });
  }

  void Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
             const DoneCallback& callback) {
    mutex_lock lock(mu_);
    // We're allowed to close twice if the first close wasn't a
    // cancel but the second one is.
    if (closed_ && (cancel_pending_enqueues_ || !cancel_pending_enqueues)) {
      ctx->SetStatus(
          errors::Cancelled("Barrier '", name_, "' is already closed."));
      callback();
      return;
    }
    cancel_pending_enqueues_ = cancel_pending_enqueues;
    closed_ = true;
    if (cancel_pending_enqueues_ || incomplete_.empty()) {
      incomplete_.clear();
      // CloseQueueLocked runs the callback
      CloseQueueLocked(ctx, cancel_pending_enqueues_, callback);
      return;
    }
    callback();
  }

  int32 ready_size() { return ready_queue_->size(); }

  int32 incomplete_size() {
    mutex_lock lock(mu_);
    return incomplete_.size();
  }

  const string& name() const { return name_; }
  int num_components() const { return value_component_types_.size(); }
  DataType component_type(int i) const {
    CHECK_GE(i, 0);
    CHECK_LT(static_cast<size_t>(i), value_component_types_.size());
    return value_component_types_[i];
  }
  const DataTypeVector component_types() const {
    return value_component_types_;
  }
  const gtl::ArraySlice<TensorShape> component_shapes() const {
    return value_component_shapes_;
  }

  ~Barrier() override EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    mutex_lock lock(mu_);
    incomplete_.clear();
    ready_queue_->Unref();
  }

  string DebugString() override { return "A barrier"; }

 protected:
  template <typename T>
  Status InsertOneLocked(OpKernelContext* ctx, const Tensor& keys,
                         const Tensor& values, const TensorShape& element_shape,
                         int component_index, int i,
                         std::vector<Tuple>* ready_tuples, bool* new_elements)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto keys_vec = keys.flat<string>();
    auto values_matrix = values.flat_outer_dims<T>();

    PersistentTuple* element_ptr;
    if (closed_) {
      element_ptr = gtl::FindOrNull(incomplete_, keys_vec(i));
      if (element_ptr == nullptr) {
        return errors::Cancelled(
            "Barrier ", name_,
            " is closed, but attempted to insert a brand new key: ",
            keys_vec(i), ".  Pending enqueues cancelled: ",
            cancel_pending_enqueues_, ".  Insertion index: ", i,
            ".  Number of incomplete keys: ", incomplete_.size(), ".");
      }
    } else {
      element_ptr =
          &gtl::LookupOrInsert(&incomplete_, keys_vec(i), PersistentTuple());
    }
    PersistentTuple& element = *element_ptr;

    if (element.empty()) {  // Never seen before key
      // Added a new element, for keeping track of the insertion index
      *new_elements = true;

      // Initialize the incomplete tuple for a new key.
      element.reserve(1 + num_components());

      // The first entry in element is the priority: the
      // input_index_, so that tensors that entered the Barrier
      // earlier have higher priority in the queue.
      PersistentTensor index_persistent_tensor;
      Tensor* allocate_index_tensor;
      TF_RETURN_IF_ERROR(ctx->allocate_persistent(DT_INT64, TensorShape({}),
                                                  &index_persistent_tensor,
                                                  &allocate_index_tensor));

      Tensor index_tensor(DT_INT64, TensorShape({}));
      allocate_index_tensor->scalar<int64>()() = input_index_;
      element.push_back(index_persistent_tensor);

      // The rest of the element stores uninitialized Tensors with
      // the appropriate dtype.
      for (int j = 0; j < num_components(); ++j) {
        Tensor uninitialized(component_type(j));
        element.push_back(PersistentTensor(uninitialized));
      }
    }
    const PersistentTensor& component = element[1 + component_index];
    if (component.IsInitialized() && component.NumElements() > 0) {
      return errors::InvalidArgument("Key ", keys_vec(i),
                                     " already has a value for component ",
                                     component_index, " in barrier ", name());
    }

    // Extract the slice corresponding to the value from the value Tensor,
    // and store it in the incomplete tuple at component_index.
    PersistentTensor next_element;
    Tensor* allocated_element;
    TF_RETURN_IF_ERROR(ctx->allocate_persistent(
        values.dtype(), element_shape, &next_element, &allocated_element));
    element[1 + component_index] = next_element;
    allocated_element->flat<T>() = values_matrix.template chip<0>(i);

    // Check the components of the tuple to see if it has become complete
    // (i.e. all of its components are initialized). If so, add it to the
    // ready queue.
    bool is_complete = true;
    for (int j = 0; is_complete && j < element.size(); ++j) {
      is_complete = element[j].IsInitialized() && element[j].NumElements() > 0;
    }
    if (is_complete) {
      // Add tuple to the ready queue. A queue tuple has the index
      // as the first element and the key as the second element,
      // followed by the value components.
      Tuple ready_tuple;
      ready_tuple.reserve(2 + num_components());  // index, key, rest
      // Build a tensor for the key. TODO(mrry): Something more efficient.
      PersistentTensor key;
      Tensor* allocated_key;
      TF_RETURN_IF_ERROR(ctx->allocate_persistent(DT_STRING, TensorShape({}),
                                                  &key, &allocated_key));
      ready_tuple.push_back(*element[0].AccessTensor(ctx));  // index
      ready_tuple.push_back(*allocated_key);                 // key
      ready_tuple[1].scalar<string>()() = keys_vec(i);       // set the key
      for (int j = 1; j < num_components() + 1; ++j) {
        ready_tuple.push_back(*element[j].AccessTensor(ctx));
      }
      incomplete_.erase(incomplete_.find(keys_vec(i)));
      TF_RETURN_IF_ERROR(ready_queue_->ValidateTuple(ready_tuple));
      ready_tuples->push_back(ready_tuple);
    }
    return Status::OK();
  }

  void CloseQueueLocked(OpKernelContext* ctx, bool cancel_pending_enqueues,
                        const DoneCallback& callback)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    // CloseQueueLocked may only be called with mu_ held.
    if (!cancel_pending_enqueues && queue_closed_) {
      callback();
      return;
    }
    if (cancel_pending_enqueues && queue_cancelled_) {
      callback();
      return;
    }
    queue_closed_ = true;
    if (cancel_pending_enqueues) queue_cancelled_ = true;
    if (!ready_queue_->is_closed()) {
      ready_queue_->Close(ctx, cancel_pending_enqueues, callback);
    }
  }

 private:
  typedef std::vector<PersistentTensor> PersistentTuple;
  mutex mu_;
  bool closed_ GUARDED_BY(mu_);
  bool queue_closed_ GUARDED_BY(mu_);
  bool queue_cancelled_ GUARDED_BY(mu_);
  bool cancel_pending_enqueues_ GUARDED_BY(mu_);
  const DataTypeVector value_component_types_;
  const std::vector<TensorShape>& value_component_shapes_;
  const string name_;
  int64 input_index_ GUARDED_BY(mu_);
  std::unordered_map<string, PersistentTuple> incomplete_ GUARDED_BY(mu_);
  PriorityQueue* ready_queue_;

  TF_DISALLOW_COPY_AND_ASSIGN(Barrier);
};

class BarrierOp : public ResourceOpKernel<Barrier> {
 public:
  explicit BarrierOp(OpKernelConstruction* context)
      : ResourceOpKernel(context) {
    OP_REQUIRES_OK(
        context, context->GetAttr("component_types", &value_component_types_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shapes", &value_component_shapes_));
    OP_REQUIRES(context,
                value_component_shapes_.size() == value_component_types_.size(),
                errors::InvalidArgument(
                    "All of the component shapes must be specified"));

    int32 value_capacity;
    OP_REQUIRES_OK(context, context->GetAttr("capacity", &value_capacity));
    OP_REQUIRES(context, value_capacity == -1,
                errors::InvalidArgument(
                    "Barrier only accepts capacity=-1.  Feed the "
                    "inputs to your Barrier through a queue to enforce a "
                    "limited capacity."));
  }

 private:
  Status CreateResource(Barrier** barrier) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    *barrier = new Barrier(value_component_types_, value_component_shapes_,
                           cinfo_.name());
    if (*barrier == nullptr) {
      return errors::ResourceExhausted("Failed to allocate barrier");
    }
    return (*barrier)->Initialize();
  }

  Status VerifyResource(Barrier* barrier) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (barrier->component_types() != value_component_types_) {
      return errors::InvalidArgument(
          "Shared barrier '", cinfo_.name(), "' has component types ",
          DataTypeSliceString(barrier->component_types()),
          " but requested component types were ",
          DataTypeSliceString(value_component_types_));
    }
    if (barrier->component_shapes() != value_component_shapes_) {
      return errors::InvalidArgument(
          "Shared barrier '", cinfo_.name(), "' has component shapes ",
          TensorShapeUtils::ShapeListString(barrier->component_shapes()),
          " but requested component shapes were ",
          TensorShapeUtils::ShapeListString(value_component_shapes_));
    }
    return Status::OK();
  }

  DataTypeVector value_component_types_;
  std::vector<TensorShape> value_component_shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(BarrierOp);
};

REGISTER_KERNEL_BUILDER(Name("Barrier").Device(DEVICE_CPU), BarrierOp);

class BarrierOpKernel : public AsyncOpKernel {
 public:
  explicit BarrierOpKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback callback) final {
    Barrier* barrier = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, GetResourceFromContext(ctx, "handle", &barrier),
                         callback);
    ComputeAsync(ctx, barrier, [this, callback, barrier]() {
      barrier->Unref();
      callback();
    });
  }

 protected:
  virtual void ComputeAsync(OpKernelContext* ctx, Barrier* barrier,
                            DoneCallback callback) = 0;
};

template <typename T>
class InsertManyOp : public BarrierOpKernel {
 public:
  explicit InsertManyOp(OpKernelConstruction* context)
      : BarrierOpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("component_index", &component_index_));
  }

 protected:
  void ComputeAsync(OpKernelContext* ctx, Barrier* barrier,
                    DoneCallback callback) override {
    OP_REQUIRES_ASYNC(
        ctx, component_index_ < barrier->num_components(),
        errors::InvalidArgument("The component ID is out of range ",
                                component_index_, " > num_components", " (= ",
                                barrier->num_components(), ")"),
        callback);
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->MatchSignature({DT_STRING_REF, DT_STRING,
                                  barrier->component_type(component_index_)},
                                 {}),
        callback);

    const Tensor* keys;
    const Tensor* values;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("keys", &keys), callback);
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("values", &values), callback);
    barrier->TryInsertMany<T>(*keys, component_index_, *values, ctx, callback);
  }

 private:
  int component_index_;
  TF_DISALLOW_COPY_AND_ASSIGN(InsertManyOp);
};

#define REGISTER_INSERTMANY(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("BarrierInsertMany").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      InsertManyOp<T>);

TF_CALL_ALL_TYPES(REGISTER_INSERTMANY);
#undef REGISTER_INSERTMANY

class TakeManyOp : public BarrierOpKernel {
 public:
  explicit TakeManyOp(OpKernelConstruction* context)
      : BarrierOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("timeout_ms", &timeout_));
    // TODO(keveman): Enable timeout.
    OP_REQUIRES(context, timeout_ == -1,
                errors::InvalidArgument("Timeout not supported yet."));

    OP_REQUIRES_OK(context,
                   context->GetAttr("allow_small_batch", &allow_small_batch_));
  }

 protected:
  void ComputeAsync(OpKernelContext* ctx, Barrier* barrier,
                    DoneCallback callback) override {
    const Tensor* Tnum_elements;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("num_elements", &Tnum_elements),
                         callback);
    OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsScalar(Tnum_elements->shape()),
                      errors::InvalidArgument("num_elements must be a scalar."),
                      callback);
    const int32 num_elements = Tnum_elements->scalar<int32>()();

    DataTypeVector expected_inputs = {DT_STRING_REF, DT_INT32};
    // The first output is the insertion index, the second output is the key.
    DataTypeVector expected_outputs = {DT_INT64, DT_STRING};
    for (DataType dt : barrier->component_types()) {
      expected_outputs.push_back(dt);
    }
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->MatchSignature(expected_inputs, expected_outputs), callback);

    barrier->TryTakeMany(
        num_elements, allow_small_batch_, timeout_, ctx,
        [ctx, callback](const Tensor& indices, const Tensor& keys,
                        const Barrier::Tuple& values) {
          if (!ctx->status().ok()) {
            callback();
            return;
          }
          // At this point, indices, keys, and values
          // have all been written to successfully.
          OP_REQUIRES_OK_ASYNC(ctx, ctx->set_output("indices", indices),
                               callback);
          OP_REQUIRES_OK_ASYNC(ctx, ctx->set_output("keys", keys), callback);
          OpOutputList values_output;
          OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("values", &values_output),
                               callback);
          for (size_t i = 0; i < values.size(); ++i) {
            values_output.set(i, values[i]);
          }
          callback();
          return;
        });
  }

 private:
  int64 timeout_;
  bool allow_small_batch_;
  TF_DISALLOW_COPY_AND_ASSIGN(TakeManyOp);
};

REGISTER_KERNEL_BUILDER(Name("BarrierTakeMany").Device(DEVICE_CPU), TakeManyOp);

class BarrierCloseOp : public BarrierOpKernel {
 public:
  explicit BarrierCloseOp(OpKernelConstruction* context)
      : BarrierOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("cancel_pending_enqueues",
                                             &cancel_pending_enqueues_));
  }

 protected:
  void ComputeAsync(OpKernelContext* ctx, Barrier* barrier,
                    DoneCallback callback) override {
    barrier->Close(ctx, cancel_pending_enqueues_, callback);
  }

 private:
  bool cancel_pending_enqueues_;
  TF_DISALLOW_COPY_AND_ASSIGN(BarrierCloseOp);
};

REGISTER_KERNEL_BUILDER(Name("BarrierClose").Device(DEVICE_CPU),
                        BarrierCloseOp);

class BarrierIncompleteSizeOp : public BarrierOpKernel {
 public:
  explicit BarrierIncompleteSizeOp(OpKernelConstruction* context)
      : BarrierOpKernel(context) {}

 protected:
  void ComputeAsync(OpKernelContext* ctx, Barrier* barrier,
                    DoneCallback callback) override {
    Tensor* Tsize = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, TensorShape({}), &Tsize),
                         callback);
    Tsize->scalar<int32>().setConstant(barrier->incomplete_size());
    callback();
  }
};

REGISTER_KERNEL_BUILDER(Name("BarrierIncompleteSize").Device(DEVICE_CPU),
                        BarrierIncompleteSizeOp);

class BarrierReadySizeOp : public BarrierOpKernel {
 public:
  explicit BarrierReadySizeOp(OpKernelConstruction* context)
      : BarrierOpKernel(context) {}

 protected:
  void ComputeAsync(OpKernelContext* ctx, Barrier* barrier,
                    DoneCallback callback) override {
    Tensor* Tsize = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, TensorShape({}), &Tsize),
                         callback);
    Tsize->scalar<int32>().setConstant(barrier->ready_size());
    callback();
  }
};

REGISTER_KERNEL_BUILDER(Name("BarrierReadySize").Device(DEVICE_CPU),
                        BarrierReadySizeOp);

}  // namespace barrier

}  // namespace tensorflow
