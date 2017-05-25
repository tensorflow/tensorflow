/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/queue_base.h"

#include <vector>
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

template <DataType DT>
Status HandleSliceToElement(const Tensor& parent, Tensor* element,
                            int64 index) {
  typedef typename EnumToDataType<DT>::Type T;
  DCHECK_NE(parent.dim_size(0), 0);
  DCHECK_GE(index, 0);
  if (element->NumElements() != (parent.NumElements() / parent.dim_size(0))) {
    TensorShape chip_shape = parent.shape();
    chip_shape.RemoveDim(0);
    return errors::Internal(
        "HandleSliceToElement Cannot copy slice: number of elements does not "
        "match.  Shapes are: [element]: ",
        element->shape().DebugString(), ", [parent slice]: ",
        chip_shape.DebugString());
  }
  auto parent_as_matrix = parent.flat_outer_dims<T>();
  element->flat<T>() = parent_as_matrix.chip(index, 0);
  return Status::OK();
}

template <DataType DT>
Status HandleElementToSlice(const Tensor& element, Tensor* parent, int index) {
  typedef typename EnumToDataType<DT>::Type T;
  DCHECK_NE(parent->dim_size(0), 0);
  DCHECK_GE(index, 0);
  if (element.NumElements() != (parent->NumElements() / parent->dim_size(0))) {
    TensorShape chip_shape = parent->shape();
    chip_shape.RemoveDim(0);
    return errors::Internal(
        "HandleElementToSlice Cannot copy slice: number of elements does not "
        "match.  Shapes are: [element]: ",
        element.shape().DebugString(), ", [parent slice]: ",
        chip_shape.DebugString());
  }
  auto parent_as_matrix = parent->flat_outer_dims<T>();
  parent_as_matrix.chip(index, 0) = element.flat<T>();
  return Status::OK();
}

}  // namespace

QueueBase::QueueBase(int32 capacity, const DataTypeVector& component_dtypes,
                     const std::vector<TensorShape>& component_shapes,
                     const string& name)
    : capacity_(capacity),
      component_dtypes_(component_dtypes),
      component_shapes_(component_shapes),
      name_(name),
      closed_(false) {}

QueueBase::~QueueBase() {}

Status QueueBase::ValidateTupleCommon(const Tuple& tuple) const {
  if (tuple.size() != static_cast<size_t>(num_components())) {
    return errors::InvalidArgument(
        "Wrong number of components in tuple. Expected ", num_components(),
        ", got ", tuple.size());
  }
  for (size_t i = 0; i < tuple.size(); ++i) {
    if (tuple[i].dtype() != component_dtypes_[i]) {
      return errors::InvalidArgument(
          "Type mismatch in tuple component ", i, ". Expected ",
          DataTypeString(component_dtypes_[i]), ", got ",
          DataTypeString(tuple[i].dtype()));
    }
  }
  return Status::OK();
}

// static
string QueueBase::ShapeListString(const gtl::ArraySlice<TensorShape>& shapes) {
  string result = "[";
  bool first = true;
  for (const TensorShape& shape : shapes) {
    strings::StrAppend(&result, (first ? "" : ", "), shape.DebugString());
    first = false;
  }
  strings::StrAppend(&result, "]");
  return result;
}

Status QueueBase::MatchesNodeDefOp(const NodeDef& node_def,
                                   const string& op) const {
  if (node_def.op() != op) {
    return errors::InvalidArgument("Shared queue '", name_, "' has type '", op,
                                   "' that does not match type of Node '",
                                   node_def.name(), "': ", node_def.op());
  }
  return Status::OK();
}

Status QueueBase::MatchesNodeDefCapacity(const NodeDef& node_def,
                                         int32 capacity) const {
  int32 requested_capacity = -1;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "capacity", &requested_capacity));
  if (requested_capacity < 0) requested_capacity = kUnbounded;
  if (requested_capacity != capacity) {
    return errors::InvalidArgument("Shared queue '", name_, "' has capacity ",
                                   capacity, " but requested capacity was ",
                                   requested_capacity);
  }
  return Status::OK();
}

Status QueueBase::MatchesNodeDefTypes(const NodeDef& node_def) const {
  DataTypeVector requested_dtypes;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node_def, "component_types", &requested_dtypes));
  if (requested_dtypes != component_dtypes_) {
    return errors::InvalidArgument("Shared queue '", name_,
                                   "' has component types ",
                                   DataTypeSliceString(component_dtypes_),
                                   " but requested component types were ",
                                   DataTypeSliceString(requested_dtypes));
  }
  return Status::OK();
}

Status QueueBase::MatchesNodeDefShapes(const NodeDef& node_def) const {
  std::vector<TensorShape> requested_shapes;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "shapes", &requested_shapes));
  if (requested_shapes != component_shapes_) {
    return errors::InvalidArgument("Shared queue '", name_,
                                   "' has component shapes ",
                                   ShapeListString(component_shapes_),
                                   " but requested component shapes were ",
                                   ShapeListString(requested_shapes));
  }
  return Status::OK();
}

// TODO(mrry): If these checks become a bottleneck, find a way to
//   reduce the number of times that they are called.
Status QueueBase::ValidateTuple(const Tuple& tuple) {
  TF_RETURN_IF_ERROR(ValidateTupleCommon(tuple));
  if (specified_shapes()) {
    for (size_t i = 0; i < tuple.size(); ++i) {
      if (!component_shapes_[i].IsSameSize(tuple[i].shape())) {
        return errors::InvalidArgument(
            "Shape mismatch in tuple component ", i, ". Expected ",
            component_shapes_[i].DebugString(), ", got ",
            tuple[i].shape().DebugString());
      }
    }
  }
  return Status::OK();
}

// TODO(mrry): If these checks become a bottleneck, find a way to
//   reduce the number of times that they are called.
Status QueueBase::ValidateManyTuple(const Tuple& tuple) {
  TF_RETURN_IF_ERROR(ValidateTupleCommon(tuple));
  const int64 batch_size = tuple[0].dim_size(0);
  if (specified_shapes()) {
    for (size_t i = 0; i < tuple.size(); ++i) {
      // Expected shape is [batch_size] + component_shapes_[i]
      const TensorShape expected_shape = ManyOutShape(i, batch_size);
      if (!expected_shape.IsSameSize(tuple[i].shape())) {
        return errors::InvalidArgument("Shape mismatch in tuple component ", i,
                                       ". Expected ",
                                       expected_shape.DebugString(), ", got ",
                                       tuple[i].shape().DebugString());
      }
    }
  } else {
    for (size_t i = 1; i < tuple.size(); ++i) {
      if (tuple[i].dim_size(0) != batch_size) {
        return errors::InvalidArgument(
            "All input tensors must have the same size in the 0th ",
            "dimension. Component ", i, " has ", tuple[i].dim_size(0),
            ", and should have ", batch_size);
      }
    }
  }
  return Status::OK();
}

void QueueBase::Cancel(Action action, CancellationManager* cancellation_manager,
                       CancellationToken token) {
  DoneCallback callback = nullptr;
  {
    mutex_lock lock(mu_);
    std::deque<Attempt>* attempts =
        action == kEnqueue ? &enqueue_attempts_ : &dequeue_attempts_;

    for (Attempt& attempt : *attempts) {
      if (attempt.cancellation_manager == cancellation_manager &&
          attempt.cancellation_token == token) {
        if (!attempt.is_cancelled) {
          attempt.is_cancelled = true;
          if (action == kEnqueue) {
            attempt.context->SetStatus(
                errors::Cancelled("Enqueue operation was cancelled"));
          } else {
            attempt.context->SetStatus(
                errors::Cancelled("Dequeue operation was cancelled"));
          }
          std::swap(callback, attempt.done_callback);
        }
        break;
      }
    }
  }
  if (callback) {
    callback();
    FlushUnlocked();
  }
}

void QueueBase::CloseAndCancel() {
  std::vector<DoneCallback> callbacks;
  {
    mutex_lock lock(mu_);
    closed_ = true;
    for (Attempt& attempt : enqueue_attempts_) {
      if (!attempt.is_cancelled) {
        attempt.is_cancelled = true;
        attempt.context->SetStatus(
            errors::Cancelled("Enqueue operation was cancelled"));
        callbacks.emplace_back(std::move(attempt.done_callback));
      }
    }
  }
  for (const DoneCallback& callback : callbacks) {
    callback();
  }
  FlushUnlocked();
}

void QueueBase::Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
                      DoneCallback callback) {
  if (cancel_pending_enqueues) {
    CloseAndCancel();
    callback();
  } else {
    {
      mutex_lock lock(mu_);
      enqueue_attempts_.emplace_back(
          0, callback, ctx, nullptr, CancellationManager::kInvalidToken,
          [this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("Queue '", name_, "' is already closed."));
            } else {
              closed_ = true;
            }
            return kComplete;
          });
    }
    FlushUnlocked();
  }
}

bool QueueBase::TryAttemptLocked(Action action,
                                 std::vector<CleanUp>* clean_up) {
  std::deque<Attempt>* attempts =
      action == kEnqueue ? &enqueue_attempts_ : &dequeue_attempts_;

  bool progress = false;
  bool done = false;
  while (!done && !attempts->empty()) {
    if (attempts->front().is_cancelled) {
      if (action == kEnqueue) {
        if (closed_) {
          VLOG(1) << "Skipping cancelled enqueue attempt";
        } else {
          LOG(WARNING)
              << name_
              << ": Skipping cancelled enqueue attempt with queue not closed";
        }
      } else {
        if (closed_) {
          VLOG(1) << "Skipping cancelled dequeue attempt";
        } else {
          LOG(WARNING)
              << name_
              << ": Skipping cancelled dequeue attempt with queue not closed";
        }
      }
      attempts->pop_front();
    } else {
      Attempt* cur_attempt = &attempts->front();
      switch (cur_attempt->run_callback(cur_attempt)) {
        case kNoProgress:
          done = true;
          break;
        case kProgress:
          done = true;
          progress = true;
          break;
        case kComplete:
          progress = true;
          clean_up->emplace_back(std::move(cur_attempt->done_callback),
                                 cur_attempt->cancellation_token,
                                 cur_attempt->context->cancellation_manager());
          attempts->pop_front();
          break;
      }
    }
  }
  return progress;
}

void QueueBase::FlushUnlocked() {
  std::vector<CleanUp> clean_up;
  Ref();
  {
    mutex_lock lock(mu_);
    bool changed;
    do {
      changed = TryAttemptLocked(kEnqueue, &clean_up);
      changed = TryAttemptLocked(kDequeue, &clean_up) || changed;
    } while (changed);
  }
  Unref();
  for (const auto& to_clean : clean_up) {
    if (to_clean.to_deregister != CancellationManager::kInvalidToken) {
      // NOTE(mrry): We can safely ignore the return value of
      // DeregisterCallback because the mutex mu_ ensures that the
      // cleanup action only executes once.
      to_clean.cm->DeregisterCallback(to_clean.to_deregister);
    }
    to_clean.finished();
  }
}

Status QueueBase::CopySliceToElement(const Tensor& parent, Tensor* element,
                                     int64 index) {
#define HANDLE_TYPE(DT)                                                   \
  if (parent.dtype() == DT) {                                             \
    TF_RETURN_IF_ERROR(HandleSliceToElement<DT>(parent, element, index)); \
    return Status::OK();                                                  \
  }
  HANDLE_TYPE(DT_FLOAT);
  HANDLE_TYPE(DT_HALF);
  HANDLE_TYPE(DT_DOUBLE);
  HANDLE_TYPE(DT_INT32);
  HANDLE_TYPE(DT_UINT8);
  HANDLE_TYPE(DT_INT16);
  HANDLE_TYPE(DT_INT8);
  HANDLE_TYPE(DT_STRING);
  HANDLE_TYPE(DT_COMPLEX64);
  HANDLE_TYPE(DT_COMPLEX128);
  HANDLE_TYPE(DT_INT64);
  HANDLE_TYPE(DT_BOOL);
  HANDLE_TYPE(DT_QINT8);
  HANDLE_TYPE(DT_QUINT8);
  HANDLE_TYPE(DT_QINT32);
  HANDLE_TYPE(DT_QINT16);
  HANDLE_TYPE(DT_QUINT16);
  HANDLE_TYPE(DT_UINT16);
#undef HANDLE_TYPE
  return errors::Unimplemented("CopySliceToElement Unhandled data type: ",
                               parent.dtype());
}

// Static method
Status QueueBase::CopyElementToSlice(const Tensor& element, Tensor* parent,
                                     int64 index) {
#define HANDLE_TYPE(DT)                                                   \
  if (element.dtype() == DT) {                                            \
    TF_RETURN_IF_ERROR(HandleElementToSlice<DT>(element, parent, index)); \
    return Status::OK();                                                  \
  }
  HANDLE_TYPE(DT_FLOAT);
  HANDLE_TYPE(DT_HALF);
  HANDLE_TYPE(DT_DOUBLE);
  HANDLE_TYPE(DT_INT32);
  HANDLE_TYPE(DT_UINT8);
  HANDLE_TYPE(DT_INT16);
  HANDLE_TYPE(DT_INT8);
  HANDLE_TYPE(DT_STRING);
  HANDLE_TYPE(DT_COMPLEX64);
  HANDLE_TYPE(DT_COMPLEX128);
  HANDLE_TYPE(DT_INT64);
  HANDLE_TYPE(DT_BOOL);
  HANDLE_TYPE(DT_QINT8);
  HANDLE_TYPE(DT_QUINT8);
  HANDLE_TYPE(DT_QINT32);
  HANDLE_TYPE(DT_QINT16);
  HANDLE_TYPE(DT_QUINT16);
  HANDLE_TYPE(DT_UINT16);
#undef HANDLE_TYPE
  return errors::Unimplemented("CopyElementToSlice Unhandled data type: ",
                               element.dtype());
}

}  // namespace tensorflow
