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

#ifndef TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_
#define TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_

#include <deque>
#include <vector>

#include "absl/base/macros.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Functionality common to asynchronous QueueInterface implementations.
class QueueBase : public QueueInterface {
 public:
  // As a possible value of 'capacity'.
  static constexpr int32_t kUnbounded = INT_MAX;

  // Args:
  //   component_dtypes: The types of each component in a queue-element tuple.
  //   component_shapes: The shapes of each component in a queue-element tuple,
  //     which must either be empty (if the shapes are not specified) or
  //     or have the same size as component_dtypes.
  //   name: A name to use for the queue.
  QueueBase(int32_t capacity, const DataTypeVector& component_dtypes,
            const std::vector<TensorShape>& component_shapes,
            const string& name);

  // Implementations of QueueInterface methods --------------------------------
  const DataTypeVector& component_dtypes() const override {
    return component_dtypes_;
  }

  Status ValidateTuple(const Tuple& tuple) override;
  Status ValidateManyTuple(const Tuple& tuple) override;

  void Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
             DoneCallback callback) override;

  // Other public methods -----------------------------------------------------
  const std::vector<TensorShape>& component_shapes() const {
    return component_shapes_;
  }

  int32 capacity() const { return capacity_; }

  bool is_closed() const override {
    mutex_lock lock(mu_);
    return closed_;
  }

  // Copies the index^th slice (in the first dimension) of parent into element.
  static Status CopySliceToElement(const Tensor& parent, Tensor* element,
                                   int64_t index);

  // Copies element into the index^th slice (in the first dimension) of parent.
  // NOTE(mrry): This method is deprecated. Use
  // `tensorflow::batch_util::CopySliceToElement()` defined in
  // "./batch_util.h" instead.
  ABSL_DEPRECATED(
      "Use `tensorflow::batch_util::CopySliceToElement()` defined in "
      "\"./batch_util.h\" instead.")
  static Status CopyElementToSlice(const Tensor& element, Tensor* parent,
                                   int64_t index);

 protected:
  enum Action { kEnqueue, kDequeue };
  enum RunResult { kNoProgress, kProgress, kComplete };

  // Tries to enqueue/dequeue (or close) based on whatever is at the
  // front of enqueue_attempts_/dequeue_attempts_.  Appends to
  // *finished the callback for any finished attempt (so it may be
  // called once mu_ is released).  Returns true if any progress was
  // made.
  struct CleanUp {
    CleanUp(DoneCallback&& f, CancellationToken ct, CancellationManager* cm)
        : finished(f), to_deregister(ct), cm(cm) {}
    DoneCallback finished;
    CancellationToken to_deregister;
    CancellationManager* cm;
  };

  // Returns the number of components in a queue-element tuple.
  int32 num_components() const { return component_dtypes_.size(); }

  // True if shapes were specified.  If so, inputs will be validated
  // against them, etc.
  bool specified_shapes() const { return component_shapes_.size() > 0; }

  // Code common to Validate*Tuple().
  Status ValidateTupleCommon(const Tuple& tuple) const;

  TensorShape ManyOutShape(int i, int64_t batch_size) {
    TensorShape shape({batch_size});
    shape.AppendShape(component_shapes_[i]);
    return shape;
  }

  void Cancel(Action action, CancellationManager* cancellation_manager,
              CancellationToken token);

  // Helper for cancelling all pending Enqueue(Many) operations when
  // Close is called with cancel_pending_enqueues.
  void CloseAndCancel();

  bool TryAttemptLocked(Action action, std::vector<CleanUp>* clean_up)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Tries to make progress on the enqueues or dequeues at the front
  // of the *_attempts_ queues.
  void FlushUnlocked();

  ~QueueBase() override;

  // Helpers for implementing MatchesNodeDef().
  static string ShapeListString(const absl::Span<const TensorShape>& shapes);
  Status MatchesNodeDefOp(const NodeDef& node_def, const string& op) const;
  Status MatchesNodeDefCapacity(const NodeDef& node_def,
                                int32_t capacity) const;
  Status MatchesNodeDefTypes(const NodeDef& node_def) const;
  Status MatchesNodeDefShapes(const NodeDef& node_def) const;

 protected:
  const int32 capacity_;
  const DataTypeVector component_dtypes_;
  const std::vector<TensorShape> component_shapes_;
  const string name_;
  mutable mutex mu_;
  bool closed_ TF_GUARDED_BY(mu_);

  struct Attempt;
  typedef std::function<RunResult(Attempt*)> RunCallback;
  struct Attempt {
    int32 elements_requested;
    DoneCallback done_callback;  // must be run outside mu_
    OpKernelContext* context;
    CancellationManager* cancellation_manager;  // not owned
    CancellationToken cancellation_token;
    RunCallback run_callback;  // must be run while holding mu_
    bool is_cancelled;
    Tuple tuple;
    // tuples is used by some implementations allowing dynamic shapes.
    std::vector<Tuple> tuples;

    Attempt(int32_t elements_requested, DoneCallback done_callback,
            OpKernelContext* context, CancellationManager* cancellation_manager,
            CancellationToken cancellation_token, RunCallback run_callback)
        : elements_requested(elements_requested),
          done_callback(done_callback),
          context(context),
          cancellation_manager(cancellation_manager),
          cancellation_token(cancellation_token),
          run_callback(run_callback),
          is_cancelled(false) {}
  };
  std::deque<Attempt> enqueue_attempts_ TF_GUARDED_BY(mu_);
  std::deque<Attempt> dequeue_attempts_ TF_GUARDED_BY(mu_);

  QueueBase(const QueueBase&) = delete;
  void operator=(const QueueBase&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_
