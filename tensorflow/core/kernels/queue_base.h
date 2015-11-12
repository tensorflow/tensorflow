#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

// Functionality common to asynchronous QueueInterface implementations.
class QueueBase : public QueueInterface {
 public:
  // As a possible value of 'capacity'.
  static const int32 kUnbounded = INT_MAX;

  // Args:
  //   component_dtypes: The types of each component in a queue-element tuple.
  //   component_shapes: The shapes of each component in a queue-element tuple,
  //     which must either be empty (if the shapes are not specified) or
  //     or have the same size as component_dtypes.
  //   name: A name to use for the queue.
  QueueBase(int32 capacity, const DataTypeVector& component_dtypes,
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

  TensorShape ManyOutShape(int i, int64 batch_size) {
    TensorShape shape({batch_size});
    shape.AppendShape(component_shapes_[i]);
    return shape;
  }

  // Copies the index^th slice (in the first dimension) of parent into element.
  static Status CopySliceToElement(const Tensor& parent, Tensor* element,
                                   int index);

  // Copies element into the index^th slice (in the first dimension) of parent.
  static Status CopyElementToSlice(const Tensor& element, Tensor* parent,
                                   int index);

  void Cancel(Action action, CancellationToken token);

  // Helper for cancelling all pending Enqueue(Many) operations when
  // Close is called with cancel_pending_enqueues.
  void CloseAndCancel();

  bool TryAttemptLocked(Action action, std::vector<CleanUp>* clean_up)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Tries to make progress on the enqueues or dequeues at the front
  // of the *_attempts_ queues.
  void FlushUnlocked();

  ~QueueBase() override {}

  // Helpers for implementing MatchesNodeDef().
  static string ShapeListString(const gtl::ArraySlice<TensorShape>& shapes);
  Status MatchesNodeDefOp(const NodeDef& node_def, const string& op) const;
  Status MatchesNodeDefCapacity(const NodeDef& node_def, int32 capacity) const;
  Status MatchesNodeDefTypes(const NodeDef& node_def) const;
  Status MatchesNodeDefShapes(const NodeDef& node_def) const;

 protected:
  const int32 capacity_;
  const DataTypeVector component_dtypes_;
  const std::vector<TensorShape> component_shapes_;
  const string name_;
  mutex mu_;
  bool closed_ GUARDED_BY(mu_);

  struct Attempt;
  typedef std::function<RunResult(Attempt*)> RunCallback;
  struct Attempt {
    int32 elements_requested;
    DoneCallback done_callback;  // must be run outside mu_
    OpKernelContext* context;
    CancellationToken cancellation_token;
    RunCallback run_callback;  // must be run while holding mu_
    bool is_cancelled;
    Tuple tuple;

    Attempt(int32 elements_requested, DoneCallback done_callback,
            OpKernelContext* context, CancellationToken cancellation_token,
            RunCallback run_callback)
        : elements_requested(elements_requested),
          done_callback(done_callback),
          context(context),
          cancellation_token(cancellation_token),
          run_callback(run_callback),
          is_cancelled(false) {}
  };
  std::deque<Attempt> enqueue_attempts_ GUARDED_BY(mu_);
  std::deque<Attempt> dequeue_attempts_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(QueueBase);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_QUEUE_BASE_H_
