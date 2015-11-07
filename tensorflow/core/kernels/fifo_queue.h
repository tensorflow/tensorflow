#ifndef TENSORFLOW_KERNELS_FIFO_QUEUE_H_
#define TENSORFLOW_KERNELS_FIFO_QUEUE_H_

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

class FIFOQueue : public QueueBase {
 public:
  FIFOQueue(int32 capacity, const DataTypeVector& component_dtypes,
            const std::vector<TensorShape>& component_shapes,
            const string& name);
  Status Initialize();  // Must be called before any other method.

  // Implementations of QueueInterface methods --------------------------------

  Status ValidateTuple(const Tuple& tuple) override;
  Status ValidateManyTuple(const Tuple& tuple) override;
  void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                  DoneCallback callback) override;
  void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                      DoneCallback callback) override;
  void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override;
  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      CallbackWithTuple callback) override;
  void Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
             DoneCallback callback) override;
  Status MatchesNodeDef(const NodeDef& node_def) override;

  int32 size() override {
    mutex_lock lock(mu_);
    return queues_[0].size();
  }

  int32 capacity() const { return capacity_; }

 private:
  enum Action { kEnqueue, kDequeue };

  ~FIFOQueue() override {}

  TensorShape ManyOutShape(int i, int64 batch_size) {
    TensorShape shape({batch_size});
    shape.AppendShape(component_shapes_[i]);
    return shape;
  }

  // Helper for dequeuing a single element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void Cancel(Action action, CancellationToken token);

  // Helper for cancelling all pending Enqueue(Many) operations when
  // Close is called with cancel_pending_enqueues.
  void CloseAndCancel();

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
  bool TryAttemptLocked(Action action, std::vector<CleanUp>* clean_up)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Tries to make progress on the enqueues or dequeues at the front
  // of the *_attempts_ queues.
  void FlushUnlocked();

  const int32 capacity_;

  mutex mu_;
  typedef std::deque<PersistentTensor> SubQueue;
  std::vector<SubQueue> queues_ GUARDED_BY(mu_);
  bool closed_ GUARDED_BY(mu_);

  enum RunResult { kNoProgress, kProgress, kComplete };
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

  static Status GetElementComponentFromBatch(const Tuple& tuple, int index,
                                             int component,
                                             OpKernelContext* ctx,
                                             PersistentTensor* out_element);

  TF_DISALLOW_COPY_AND_ASSIGN(FIFOQueue);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FIFO_QUEUE_H_
