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

#ifndef TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_H_
#define TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_H_

#include <deque>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/numeric_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

/**
 * ConditionalAccumulator/ConditionalAccumulatorBase implements an aggregation
 * object for adding gradients.
 * The two main methods of this class are TryApplyGrad and TryTakeGrad.
 *
 * TryApplyGrad tries add a gradient to the accumulator. The attempt is
 * successful if local_step >= global_step, i.e., if the gradient is not stale,
 * having been computed using up-to-date information. Otherwise, the gradient is
 * silently dropped.
 *
 * TryTakeGrad logs an attempt to read the average gradient. The attempt is
 * blocked until the number of gradients accumulated (via TryApplyGrad) is equal
 * or exceeds the number requested by TryTakeGrad.
 * Once this condition is satisfied, the following actions are taken:
 * (1) the value of the average gradient is returned
 * (2) the count of accumulated gradients is reset to 0
 * (3) the internal global_step value (current_global_step_) is incremented by 1
 */
class ConditionalAccumulatorBase : public ResourceBase {
 public:
  // Args:
  //   dtype: The datatype of the gradients to be accumulated.
  //   shape: The shape of the accumulated gradients.
  //   name:  A name to use for the ConditionalAccumulator.
  ConditionalAccumulatorBase(const DataType& dtype,
                             const PartialTensorShape& shape,
                             const string& name, const string& reduction_type);

  typedef AsyncOpKernel::DoneCallback DoneCallback;

  virtual void TryApplyGrad(int64_t local_step, OpKernelContext* ctx) = 0;
  void TryTakeGrad(int num_required, OpKernelContext* ctx,
                   DoneCallback callback);

  // Accessor methods
  uint32 num_accumulated() {
    mutex_lock lock(mu_);
    return counter_;
  }

  const DataType& dtype() const { return dtype_; }

  string DebugString() const override { return "A conditional accumulator"; }

  // SetGlobalStep is a modifier method for current_global_step.
  // It returns an InvalidArgument error if the new_global_step is less than
  // current_global_step.
  absl::Status SetGlobalStep(int64_t new_global_step);

  absl::Status MatchesNodeDef(const NodeDef& node_def);

 protected:
  // Virtual methods to be implemented by sub-classes for different datatypes.
  // Implements arithmetic operations specific to datatype.
  virtual void DivideAccumGradByCounter(OpKernelContext* ctx)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;
  virtual bool SetOutput(OpKernelContext* ctx) = 0;

  enum RunResult { kNoProgress, kComplete };

  // Helper struct holding information about a TakeGrad attempt
  struct Attempt;
  typedef std::function<RunResult(Attempt*)> RunCallback;
  struct Attempt {
    int elements_requested;
    DoneCallback done_callback;  // must be run outside mu_
    OpKernelContext* context;
    CancellationManager* cancellation_manager;  // not owned
    CancellationToken cancellation_token;
    RunCallback run_callback;  // must be run while holding mu_
    bool is_cancelled;

    Attempt(int elements_requested, DoneCallback done_callback,
            OpKernelContext* context, CancellationManager* cancellation_manager,
            CancellationToken cancellation_token, RunCallback run_callback)
        : elements_requested(elements_requested),
          done_callback(std::move(done_callback)),
          context(context),
          cancellation_manager(cancellation_manager),
          cancellation_token(cancellation_token),
          run_callback(std::move(run_callback)),
          is_cancelled(false) {}
  };

  // Helper struct for deregistration of a cancellation token and executing a
  // DoneCallback after a TakeGrad attempt is complete.
  struct CleanUp {
    CleanUp(DoneCallback&& f, CancellationToken ct, CancellationManager* cm)
        : finished(f), to_deregister(ct), cm(cm) {}
    DoneCallback finished;
    CancellationToken to_deregister;
    CancellationManager* cm;
  };

  // Fields

  const DataType dtype_;
  const PartialTensorShape shape_;
  const string name_;
  const string reduction_type_;
  mutex mu_;
  int counter_ TF_GUARDED_BY(mu_);
  int64_t current_global_step_ TF_GUARDED_BY(mu_);

  std::deque<Attempt> takegrad_attempts_ TF_GUARDED_BY(mu_);

  // Methods

  // Helper function for creating cancellation callback
  void Cancel(CancellationManager* cancellation_manager,
              CancellationToken token);

  // Helper functions to process TakeGrad attempts.
  // FlushUnlocked is called at the end of each TryApplyGrad and TryTakeGrad
  // calls to try to clear the TakeGrad attempts. This in turn calls
  // TryAttemptLocked, which then executes the RunCallback of the logged
  // attempts.
  // Both functions are modeled after core/kernels/queue_base.
  // Note: ApplyGrad attempts never block -- unlike in a queue with limited
  //       capacity, we can always add the newest gradient to our accumulator
  //       (if it is not stale) or drop it silently (if it is stale).
  void FlushUnlocked();
  bool TryAttemptLocked(std::vector<CleanUp>* clean_up)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Helper methods
  //  void DeepCopy(Tensor* dst);
  bool TakeGradLockedHelper(OpKernelContext* ctx, DoneCallback callback)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
};

/*
 * Modifications to convenience macros defined in core/framework/op_kernel.h.
 * The below macros return a boolean if the test fails, so that the calling
 * function can get an indication that a failure has occurred.
 */
#define OP_REQUIRES_BOOLEAN(CTX, EXP, STATUS)          \
  do {                                                 \
    if (!TF_PREDICT_TRUE(EXP)) {                       \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS)); \
      return false;                                    \
    }                                                  \
  } while (0)

#define OP_REQUIRES_OK_BOOLEAN(CTX, STATUS)                 \
  do {                                                      \
    ::tensorflow::Status _s(STATUS);                        \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      return false;                                         \
    }                                                       \
  } while (0)

/*
 * Convenience classes for helping to convert between numeric types.
 * The specialization for Eigen::half here simplifies specialization of
 * ConditionalAccumulator classes later.
 */
template <typename T, typename U>
class TypeConverter {
 public:
  static T ConvertUToT(U c) { return c; /* implicit conversion */ }
};

template <typename U>
class TypeConverter<Eigen::half, U> {
 public:
  static Eigen::half ConvertUToT(U c) { return static_cast<Eigen::half>(c); }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_H_
