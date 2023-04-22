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

#include "tensorflow/core/kernels/conditional_accumulator_base.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

ConditionalAccumulatorBase::ConditionalAccumulatorBase(
    const DataType& dtype, const PartialTensorShape& shape, const string& name,
    const string& reduction_type)
    : dtype_(dtype),
      shape_(shape),
      name_(name),
      reduction_type_(reduction_type) {
  counter_ = 0;
  current_global_step_ = 0;
}

Status ConditionalAccumulatorBase::MatchesNodeDef(const NodeDef& node_def) {
  // TODO(xinghao@): implement the checks for the node definition
  return Status::OK();
}

/**
 * Sets the time step of the accumulator to be in line with the global time
 * step. Logs warning if the accumulator's time step is already larger than the
 * provided time step.
 */
Status ConditionalAccumulatorBase::SetGlobalStep(int64 new_global_step) {
  mutex_lock lock(mu_);
  if (new_global_step < current_global_step_) {
    LOG(WARNING) << "Attempt to set current_global_step_ to smaller value: "
                 << "current_global_step_ = " << current_global_step_
                 << " >= " << new_global_step << " = new_global_step.";
  }
  current_global_step_ = new_global_step;
  return Status::OK();
}

/**
 * Logs an attempt to extract the average gradient, and tries to flush all
 * TakeGrad attempts.
 * A TakeGrad attempt is blocked until num_required > counter_, i.e.,
 * sufficient gradients have been accumulated.
 *
 * num_required: Number of gradients that needs to be accumulated before the
 *               attempt is unblocked.
 * ctx:          Context in which the op is executed.
 * callback:     A callback to be executed after the attempt has been completed.
 */
void ConditionalAccumulatorBase::TryTakeGrad(int num_required,
                                             OpKernelContext* ctx,
                                             DoneCallback callback) {
  if (num_required <= 0) {
    ctx->CtxFailureWithWarning(errors::InvalidArgument(
        "Argument num_required must be positive, but was ", num_required));
    callback();
  } else {
    CancellationManager* cm = ctx->cancellation_manager();
    CancellationToken token = cm->get_cancellation_token();
    bool already_cancelled;
    {
      mutex_lock l(mu_);
      already_cancelled = !cm->RegisterCallback(
          token, [this, cm, token]() { Cancel(cm, token); });
      if (!already_cancelled) {
        takegrad_attempts_.emplace_back(
            num_required, callback, ctx, cm, token,
            [this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              if (counter_ >= attempt->elements_requested) {
                bool successful_take_grad = TakeGradLockedHelper(
                    attempt->context, attempt->done_callback);
                if (successful_take_grad) {
                  return kComplete;
                } else {
                  // Try again
                  return kNoProgress;
                }
              } else {
                return kNoProgress;
              }
            });
      }
    }
    if (!already_cancelled) {
      FlushUnlocked();
    } else {
      ctx->SetStatus(errors::Cancelled("TakeGrad operation was cancelled"));
      callback();
    }
  }
}

/**
 * Cancellation callback.
 */
void ConditionalAccumulatorBase::Cancel(
    CancellationManager* cancellation_manager, CancellationToken token) {
  DoneCallback callback = nullptr;
  {
    mutex_lock lock(mu_);

    for (Attempt& attempt : takegrad_attempts_) {
      if (attempt.cancellation_manager == cancellation_manager &&
          attempt.cancellation_token == token) {
        if (!attempt.is_cancelled) {
          attempt.is_cancelled = true;
          attempt.context->SetStatus(
              errors::Cancelled("TakeGrad operation was cancelled"));
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

/**
 * Try to flush logged, blocked TakeGrad attempts.
 */
bool ConditionalAccumulatorBase::TryAttemptLocked(
    std::vector<CleanUp>* clean_up) {
  bool progress = false;
  bool done = false;
  while (!done && !takegrad_attempts_.empty()) {
    if (takegrad_attempts_.front().is_cancelled) {
      VLOG(1) << "Skipping cancelled TakeGrad attempt";
      takegrad_attempts_.pop_front();
    } else {
      Attempt* cur_attempt = &takegrad_attempts_.front();
      switch (cur_attempt->run_callback(cur_attempt)) {
        case kNoProgress:
          done = true;
          break;
        case kComplete:
          progress = true;
          clean_up->emplace_back(std::move(cur_attempt->done_callback),
                                 cur_attempt->cancellation_token,
                                 cur_attempt->context->cancellation_manager());
          takegrad_attempts_.pop_front();
          break;
      }
    }
  }
  return progress;
}

/**
 * Try to flush logged, blocked TakeGrad attempts.
 */
void ConditionalAccumulatorBase::FlushUnlocked() {
  std::vector<CleanUp> clean_up;
  Ref();
  {
    mutex_lock lock(mu_);
    bool changed;
    do {
      changed = TryAttemptLocked(&clean_up);
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

bool ConditionalAccumulatorBase::TakeGradLockedHelper(OpKernelContext* ctx,
                                                      DoneCallback callback) {
  // At this point, the conditional should have been passed

  // Implicitly increment global_step
  current_global_step_++;

  // Average the accumulated gradient
  if (reduction_type_ == "MEAN") {
    DivideAccumGradByCounter(ctx);
  }

  // Set output for accumulated gradient tensor
  bool successful_set_output = SetOutput(ctx);

  // Reset counter
  if (successful_set_output) counter_ = 0;

  return successful_set_output;
}

}  // namespace tensorflow
