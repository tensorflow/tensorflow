/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/base_collective_executor.h"

#include <algorithm>
#include <functional>
#include <utility>

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"

#define VALUE_IN_DEBUG_STRING false

namespace tensorflow {

namespace {
bool IsCancelled(CancellationManager* cancel_mgr) {
  return cancel_mgr != nullptr &&
         (cancel_mgr->IsCancelled() || cancel_mgr->IsCancelling());
}
}  // namespace

/*static*/
int64_t CollectiveAdapter::AlignedChunkElts(int64_t elt_bytes,
                                            int64_t total_elts,
                                            int64_t num_chunks) {
  DCHECK_GT(num_chunks, 0);
  int64_t base_chunk_elts = (total_elts + (num_chunks - 1)) / num_chunks;
  if (EIGEN_MAX_ALIGN_BYTES == 0) return base_chunk_elts;
  if (EIGEN_MAX_ALIGN_BYTES <= elt_bytes) {
    // Tolerate weird small values of EIGEN_MAX_ALIGN_BYTES
    DCHECK_EQ(0, elt_bytes % EIGEN_MAX_ALIGN_BYTES);
    return base_chunk_elts;
  }
  // elt_bytes < EIGEN_MAX_ALIGN_BYTES, which
  // must be a common multiple of the various atomic data types.
  DCHECK_EQ(0, EIGEN_MAX_ALIGN_BYTES % elt_bytes)
      << "total_elts=" << total_elts << " num_chunks=" << num_chunks
      << " EIGEN_MAX_ALIGN_BYTES=" << EIGEN_MAX_ALIGN_BYTES
      << " elt_bytes=" << elt_bytes;
  // Round bytes per chunk up to the next multiple of EIGEN_MAX_ALIGN_BYTES.
  int64_t chunk_bytes = base_chunk_elts * elt_bytes;
  int64_t diff =
      (chunk_bytes < EIGEN_MAX_ALIGN_BYTES)
          ? (EIGEN_MAX_ALIGN_BYTES - chunk_bytes)
          : (EIGEN_MAX_ALIGN_BYTES - (chunk_bytes % EIGEN_MAX_ALIGN_BYTES));
  DCHECK_EQ(0, diff % elt_bytes);
  base_chunk_elts += (diff / elt_bytes);
  DCHECK_EQ(0, ((base_chunk_elts * elt_bytes) % EIGEN_MAX_ALIGN_BYTES))
      << "total_elts=" << total_elts << " num_chunks=" << num_chunks
      << " EIGEN_MAX_ALIGN_BYTES=" << EIGEN_MAX_ALIGN_BYTES
      << " base_chunk_elts=" << base_chunk_elts << " elt_bytes=" << elt_bytes;
  return base_chunk_elts;
}

namespace {
template <typename T>
class CollectiveAdapterImpl : public CollectiveAdapter {
 public:
  // Takes ownership of output and prepares to properly alias its chunks.
  // Ownership is taken because the shape may temporarily change.
  CollectiveAdapterImpl(Tensor* output, int64_t num_chunks,
                        Allocator* allocator, bool align_chunks)
      : output_(std::move(*output)),
        dt_(output_.dtype()),
        old_shape_(output_.shape()),
        num_chunks_(num_chunks),
        allocator_(allocator),
        total_elts_(output_.NumElements()),
        chunk_elts_(align_chunks
                        ? AlignedChunkElts(sizeof(T), total_elts_, num_chunks_)
                        : total_elts_ / num_chunks_),
        data_start_(reinterpret_cast<T*>(DMAHelper::base(&output_))),
        data_end_(data_start_ + total_elts_) {
    if (!align_chunks) {
      DCHECK_EQ(total_elts_, num_chunks_ * chunk_elts_);
    }
    DCHECK_GT(chunk_elts_, 0);
    Flatten();
  }

  ~CollectiveAdapterImpl() override {}

  const Tensor& Value() const override { return output_; }

  // If necessary, flatten output.
  void Flatten() {
    if (old_shape_.dims() != 1) {
      TensorShape new_shape = TensorShape({old_shape_.num_elements()});
      DMAHelper::UnsafeSetShape(&output_, new_shape);
    }
  }

  void ConsumeFinalValue(Tensor* output) override {
    if (old_shape_ != output_.shape()) {
      DMAHelper::UnsafeSetShape(&output_, old_shape_);
    }
    *output = std::move(output_);
  }

  // Number of T elements in a particular chunk.
  inline int64_t ChunkElts(int i) const {
    DCHECK_LT(i, num_chunks_);
    const T* chunk_start = std::min(data_end_, data_start_ + i * chunk_elts_);
    const T* chunk_end = std::min(data_end_, chunk_start + chunk_elts_);
    return chunk_end - chunk_start;
  }

  int64_t ChunkBytes(int i) const override { return sizeof(T) * ChunkElts(i); }

  // Returns a new Tensor that aliases the required chunk.
  Tensor ChunkAlias(int i) override {
    int64_t start = chunk_elts_ * i;
    int64_t num_elts = ChunkElts(i);
    // If this chunk is empty the prior chunk might also be short
    // so always take an empty slice from the front of the tensor
    // to avoid an illegal offset check failure somewhere.
    return (num_elts > 0) ? output_.Slice(start, start + num_elts)
                          : output_.Slice(0, 0);
  }

  Tensor TempChunk(int i) const override {
    AllocationAttributes empty;
    profiler::ScopedMemoryDebugAnnotation op_annotation(
        "CollectiveAdapterImpl::TempChunk");
    return Tensor(allocator_, dt_, {ChunkElts(i)}, empty);
  }

  string DebugString() const override {
    return strings::StrCat(
        "base addr ", reinterpret_cast<int64_t>(DMAHelper::base(&output_)),
        " num_chunks ", num_chunks_, " total_elts ", total_elts_, " chunk_elts",
        chunk_elts_, " value ",
        VALUE_IN_DEBUG_STRING ? output_.SummarizeValue(1024) : "<hidden>");
  }

  string TBounds(const Tensor& t) const override {
    int64_t base_addr = reinterpret_cast<int64_t>(DMAHelper::base(&t));
    return strings::StrCat("(", base_addr, ", ", (base_addr + t.TotalBytes()),
                           ")");
  }

  Tensor Scalar(int v) const override { return Tensor(static_cast<T>(v)); }

  Tensor Scalar(Allocator* a, const AllocationAttributes& attr) const override {
    Tensor t(a, dt_, TensorShape({}), attr);
    return t;
  }

  Tensor output_;
  const DataType dt_;
  const TensorShape old_shape_;
  const int64_t num_chunks_;
  Allocator* allocator_;
  const int64_t total_elts_;
  const int64_t chunk_elts_;
  const T* data_start_;
  const T* data_end_;
};

}  // namespace

CollectiveAdapter* MakeCollectiveAdapter(Tensor* output, int num_chunks,
                                         Allocator* allocator,
                                         bool align_chunks) {
  switch (output->dtype()) {
    case DT_BFLOAT16:
      return new CollectiveAdapterImpl<Eigen::bfloat16>(
          output, num_chunks, allocator, align_chunks);
      break;
    case DT_HALF:
      return new CollectiveAdapterImpl<Eigen::half>(output, num_chunks,
                                                    allocator, align_chunks);
      break;
    case DT_FLOAT:
      return new CollectiveAdapterImpl<float>(output, num_chunks, allocator,
                                              align_chunks);
      break;
    case DT_DOUBLE:
      return new CollectiveAdapterImpl<double>(output, num_chunks, allocator,
                                               align_chunks);
      break;
    case DT_INT32:
      return new CollectiveAdapterImpl<int32>(output, num_chunks, allocator,
                                              align_chunks);
      break;
    case DT_INT64:
      return new CollectiveAdapterImpl<int64_t>(output, num_chunks, allocator,
                                                align_chunks);
      break;
    default:
      LOG(FATAL) << "Unsupported type " << DataTypeString(output->dtype())
                 << " to MakeCollectiveAdapter";
      return nullptr;
  }
}

BaseCollectiveExecutor::~BaseCollectiveExecutor() {}

void BaseCollectiveExecutor::StartAbort(const Status& s) {
  Status status;
  {
    mutex_lock l(status_mu_);
    if (!status_.ok()) {
      VLOG(2) << "BaseCollectiveExecutor already aborted, ignoring StartAbort: "
              << s;
      return;
    }
    status_ = StatusGroup::MakeDerived(Status(
        s.code(),
        absl::StrCat(
            "Collective ops is aborted by: ", s.error_message(),
            "\nThe error could be from a previous operation. Restart your "
            "program to reset.")));
    status = status_;
  }
  LOG(ERROR) << "BaseCollectiveExecutor::StartAbort " << s;
  cem_->GetParamResolver()->StartAbort(status);
  remote_access_->StartAbort(status);
  if (cem_->GetNcclCommunicator() != nullptr) {
    cem_->GetNcclCommunicator()->StartAbort(status);
  }
}

Status BaseCollectiveExecutor::GetStatus(const Status& s) {
  if (s.ok()) return s;
  mutex_lock l(status_mu_);
  // If the collective executor is already aborted, use the aborted status
  // which is more likely the actual error instead of an artifact of an
  // abortion.
  if (!status_.ok()) {
    VLOG(2) << "Overriding status with collective ops executor status. "
               "Original status: "
            << s;
    return status_;
  }
  return s;
}

void BaseCollectiveExecutor::ExecuteAsync(OpKernelContext* ctx,
                                          const CollectiveParams* col_params,
                                          const string& exec_key,
                                          StatusCallback done) {
  // See CompleteParamsAsync() how done() and the timeout callback interacts.
  const auto is_callback_called = std::make_shared<std::atomic<bool>>(false);
  auto done_safe = [this, done, ctx, is_callback_called](const Status& s) {
    bool called = is_callback_called->exchange(true);
    if (!called) {
      if (!s.ok() && !IsCancelled(ctx->cancellation_manager())) {
        // This is a collective error. Abort CollectiveExecutor so that this
        // error can propagate to other workers.
        StartAbort(s);
      }
      done(GetStatus(s));
    }
  };
  auto timeout_microseconds = static_cast<int64_t>(
      col_params->instance.impl_details.timeout_seconds * 1'000'000);
  if (timeout_microseconds > 0) {
    // TODO(xldrx): Share the timeout watchdog thread among collectives.
    SchedNonBlockingClosureAfter(
        timeout_microseconds, [this, is_callback_called, done] {
          bool called = is_callback_called->exchange(true);
          if (!called) {
            Status status(error::DEADLINE_EXCEEDED,
                          "Collective has timed out during execution.");
            StartAbort(status);
            done(status);
          }
        });
  }

  Tensor* output = ctx->mutable_output(0);
  const Tensor* input = (col_params->instance.type == REDUCTION_COLLECTIVE ||
                         col_params->instance.type == GATHER_COLLECTIVE ||
                         col_params->instance.type == PERMUTE_COLLECTIVE ||
                         col_params->instance.type == ALL_TO_ALL_COLLECTIVE ||
                         (col_params->instance.type == BROADCAST_COLLECTIVE &&
                          col_params->is_source))
                            ? &ctx->input(0)
                            : nullptr;
  CollectiveImplementationInterface* col_impl = nullptr;
  Status status = CreateCollective(*col_params, &col_impl);
  if (!status.ok()) {
    done_safe(status);
    DCHECK_EQ(nullptr, col_impl);
    return;
  }
  core::ScopedUnref unref(col_impl);
  auto col_ctx = std::make_shared<CollectiveContext>(
      this, cem_->GetNcclCommunicator(), dev_mgr_, ctx, CtxParams(ctx),
      col_params, exec_key, step_id_, input, output);
  status = col_impl->InitializeCollectiveContext(col_ctx);
  if (!status.ok()) {
    done_safe(status);
    return;
  }
  // Run on an unbounded work queue that can handle blocking work so as to not
  // starve executor threads.
  col_impl->Ref();
  profiler::TraceMeProducer producer("BaseCollectiveExecutor::ExecuteAsync");
  RunClosure([col_impl, col_ctx, done_safe, ctx,
              context_id = producer.GetContextId()]() {
    core::ScopedUnref unref(col_impl);
    profiler::TraceMeConsumer consumer(
        [ctx, col_ctx] {
          string op = profiler::TraceMeOp(ctx->op_kernel().name_view(),
                                          ctx->op_kernel().type_string_view());
          return profiler::TraceMeEncode(
              std::move(op),
              {{"id", ctx->step_id()},
               {"instance_key", col_ctx->col_params->instance.instance_key},
               {"collective", col_ctx->col_params->instance.type}});
        },
        context_id);
    col_impl->Ref();
    col_impl->Run([col_impl, col_ctx, done_safe](const Status& s) {
      core::ScopedUnref unref(col_impl);
      done_safe(s);
    });
  });
}

void BaseCollectiveExecutor::CompleteParamsAsync(
    const DeviceAttributes& device, CollectiveParams* cp,
    CancellationManager* cancel_mgr, StatusCallback done) {
  // We need to make sure that when the timeout callback executes,
  // CollectiveExecutor and CollectiveExecutorMgr are both alive. After done()
  // is called, CollectiveExecutorMgr may be destructed and we don't have a way
  // to keep it without making the ownerships more complicated. Therefore if the
  // timeout callback executes, done_safe will become a no-op and the timeout
  // callback is responsible for invoking done() at the end.
  const auto is_callback_called = std::make_shared<std::atomic<bool>>(false);
  int64_t trace_id = profiler::TraceMe::ActivityStart([cp]() {
    return profiler::TraceMeEncode("CollectiveExecutor::CompleteParams",
                                   {{"group_key", cp->group.group_key},
                                    {"group_size", cp->group.group_size}});
  });

  auto done_safe = [this, is_callback_called, cancel_mgr, trace_id,
                    done](const Status& s) {
    profiler::TraceMe::ActivityEnd(trace_id);
    bool called = is_callback_called->exchange(true);
    if (!called) {
      if (!s.ok() && !IsCancelled(cancel_mgr)) {
        // This is a collective error. Abort CollectiveExecutor so that this
        // error can propagate to other workers.
        StartAbort(s);
      }
      done(GetStatus(s));
    }
  };
  auto timeout_microseconds = static_cast<int64_t>(
      cp->instance.impl_details.timeout_seconds * 1'000'000);
  if (timeout_microseconds > 0) {
    // TODO(xldrx): Share the timeout watchdog thread among collectives.
    SchedNonBlockingClosureAfter(
        timeout_microseconds, [this, is_callback_called, done]() {
          bool called = is_callback_called->exchange(true);
          if (!called) {
            Status status(
                error::DEADLINE_EXCEEDED,
                "Collective has timed out waiting for other workers.");
            StartAbort(status);
            done(status);
          }
        });
  }
  cem_->GetParamResolver()->CompleteParamsAsync(device, cp, cancel_mgr,
                                                done_safe);
}

Status BaseCollectiveExecutor::CreateCollective(
    const CollectiveParams& col_params,
    CollectiveImplementationInterface** col_impl) {
  VLOG(2) << "CreateCollective type "
          << DataTypeString(col_params.instance.data_type) << " name "
          << col_params.instance.impl_details.collective_name;
  *col_impl = nullptr;
  switch (col_params.instance.data_type) {
    case DT_BOOL:
      if (col_params.instance.type == BROADCAST_COLLECTIVE) {
        return CollectiveRegistry::Lookup(
            col_params.instance.impl_details.collective_name, col_impl);
      } else {
        return errors::Internal(
            "No collective other than broadcast supports DT_BOOL");
      }
    case DT_INT32:
      if (col_params.group.device_type == DEVICE_GPU &&
          col_params.instance.type == REDUCTION_COLLECTIVE) {
        // TODO(b/139421603): enable int32 all-reduce on GPU.
        return errors::Internal(
            "Collective all-reduce does not support datatype DT_INT32 on "
            "DEVICE_GPU");
      } else {
        return CollectiveRegistry::Lookup(
            col_params.instance.impl_details.collective_name, col_impl);
      }
    case DT_BFLOAT16:
      if (col_params.group.device_type == DEVICE_GPU &&
          col_params.instance.type == REDUCTION_COLLECTIVE) {
        return errors::Internal(
            "Collective all-reduce does not support datatype DT_BFLOAT16 on "
            "DEVICE_GPU");
      } else {
        return CollectiveRegistry::Lookup(
            col_params.instance.impl_details.collective_name, col_impl);
      }
    case DT_HALF:
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_INT64: {
      return CollectiveRegistry::Lookup(
          col_params.instance.impl_details.collective_name, col_impl);
    }
    default:
      return errors::Internal(
          "CollectiveImplementation does not support datatype ",
          DataTypeString(col_params.instance.data_type));
  }
}

bool BaseCollectiveExecutor::CheckDependencies(
    const CollectiveParams& col_params) {
  for (int32_t instance : col_params.instance.impl_details.dependencies) {
    auto find_iter = launched_.find(instance);
    if (find_iter == launched_.end() || find_iter->second != 0) {
      VLOG(1) << "Collective " << col_params.ToString()
              << " blocked by instance " << instance;
      return false;
    }
  }
  return true;
}

void BaseCollectiveExecutor::WaitForDependencies(
    const CollectiveParams& col_params) {
  mutex_lock l(launch_mu_);
  while (!CheckDependencies(col_params)) {
    launch_cv_.wait(l);
  }
  VLOG(1) << "Unblocking collective " << col_params.ToString();
}

void BaseCollectiveExecutor::UnblockDependencies(
    const CollectiveParams& col_params) {
  mutex_lock l(launch_mu_);
  if (launched_.find(col_params.instance.instance_key) == launched_.end()) {
    const string& task_name =
        col_params.group.members[col_params.default_rank].task;
    const int32_t num_devices =
        col_params.group.num_devices_per_task.at(task_name);
    launched_[col_params.instance.instance_key] = num_devices;
  }
  if (--launched_[col_params.instance.instance_key] == 0) {
    VLOG(1) << "Unblocking dependencies for collective instance "
            << col_params.instance.instance_key;
    launch_cv_.notify_all();
  }
}

}  // namespace tensorflow
