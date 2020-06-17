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
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"

#define VALUE_IN_DEBUG_STRING false

namespace tensorflow {
/*static*/
int64 CollectiveAdapter::AlignedChunkElts(int64 elt_bytes, int64 total_elts,
                                          int64 num_chunks) {
  DCHECK_GT(num_chunks, 0);
  int64 base_chunk_elts = (total_elts + (num_chunks - 1)) / num_chunks;
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
  int64 chunk_bytes = base_chunk_elts * elt_bytes;
  int64 diff =
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
  CollectiveAdapterImpl(Tensor* output, int64 num_chunks, Allocator* allocator,
                        bool align_chunks)
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
  inline int64 ChunkElts(int i) const {
    DCHECK_LT(i, num_chunks_);
    const T* chunk_start = std::min(data_end_, data_start_ + i * chunk_elts_);
    const T* chunk_end = std::min(data_end_, chunk_start + chunk_elts_);
    return chunk_end - chunk_start;
  }

  int64 ChunkBytes(int i) const override { return sizeof(T) * ChunkElts(i); }

  // Returns a new Tensor that aliases the required chunk.
  Tensor ChunkAlias(int i) override {
    int64 start = chunk_elts_ * i;
    int64 num_elts = ChunkElts(i);
    // If this chunk is empty the prior chunk might also be short
    // so always take an empty slice from the front of the tensor
    // to avoid an illegal offset check failure somewhere.
    return (num_elts > 0) ? output_.Slice(start, start + num_elts)
                          : output_.Slice(0, 0);
  }

  Tensor TempChunk(int i) const override {
    AllocationAttributes empty;
    ScopedMemoryDebugAnnotation op_annotation(
        "CollectiveAdapterImpl::TempChunk");
    return Tensor(allocator_, dt_, {ChunkElts(i)}, empty);
  }

  string DebugString() const override {
    return strings::StrCat(
        "base addr ", reinterpret_cast<int64>(DMAHelper::base(&output_)),
        " num_chunks ", num_chunks_, " total_elts ", total_elts_, " chunk_elts",
        chunk_elts_, " value ",
        VALUE_IN_DEBUG_STRING ? output_.SummarizeValue(1024) : "<hidden>");
  }

  string TBounds(const Tensor& t) const override {
    int64 base_addr = reinterpret_cast<int64>(DMAHelper::base(&t));
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
  const int64 num_chunks_;
  Allocator* allocator_;
  const int64 total_elts_;
  const int64 chunk_elts_;
  const T* data_start_;
  const T* data_end_;
};

}  // namespace

CollectiveAdapter* MakeCollectiveAdapter(Tensor* output, int num_chunks,
                                         Allocator* allocator,
                                         bool align_chunks) {
  switch (output->dtype()) {
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
      return new CollectiveAdapterImpl<int64>(output, num_chunks, allocator,
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
  VLOG(1) << "BaseCollectiveExecutor::StartAbort " << s;
  remote_access_->StartAbort(s);
}

void BaseCollectiveExecutor::ExecuteAsync(OpKernelContext* ctx,
                                          const CollectiveParams& col_params,
                                          const string& exec_key,
                                          StatusCallback done) {
  const auto is_callback_called = std::make_shared<std::atomic<bool>>(false);

  // On any individual collective Op failure we need to abort the
  // BufRendezvous so that other Ops in the instance don't hang
  // waiting for transmissions that will never happen.  Do so after a
  // delay so that the original error status is more likely to
  // propagate up, and peers are unlikely to re-create the purged
  // BufRendezvous by late-arriving requests.
  StatusCallback done_safe = [this, done, is_callback_called](const Status& s) {
    auto should_call_callback = !is_callback_called->exchange(true);
    if (should_call_callback) {
      if (!s.ok()) {
        Ref();  // Ensure this lasts until the closure executes.
        SchedNonBlockingClosureAfter(1000000, [this, s] {
          remote_access_->buf_rendezvous()->StartAbort(s);
          Unref();
        });
      }
      done(s);
    }
  };

  auto timeout_microseconds = static_cast<int64>(
      col_params.instance.impl_details.timeout_seconds * 1'000'000);
  if (timeout_microseconds > 0) {
    // TODO(xldrx): Share the timeout watchdog thread among collectives.
    SchedNonBlockingClosureAfter(
        timeout_microseconds, [is_callback_called, done_safe] {
          if (!is_callback_called->load()) {
            auto status = Status(error::DEADLINE_EXCEEDED,
                                 "Collective has timed out during execution.");
            done_safe(status);
          }
        });
  }

  Tensor* output = ctx->mutable_output(0);
  const Tensor* input = (col_params.instance.type == REDUCTION_COLLECTIVE ||
                         col_params.instance.type == GATHER_COLLECTIVE ||
                         (col_params.instance.type == BROADCAST_COLLECTIVE &&
                          col_params.is_source))
                            ? &ctx->input(0)
                            : nullptr;
  CollectiveImplementationInterface* col_impl = nullptr;
  Status status = CreateCollective(col_params, &col_impl);
  if (!status.ok()) {
    done_safe(status);
    DCHECK_EQ(nullptr, col_impl);
    return;
  }
  CollectiveContext* col_ctx =
      new CollectiveContext(this, dev_mgr_, ctx, CtxParams(ctx), col_params,
                            exec_key, step_id_, input, output);
  status = col_impl->InitializeCollectiveContext(col_ctx);
  if (!status.ok()) {
    done_safe(status);
    delete col_ctx;
    delete col_impl;
    return;
  }
  // Run on an unbounded work queue that can handle blocking work so as to not
  // starve executor threads.
  remote_access_->RunClosure([col_impl, col_ctx, done_safe, ctx]() {
    profiler::TraceMe activity(
        [&] {
          return strings::StrCat(ctx->op_kernel().name_view(), ":",
                                 ctx->op_kernel().type_string_view(),
                                 "#id=", ctx->step_id(), "#");
        },
        profiler::TraceMeLevel::kInfo);
    col_impl->Run([col_impl, col_ctx, done_safe](const Status& s) {
      done_safe(s);
      delete col_ctx;
      delete col_impl;
    });
  });
}

void BaseCollectiveExecutor::CompleteParamsAsync(
    const string& device, CollectiveParams* cp, CancellationManager* cancel_mgr,
    StatusCallback done) {
  cp->instance.gpu_ring_order = *gpu_ring_order_;
  const auto is_callback_called = std::make_shared<std::atomic<bool>>(false);
  auto done_with_timeout = done;
  auto timeout_microseconds =
      static_cast<int64>(cp->instance.impl_details.timeout_seconds * 1'000'000);
  if (timeout_microseconds > 0) {
    // TODO(xldrx): Share the timeout watchdog thread among collectives.
    SchedNonBlockingClosureAfter(
        timeout_microseconds, [is_callback_called, done] {
          if (!is_callback_called->load()) {
            auto status =
                Status(error::DEADLINE_EXCEEDED,
                       "Collective has timed out waiting for other workers.");
            done(status);
          }
        });
    done_with_timeout = [is_callback_called, done](const Status& s) {
      auto should_call_callback = !is_callback_called->exchange(true);
      if (should_call_callback) {
        done(s);
      }
    };
  }
  cem_->GetParamResolver()->CompleteParamsAsync(device, cp, cancel_mgr,
                                                done_with_timeout);
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
  for (int32 instance : col_params.instance.impl_details.dependencies) {
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
        col_params.instance.task_names[col_params.default_rank];
    const int32 num_devices =
        col_params.instance.num_devices_per_task.at(task_name);
    launched_[col_params.instance.instance_key] = num_devices;
  }
  if (--launched_[col_params.instance.instance_key] == 0) {
    VLOG(1) << "Unblocking dependencies for collective instance "
            << col_params.instance.instance_key;
    launch_cv_.notify_all();
  }
}

}  // namespace tensorflow
