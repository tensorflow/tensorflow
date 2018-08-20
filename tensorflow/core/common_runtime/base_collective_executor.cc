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

#include "tensorflow/core/common_runtime/broadcaster.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/ring_reducer.h"
#include "tensorflow/core/lib/core/notification.h"

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
  CHECK_EQ(0, diff % elt_bytes);
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
  CollectiveAdapterImpl(Tensor* output, int64 num_chunks, Allocator* allocator)
      : output_(std::move(*output)),
        dt_(output_.dtype()),
        old_shape_(output_.shape()),
        num_chunks_(num_chunks),
        allocator_(allocator),
        total_elts_(output_.NumElements()),
        chunk_elts_(AlignedChunkElts(sizeof(T), total_elts_, num_chunks_)),
        data_start_(reinterpret_cast<T*>(DMAHelper::base(&output_))),
        data_end_(data_start_ + total_elts_) {
    CHECK_GT(chunk_elts_, 0);
    Flatten();
  }

  ~CollectiveAdapterImpl() override {}

  const Tensor& Value() const override { return output_; }

  // If necessary, flatten output.
  void Flatten() {
    if (old_shape_.dims() > 1) {
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

  Tensor Scalar(int v) const override {
    Tensor t(dt_, TensorShape({}));
    t.scalar<T>()() = v;
    return t;
  }

  Tensor Scalar(Allocator* a) const override {
    Tensor t(a, dt_, TensorShape({}));
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
                                         Allocator* allocator) {
  switch (output->dtype()) {
    case DT_FLOAT:
      return new CollectiveAdapterImpl<float>(output, num_chunks, allocator);
      break;
    case DT_DOUBLE:
      return new CollectiveAdapterImpl<double>(output, num_chunks, allocator);
      break;
    case DT_INT32:
      return new CollectiveAdapterImpl<int32>(output, num_chunks, allocator);
      break;
    case DT_INT64:
      return new CollectiveAdapterImpl<int64>(output, num_chunks, allocator);
      break;
    default:
      LOG(FATAL) << "Unsupported type " << output->dtype()
                 << " to MakeCollectiveAdapter";
      return nullptr;
  }
}

BaseCollectiveExecutor::~BaseCollectiveExecutor() {}

void BaseCollectiveExecutor::StartAbort(const Status& s) {
  LOG(WARNING) << "BaseCollectiveExecutor::StartAbort " << s;
  remote_access_->StartAbort(s);
}

void BaseCollectiveExecutor::ExecuteAsync(OpKernelContext* ctx,
                                          const CollectiveParams& col_params,
                                          const string& exec_key,
                                          StatusCallback done) {
  // On any individual collective Op failure we need to abort the
  // BufRendezvous so that other Ops in the instance don't hang
  // waiting for transmissions that will never happen.  Do so after a
  // delay so that the original error status is more likely to
  // propagate up, and peers are unlikely to re-create the purged
  // BufRendezvous by late-arriving requests.
  StatusCallback done_safe = [this, done](const Status& s) {
    if (!s.ok()) {
      Ref();  // Ensure this lasts until the closure executes.
      SchedNonBlockingClosureAfter(1000000, [this, s] {
        remote_access_->buf_rendezvous()->StartAbort(s);
        Unref();
      });
    }
    done(s);
  };

  Tensor* output = ctx->mutable_output(0);
  string error;
  switch (col_params.instance.type) {
    case REDUCTION_COLLECTIVE: {
      // TODO(tucker): support other reduction algorithms,
      // e.g. tree-reduce, hybrid tree/ring, delegate-to-NCCL, etc.
      const Tensor* input = &ctx->input(0);
      RingReducer* reducer =
          CreateReducer(ctx, CtxParams(ctx), col_params, exec_key, step_id_,
                        input, output, &error);
      if (!reducer) {
        done_safe(errors::Internal(error));
        return;
      }
      // Run in an I/O thread, so as not to starve the executor threads.
      // TODO(tucker): Instead of forking every per-device Collective
      // Op off into its own thread, consider queuing them on a
      // fixed-size thread-pool dedicated to running CollectiveOps.
      SchedClosure([reducer, done_safe]() {
        reducer->Run([reducer, done_safe](const Status& s) {
          done_safe(s);
          delete reducer;
        });
      });
    } break;

    case BROADCAST_COLLECTIVE: {
      Broadcaster* broadcaster = CreateBroadcaster(
          ctx, CtxParams(ctx), col_params, exec_key, step_id_, output, &error);
      if (!broadcaster) {
        done_safe(errors::Internal(error));
        return;
      }
      // Run in an I/O thread, so as not to starve the executor threads.
      SchedClosure([broadcaster, done_safe]() {
        broadcaster->Run([broadcaster, done_safe](const Status& s) {
          done_safe(s);
          delete broadcaster;
        });
      });
    } break;

    default:
      done_safe(errors::Internal("Unimplemented CollectiveType ",
                                 col_params.instance.type));
  }
}

RingReducer* BaseCollectiveExecutor::CreateReducer(
    OpKernelContext* ctx, OpKernelContext::Params* params,
    const CollectiveParams& col_params, const string& exec_key, int64 step_id,
    const Tensor* input, Tensor* output, string* error) {
  switch (col_params.instance.data_type) {
    case DT_INT32:
      if (col_params.group.device_type == DEVICE_GPU) {
        *error =
            "Collective Reduce does not support datatype DT_INT32 on "
            "DEVICE_GPU";
        return nullptr;
      }
      TF_FALLTHROUGH_INTENDED;
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_INT64:
      return new RingReducer(this, dev_mgr_, ctx, params, col_params, exec_key,
                             step_id, input, output);
      break;
    default:
      *error = strings::StrCat("Collective Reduce does not support datatype ",
                               col_params.instance.data_type);
      return nullptr;
  }
}

Broadcaster* BaseCollectiveExecutor::CreateBroadcaster(
    OpKernelContext* ctx, OpKernelContext::Params* params,
    const CollectiveParams& col_params, const string& exec_key, int64 step_id,
    Tensor* output, string* error) {
  switch (col_params.instance.data_type) {
    case DT_INT32:
      if (col_params.group.device_type == DEVICE_GPU) {
        *error =
            "Collective Broadcast does not support datatype DT_INT32 on "
            "DEVICE_GPU";
        return nullptr;
      }
      TF_FALLTHROUGH_INTENDED;
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_INT64: {
      return new Broadcaster(this, dev_mgr_, ctx, params, col_params, exec_key,
                             step_id, output);
    } break;
    default:
      *error =
          strings::StrCat("Collective Broadcast does not support datatype ",
                          DataTypeString(col_params.instance.data_type));
      return nullptr;
  }
}

}  // namespace tensorflow
