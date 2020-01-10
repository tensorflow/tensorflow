#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/concat_op.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

template <typename T>
static inline void Copy(T* dst, const T* src, int n) {
  if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
    memcpy(dst, src, n * sizeof(T));
  } else {
    for (int k = 0; k < n; ++k) {
      *dst++ = *src++;
    }
  }
}

template <typename T>
void ConcatCPU(DeviceBase* d,
               const std::vector<
                   std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& inputs,
               typename TTypes<T, 2>::Matrix* output) {
  int num_inputs = inputs.size();
  std::vector<ptrdiff_t> sizes;
  sizes.reserve(num_inputs);
  int row_size = 0;
  for (int j = 0; j < num_inputs; ++j) {
    sizes.push_back(inputs[j]->dimension(1));
    row_size += sizes.back();
  }

  auto worker_threads = d->tensorflow_cpu_worker_threads();
  int num_threads = std::min<int>(std::min(4, worker_threads->num_threads),
                                  output->size() / 4096);
  // Single threaded mode.
  if (num_threads == 0) {
    T* out = &(*output)(0, 0);
    std::vector<const T*> inp;
    inp.reserve(num_inputs);
    for (int j = 0; j < num_inputs; ++j) {
      inp.push_back(&(*inputs[j])(0, 0));
    }
    const int dim0 = output->dimension(0);
    for (int i = 0; i < dim0; ++i) {
      for (int j = 0; j < num_inputs; ++j) {
        auto size = sizes[j];
        Copy(out, inp[j], size);
        out += size;
        inp[j] += size;
      }
    }
    return;
  }

  // Sharded mode.
  auto work = [&row_size, &sizes, &inputs, &output, &num_inputs](int64 start,
                                                                 int64 end) {
    int64 skipped_rows = start / row_size;
    T* out = output->data() + skipped_rows * row_size;
    T* out_start = output->data() + start;
    T* out_end = output->data() + end;

    // Handle partial row at start
    if (out < out_start) {
      for (int j = 0; j < num_inputs; ++j) {
        ptrdiff_t size = sizes[j];
        ptrdiff_t offset = out_start - out;
        if (size <= offset) {
          out += size;
          continue;
        }
        const T* inp = &(*inputs[j])(skipped_rows, 0);
        if (offset > 0) {
          out += offset;
          inp += offset;
          size -= offset;
        }
        size = std::min(size, out_end - out);
        if (size <= 0) break;
        Copy(out, inp, size);
        out += size;
      }
      ++skipped_rows;
    }
    if (out == out_end) return;
    CHECK(out >= out_start);
    CHECK(out < out_end);

    // Copy remaining data.
    std::vector<const T*> inp;
    inp.reserve(num_inputs);
    for (int j = 0; j < num_inputs; ++j) {
      inp.push_back(&(*inputs[j])(skipped_rows, 0));
    }
    const int dim0 = output->dimension(0);
    for (int i = skipped_rows; i < dim0; ++i) {
      for (int j = 0; j < num_inputs; ++j) {
        ptrdiff_t size = std::min(sizes[j], out_end - out);
        Copy(out, inp[j], size);
        out += size;
        inp[j] += size;
        if (out == out_end) return;
      }
    }
  };
  Shard(num_threads, worker_threads->workers, output->size(), 100, work);
}

#define REGISTER(T)                                                            \
  template void ConcatCPU<T>(                                                  \
      DeviceBase*,                                                             \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&, \
      typename TTypes<T, 2>::Matrix* output);
TF_CALL_ALL_TYPES(REGISTER)
REGISTER(quint8)
REGISTER(qint8)
REGISTER(qint32)
REGISTER(bfloat16)

}  // namespace tensorflow
