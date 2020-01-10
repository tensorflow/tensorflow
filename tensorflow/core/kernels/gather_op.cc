// See docs in ../ops/array_ops.cc.

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

namespace {
template <typename T, typename Index, int static_slice_elems>
void HandleCopies(const Tensor& Tparams,
                  typename TTypes<Index>::ConstVec& Tindices, int slice_elems,
                  typename TTypes<T>::Matrix Tout) {
  const int N = Tindices.dimension(0);
  const auto& Tparams_flat = Tparams.flat_outer_dims<T>();
  T* Tout_base = &Tout(0, 0);
  const T* Tparams_base = &Tparams_flat(0, 0);
  const size_t slice_bytes = slice_elems * sizeof(T);
  if (static_slice_elems >= 0) {
    // Give compiler static knowledge of the number of elements/bytes
    CHECK_EQ(static_slice_elems, slice_elems);
    slice_elems = static_slice_elems;
  }
  for (int i = 0; i < N; i++) {
    int j = i + 1;
    if (j < N) {
      port::prefetch<port::PREFETCH_HINT_T0>(&Tparams_flat(Tindices(j), 0));
      port::prefetch<port::PREFETCH_HINT_T0>(&Tout(j, 0));
    }
    memcpy(Tout_base + i * slice_elems,
           Tparams_base + Tindices(i) * slice_elems, slice_bytes);
  }
}

}  // anonymous namespace

template <typename T, typename Index>
class GatherOp : public OpKernel {
 public:
  //   QUESTION: It'd be nice to support DT_INT16, DT_UINT8,
  //   etc. here for the type of the second input argument.  Should
  //   we have the framework do some sort of integer promotion
  //   automatically, or should that be something that users have to
  //   do explicitly with a conversion operator in the graph?
  explicit GatherOp(OpKernelConstruction* c) : OpKernel(c) {
    const DataType dt = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<Index>::v();
    OP_REQUIRES_OK(c, c->MatchSignature({dt, index_t}, {dt}));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& Tparams = c->input(0);
    const Tensor& Tindices = c->input(1);
    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(Tparams.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));
    const int64 N = Tindices.NumElements();
    const int64 first_dim_size = Tparams.dim_size(0);

    // Validate all the indices are in range
    auto Tindices_vec = Tindices.flat<Index>();
    for (int64 i = 0; i < N; i++) {
      const Index index = Tindices_vec(i);
      OP_REQUIRES(c, index >= 0 && index < first_dim_size,
                  errors::InvalidArgument(
                      strings::StrCat("Index ", index, " at offset ", i,
                                      " in Tindices is out of range")));
    }

    // The result shape is indices.shape + params.shape[1:].
    TensorShape result_shape = Tindices.shape();
    for (int i = 1; i < Tparams.dims(); i++) {
      result_shape.AddDim(Tparams.dim_size(i));
    }

    Tensor* Tout = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &Tout));
    const auto& Tparams_flat = Tparams.flat_outer_dims<T>();
    if (N > 0) {
      auto Tindices_flat = Tindices.flat<Index>();
      auto Tout_flat = Tout->shaped<T, 2>({N, Tout->NumElements() / N});
      if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
        const int64 slice_size = Tout->NumElements() / N;
#define SPECIALIZE(elems)                                               \
  do {                                                                  \
    if (slice_size == elems) {                                          \
      HandleCopies<T, Index, elems>(Tparams, Tindices_flat, slice_size, \
                                    Tout_flat);                         \
      return;                                                           \
    }                                                                   \
  } while (0)

        SPECIALIZE(10);
        SPECIALIZE(20);

#undef SPECIALIZE

        HandleCopies<T, Index, -1>(Tparams, Tindices_flat, slice_size,
                                   Tout_flat);
      } else {
        for (int i = 0; i < N; i++) {
          int j = i + 1;
          if (j < N) {
            port::prefetch<port::PREFETCH_HINT_T0>(
                &Tparams_flat(Tindices_vec(j), 0));
            port::prefetch<port::PREFETCH_HINT_T0>(&Tout_flat(j, 0));
          }
          // Copy last Ndim-1 dimensions of Tparams[Tindices[i]] to Tout[i]
          Tout_flat.template chip<0>(i) =
              Tparams_flat.template chip<0>(Tindices_vec(i));
        }
      }
    }
  }
};

#define REGISTER_GATHER(type, index_type)                              \
  REGISTER_KERNEL_BUILDER(Name("Gather")                               \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherOp<type, index_type>)

#define REGISTER_GATHER_INT32(type) REGISTER_GATHER(type, int32)
#define REGISTER_GATHER_INT64(type) REGISTER_GATHER(type, int64)

TF_CALL_ALL_TYPES(REGISTER_GATHER_INT32);
TF_CALL_ALL_TYPES(REGISTER_GATHER_INT64);

#undef REGISTER_GATHER_INT32
#undef REGISTER_GATHER_INT64
#undef REGISTER_GATHER

}  // namespace tensorflow
