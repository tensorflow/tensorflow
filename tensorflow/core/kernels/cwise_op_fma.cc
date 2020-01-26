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

#include "tensorflow/core/kernels/cwise_op_fma.h"

#include <vector>

#include "tensorflow/core/kernels/cwise_ops_common.h"
namespace tensorflow {

//#define FMA_TRACE

typedef Eigen::ThreadPoolDevice CPUDevice;

template <FMAType Type, typename T>
T fma_op(T m1, T m2) {
  if (Type == FMAType_Add)
    return m1 + m2;
  else if (Type == FMAType_Sub)
    return m1 - m2;
  else
    return m2 - m1;
}

template <typename Device, typename T, FMAType Type>
void LaunchFusedMulAddOp<Device, T, Type>::operator()(
    const Device& device, T* out, const T* x1, const T* y1, const T* x2,
    uint64 elements, bool broadcast_x1, bool broadcast_y1, bool broadcast_x2) {
  for (uint64 i = 0; i < elements; i++) {
    out[i] = fma_op<Type>(x1[broadcast_x1 ? 0 : i] * y1[broadcast_y1 ? 0 : i],
                          x2[broadcast_x2 ? 0 : i]);
  }
}

template <typename Device, typename T, FMAType Type>
void LaunchFusedMulAdd2Op<Device, T, Type>::operator()(
    const Device& device, T* out, const T* x1, const T* y1, const T* x2,
    const T* y2, uint64 elements, bool broadcast_x1, bool broadcast_y1,
    bool broadcast_x2, bool broadcast_y2) {
  for (uint64 i = 0; i < elements; i++) {
    out[i] = fma_op<Type>(x1[broadcast_x1 ? 0 : i] * y1[broadcast_y1 ? 0 : i],
                          x2[broadcast_x2 ? 0 : i] * y2[broadcast_y2 ? 0 : i]);
  }
}

template <typename Device, typename T, FMAType Type>
void FallbackLaunchFusedMulAddOp<Device, T, Type>::operator()(
    const Device& device, T* out, const T* x1, const T* y1, const T* x2,
    int64 dims[6], uint8 broadcast_masks[6]) {
#ifdef FMA_TRACE
  printf("FallbackLaunchFusedMulAddOpCPU %p %p %p %p\n", out, x1, y1, x2);
#endif
  int64 strides[3][5];
  for (int i = 0; i < 3; i++) {
    int64 s = 1;
    for (int j = 0; j < 5; j++) {
      int b = broadcast_masks[j] & (1 << i);
      int b_next = broadcast_masks[j + 1] & (1 << i);
      s *= b ? dims[j] : 1;
      strides[i][j] = s * ((b_next - b) >> i);
    }
  };
  for (uint64 t = 0; t < dims[5]; t++) {
    for (uint64 u = 0; u < dims[4]; u++) {
      for (uint64 v = 0; v < dims[3]; v++) {
        for (uint64 z = 0; z < dims[2]; z++) {
          for (uint64 y = 0; y < dims[1]; y++) {
            for (uint64 x = 0; x < dims[0]; x++) {
              *out = fma_op<Type>((*x1) * (*y1), (*x2));
              out++;
              if (broadcast_masks[0] & 1) x1++;
              if (broadcast_masks[0] & 2) y1++;
              if (broadcast_masks[0] & 4) x2++;
            }
            x1 += strides[0][0];
            y1 += strides[1][0];
            x2 += strides[2][0];
          }
          x1 += strides[0][1];
          y1 += strides[1][1];
          x2 += strides[2][1];
        }
        x1 += strides[0][2];
        y1 += strides[1][2];
        x2 += strides[2][2];
      }
      x1 += strides[0][3];
      y1 += strides[1][3];
      x2 += strides[2][3];
    }
    x1 += strides[0][4];
    y1 += strides[1][4];
    x2 += strides[2][4];
  }
}

template <typename Device, typename T, FMAType Type>
void FallbackLaunchFusedMulAdd2Op<Device, T, Type>::operator()(
    const Device& device, T* out, const T* x1, const T* y1, const T* x2,
    const T* y2, int64 dims[6], uint8 broadcast_masks[6]) {
  int64 strides[4][4];
  for (int i = 0; i < 4; i++) {
    int64 s = 1;
    for (int j = 0; j < 4; j++) {
      int b = broadcast_masks[j] & (1 << i);
      int b_next = broadcast_masks[j + 1] & (1 << i);
      s *= b ? dims[j] : 1;
      strides[i][j] = s * ((b_next - b) >> i);
    }
  };
  for (uint64 u = 0; u < dims[4]; u++) {
    for (uint64 v = 0; v < dims[3]; v++) {
      for (uint64 z = 0; z < dims[2]; z++) {
        for (uint64 y = 0; y < dims[1]; y++) {
          for (uint64 x = 0; x < dims[0]; x++) {
            *out = fma_op<Type>((*x1) * (*y1), (*x2) * (*y2));
            out++;
            if (broadcast_masks[0] & 1) x1++;
            if (broadcast_masks[0] & 2) y1++;
            if (broadcast_masks[0] & 4) x2++;
            if (broadcast_masks[0] & 8) y2++;
          }
          x1 += strides[0][0];
          y1 += strides[1][0];
          x2 += strides[2][0];
          y2 += strides[3][0];
        }
        x1 += strides[0][1];
        y1 += strides[1][1];
        x2 += strides[2][1];
        y2 += strides[3][1];
      }
      x1 += strides[0][2];
      y1 += strides[1][2];
      x2 += strides[2][2];
      y2 += strides[3][2];
    }
    x1 += strides[0][3];
    y1 += strides[1][3];
    x2 += strides[2][3];
    y2 += strides[3][3];
  }
}

template <typename Device, int N>
class FusedMulAddBase {
 public:
  // Analyze the incoming shapes for compatibility, calculate
  // the output shape and the necessary broadcasts.
  bool DoShapeAnalysis(OpKernelContext* ctx, const Tensor** inputs,
                       bool& pure_broadcast,
                       std::vector<uint8>& broadcast_masks,
                       std::vector<int64>& out_dims, TensorShape& out_shape,
                       int64& out_elements) {
    int64 in_elements[N];
    int in_ranks[N];
    int rank = 0;
    bool scalars[N];

    for (int i = 0; i < N; i++) {
      in_elements[i] = inputs[i]->NumElements();
      in_ranks[i] = inputs[i]->dims();
      scalars[i] = (in_ranks[i] == 0);
      rank = (rank > in_ranks[i]) ? rank : in_ranks[i];
#ifdef FMA_TRACE
      printf("Input %d: %d dimensions, %d elements\n", i, in_ranks[i],
             in_elements[i]);
      for (int j = 0; j < in_ranks[i]; j++)
        printf("%d ", inputs[i]->shape().dim_size(j));
      printf("\n");
#endif
    }
    broadcast_masks.resize(rank);
    out_dims.resize(rank);

    for (int i = 0; i < rank; i++) {
      int64 xds[N];
      int64 max_dim = 0;
      int ii = rank - i - 1;
      bool null_shape = false;
      // find the largest dimension of all inputs at this index
      for (int j = 0; j < N; j++) {
        xds[j] = (scalars[j] || ii >= in_ranks[j])
                     ? 1
                     : inputs[j]->shape().dim_size(in_ranks[j] - ii - 1);
        max_dim = (max_dim > xds[j]) ? max_dim : xds[j];
      }

      for (int j = 0; j < N; j++) {
        // make sure dimensions are compatible
        if (!(xds[j] == 0 || xds[j] == 1 || xds[j] == max_dim)) return false;
        if (xds[j] == 0) null_shape = true;
        broadcast_masks[rank - i - 1] |= (xds[j] != 1 ? 1 : 0) << j;
      }
      if (null_shape) max_dim = 0;
      out_shape.AddDim(max_dim);
      out_dims[rank - i - 1] = max_dim;
    }
    out_elements = out_shape.num_elements();
    pure_broadcast = true;
    if (out_elements <= 1) return true;

    for (int i = 0; i < N; i++)
      if (in_elements[i] != 1 && in_elements[i] != out_elements)
        pure_broadcast = false;
    if (pure_broadcast) {
      broadcast_masks[0] = 0;
      for (int i = 0; i < N; i++)
        if (in_elements[i] != 1) broadcast_masks[0] |= 1 << i;
#ifdef FMA_TRACE
      printf("%d elements, pure broadcast\n", out_elements);
#endif
      return true;
    }
#ifdef FMA_TRACE
    printf("%d elements, broadcast %s\n", out_elements,
           pure_broadcast ? "true" : "false");
    for (int i = 0; i < rank; i++)
      printf("out_dim[%d] = %d mask %d\n", i, out_dims[i],
             (int)broadcast_masks[i]);
#endif
    // folding dimensions from the highest down
    // [50,10,10]x[50,10,10]x[50,1,1] -> [50,100]x[50,100]x[50,1]
    bool folded = false;

    while (rank > 1 && out_dims.back() == 1) {
      out_dims.pop_back();
      broadcast_masks.pop_back();
      rank--;
    }

    for (int i = rank - 1; i > 0; i--) {
      if (out_dims[i] == 1 || broadcast_masks[i] == broadcast_masks[i - 1]) {
        folded = true;
        out_dims[i - 1] *= out_dims[i];
        for (int j = i; j < rank - 1; j++) {
          out_dims[j] = out_dims[j + 1];
          broadcast_masks[j] = broadcast_masks[j + 1];
        }
        out_dims.pop_back();
        broadcast_masks.pop_back();
        rank--;
      }
    }
#ifdef FMA_TRACE
    if (folded) {
      printf("After folding:\n");
      for (int i = 0; i < rank; i++)
        printf("out_dim[%d] = %d mask %d\n", i, out_dims[i],
               (int)broadcast_masks[i]);
    }
#endif
    return (rank <= 6);
  }
};

template <typename Device, typename T, FMAType Type>
class FusedMulAddOp : public OpKernel, public FusedMulAddBase<Device, 3> {
 public:
  explicit FusedMulAddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs[3];
    // printf("FusedMulAddOp\n");
    OP_REQUIRES_OK(ctx, ctx->input("x1", &inputs[0]));
    OP_REQUIRES_OK(ctx, ctx->input("y1", &inputs[1]));
    OP_REQUIRES_OK(ctx, ctx->input("x2", &inputs[2]));
    bool pure_broadcast = true;

    std::vector<uint8> broadcast_masks;
    std::vector<int64> out_dims;
    //    uint8 broadcast_masks[6]={0,0,0,0,0,0};
    //    int64 out_dims[6]={1,1,1,1,1,1};
    TensorShape out_shape;
    int64 out_elements = 0;
    bool ok = FusedMulAddBase<Device, 3>::DoShapeAnalysis(
        ctx, inputs, pure_broadcast, broadcast_masks, out_dims, out_shape,
        out_elements);
    OP_REQUIRES(
        ctx, ok,
        errors::InvalidArgument("FusedMulAdd with incompatible shapes"));
    OP_REQUIRES(
        ctx, pure_broadcast || broadcast_masks.size() <= 6,
        errors::InvalidArgument("FusedMulAdd with ", broadcast_masks.size(),
                                " broadcast ranks not supported"));
    while (broadcast_masks.size() < 6) {
      broadcast_masks.push_back(0);
      out_dims.push_back(1);
    }
    // pure_broadcast=false;
    Tensor* output = nullptr;
    // todo: an OP_REQUIRES to check that all dims fit in 32 bit
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    if (out_elements == 0) return;
    if (pure_broadcast)
      LaunchFusedMulAddOp<Device, T, Type>()(
          ctx->eigen_device<Device>(), output->flat<T>().data(),
          inputs[0]->flat<T>().data(), inputs[1]->flat<T>().data(),
          inputs[2]->flat<T>().data(), out_elements, !(broadcast_masks[0] & 1),
          !(broadcast_masks[0] & 2), !(broadcast_masks[0] & 4));
    else
      FallbackLaunchFusedMulAddOp<Device, T, Type>()(
          ctx->eigen_device<Device>(), output->flat<T>().data(),
          inputs[0]->flat<T>().data(), inputs[1]->flat<T>().data(),
          inputs[2]->flat<T>().data(), &out_dims[0], &broadcast_masks[0]);
  }
};

template <typename Device, typename T, FMAType Type>
class FusedMulAdd2Op : public OpKernel, public FusedMulAddBase<Device, 4> {
 public:
  explicit FusedMulAdd2Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs[4];
    OP_REQUIRES_OK(ctx, ctx->input("x1", &inputs[0]));
    OP_REQUIRES_OK(ctx, ctx->input("y1", &inputs[1]));
    OP_REQUIRES_OK(ctx, ctx->input("x2", &inputs[2]));
    OP_REQUIRES_OK(ctx, ctx->input("y2", &inputs[3]));
    bool pure_broadcast = true;
    std::vector<uint8> broadcast_masks;
    std::vector<int64> out_dims;
    TensorShape out_shape;
    int64 out_elements = 0;
    bool ok = FusedMulAddBase<Device, 4>::DoShapeAnalysis(
        ctx, inputs, pure_broadcast, broadcast_masks, out_dims, out_shape,
        out_elements);
#ifdef TRACE_FMA
    fflush(stdout);
#endif
    OP_REQUIRES(
        ctx, ok,
        errors::InvalidArgument("FusedMulAdd2 with incompatible shapes"));
    OP_REQUIRES(
        ctx, pure_broadcast || broadcast_masks.size() <= 6,
        errors::InvalidArgument("FusedMulAdd with ", broadcast_masks.size(),
                                " broadcast ranks not supported"));
    while (broadcast_masks.size() < 6) {
      broadcast_masks.push_back(0);
      out_dims.push_back(1);
    }
    // pure_broadcast=false;
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    if (out_elements == 0) return;
    if (pure_broadcast)
      LaunchFusedMulAdd2Op<Device, T, Type>()(
          ctx->eigen_device<Device>(), output->flat<T>().data(),
          inputs[0]->flat<T>().data(), inputs[1]->flat<T>().data(),
          inputs[2]->flat<T>().data(), inputs[3]->flat<T>().data(),
          out_elements, !(broadcast_masks[0] & 1), !(broadcast_masks[0] & 2),
          !(broadcast_masks[0] & 4), !(broadcast_masks[0] & 8));
    else
      FallbackLaunchFusedMulAdd2Op<Device, T, Type>()(
          ctx->eigen_device<Device>(), output->flat<T>().data(),
          inputs[0]->flat<T>().data(), inputs[1]->flat<T>().data(),
          inputs[2]->flat<T>().data(), inputs[3]->flat<T>().data(),
          &out_dims[0], &broadcast_masks[0]);
  }
};

#define REGISTER_CPU_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedMulAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"),    \
      FusedMulAddOp<CPUDevice, type, FMAType_Add>);                         \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedMulAdd2").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      FusedMulAdd2Op<CPUDevice, type, FMAType_Add>);                        \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedMulSub").Device(DEVICE_CPU).TypeConstraint<type>("T"),    \
      FusedMulAddOp<CPUDevice, type, FMAType_Sub>);                         \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedMulSubRev").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      FusedMulAddOp<CPUDevice, type, FMAType_SubRev>);                      \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedMulSub2").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      FusedMulAdd2Op<CPUDevice, type, FMAType_Sub>);

REGISTER_CPU_KERNEL(Eigen::half);
REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedMulAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"),    \
      FusedMulAddOp<GPUDevice, type, FMAType_Add>);                         \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedMulAdd2").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      FusedMulAdd2Op<GPUDevice, type, FMAType_Add>);                        \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedMulSub").Device(DEVICE_GPU).TypeConstraint<type>("T"),    \
      FusedMulAddOp<GPUDevice, type, FMAType_Sub>);                         \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedMulSubRev").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      FusedMulAddOp<GPUDevice, type, FMAType_SubRev>);                      \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_FusedMulSub2").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      FusedMulAdd2Op<GPUDevice, type, FMAType_Sub>);

REGISTER_GPU_KERNEL(Eigen::half);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
#endif
};  // namespace tensorflow