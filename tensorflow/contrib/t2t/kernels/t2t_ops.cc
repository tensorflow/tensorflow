
#include "t2t_ops.h"
#include <cuda_fp16.h>

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename T>
struct CustomL2NormFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, float* temp, T* out,
    const float* _eps, const float* _bias, const float* _scale)
    {
      float eps = *_eps;
      float bias = *_bias;
      float scale = *_scale;

      T averager = 1./ k;
      for (uint64_t i = 0; i < N; i++) {
        T sum=0, sumsq=0;
        for(uint64_t j = 0; j<k; j++)
        { 
          T t = in[i*k+j];
          sum += t;
          sumsq += t*t;
        }
        sumsq -= sum*sum*averager;
        T mean = sum*averager;
        T sigma = sumsq*averager;
        sigma = 1./sqrt(sigma+eps);
        temp[i*2+0] = scale*sigma;
        temp[i*2+1] = bias-mean*scale*sigma;
      }

      for (uint64_t i = 0; i < N; i++) {
        T c1 = temp[i*2+0];
        T c2 = temp[i*2+1];
        for(uint64_t j = 0; j<k; j++)
          out[i*k+j] = in[i*k+j] * c1 + c2;
      }
  }
};

template <typename T>
struct CustomL2NormGradFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, const T* outgrad, float* temp, T* out,
    const float* _eps, const float* _bias, const float* _scale)
    {
      float eps = *_eps;
      float bias = *_bias;
      float scale = *_scale;

      T a = 1./ k;
      for (uint64_t i = 0; i < N; i++) {
        T sum=0, sumsq=0;
        for(uint64_t j = 0; j<k; j++)
        { 
          T t = in[i*k+j];
          sum += t;
          sumsq += t*t;
        }

        sumsq -= sum*sum*a;
        T mean = sum*a;
        T sigma = sumsq*a;
        sigma = 1./sqrt(sigma+eps);
        T c1 = scale*sigma;
        T c2 = bias-mean*scale*sigma;
        temp[i*2+0] = c1;
        temp[i*2+1] = c2;

        T c3 = a*(c1*c1*c1)/(scale*scale);
        T* op = out+i*k;
        const T* ip = in+i*k;
        const T* ogp = outgrad+i*k;

        T s=0;
        for(int y=0 ;y<k; y++)
          s += ogp[y];

        for(int y=0; y<k; y++)
          op[y] = ogp[y] * c1 - s*c1*a;

        for(int y=0; y<k; y++)
          for(uint64_t z = 0; z<k; z++)
            op[y] -= ogp[z] * c3 * (ip[y]-mean) * (ip[z]-mean);
    }
  }
};

template <typename Device, typename T>
class CustomL2NormOp : public OpKernel {
 public:
  explicit CustomL2NormOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);

    const Tensor& eps = ctx->input(1);
    const Tensor& scale = ctx->input(2);
    const Tensor& bias = ctx->input(3);


    auto Nd = input_tensor.dims();

    const AllocatorAttributes alloc_attr = ctx->output_alloc_attr(0);

    TensorShape mod_shape = input_tensor.shape();
    mod_shape.set_dim(Nd-1, 2);
    Tensor tmp_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, mod_shape, &tmp_out, alloc_attr));

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
    
    auto tmp_flat = tmp_out.flat<float>();
    const uint64_t N = tmp_flat.size() >> 1;
    const uint64_t k = input_tensor.shape().dim_size(Nd-1);

    //OP_REQUIRES(ctx, eps.dims()==0, errors::InvalidArgument("Bad eps argument"));
    //OP_REQUIRES(ctx, bias.dims()==1 && bias.shape().dim_size(0)==k, errors::InvalidArgument("Bad bias argument"));
    //OP_REQUIRES(ctx, scale.dims()==1  && scale.shape().dim_size(0)==k, errors::InvalidArgument("Bad scale argument"));

    CustomL2NormFunctor<Device,T>()(ctx->eigen_device<Device>(), N, k, 
      input_tensor.flat<T>().data(),
      tmp_out.flat<float>().data(),
      output_tensor->flat<T>().data(),
      eps.flat<float>().data(),
      bias.flat<float>().data(),
      scale.flat<float>().data()
      );
  }
};

template <typename Device, typename T>
class CustomL2NormGradOp : public OpKernel {
 public:
  explicit CustomL2NormGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    const Tensor& eps = ctx->input(1);
    const Tensor& scale = ctx->input(2);
    const Tensor& bias = ctx->input(3);
    const Tensor& outgrad = ctx->input(4);
    auto Nd = in.dims();
    const uint64_t k = in.shape().dim_size(Nd-1);

    const AllocatorAttributes alloc_attr = ctx->output_alloc_attr(0);

    auto mod_shape = in.shape();
    mod_shape.set_dim(Nd-1, 2);
    Tensor tmp_out;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, mod_shape, &tmp_out, alloc_attr));

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &output_tensor));
    
    auto tmp_flat = tmp_out.flat<float>();
    const uint64_t N = tmp_flat.size() >> 1;
    CustomL2NormGradFunctor<Device,T>()(ctx->eigen_device<Device>(), N, k,  
      in.flat<T>().data(),
      outgrad.flat<T>().data(),
      tmp_out.flat<float>().data(),
      output_tensor->flat<T>().data(),
      eps.flat<float>().data(),
      bias.flat<float>().data(),
      scale.flat<float>().data()
      );
  }
};

//REGISTER_KERNEL_BUILDER(Name("CustomL2Norm").Device(DEVICE_CPU), CustomL2NormOp<CPUDevice,float>);
//REGISTER_KERNEL_BUILDER(Name("CustomL2NormGrad").Device(DEVICE_CPU), CustomL2NormGradOp<CPUDevice,float>);

#ifdef GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("CustomL2Norm").Device(DEVICE_GPU).TypeConstraint<float>("T"), CustomL2NormOp<Eigen::GpuDevice,float>);
REGISTER_KERNEL_BUILDER(Name("CustomL2NormGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), CustomL2NormGradOp<Eigen::GpuDevice,float>);

REGISTER_KERNEL_BUILDER(Name("CustomL2Norm").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), CustomL2NormOp<Eigen::GpuDevice,Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("CustomL2NormGrad").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), CustomL2NormGradOp<Eigen::GpuDevice,Eigen::half>);

#endif

}