
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
    // Set all but the first element of the output tensor to 0.
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
      //  Sum (x-m)^2 = Sum x^2 - 2 Sum x m + Sum m^2 = Sum x^2 - m^2 N = Sum x^2 - (Sum x)^2 / N
    }
    //return (x - mean) * scale * tf.rsqrt(variance + epsilon) + bias
    // = x*(scale*sigma) + (bias-mean*scale*sigma);

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

    // Set all but the first element of the output tensor to 0.
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
//      for(int y=0; y<k; y++)
//        for(uint64_t z = 0; z<k; z++)
//          txz[i*k*k + y*k + z] = - c1 * a - c3*(in[i*k+y]-mean)*(in[i*k+z]-mean);
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
    // Grab the input tensor
    const Tensor& input_tensor = ctx->input(0);

    const Tensor& eps = ctx->input(1);
    const Tensor& scale = ctx->input(2);
    const Tensor& bias = ctx->input(3);
    //printf("Input: %d dims\n", input_tensor.dims());
    //printf("Eps: %d dims\n", eps.dims());
    //printf("Bias: %d dims\n", bias.dims());
    //printf("Scale: %d dims\n", scale.dims());
    //printf("%d\n", bias.shape().dim_size(0));

    //OP_REQUIRES(ctx, bias.dims()==1 && bias.shape().dim_size(0)==1, errors::InvalidArgument("Bad bias argument"));
    //OP_REQUIRES(ctx, eps.dims()==1 && eps.shape().dim_size(0)==1, errors::InvalidArgument("Bad eps argument"));
    //OP_REQUIRES(ctx, scale.dims()==1  && scale.shape().dim_size(0)==1, errors::InvalidArgument("Bad scale argument"));

    //auto input = input_tensor.flat<T>();
    auto Nd = input_tensor.dims();

    const AllocatorAttributes alloc_attr = ctx->output_alloc_attr(0);

    TensorShape mod_shape = input_tensor.shape();
    mod_shape.set_dim(Nd-1, 2);
    // A temporary tensor whose size matches the size of the reduced
    // output.
    Tensor tmp_out;


    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, mod_shape, &tmp_out, alloc_attr));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
    
    auto tmp_flat = tmp_out.flat<float>();
    //auto input_flat = input_tensor.flat<T>();
    //auto output_flat = output_tensor->flat<T>();

    const uint64_t N = tmp_flat.size() >> 1;
    const uint64_t k = input_tensor.shape().dim_size(Nd-1);
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

/***

out[x,y] = in[x,y] * c1(in[x,0],in[x,1],...) + c2(in[x,0],in[x,1],...)

d out[x,y] / d in[x,y] = c1(in[x]) + in[x,y] {d c1(in[x]) / d in[x,y]} + {d c2(in[x]) / d in[x,y]} 

d out[x,y] / d in[x,z] = in[x,y] {d c1(in[x]) / d in[x,z]} + {d c2(in[x]) / d in[x,z]} 
  for y!=z

-> d in[x,y] / d out[x,y] = 1 / [c1(in[x]) + in[x,y] {d c1(in[x]) / d in[x,y]} + {d c2(in[x]) / d in[x,y]}]
   d in[x,z] / d out[x,y] = 1 / [in[x,y] {d c1(in[x]) / d in[x,z]} + {d c2(in[x]) / d in[x,z]} ]


in_grad[x,y] = Sum out_grad[x,z] / [c1(in[x])*(x==z) + in[x,y] {d c1(in[x]) / d in[x,z]} + {d c2(in[x]) / d in[x,z]}]

      a = 1./k
      sumsq = sum(in[x,y]*in[x.y]) - (sum(in[x,y]))^2*a
      mean = sum(in[x,y])*a
      c1 = scale/sqrt(sumsq*a+eps);
      c2 = bias-mean*c1;

      d mean / d in[x,y] = a
      d sumsq / d in[x,y] = 2 in[x,y] - 2 a sum(in[x,y]) = 2 (in[x,y] - mean)

      d c1(in[x]) / d in[x,y] =  [-0.5 a  scale/(sumsq*a+eps)^1.5 ] d (sumsq) / d in[x,y]
          = - a  (in[x,y] - mean) scale/(sumsq*a+eps)^1.5
      d c2(in[x]) / d in[x,y] = - mean d c1(in[x]) / d in[x,y] - a c1
        
      - t[x,y,z] = c1*((y==z)-a) + a (c1^3/scale^2) (in[x,z]-mean)^2 
****/

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
    //const Tensor& outorig = ctx->input(2);

    //auto input = input_tensor.flat<T>();
    auto Nd = in.dims();
    const uint64_t k = in.shape().dim_size(Nd-1);

    const AllocatorAttributes alloc_attr = ctx->output_alloc_attr(0);

    auto mod_shape = in.shape();
    mod_shape.set_dim(Nd-1, 2);
    Tensor tmp_out;
    //OP_REQUIRES_OK(ctx, ctx->allocate_temp(ctx->input_dtype(1), mod_shape, &tmp_out, alloc_attr));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, mod_shape, &tmp_out, alloc_attr));

    //auto mod_shape2 = in.shape();
    //mod_shape2.AddDim(k);
    //Tensor tmp2;
    //OP_REQUIRES_OK(ctx, ctx->allocate_temp(ctx->expected_output_dtype(0), mod_shape2, &tmp2, alloc_attr));


    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &output_tensor));
    
    auto tmp_flat = tmp_out.flat<float>();
    //auto input_flat = input_tensor.flat<T>();
    //auto output_flat = output_tensor->flat<T>();

    const uint64_t N = tmp_flat.size() >> 1;
    CustomL2NormGradFunctor<Device,T>()(ctx->eigen_device<Device>(), N, k,  
      in.flat<T>().data(),
      outgrad.flat<T>().data(),
      //outorig.flat<T>().data(),
      tmp_out.flat<float>().data(),
      //tmp2.flat<T>().data(),
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
//extern template CustomL2NormFunctor<Eigen::GpuDevice, float>;
REGISTER_KERNEL_BUILDER(Name("CustomL2Norm").Device(DEVICE_GPU).TypeConstraint<float>("T"), CustomL2NormOp<Eigen::GpuDevice,float>);
REGISTER_KERNEL_BUILDER(Name("CustomL2NormGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), CustomL2NormGradOp<Eigen::GpuDevice,float>);

REGISTER_KERNEL_BUILDER(Name("CustomL2Norm").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), CustomL2NormOp<Eigen::GpuDevice,Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("CustomL2NormGrad").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), CustomL2NormGradOp<Eigen::GpuDevice,Eigen::half>);

#endif

}