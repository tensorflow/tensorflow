
#include "t2t_ops.h"
#include <cuda_fp16.h>

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

typedef float CT; // cast everthing up to float for internal calculations, for extra precision

template <typename T, typename U>
struct CustomL2NormFunctor<CPUDevice, T, U> {
  void operator()(const CPUDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, CT* temp, T* out,
    const U* _eps, const U* _bias, const U* _scale)
    {
      CT eps = *_eps;

      CT averager = 1./ k;
      for (uint64_t i = 0; i < N; i++) {
        CT sum=0, sumsq=0;
        for(uint64_t j = 0; j<k; j++)
        { 
          CT t = (CT)in[i*k+j];
          sum += t;
          sumsq += t*t;
        }
        sumsq -= sum*sum*averager;
        CT mean = sum*averager;
        CT sigma = sumsq*averager;
        sigma = 1./sqrt(sigma+eps);
//        temp[i*2+0] = scale*sigma;
//        temp[i*2+1] = bias-mean*scale*sigma;
        temp[i*2+0] = (CT)mean;
        temp[i*2+1] = (CT)sigma;
      }

      for (uint64_t i = 0; i < N; i++) {
        CT mean = (CT)temp[i*2+0];
        CT sigma = (CT)temp[i*2+1];
        for(uint64_t j = 0; j<k; j++)
        {
          CT bias = _bias[j];
          CT scale = _scale[j];
          CT c1 = scale*sigma;
          CT c2 = bias-mean*scale*sigma;
          out[i*k+j] = (T) ((CT)in[i*k+j] * c1 + c2);
        }
      }
  }
};

template <typename T, typename U>
struct CustomL2NormGradFunctor<CPUDevice, T, U> {
  void operator()(const CPUDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, const T* outgrad, CT* temp, T* out,
    const U* _eps, const U* _bias, const U* _scale)
    {
      float eps = *_eps;
      //float bias = *_bias;
      //float scale = *_scale;

      float a = 1./ k;
      for (uint64_t i = 0; i < N; i++) {
        CT sum=0, sumsq=0;
        for(uint64_t j = 0; j<k; j++)
        { 
          CT t = (CT)in[i*k+j];
          sum += t;
          sumsq += t*t;
        }

        sumsq -= sum*sum*a;
        CT mean = sum*a;
        CT sigma = sumsq*a;
        sigma = 1./sqrt(sigma+eps);
        //T c1 = scale*sigma;
        //T c2 = bias-mean*scale*sigma;
        //temp[i*2+0] = mean;
        //temp[i*2+1] = sigma;

        //T c3 = a*(c1*c1*c1)/(scale*scale);
        T* op = out+i*k;
        const T* ip = in+i*k;
        const T* ogp = outgrad+i*k;
/*
        T s=0;
        for(int y=0 ;y<k; y++)
          s += ogp[y];

        for(int y=0; y<k; y++)
          op[y] = ogp[y] * c1 - s*c1*a;

        for(int y=0; y<k; y++)
          for(uint64_t z = 0; z<k; z++)
            op[y] -= ogp[z] * c3 * (ip[y]-mean) * (ip[z]-mean);
*/
        CT s1 = 0;
        CT s2 = 0;
        for(int y=0; y<k; y++)
        {
          s1 += (CT)ogp[y] * (CT)_scale[y];
          s2 += (CT)ogp[y] * (CT)_scale[y] * ((CT)ip[y]-mean);
        }

        s1 *= a*sigma;
        s2 *= a*sigma*sigma*sigma;

        for(int y=0; y<k; y++)
            op[y] = (T) ((CT)ogp[y] * (CT)_scale[y] * sigma - s1 - s2*((CT)ip[y]-mean));

    }
  }
};

template <typename Device, typename T, typename U>
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

    CustomL2NormFunctor<Device,T,U>()(ctx->eigen_device<Device>(), N, k, 
      input_tensor.flat<T>().data(),
      tmp_out.flat<float>().data(),
      output_tensor->flat<T>().data(),
      eps.flat<U>().data(),
      bias.flat<U>().data(),
      scale.flat<U>().data()
      );
  }
};

template <typename Device, typename T, typename U>
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
    CustomL2NormGradFunctor<Device,T,U>()(ctx->eigen_device<Device>(), N, k,  
      in.flat<T>().data(),
      outgrad.flat<T>().data(),
      tmp_out.flat<float>().data(),
      output_tensor->flat<T>().data(),
      eps.flat<U>().data(),
      bias.flat<U>().data(),
      scale.flat<U>().data()
      );
  }
};

template <typename T>
struct CustomDropoutFunctor2<CPUDevice, T> {
  void operator()(const CPUDevice& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1,
    int s0, int s1,
    int r0, int r1
    )
  {
    float threshold = (float)*pthr;
    float scale = 1./(1-threshold);

    for(int i=0; i<d0; i++)
      for(int j=0; j<d1; j++)
          out[i*s0+j] = in[i*s0+j] * T(float(rng[i*r0+j*r1])>=threshold ? scale : 0.0);
  }
};

template <typename T>
struct CustomDropoutFunctor3<CPUDevice, T> {
  void operator()(const CPUDevice& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2,
    int s0, int s1, int s2,
    int r0, int r1, int r2
    )
  {
    float threshold = (float)*pthr;
    float scale = 1./(1-threshold);

    for(int i=0; i<d0; i++)
      for(int j=0; j<d1; j++)
        for(int k=0; k<d2; k++)
          out[i*s0+j*s1+k] = in[i*s0+j*s1+k] * T(float(rng[i*r0+j*r1+k*r2])>=threshold ? scale : 0.0);
  }
};


template <typename T>
struct CustomDropoutFunctor4<CPUDevice, T> {
  void operator()(const CPUDevice& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2, int d3,
    int s0, int s1, int s2, int s3,
    int r0, int r1, int r2, int r3
    )
  {
    float threshold = (float)*pthr;
    float scale = 1./(1-threshold);

    for(int i=0; i<d0; i++)
      for(int j=0; j<d1; j++)
        for(int k=0; k<d2; k++)
          for(int m=0; m<d3; m++)
            out[i*s0+j*s1+k*s2+m] = in[i*s0+j*s1+k*s2+m] * T(float(rng[i*r0+j*r1+k*r2+m*r3])>=threshold ? scale : 0.0);
  }
};

template <typename Device, typename T>
class CustomDropoutOp : public OpKernel {
 public:
  explicit CustomDropoutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    const Tensor& rng = ctx->input(1);
    const Tensor& threshold = ctx->input(2);

    auto Nd = input_tensor.dims();
    auto NdR = rng.dims();
    OP_REQUIRES(ctx, Nd == NdR, errors::InvalidArgument("input and rng must have the same number of dimensions"));
    OP_REQUIRES(ctx, threshold.dims()==0, errors::InvalidArgument("threshold must be a scalar"));
    OP_REQUIRES(ctx, Nd >= 2 && Nd <=4 , errors::InvalidArgument("must have 2..4 dim"));
    OP_REQUIRES(ctx, input_tensor.dim_size(Nd-1)<=1048576, errors::InvalidArgument("last dimension must be no more than 2^20"));

    for(int i=0; i<Nd; i++)
    {
      OP_REQUIRES(ctx, rng.dim_size(i)==1 || rng.dim_size(i)==input_tensor.dim_size(i), errors::InvalidArgument("dimension %d has size %d in input, size %d in rng", i, input_tensor.dim_size(i), rng.dim_size(i)));
    }

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
  
    int strides_in[4]={1,1,1,1};
    int strides_rng[4]={1,1,1,1};
    for(int i=Nd-1; i>0; i--)
    {
      strides_in[i-1] = strides_in[i] * input_tensor.dim_size(i);
      strides_rng[i-1] = strides_rng[i] * rng.dim_size(i);
      if(rng.dim_size(i)==1)
        strides_rng[i]=0;
    }
    if(rng.dim_size(0)==1)
      strides_rng[0]=0;
    if(Nd==2)
      CustomDropoutFunctor2<Device,T>()(ctx->eigen_device<Device>(), 
        input_tensor.flat<T>().data(),
        rng.flat<T>().data(),
        output_tensor->flat<T>().data(),
        threshold.flat<T>().data(),
        input_tensor.dim_size(0), input_tensor.dim_size(1),
        strides_in[0], strides_in[1], 
        strides_rng[0], strides_rng[1]
        );
    else if(Nd==3)
      CustomDropoutFunctor3<Device,T>()(ctx->eigen_device<Device>(), 
        input_tensor.flat<T>().data(),
        rng.flat<T>().data(),
        output_tensor->flat<T>().data(),
        threshold.flat<T>().data(),
        input_tensor.dim_size(0), input_tensor.dim_size(1), input_tensor.dim_size(2), 
        strides_in[0], strides_in[1], strides_in[2],
        strides_rng[0], strides_rng[1], strides_rng[2]
        );
    else
        CustomDropoutFunctor4<Device,T>()(ctx->eigen_device<Device>(), 
        input_tensor.flat<T>().data(),
        rng.flat<T>().data(),
        output_tensor->flat<T>().data(),
        threshold.flat<T>().data(),
        input_tensor.dim_size(0), input_tensor.dim_size(1), input_tensor.dim_size(2), input_tensor.dim_size(3),
        strides_in[0], strides_in[1], strides_in[2], strides_in[3],
        strides_rng[0], strides_rng[1], strides_rng[2], strides_rng[3]
        );

  }
};

#define DO_REGISTER(X, X1, Y, Z) \
REGISTER_KERNEL_BUILDER(Name("CustomL2Norm").Device(X).TypeConstraint<Y>("T").TypeConstraint<Z>("U"), CustomL2NormOp<X1,Y,Z>); \
REGISTER_KERNEL_BUILDER(Name("CustomL2NormGrad").Device(X).TypeConstraint<Y>("T").TypeConstraint<Z>("U"), CustomL2NormGradOp<X1,Y,Z>);

#define DO_REGISTER_ALL(X, X1) \
DO_REGISTER(X, X1, float, float); \
DO_REGISTER(X, X1, Eigen::half, float); \
DO_REGISTER(X, X1, Eigen::half, Eigen::half);

DO_REGISTER_ALL(DEVICE_CPU, CPUDevice);

REGISTER_KERNEL_BUILDER(Name("CustomDropout").Device(DEVICE_CPU).TypeConstraint<float>("T"), CustomDropoutOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("CustomDropout").Device(DEVICE_CPU).TypeConstraint<Eigen::half>("T"), CustomDropoutOp<CPUDevice, Eigen::half>);
//REGISTER_KERNEL_BUILDER(Name("CustomDropoutGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), CustomDropoutGradOp<CPUDevice, float>);
//REGISTER_KERNEL_BUILDER(Name("CustomDropoutGrad").Device(DEVICE_CPU).TypeConstraint<Eigen::half>("T"), CustomDropoutGradOp<CPUDevice, Eigen::half>);

#ifdef GOOGLE_CUDA

DO_REGISTER_ALL(DEVICE_GPU, Eigen::GpuDevice);
REGISTER_KERNEL_BUILDER(Name("CustomDropout").Device(DEVICE_GPU).TypeConstraint<float>("T"), CustomDropoutOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("CustomDropout").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), CustomDropoutOp<Eigen::GpuDevice, Eigen::half>);

#endif

}