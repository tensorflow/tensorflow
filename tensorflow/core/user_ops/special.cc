#include "tensorflow/core/framework/op.h"
#include "boost/math/special_functions/digamma.hpp"

#include "boost/exception/error_info.hpp"

#include <math.h> 

#define UNARY_REAL()                      \
  Input("x: T").Output("y: T").Attr( \
      "T: {float, double}")

REGISTER_OP("LogGamma")
    .UNARY_REAL()
    .Doc(R"doc(Computes natural logarithm of the Gamma function of value element-wise.)doc");

REGISTER_OP("Gamma")
    .UNARY_REAL()
    .Doc(R"doc(Computes Gamma function of value element-wise.)doc");

REGISTER_OP("Digamma")
    .UNARY_REAL()
    .Doc(R"doc(Computes Digamma function of value element-wise.)doc");


#undef UNARY_REAL

#include "tensorflow/core/framework/op_kernel.h"

namespace boost
{
void throw_exception(std::exception const& e) {} //TODO fix this.

}

using namespace tensorflow;

template <typename T>
class LogGamma : public OpKernel {
    public:
    
    explicit LogGamma(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {

    const Tensor& input_tensor = context->input(0); //Grab the input tensor.
    auto input = input_tensor.flat<T>();
    
    // Create an output tensor
    Tensor* output_tensor = NULL;
    
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<T>();
    
    const int N = input.size();

    for (int i = 0; i < N; i++) {
      output(i) = lgamma( input(i) );
    }
        
    
    }
};

template <typename T>
class Gamma : public OpKernel {
    public:
    
    explicit Gamma(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {

    const Tensor& input_tensor = context->input(0); //Grab the input tensor.
    auto input = input_tensor.flat<T>();
    
    // Create an output tensor
    Tensor* output_tensor = NULL;
    
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<T>();
    
    const int N = input.size();

    for (int i = 0; i < N; i++) {
      output(i) = tgamma( input(i) );
    }
        
    
    }
};

template <typename T>
class Digamma : public OpKernel {
    public:
    
    explicit Digamma(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {

    const Tensor& input_tensor = context->input(0); //Grab the input tensor.
    auto input = input_tensor.flat<T>();
    
    // Create an output tensor
    Tensor* output_tensor = NULL;
    
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<T>();
    
    const int N = input.size();

    for (int i = 0; i < N; i++) {
      output(i) = boost::math::digamma( input(i) ); 
    }
        
    
    }
};

REGISTER_KERNEL_BUILDER(
    Name("LogGamma")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    LogGamma<float>);
    
REGISTER_KERNEL_BUILDER(
    Name("LogGamma")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    LogGamma<double>);

REGISTER_KERNEL_BUILDER(
    Name("Gamma")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    Gamma<float>);
    
REGISTER_KERNEL_BUILDER(
    Name("Gamma")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    Gamma<double>);

REGISTER_KERNEL_BUILDER(
    Name("Digamma")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    Digamma<float>);
    
REGISTER_KERNEL_BUILDER(
    Name("Digamma")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    Digamma<double>);
