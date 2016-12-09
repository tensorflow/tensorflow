#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;


  REGISTER_OP("LayerNormCustom")
      .Input("input: T")
      .Output("output: T")
      .Attr("T: {float, double}")
      .Attr("epsilon: float = 0.0000001")
      .SetShapeFn([](InferenceContext* c){
        c->set_output(0,c->input(0));        
        return Status::OK();
      })
      .Doc(R"doc(
        Custom efficient Layer Normalization GPU kernel.
        Normalizes along last dimension.Uses two-pass algorithm to calculate variance.

        epsilon: tiny number added before taking rsqrt(variance) to prevent division of zero.
  )doc");

  REGISTER_OP("LayerNormBackpropCustom")
      .Input("input: T")
      .Input("out_back: T")
      .Output("in_back: T")
      .Attr("T: {float, double}")
      .Attr("epsilon: float = 0.0000001")
      .SetShapeFn([](InferenceContext* c){
        c->set_output(0,c->input(0));        
        return Status::OK();
      })
      .Doc(R"doc(
        Custom efficient Layer Normalization GPU kernel for back propagation.
        Normalizes along last dimension.Uses two-pass algorithm to calculate variance. 

        epsilon: tiny number added before taking rsqrt(variance) to prevent division of zero.
  )doc");

  REGISTER_OP("LayerNormBiasAddCustom")
      .Input("input: T")
      .Input("beta: T")
      .Output("output: T")
      .Attr("T: {float, double}")
      .Attr("epsilon: float = 0.0000001")
      .SetShapeFn([](InferenceContext* c){
        c->set_output(0,c->input(0));        
        return Status::OK();
      })
      .Doc(R"doc(
        Custom efficient Layer Normalization fused with center(beta) op.
        Normalizes along last dimension.Uses two-pass algorithm to calculate variance.

        With only CUDA kernel for GPU.

        epsilon: tiny number added before taking rsqrt(variance) to prevent division of zero.
  )doc");

  REGISTER_OP("LayerNormBiasAddBackpropCustom")
      .Input("input: T")
      .Input("out_back: T")
      .Input("beta: T")
      .Output("in_back: T")
      .Output("beta_back: T")
      .Attr("T: {float, double}")
      .Attr("epsilon: float = 0.0000001")
      .SetShapeFn([](InferenceContext* c){
        c->set_output(0,c->input(0));        
        c->set_output(1,c->input(2));        
        return Status::OK();
      })
      .Doc(R"doc(
        Custom efficient Layer Normalization backprop fused with center(beta) op.
        Normalizes along last dimension.Uses two-pass algorithm to calculate variance. 

        With only CUDA kernel for GPU.

        TODO: We don't really need beta as input to calculate gradient, I am temporarily requesting
        beta to set output shape for beta_back before I find better method.

        epsilon: tiny number added before taking rsqrt(variance) to prevent division of zero.
  )doc");      

  REGISTER_OP("LayerNormFusedCustom")
      .Input("input: T")
      .Input("gamma: T")
      .Input("beta: T")
      .Output("output: T")
      .Attr("T: {float, double}")
      .Attr("epsilon: float = 0.0000001")
      .SetShapeFn([](InferenceContext* c){
        c->set_output(0,c->input(0));        
        return Status::OK();
      })
      .Doc(R"doc(
        Custom efficient Layer Normalization fused with scale(gamma) and center(beta) ops.
        Normalizes along last dimension.Uses two-pass algorithm to calculate variance.

        With only CUDA kernel for GPU.

        epsilon: tiny number added before taking rsqrt(variance) to prevent division of zero.
  )doc");

  REGISTER_OP("LayerNormFusedBackpropCustom")
      .Input("input: T")
      .Input("out_back: T")
      .Input("gamma: T")
      .Output("in_back: T")
      .Output("gamma_back: T")
      .Output("beta_back: T")
      .Attr("T: {float, double}")
      .Attr("epsilon: float = 0.0000001")
      .SetShapeFn([](InferenceContext* c){
        c->set_output(0,c->input(0));        
        c->set_output(1,c->input(2));        
        c->set_output(2,c->input(2));        
        return Status::OK();
      })
      .Doc(R"doc(
        Custom efficient Layer Normalization backprop fused with scale(gamma) and center(beta) ops.
        Normalizes along last dimension.Uses two-pass algorithm to calculate variance. 

        With only CUDA kernel for GPU.

        epsilon: tiny number added before taking rsqrt(variance) to prevent division of zero.
  )doc");

}// namespace tensorflow