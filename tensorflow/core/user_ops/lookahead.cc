//#include "lookahead.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
REGISTER_OP("Lookahead")
	.Attr("T: realnumbertype")
	.Input("input: T")
	.Input("filter: T")
	.Output("output: T");

REGISTER_OP("Lookaheadgrad")
	.Attr("T: realnumbertype")
	.Input("input: T")
	.Input("filter: T")
	.Input("backprop_output: T")
	.Output("output1: T")
	.Output("output2: T");
