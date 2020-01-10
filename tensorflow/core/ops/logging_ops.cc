#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("Assert")
    .Input("condition: bool")
    .Input("data: T")
    .Attr("T: list(type)")
    .Attr("summarize: int = 3")
    .Doc(R"doc(
Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

condition: The condition to evaluate.
data: The tensors to print out when condition is false.
summarize: Print this many entries of each tensor.
)doc");

REGISTER_OP("Print")
    .Input("input: T")
    .Input("data: U")
    .Output("output: T")
    .Attr("T: type")
    .Attr("U: list(type)")
    .Attr("message: string = ''")
    .Attr("first_n: int = -1")
    .Attr("summarize: int = 3")
    .Doc(R"doc(
Prints a list of tensors.

Passes `input` through to `output` and prints `data` when evaluating.

input: The tensor passed to `output`
data: A list of tensors to print out when op is evaluated.
output:= The unmodified `input` tensor
message: A string, prefix of the error message.
first_n: Only log `first_n` number of times. -1 disables logging.
summarize: Only print this many entries of each tensor.
)doc");

}  // end namespace tensorflow
