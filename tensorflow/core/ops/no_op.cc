#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("NoOp")
    .Doc(R"doc(
Does nothing. Only useful as a placeholder for control edges.
)doc");

}  // namespace tensorflow
