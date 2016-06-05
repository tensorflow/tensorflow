bazel build -c opt --config=cuda //tensorflow/core/user_ops:roi_pooling_op.so  
bazel build -c opt --config=cuda //tensorflow/core/user_ops:roi_pooling_op_grad.so  