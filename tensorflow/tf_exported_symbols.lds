*tensorflow*
*toco*
*perftools*gputools*
*tf_*
*TF_*
*Eager*
*TFE_*
*nsync_*
*stream_executor*
*xla*
# Though it may seem unsafe to export grpc, any .so files that rely on gRPC
# MUST do it through tensorflow's version, because gRPC uses link-time
# object registration, and if they hard link their own gRPC dependencies
# then at load time you get a segfault due to registration collissions.
# As a result, any dynamic libraries relying on gRPC must build with gRPC
# deps in header-only mode, and rely on tensorflow's symbol exports at
# link-time.
*grpc*

