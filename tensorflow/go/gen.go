//go:generate bazel build //tensorflow:libtensorflow.so
//go:generate mkdir -p /usr/local/tensorlow/
//go:generate cp ../../tensorflow/core/ops/ops.pbtxt /usr/local/tensorlow/

package tensorflow
