//go:generate bazel build //tensorflow:libtensorflow.so
//go:generate mkdir -p /usr/local/tensorlow/
//go:generate cp ../../core/ops/ops.pbtxt /usr/local/tensorlow/
//go:generate sh -c "godocdown github.com/tensorflow/tensorflow/tensorflow/contrib/go/ > g3doc/index.md"

package tensorflow
