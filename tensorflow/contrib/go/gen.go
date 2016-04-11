//go:generate bazel build //tensorflow:libtensorflow.so
//go:generate mkdir -p /usr/local/tensorlow/
//go:generate cp ../../core/ops/ops.pbtxt /usr/local/tensorlow/
//go:generate sh -c "godoc  -ex=true -templates=godoc_tmpl/ github.com/tensorflow/tensorflow/tensorflow/contrib/go > g3doc/index.md"

package tensorflow
