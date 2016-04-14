//go:generate bazel build //tensorflow:libtensorflow.so
//go:generate /bin/bash proto/generate.sh
//go:generate sh -c "godoc  -ex=true -templates=godoc_tmpl/ cmd/github.com/tensorflow/tensorflow/tensorflow/contrib/go Tensor > g3doc/tensor.md"
//go:generate sh -c "godoc  -ex=true -templates=godoc_tmpl/ cmd/github.com/tensorflow/tensorflow/tensorflow/contrib/go Session > g3doc/session.md"
//go:generate sh -c "godoc  -ex=true -templates=godoc_tmpl/ cmd/github.com/tensorflow/tensorflow/tensorflow/contrib/go Graph > g3doc/graph.md"

package tensorflow
