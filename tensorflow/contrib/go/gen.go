// Operation definitions file.
//go:generate sh proto/generate.sh

// Documentation.
//go:generate sh -c "godoc  -ex=true -templates=godoc_tmpl/ cmd/github.com/tensorflow/tensorflow/tensorflow/contrib/go Tensor > g3doc/tensor.md"
//go:generate sh -c "godoc  -ex=true -templates=godoc_tmpl/ cmd/github.com/tensorflow/tensorflow/tensorflow/contrib/go Session > g3doc/session.md"
//go:generate sh -c "godoc  -ex=true -templates=godoc_tmpl/ cmd/github.com/tensorflow/tensorflow/tensorflow/contrib/go Graph > g3doc/graph.md"

// Protobuf definitions.
//go:generate sh -c "cd ../../../ && protoc --go_out=tensorflow/contrib/go/proto/ tensorflow/core/framework/allocation_description.proto tensorflow/core/framework/attr_value.proto tensorflow/core/framework/device_attributes.proto tensorflow/core/framework/function.proto tensorflow/core/framework/graph.proto tensorflow/core/framework/kernel_def.proto tensorflow/core/framework/log_memory.proto tensorflow/core/framework/op_def.proto tensorflow/core/framework/step_stats.proto tensorflow/core/framework/summary.proto tensorflow/core/framework/tensor.proto tensorflow/core/framework/tensor_description.proto tensorflow/core/framework/tensor_shape.proto tensorflow/core/framework/tensor_slice.proto tensorflow/core/framework/types.proto tensorflow/core/framework/versions.proto tensorflow/core/protobuf/config.proto tensorflow/core/protobuf/saver.proto tensorflow/core/util/saved_tensor_slice.proto"
//go:generate sh -c "mv proto/tensorflow/core/framework/log_memory.pb.go proto/tensorflow/core/framework/op_def.pb.go proto/tensorflow/core/framework/attr_value.pb.go proto/tensorflow/core/framework/function.pb.go proto/tensorflow/core/framework/allocation_description.pb.go proto/tensorflow/core/framework/device_attributes.pb.go proto/tensorflow/core/framework/kernel_def.pb.go proto/tensorflow/core/framework/tensor_shape.pb.go proto/tensorflow/core/framework/tensor_slice.pb.go proto/tensorflow/core/framework/versions.pb.go proto/tensorflow/core/framework/step_stats.pb.go proto/tensorflow/core/framework/tensor.pb.go proto/tensorflow/core/framework/tensor_description.pb.go proto/tensorflow/core/framework/types.pb.go proto/tensorflow/core/framework/graph.pb.go proto/tensorflow/core/framework/summary.pb.go proto/tensorflow/core/protobuf/saver.pb.go proto/tensorflow/core/protobuf/config.pb.go proto/tensorflow/core/util/saved_tensor_slice.pb.go proto/ && rm -rf proto/tensorflow/"

package tensorflow
