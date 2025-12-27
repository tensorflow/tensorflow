# TensorFlow in Go

Construct and execute TensorFlow graphs in Go.

[![GoDoc](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go?status.svg)](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)

> *WARNING*: The API defined in this package is not stable and can change
> without notice. The same goes for the package path:
> (`github.com/tensorflow/tensorflow/tensorflow/go`).

# WARNING:

The TensorFlow team is not currently maintaining the Documentation for installing the Go bindings for TensorFlow.

The instructions has been maintained by the third party contributor: @wamuir

Please follow this [source](https://github.com/tensorflow/build/tree/master/golang_install_guide) by @wamuir for the installation of Golang with Tensorflow.


## GPU Configuration

The TensorFlow Go bindings do not currently expose `GPUOptions` as a
first-class API.

However, GPU-related options can be configured by passing a serialized
`ConfigProto` to `SessionOptions.SetConfig`.

### Example: Enable GPU memory growth

```go
import (
    tf "github.com/tensorflow/tensorflow/tensorflow/go"
    "google.golang.org/protobuf/proto"
    "tensorflow/core/protobuf/config_pb2"
)

config := &config_pb2.ConfigProto{
    GpuOptions: &config_pb2.GPUOptions{
        AllowGrowth: true,
    },
}

configBytes, err := proto.Marshal(config)
if err != nil {
    panic(err)
}

opts := tf.NewSessionOptions()
opts.SetConfig(configBytes)
