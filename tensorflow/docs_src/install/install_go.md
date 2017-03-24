# Installing TensorFlow for Go

TensorFlow provides APIs for use in Go programs. These APIs are particularly
well-suited to loading models created in Python and executing them within
a Go application. This guide explains how to install and set up the
[TensorFlow Go package](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go).

**WARNING:** The TensorFlow Go API is *not* covered by the TensorFlow
[API stability guarantees](https://www.tensorflow.org/programmers_guide/version_semantics).


## Supported Platforms

You may install TensorFlow for Go on the following operating systems:

  * Linux
  * Mac OS X


## Installation

TensorFlow for Go depends on the TensorFlow C library. Take the following
steps to install this library and enable TensorFlow for Go:

  1. Decide whether you will run TensorFlow for Go on CPU(s) only or with
     the help of GPU(s). To help you decide, read the section entitled
     "Determine which TensorFlow to install" in one of the following guides:

     * @{$install_linux#determine_which_tensorflow_to_install$Installing TensorFlow on Linux}
     * @{$install_mac#determine_which_tensorflow_to_install$Installing TensorFlow on Mac OS}

  2. Download and extract the TensorFlow C library into `/usr/local/lib` by
     invoking the following shell commands:

     ```sh
     TF_TYPE="cpu" # Change to "gpu" for GPU support
     TARGET_DIRECTORY='/usr/local'
     curl -L \
       "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.1.0.tar.gz" |
     sudo tar -C $TARGET_DIRECTORY -xz
     ```

     The `tar` command extracts the TensorFlow C library into the `lib`
     subdirectory of `TARGET_DIRECTORY`. For example, specifying `/usr/local`
     as `TARGET_DIRECTORY` causes `tar` to extract the TensorFlow C library
     into `/usr/local/lib`.

     If you'd prefer to extract the library into a different directory,
     adjust `TARGET_DIRECTORY` accordingly.

  3. In Step 2, if you specified a system directory (for example, `/usr/local`)
     as the `TARGET_DIRECTORY`, then run `ldconfig` to configure the linker.
     For example:

     ```sh
     sudo ldconfig
     ```

     If you assigned a `TARGET_DIRECTORY` other than a system
     directory (for example, `~/mydir`), then you must append the extraction
     directory (for example, `~/mydir/lib`) to two environment variables.
     For example:

     ```sh
     export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib # For both Linux and Mac OS X
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib # For Linux only
     export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/mydir/lib # For Mac OS X only
     ```

  4. Now that the TensorFlow C library is installed, invoke `go get` as follows
     to download the appropriate packages and their dependencies:

     ```sh
     go get github.com/tensorflow/tensorflow/tensorflow/go
     ```

  5. Invoke `go test` as follows to validate the TensorFlow for Go
     installation:

     ```sh
     go test github.com/tensorflow/tensorflow/tensorflow/go
     ```

If `go get` or `go test` generate error messages, search (or post to)
[StackOverflow](http://www.stackoverflow.com/questions/tagged/tensorflow)
for possible solutions.


## Hello World

After installing TensorFlow for Go, enter the following code into a
file named `hello_tf.go`:

```go
package main

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"fmt"
)

func main() {
	// Construct a graph with an operation that produces a string constant.
	s := op.NewScope()
	c := op.Const(s, "Hello from TensorFlow version " + tf.Version())
	graph, err := s.Finalize()
	if err != nil {
		panic(err)
	}

	// Execute the graph in a session.
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}
	output, err := sess.Run(nil, []tf.Output{c}, nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(output[0].Value())
}
```

For a more advanced example of TensorFlow in Go, look at the
[example in the API documentation](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go#ex-package),
which uses a pre-trained TensorFlow model to label contents of an image.


### Running

Run `hello_tf.go` by invoking the following command:

```sh
go run hello_tf.go
```

The program should print the following output:

```sh
Hello from TensorFlow version *number*
```

The program might also generate multiple warning messages of the
following form, which you can ignore:

```sh
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library
wasn't compiled to use *Type* instructions, but these are available on your
machine and could speed up CPU computations.
```


## Building from source code

TensorFlow is open-source. You may build TensorFlow for Go from the
TensorFlow source code by following the instructions in a
[separate document](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/README.md).
