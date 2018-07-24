# Installing TensorFlow for Go

TensorFlow provides APIs for use in Go programs. These APIs are particularly
well-suited to loading models created in Python and executing them within
a Go application. This guide explains how to install and set up the
[TensorFlow Go package](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go).

Warning: The TensorFlow Go API is *not* covered by the TensorFlow
[API stability guarantees](../guide/version_semantics.md).


## Supported Platforms

This guide explains how to install TensorFlow for Go.  Although these
instructions might also work on other variants, we have only tested
(and we only support) these instructions on machines meeting the
following requirements:

  * Linux, 64-bit, x86
  * macOS X, 10.12.6 (Sierra) or higher


## Installation

TensorFlow for Go depends on the TensorFlow C library. Take the following
steps to install this library and enable TensorFlow for Go:

  1. Decide whether you will run TensorFlow for Go on CPU(s) only or with
     the help of GPU(s). To help you decide, read the section entitled
     "Determine which TensorFlow to install" in one of the following guides:

     * @{$install_linux#determine_which_tensorflow_to_install$Installing TensorFlow on Linux}
     * @{$install_mac#determine_which_tensorflow_to_install$Installing TensorFlow on macOS}

  2. Download and extract the TensorFlow C library into `/usr/local/lib` by
     invoking the following shell commands:

         TF_TYPE="cpu" # Change to "gpu" for GPU support
         TARGET_DIRECTORY='/usr/local'
         curl -L \
           "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.10.0-rc1.tar.gz" |
         sudo tar -C $TARGET_DIRECTORY -xz

     The `tar` command extracts the TensorFlow C library into the `lib`
     subdirectory of `TARGET_DIRECTORY`. For example, specifying `/usr/local`
     as `TARGET_DIRECTORY` causes `tar` to extract the TensorFlow C library
     into `/usr/local/lib`.

     If you'd prefer to extract the library into a different directory,
     adjust `TARGET_DIRECTORY` accordingly.

  3. In Step 2, if you specified a system directory (for example, `/usr/local`)
     as the `TARGET_DIRECTORY`, then run `ldconfig` to configure the linker.
     For example:

     <pre><b>sudo ldconfig</b></pre>

     If you assigned a `TARGET_DIRECTORY` other than a system
     directory (for example, `~/mydir`), then you must append the extraction
     directory (for example, `~/mydir/lib`) to two environment variables
     as follows:

     <pre> <b>export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib</b> # For both Linux and macOS X
     <b>export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib</b> # For Linux only
     <b>export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/mydir/lib</b> # For macOS X only</pre>

  4. Now that the TensorFlow C library is installed, invoke `go get` as follows
     to download the appropriate packages and their dependencies:

     <pre><b>go get github.com/tensorflow/tensorflow/tensorflow/go</b></pre>

  5. Invoke `go test` as follows to validate the TensorFlow for Go
     installation:

     <pre><b>go test github.com/tensorflow/tensorflow/tensorflow/go</b></pre>

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

<pre><b>go run hello_tf.go</b>
Hello from TensorFlow version <i>number</i></pre>

The program might also generate multiple warning messages of the
following form, which you can ignore:

<pre>W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library
wasn't compiled to use *Type* instructions, but these are available on your
machine and could speed up CPU computations.</pre>


## Building from source code

TensorFlow is open-source. You may build TensorFlow for Go from the
TensorFlow source code by following the instructions in a
[separate document](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/README.md).
