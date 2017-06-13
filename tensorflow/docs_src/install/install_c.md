# Installing TensorFlow for C

TensorFlow provides a C API defined in
[`c_api.h`](https://github.com/tensorflow/tensorflow/tree/master/c/c_api.h),
which is suitable for
[building bindings for other languages](https://www.tensorflow.org/extend/language_bindings).
The API leans towards simplicity and uniformity rather than convenience.


## Supported Platforms

You may install TensorFlow for C on the following operating systems:

  * Linux
  * Mac OS X


## Installation

Take the following steps to install the TensorFlow for C library and
enable TensorFlow for C:

  1. Decide whether you will run TensorFlow for C on CPU(s) only or
     with the help of GPU(s). To help you decide, read the section
     entitled "Determine which TensorFlow to install" in one of the
     following guides:

       * @{$install_linux#determine_which_tensorflow_to_install$Installing TensorFlow on Linux}
       * @{$install_mac#determine_which_tensorflow_to_install$Installing TensorFlow on Mac OS}

  2. Download and extract the TensorFlow C library into `/usr/local/lib` by
     invoking the following shell commands:

         TF_TYPE="cpu" # Change to "gpu" for GPU support
         OS="linux" # Change to "darwin" for Mac OS
         TARGET_DIRECTORY="/usr/local"
         curl -L \
           "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.2.0-rc2.tar.gz" |
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
     directory (for example, `~/mydir/lib`) to two environment variables.
     For example:

     <pre> <b>export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib</b> # For both Linux and Mac OS X
     <b>export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib</b> # For Linux only
     <b>export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/mydir/lib</b> # For Mac OS X only</pre>



## Validate your installation

After installing TensorFlow for C, enter the following code into a file named
`hello_tf.c`:

```c
#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());
  return 0;
}
```

### Build and Run

Build `hello_tf.c` by invoking the following command:


<pre><b>gcc hello_tf.c</b></pre>


Running the resulting executable should output the following message:


<pre><b>a.out</b>
Hello from TensorFlow C library version <i>number</i></pre>


### Troubleshooting

If building the program fails, the most likely culprit is that `gcc` cannot
find the TensorFlow C library.  One way to fix this problem is to specify
the `-I` and `-L` options to `gcc`.  For example, if the `TARGET_LIBRARY`
was `/usr/local`, you would invoke `gcc` as follows:

<pre><b>gcc -I/usr/local/include -L/usr/local/lib hello_tf.c -ltensorflow</b></pre>

If executing `a.out` fails, ask yourself the following questions:

  * Did the program build without error?
  * Have you assigned the correct directory to the environment variables
    noted in Step 3 of [Installation](#installation)?
  * Did you export those environment variables?

If you are still seeing build or execution error messages, search (or post to)
[StackOverflow](www.stackoverflow.com/questions/tagged/tensorflow) for
possible solutions.

