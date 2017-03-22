# Installing TensorFlow for Java

TensorFlow provides APIs for use in Java programs. These APIs are particularly
well-suited to loading models created in Python and executing them within a
Java application. This guide explains how to install
[TensorFlow for Java](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)
and use it in a Java application.

**WARNING:** The TensorFlow Java API is *not* covered by the TensorFlow
[API stability guarantees](https://www.tensorflow.org/programmers_guide/version_semantics).


## Supported Platforms

TensorFlow for Java is supported on the following operating systems:

  * Linux
  * Mac OS X
  * Windows
  * Android

The installation instructions for Android are in a separate
[Android TensorFlow Support page](https://www.tensorflow.org/code/tensorflow/contrib/android).
After installation, please see this
[complete example](https://www.tensorflow.org/code/tensorflow/examples/android)
of TensorFlow on Android.


## Install on Linux or Mac OS

Take the following steps to install TensorFlow for Java on Linux or Mac OS:

  1. Download
     [libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.1.0.jar),
     which is the TensorFlow Java Archive (JAR).

  2. Decide whether you will run TensorFlow for Java on CPU(s) only or with
     the help of GPU(s). To help you decide, read the section entitled
     "Determine which TensorFlow to install" in one of the following guides:

     * @{$install_linux#determine_which_tensorflow_to_install$Installing TensorFlow on Linux}
     * @{$install_mac#determine_which_tensorflow_to_install$Installing TensorFlow on Mac OS}

  3. Download and extract the appropriate Java Native Interface (JNI)
     file for your operating system and processor support by running the
     following shell commands:

     ```sh
     TF_TYPE="cpu" # Default processor is CPU. If you want GPU, set to "gpu"
     OS=$(uname -s | tr '[:upper:]' '[:lower:]')
     mkdir -p ./jni
     curl -L \
       "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-${TF_TYPE}-${OS}-x86_64-1.1.0.tar.gz" |
          tar -xz -C ./jni
     ```


## Install on Windows

Take the following steps to install TensorFlow for Java on Windows:

  1. Download
     [libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.1.0.jar),
     which is the TensorFlow Java Archive (JAR).
  2. Download the following Java Native Interface (JNI) file appropriate for
     [TensorFlow for Java on Windows](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.1.0.zip).
  3. Extract this .zip file.



## Validate the installation

After installing TensorFlow for Java, validate your installation by entering
the following code into a file named `HelloTF.java`:

```java
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class HelloTF {
  public static void main(String[] args) throws Exception {
    try (Graph g = new Graph()) {
      final String value = "Hello from " + TensorFlow.version();

      // Construct the computation graph with a single operation, a constant
      // named "MyConst" with a value "value".
      try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
        // The Java API doesn't yet include convenience functions for adding operations.
        g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
      }

      // Execute the "MyConst" operation in a Session.
      try (Session s = new Session(g);
           Tensor output = s.runner().fetch("MyConst").run().get(0)) {
        System.out.println(new String(output.bytesValue(), "UTF-8"));
      }
    }
  }
}
```


### Compiling

When compiling a TensorFlow program written in Java, the downloaded `.jar`
must be part of your `classpath`. For example, you can include the
downloaded `.jar` in your `classpath` by using the `-cp` compilation flag
as follows:

```sh
javac -cp libtensorflow-1.1.0.jar HelloTF.java
```


### Running

To execute a TensorFlow program written in Java, ensure that the following
two files are both in your `classpath`:

  * the downloaded `.jar` file
  * the extracted JNI library

For example, the following command line executes the `HelloTF` program:

```sh
java -cp libtensorflow-1.1.0.jar:. -Djava.library.path=./jni HelloTF
```

If the program prints `Hello from *version*`, you've successfully installed
TensorFlow for Java and are ready to use the API.  If the program outputs
something else, check
[Stack Overflow](http://stackoverflow.com/questions/tagged/tensorflow)
for possible solutions.


### Advanced Example

For a more sophisticated example, see
[LabelImage.java](https://www.tensorflow.org/code/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java),
which recognizes objects in an image.


## Building from source code

TensorFlow is open-source. You may build TensorFlow for Java from the
TensorFlow source code by following the instructions in a
[separate document](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/README.md).
