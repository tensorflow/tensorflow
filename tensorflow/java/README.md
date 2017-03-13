# TensorFlow for Java

Java bindings for TensorFlow. ([Javadoc](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary))

> *WARNING*: The TensorFlow Java API is not currently covered by the TensorFlow
> [API stability guarantees](https://www.tensorflow.org/programmers_guide/version_semantics).
>
> For using TensorFlow on Android refer to
> [contrib/android](https://www.tensorflow.org/code/tensorflow/contrib/android),
> [makefile](https://www.tensorflow.org/code/tensorflow/contrib/makefile#android)
> and/or the [Android
> demo](https://www.tensorflow.org/code/tensorflow/examples/android).

## Quickstart

1.  Download the Java archive (JAR):
    [libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.0.0-PREVIEW1.jar)
    (optionally, the Java sources:
    [libtensorflow-src.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-src-1.0.0-PREVIEW1.jar)).

2.  Download the native library. GPU-enabled versions required CUDA 8 and cuDNN
    5.1. For other versions, the native library will need to be built from
    source (see below).

    -   Linux:
        [CPU-only](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.0.0-PREVIEW1.tar.gz),
        [GPU-enabled](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.0.0-PREVIEW1.tar.gz)
    -   OS X:
        [CPU-only](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.0.0-PREVIEW1.tar.gz),
        [GPU-enabled](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-darwin-x86_64-1.0.0-PREVIEW1.tar.gz)

    The following shell snippet downloads and extracts the native library:

    ```sh
    TF_TYPE="cpu" # Set to "gpu" to enable GPU support
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    mkdir -p ./jni
    curl -L \
      "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-${TF_TYPE}-${OS}-x86_64-1.0.0-PREVIEW1.tar.gz" |
    tar -xz -C ./jni
    ```

3.  Include the downloaded `.jar` in the classpath during compilation. For
    example, if your program looks like the following:

    ```java
    import org.tensorflow.TensorFlow;

    public class MyClass {
      public static void main(String[] args) {
        System.out.println("I'm using TensorFlow version: " +  TensorFlow.version());
      }
    }
    ```

    then it should be compiled with:

    ```sh
    javac -cp libtensorflow-1.0.0-PREVIEW1.jar MyClass.java
    ```

    For a more sophisticated example, see
    [LabelImage.java](https://www.tensorflow.org/code/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java),
    which can be compiled with:

    ```sh
    javac \
      -cp libtensorflow-1.0.0-PREVIEW1.jar \
      ./src/main/java/org/tensorflow/examples/LabelImage.java
    ```

4.  Include the downloaded `.jar` in the classpath and the native library in the
    library path during execution. For example:

    ```sh
    java -cp libtensorflow-1.0.0-PREVIEW1.jar:. -Djava.library.path=./jni MyClass
    ```

    or for the `LabelImage` example:

    ```sh
    java \
      -Djava.library.path=./jni \
      -cp libtensorflow-1.0.0-PREVIEW1.jar:./src/main/java \
      org.tensorflow.examples.LabelImage
    ```

That's all. These artifacts are not yet available on Maven central, see
[#6926](https://github.com/tensorflow/tensorflow/issues/6926).

## Building from source

If the quickstart instructions above do not work out, the TensorFlow native
libraries will need to be built from source.

1.  Install [bazel](https://www.bazel.build/versions/master/docs/install.html)

2.  Setup the environment to buile TensorFlow from source code
    ([Linux](https://www.tensorflow.org/versions/master/get_started/os_setup.html#prepare-environment-for-linux)
    or [Mac OS
    X](https://www.tensorflow.org/versions/master/get_started/os_setup.html#prepare-environment-for-mac-os-x)).
    If you'd like to skip reading those details and do not care about GPU
    support, try the following:

    ```sh
    # On Linux
    sudo apt-get install python swig python-numpy

    # On Mac OS X with homebrew
    brew install swig
    ```

3.  [Configure](https://www.tensorflow.org/install/install_sources#configure_the_installation)
    (e.g., enable GPU support) and build:

    ```sh
    ./configure
    bazel build --config opt \
      //tensorflow/java:tensorflow \
      //tensorflow/java:libtensorflow_jni
    ```

The JAR (`libtensorflow.jar`) and native library (`libtensorflow_jni.so` on Linux or `libtensorflow_jni.dylib` on OS X) will 
be in `bazel-bin/tensorflow/java`. Using these artifacts follow both steps 3 and 4 in the [quickstart](#quickstart) section in order to get your application up and running.

### Maven

To use the library in an external Java project, publish the library to a Maven
repository. For example, publish the library to the local Maven repository using
the `mvn` tool (installed separately):

```sh
bazel build -c opt //tensorflow/java:pom
mvn install:install-file \
  -Dfile=../../bazel-bin/tensorflow/java/libtensorflow.jar \
  -DpomFile=../../bazel-bin/tensorflow/java/pom.xml
```

Refer to the library using Maven coordinates. For example, if you're using Maven
then place this dependency into your `pom.xml` file (replacing 1.0.head with
the version of the TensorFlow runtime you wish to use).

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow</artifactId>
  <version>1.0.head</version>
</dependency>
```

### Bazel

If your project uses bazel for builds, add a dependency on
`//tensorflow/java:tensorflow` to the `java_binary` or `java_library` rule. For
example:

```sh
bazel run -c opt //tensorflow/java/src/main/java/org/tensorflow/examples:label_image
```
