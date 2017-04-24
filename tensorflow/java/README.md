# TensorFlow for Java

Java bindings for TensorFlow. ([Javadoc](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary))

[![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.tensorflow/tensorflow/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.tensorflow/tensorflow)

> *WARNING*: The TensorFlow Java API is not currently covered by the TensorFlow
> [API stability guarantees](https://www.tensorflow.org/programmers_guide/version_semantics).
>
> For using TensorFlow on Android refer to
> [contrib/android](https://www.tensorflow.org/code/tensorflow/contrib/android),
> [makefile](https://www.tensorflow.org/code/tensorflow/contrib/makefile#android)
> and/or the [Android
> demo](https://www.tensorflow.org/code/tensorflow/examples/android).

## Quickstart: Using [Apache Maven](https://maven.apache.org)

TensorFlow for Java releases are included in
[Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22org.tensorflow%22%20AND%20a%3A%22tensorflow%22)
and support Linux, OS X and Windows. To use it, add the following dependency to
your project's `pom.xml`:

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow</artifactId>
  <version>1.1.0-rc2</version>
</dependency>
```

That's all. As an example, to create a Maven project for the
[label image example](https://www.tensorflow.org/code/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java):

1.  Create a `pom.xml`:

    ```xml
    <project>
        <modelVersion>4.0.0</modelVersion>
        <groupId>org.myorg</groupId>
        <artifactId>label-image</artifactId>
        <version>1.0-SNAPSHOT</version>
        <properties>
          <exec.mainClass>org.tensorflow.examples.LabelImage</exec.mainClass>
          <!-- The LabelImage example code requires at least JDK 1.7. -->
          <!-- The maven compiler plugin defaults to a lower version -->
          <maven.compiler.source>1.7</maven.compiler.source>
          <maven.compiler.target>1.7</maven.compiler.target>
        </properties>
        <dependencies>
          <dependency>
            <groupId>org.tensorflow</groupId>
            <artifactId>tensorflow</artifactId>
            <version>1.1.0-rc2</version>
          </dependency>
        </dependencies>
    </project>
    ```

2.  Download the [example source](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java)
    into `src/main/java/org/tensorflow/examples`. On Linux and OS X, the following script should work:

    ```sh
    mkdir -p src/main/java/org/tensorflow/examples
    curl -L "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java" -o src/main/java/org/tensorflow/examples/LabelImage.java
    ```

3.  Compile and execute:

    ```sh
    mvn compile exec:java
    ```

## Quickstart: Using `java` and `javac`

This section describes how to use TensorFlow armed with just a JDK installation.

1.  Download the Java archive (JAR):
    [libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.1.0-rc2.jar)
    (optionally, the Java sources:
    [libtensorflow-src.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-src-1.1.0-rc2.jar)).

2.  Download the native library. GPU-enabled versions required CUDA 8 and cuDNN
    5.1. For other versions, the native library will need to be built from
    source (see below).

    -   Linux:
        [CPU-only](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.1.0-rc2.tar.gz),
        [GPU-enabled](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.1.0-rc2.tar.gz)
    -   OS X:
        [CPU-only](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.1.0-rc2.tar.gz),
        [GPU-enabled](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-darwin-x86_64-1.1.0-rc2.tar.gz)
    -   Windows:
        [CPU-only](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.1.0-rc2.zip)


    The following shell snippet downloads and extracts the native library on
    Linux and OS X. For Windows, download and extract manually.

    ```sh
    TF_TYPE="cpu" # Set to "gpu" to enable GPU support
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    mkdir -p ./jni
    curl -L \
      "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-${TF_TYPE}-${OS}-x86_64-1.1.0-rc2.tar.gz" |
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
    javac -cp libtensorflow-1.1.0-rc2.jar MyClass.java
    ```

    For a more sophisticated example, see
    [LabelImage.java](https://www.tensorflow.org/code/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java),
    which can be compiled with:

    ```sh
    javac \
      -cp libtensorflow-1.1.0-rc2.jar \
      ./src/main/java/org/tensorflow/examples/LabelImage.java
    ```

4.  Include the downloaded `.jar` in the classpath and the native library in the
    library path during execution. For example:

    ```sh
    java -cp libtensorflow-1.1.0-rc2.jar:. -Djava.library.path=./jni MyClass
    ```

    or for the `LabelImage` example:

    ```sh
    java \
      -Djava.library.path=./jni \
      -cp libtensorflow-1.1.0-rc2.jar:./src/main/java \
      org.tensorflow.examples.LabelImage
    ```

## Building from source

If the quickstart instructions above do not work out, the TensorFlow Java and
native libraries will need to be built from source.

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

The JAR (`libtensorflow.jar`) and native library (`libtensorflow_jni.so` on
Linux, `libtensorflow_jni.dylib` on OS X, `tensorflow_jni.dll` on Windows) will
be in `bazel-bin/tensorflow/java`. Using these artifacts follow both steps 3
and 4 in the previous section in order to get your application
up and running.

Installation on Windows requires the more experimental [bazel on Windows](https://bazel.build/versions/master/docs/windows.html).
Details are elided here, but find inspiration in the script used for
building the release archive:
[`tensorflow/tools/ci_build/windows/libtensorflow_cpu.sh`](https://www.tensorflow.org/code/tensorflow/tools/ci_build/windows/libtensorflow_cpu.sh).

### Maven

Details of the release process for Maven Central are in [`maven/README.md`](https://www.tensorflow.org/code/tensorflow/java/maven/README.md).
However, for development, you can push the library built from source to a local
Maven repository with:

```sh
bazel build -c opt //tensorflow/java:pom
mvn install:install-file \
  -Dfile=../../bazel-bin/tensorflow/java/libtensorflow.jar \
  -DpomFile=../../bazel-bin/tensorflow/java/pom.xml
```

And then rever to this library in a project's `pom.xml` with:
(replacing 1.0.head with the appropriate version):

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
