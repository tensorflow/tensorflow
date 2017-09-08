# TensorFlow for Java

> *WARNING*: The TensorFlow Java API is not currently covered by the TensorFlow
> [API stability guarantees](https://www.tensorflow.org/programmers_guide/version_semantics).
>
> For using TensorFlow on Android refer instead to
> [contrib/android](https://www.tensorflow.org/code/tensorflow/contrib/android),
> [makefile](https://www.tensorflow.org/code/tensorflow/contrib/makefile#android)
> and/or the [Android demo](https://www.tensorflow.org/code/tensorflow/examples/android).

## Quickstart

-   Refer to [Installing TensorFlow for Java](https://www.tensorflow.org/install/install_java)
-   [Javadoc](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)
-   [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.tensorflow/tensorflow/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.tensorflow/tensorflow)

## Building from source

If the quickstart instructions above do not work out, the TensorFlow Java and
native libraries will need to be built from source.

1.  Install [bazel](https://www.bazel.build/versions/master/docs/install.html)

2.  Setup the environment to build TensorFlow from source code
    ([Linux](https://www.tensorflow.org/install/install_sources#PrepareLinux)
    or [Mac OS
    X](https://www.tensorflow.org/install/install_sources#PrepareMac)).
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

The command above will produce two files in the `bazel-bin/tensorflow/java`
directory:

*   An archive of Java classes: `libtensorflow.jar`
*   A native library: `libtensorflow_jni.so` on Linux, `libtensorflow_jni.dylib`
    on OS X, or `tensorflow_jni.dll` on Windows.

To compile Java code that uses the TensorFlow Java API, include
`libtensorflow.jar` in the classpath. For example:

```sh
javac -cp bazel-bin/tensorflow/java/libtensorflow.jar ...
```

To execute the compiled program, include `libtensorflow.jar` in the classpath
and the native library in the library path. For example:

```sh
java -cp bazel-bin/tensorflow/java/libtensorflow.jar \
  -Djava.library.path=bazel-bin/tensorflow/java \
  ...
```

Installation on Windows requires the more experimental [bazel on
Windows](https://bazel.build/versions/master/docs/windows.html). Details are
omitted here, but find inspiration in the script used for building the release
archive:
[`tensorflow/tools/ci_build/windows/libtensorflow_cpu.sh`](https://www.tensorflow.org/code/tensorflow/tools/ci_build/windows/libtensorflow_cpu.sh).

### Maven

Details of the release process for Maven Central are in
[`maven/README.md`](https://www.tensorflow.org/code/tensorflow/java/maven/README.md).
However, for development, you can push the library built from source to a local
Maven repository with:

```sh
bazel build -c opt //tensorflow/java:pom
mvn install:install-file \
  -Dfile=../../bazel-bin/tensorflow/java/libtensorflow.jar \
  -DpomFile=../../bazel-bin/tensorflow/java/pom.xml
```

And then refer to this library in a project's `pom.xml` with: (replacing
VERSION with the appropriate version of TensorFlow):

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow</artifactId>
  <version>VERSION</version>
</dependency>
```

### Bazel

If your project uses bazel for builds, add a dependency on
`//tensorflow/java:tensorflow` to the `java_binary` or `java_library` rule. For
example:

```sh
bazel run -c opt //tensorflow/java/src/main/java/org/tensorflow/examples:label_image
```
