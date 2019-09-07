# Build TensorFlow Lite C library from source

```bash
cd <TENSORFLOW_DIRECTORY>/
./tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
```

then execute one of:
```bash
# for current platform
$ ./tensorflow/tensorflow/lite/tools/make/build_lib.sh
# OR, for Raspberry Pi
$ ./tensorflow/tensorflow/lite/tools/make/build_rpi_lib.sh
# OR, for ARM devices 
$ ./tensorflow/tensorflow/lite/tools/make/build_aarch64_lib.sh
# OR, for all iOS platforms
$ ./tensorflow/tensorflow/lite/tools/make/build_ios_universal_lib.sh
```

this will produce a static lib located at `<TENSORFLOW_DIRECTORY>/tensorflow/tensorflow/lite/tools/make/gen/<PLATFORM>/lib/libtensorflow-lite.a`.

From there, we can build the shared library:

```bash
cd <TENSORFLOW_DIRECTORY>/tensorflow/tensorflow/lite/experimental/c/
make
```
this will output the C shared library located at `<TENSORFLOW_DIRECTORY>/tensorflow/lite/experimental/c/libtensorflowlite_c.so`


# Unofficial related projects using this library

- [Go-tflite](https://github.com/mattn/go-tflite): Go binding for TensorFlow Lite

- [Rust tflite](https://crates.io/crates/tflite): Rust binding for TensorFlow Lite

- [Tensorflow Lite for Alpine Linux](https://github.com/Jonarod/tensorflow_lite_alpine): Tiny `musl`-built TensorFlow Lite for Alpine Linux.
