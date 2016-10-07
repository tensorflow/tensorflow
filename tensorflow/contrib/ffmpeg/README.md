# FFmpeg TensorFlow integration

Decoding audio files can be done using a new op that uses
[FFmpeg](http://www.ffmpeg.org) to convert various audio file formats into
tensors.

tf.audio.decode_audio accepts MP3, WAV, and OGG file formats.

FFmpeg must be installed before these ops can be used. The ops will look for the
ffmpeg binary somewhere in `$PATH`. When the binary is unavailable, the error
`FFmpeg must be installed to run this op. FFmpeg can be found at
http://www.ffmpeg.org.` will be returned.

## Testing

In addition to the regular tests, the integration tests should also be
run on this code. First, install `docker`. Then run the integration tests:

```shell
export TF_BUILD_CONTAINER_TYPE=CPU  # or GPU
export TF_BUILD_PYTHON_VERSION=PYTHON2  # or PYTHON3
export TF_BUILD_IS_OPT=OPT
export TF_BUILD_IS_PIP=PIP
export TF_BUILD_INTEGRATION_TESTS=1
tensorflow/tools/ci_build/ci_parameterized_build.sh
```
