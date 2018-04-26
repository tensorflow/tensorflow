SET GOOGLE_CLOUD_CREDENTIAL=%KOKORO_BAZEL_AUTH_CREDENTIAL:\=/%
c:\tools\msys64\usr\bin\bash -l %cd%/tensorflow/tools/ci_build/windows/cpu/pip/build_tf_windows.sh %*
