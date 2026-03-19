WIP ML Build Docker container for ML repositories (Tensorflow, JAX and XLA).

This container branches off from
/tensorflow/tools/tf_sig_build_dockerfiles/. However, since
hermetic CUDA and hermetic Python is now available for Tensorflow, a lot of the
requirements installed on the original container can be removed to reduce the
footprint of the container and make it more reusable across different ML
repositories.
