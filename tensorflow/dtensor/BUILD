# DTensor is a TensorFlow extension that enables large-scale modeling with
# minimal changes to user code.

package(
    licenses = ["notice"],
)

package_group(
    name = "dtensor-users",
    packages = [
        "//tensorflow_models/google/...",
        "//third_party/aimee/clara2_labs/...",
        "//third_party/py/keras/dtensor/...",
    ],
)

package_group(
    name = "dtensor-internal",
    packages = [
        "//learning/brain/experimental/jax_tpu_embedding/...",
        "//learning/brain/public/...",
        "//tensorflow/core/...",
        "//tensorflow/dtensor/...",
        "//tensorflow/python/...",
    ],
)
