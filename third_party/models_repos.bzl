"""Imports TensorFlow models repositories."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def models_repositories():
    """Load TensorFlow models repositories."""
    http_archive(
        name = "inception_v1",
        build_file = "//:models.BUILD",
        sha256 = "7efe12a8363f09bc24d7b7a450304a15655a57a7751929b2c1593a71183bb105",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/inception_v1.zip",
        ],
    )

    http_archive(
        name = "mobile_ssd",
        build_file = "//:models.BUILD",
        sha256 = "bddd81ea5c80a97adfac1c9f770e6f55cbafd7cce4d3bbe15fbeb041e6b8f3e8",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip",
        ],
    )

    http_archive(
        name = "mobile_multibox",
        build_file = "//:models.BUILD",
        sha256 = "859edcddf84dddb974c36c36cfc1f74555148e9c9213dedacf1d6b613ad52b96",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip",
        ],
    )

    http_archive(
        name = "stylize",
        build_file = "//:models.BUILD",
        sha256 = "3d374a730aef330424a356a8d4f04d8a54277c425e274ecb7d9c83aa912c6bfa",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip",
        ],
    )

    http_archive(
        name = "speech_commands",
        build_file = "//:models.BUILD",
        sha256 = "c3ec4fea3158eb111f1d932336351edfe8bd515bb6e87aad4f25dbad0a600d0c",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip",
        ],
    )

    http_archive(
        name = "person_detect_data",
        sha256 = "170542270da256994ce24d1e357f6e84a54fdaf7d28ff2b74725a40b70b082cf",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_grayscale_2020_05_24.zip",
        ],
    )
