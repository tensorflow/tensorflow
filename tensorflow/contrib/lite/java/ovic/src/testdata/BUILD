# Testdata for OVIC benchmarker demo App and tests.
licenses(["notice"])  # Apache 2.0

filegroup(
    name = "ovic_testdata",
    srcs = [
        "float_model.lite",
        "low_res_model.lite",
        "quantized_model.lite",
        "test_image_128.jpg",
        "test_image_224.jpg",
    ],
    visibility = ["//visibility:public"],
)

exports_files(
    ["labels.txt"],
    visibility = ["//visibility:public"],
)
