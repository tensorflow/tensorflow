# Testdata for OVIC benchmarker demo App and tests.

package(
    licenses = ["notice"],
)

filegroup(
    name = "ovic_testdata",
    srcs = [
        "@tflite_ovic_testdata//:detect.lite",
        "@tflite_ovic_testdata//:float_model.lite",
        "@tflite_ovic_testdata//:low_res_model.lite",
        "@tflite_ovic_testdata//:quantized_detect.lite",
        "@tflite_ovic_testdata//:quantized_model.lite",
        "@tflite_ovic_testdata//:test_image_128.jpg",
        "@tflite_ovic_testdata//:test_image_224.jpg",
    ],
    visibility = ["//visibility:public"],
)

exports_files(
    [
        "labels.txt",
        "coco_labels.txt",
    ],
    visibility = ["//visibility:public"],
)
