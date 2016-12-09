# Description: SavedModel half plus two example.

package(
    default_visibility = ["//tensorflow/python/saved_model:__subpackages__"],
    features = [
        "-layering_check",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
    visibility = ["//visibility:public"],
)

py_binary(
    name = "saved_model_half_plus_two",
    srcs = [
        "saved_model_half_plus_two.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
    ],
)

# Genrule for SavedModel half-plus-two test data. Specifically, this genrule
# exports the test SavedModel to a versioned directory in order to be compatible
# with TensorFlow Serving model server requirements of a versioned subdirectory.
# Please note that SavedModel itself accepts any valid directory as the save
# location and does not perform any versioning.
# genrule(
#     name = "saved_model_half_plus_two_data",
#     outs = [
#         "saved_model_half_plus_two/00000123/saved_model.pb",
#         "saved_model_half_plus_two/00000123/assets/foo.txt",
#         "saved_model_half_plus_two/00000123/variables/variables.data-00000-of-00001",
#         "saved_model_half_plus_two/00000123/variables/variables.index",
#         "saved_model_half_plus_two_pbtxt/00000123/saved_model.pbtxt",
#         "saved_model_half_plus_two_pbtxt/00000123/assets/foo.txt",
#         "saved_model_half_plus_two_pbtxt/00000123/variables/variables.data-00000-of-00001",
#         "saved_model_half_plus_two_pbtxt/00000123/variables/variables.index",
#     ],
#     cmd =
#         "rm -rf $(@D)/saved_model_half_plus_two $(@D)/saved_model_half_plus_two_pbtxt; " +
#         "./$(locations :saved_model_half_plus_two) --output_dir=$(@D)/saved_model_half_plus_two/00000123 --output_dir_pbtxt=$(@D)/saved_model_half_plus_two_pbtxt/00000123",
#     tools = [
#         ":saved_model_half_plus_two",
#     ],
#     visibility = ["//visibility:public"],
# )
