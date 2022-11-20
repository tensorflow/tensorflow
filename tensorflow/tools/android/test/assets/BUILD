package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# It is necessary to use this filegroup rather than globbing the files in this
# folder directly the examples/android:tensorflow_demo target due to the fact
# that assets_dir is necessarily set to "" there (to allow using other
# arbitrary targets as assets).
filegroup(
    name = "asset_files",
    srcs = glob(
        ["**/*"],
        exclude = ["BUILD"],
    ),
)
