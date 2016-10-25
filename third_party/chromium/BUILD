load("@io_bazel_rules_webtesting//web:web.bzl", "web_test_archive")

licenses(["notice"])

web_test_archive(
    name = "chromium",
    archive = "headless_chromium.tar.gz",
    named_files = {
        "CHROME": "headless_shell",
    },
    visibility = ["//tensorflow/tensorboard/functionaltests/browsers:__pkg__"],
)

web_test_archive(
    name = "chromedriver",
    archive = "@org_chromium_chromedriver_linux//file",
    named_files = {
        "CHROMEDRIVER": "chromedriver",
    },
    visibility = ["//tensorflow/tensorboard/functionaltests/browsers:__pkg__"],
)