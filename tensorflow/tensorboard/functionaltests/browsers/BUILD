load("@io_bazel_rules_webtesting//web:web.bzl", "browser")

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
)

browser(
    name = "headless-chrome",
    data = select({
        "linux_x86_64": [
            "//third_party/noto:hinted",
            "//third_party/noto:mono",
            "//third_party/chromium",
            "//third_party/chromium:chromedriver",
        ],
        "//conditions:default": None,
    }),
    disabled = select({
        ":linux_x86_64": None,
        "//conditions:default": "headless-chrome is only supported on Linux.",
    }),
    environment = {"DISABLE_X_DISPLAY": "1"},
    metadata = "headless-chrome.json",
    visibility = ["//visibility:public"],
)
