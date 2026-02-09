"""Darts-clone is a clone of Darts (Double-ARray Trie System)."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "darts_clone",
    hdrs = [
        "include/darts.h",
    ],
)
