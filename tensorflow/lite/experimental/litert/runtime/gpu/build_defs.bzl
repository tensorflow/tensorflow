"""Additional build options needed for the LiteRT GPU runtime."""

def litert_angle_heapcheck_deps():
    # copybara:uncomment_begin(google-only)
    # return select({
    # "//tensorflow/lite/experimental/litert/runtime/gpu:angle_on_linux-google": [
    # "@com_google_googletest//:gtest_main_no_heapcheck",
    # ],
    # "//conditions:default": [
    # "@com_google_googletest//:gtest_main",
    # ],
    # })
    # copybara:uncomment_end
    # copybara:comment_begin(oss-only)
    return ["@com_google_googletest//:gtest_main"]
    # copybara:comment_end
