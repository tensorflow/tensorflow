"""Additional build options needed for the GPU Delegate."""

# copybara:uncomment_begin(google-only)
# load("//third_party/android/ndk/platforms:grte_top.bzl", "min_supported_ndk_api")
# copybara:uncomment_end

def nativewindow_linkopts():
    # copybara:uncomment_begin(google-only)
    # return min_supported_ndk_api("26", ["-lnativewindow"])
    # copybara:uncomment_end
    # copybara:comment_begin(oss-only)
    return ["-lnativewindow"]
    # copybara:comment_end

def gpu_delegate_linkopts():
    """Additional link options needed when linking in the GPU Delegate."""
    return select({
        "//tensorflow:android": [
            "-lEGL",
            # We don't need to link libGLESv3, because if it exists,
            # it is a symlink to libGLESv2.
            # See Compatibility Definition Document:
            # https://source.android.com/compatibility/10/android-10-cdd#7_1_4_1_opengl_es
            "-lGLESv2",
        ],
        "//conditions:default": [],
    }) + nativewindow_linkopts()
