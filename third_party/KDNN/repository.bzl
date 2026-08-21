"""Bazel repository rule for the KDNN v3.1.0 binary distribution."""

_KDNN_URL = "https://raw.gitcode.com/boostkit/kdnn/archive/refs/heads/v3.1.0.zip"
_KDNN_SHA256 = "6c5acd1aab85f9dbfe7cf81e253469e7370274f6182dad4349a2c96c3d22a882"

def _find_static_library(root):
    for relative in [
        "lib/threadpool/libkdnn.a",
        "lib/sve/threadpool/libkdnn.a",
        "lib/sve2/threadpool/libkdnn.a",
    ]:
        library = root.get_child(relative)
        if library.exists:
            return library
    return None

def _stage_installation(ctx, root, source_name):
    include = root.get_child("include")
    if not include.exists:
        fail("%s does not contain include/" % source_name)
    static_lib = _find_static_library(root)
    if static_lib == None:
        fail("%s does not contain a threadpool libkdnn.a" % source_name)
    result = ctx.execute(["cp", "-rL", str(include), "include"])
    if result.return_code:
        fail("failed to stage KDNN headers: %s" % result.stderr)
    result = ctx.execute(["cp", "-Lf", str(static_lib), "lib/libkdnn.a"])
    if result.return_code:
        fail("failed to stage KDNN static library: %s" % result.stderr)

def _kdnn_repository_impl(ctx):
    result = ctx.execute(["mkdir", "-p", "lib"])
    if result.return_code:
        fail("failed to create KDNN library staging directory: %s" % result.stderr)

    kdnn_root = ctx.os.environ.get("KDNN_ROOT", "")
    if kdnn_root:
        _stage_installation(ctx, ctx.path(kdnn_root), "KDNN_ROOT")
    else:
        archive = ctx.os.environ.get("KDNN_ARCHIVE", "")
        if archive:
            archive_path = ctx.path(archive)
            if not archive_path.exists:
                fail("KDNN_ARCHIVE does not exist: %s" % archive)
            ctx.symlink(archive_path, "kdnn.zip")
        else:
            ctx.download(url = _KDNN_URL, output = "kdnn.zip", sha256 = _KDNN_SHA256)

        ctx.extract("kdnn.zip", output = "payload")
        rpm_search = ctx.execute(["sh", "-c", "find payload -type f -name '*.rpm' -print"])
        if rpm_search.return_code:
            fail("failed to search KDNN archive for RPMs: %s" % rpm_search.stderr)
        rpms = [line for line in rpm_search.stdout.split("\n") if line]
        if len(rpms) != 1:
            fail("expected one KDNN RPM in v3.1.0 archive, found: %s" % rpms)
        rpm = rpms[0]
        result = ctx.execute(
            ["sh", "-c", "rpm2archive -n \"$1\" > \"$2\"", "sh", rpm, "payload/package.tar"],
            working_directory = str(ctx.path(".")),
        )
        if result.return_code:
            fail("rpm2archive failed: %s" % result.stderr)
        result = ctx.execute(["mkdir", "-p", "payload/root"], working_directory = str(ctx.path(".")))
        if result.return_code:
            fail("failed to create KDNN extraction directory: %s" % result.stderr)
        result = ctx.execute(
            ["bsdtar", "--no-same-owner", "-xf", "payload/package.tar", "-C", "payload/root"],
            working_directory = str(ctx.path(".")),
        )
        if result.return_code:
            fail("failed to extract KDNN RPM: %s" % result.stderr)
        _stage_installation(ctx, ctx.path("payload/root/usr/local/kdnn"), "KDNN RPM")

    # Apply the published adapter to the staged headers only. The equivalent
    # content transformation avoids carrying RPM ownership metadata through
    # patch(1) in Bazel's repository directory.
    replacements = {
        "service/kdnn_service.hpp": [
            ("throw BadArrayNewLength();", "return nullptr;"),
        ],
        "types/kdnn_data_type.hpp": [
            ("        if (type == TypeT::UNDEFINED) {\n            throw Service::LogicError {\"Type: unsupported data type\"};\n        }\n", ""),
        ],
        "types/kdnn_shape.hpp": [
            ("    Shape(T *ptr, const SizeType size) noexcept(false) : numDims(size)\n    {\n        if (ptr == nullptr) {\n            throw Service::LogicError(\"Shape: ptr is nullptr\");\n        }", "    Shape(T *ptr, const SizeType size) noexcept(false) : numDims(size)\n    {\n        if (ptr == nullptr) {\n            return;\n        }"),
            ("    Shape& ResetShape(T *ptr, const SizeType size) noexcept(false)\n    {\n        if (ptr == nullptr) {\n            throw Service::LogicError(\"Shape: ptr is nullptr\");\n        }", "    Shape& ResetShape(T *ptr, const SizeType size) noexcept(false)\n    {\n        if (ptr == nullptr) {\n            return *this;\n        }"),
            ("        if (adder.GetNumDims() !=  this->GetNumDims()) {\n            throw Service::LogicError(\"Shape: different size of base and adder shapes\");\n        }", "        if (adder.GetNumDims() !=  this->GetNumDims()) {\n            return *this;\n        }"),
            ("        if (adder.GetNumDims() != this->GetNumDims()) {\n            throw Service::LogicError(\"Shape: different size of base and adder shapes\");\n        }\n", ""),
            ("        if (id >= numDims) {\n            throw Service::LogicError(\"Shape: index >= num_dims\");\n        }\n", ""),
            ("        if (Service::WillIntMultOverflow(dimsArray.begin(), dimsArray.begin() + numDims)) {\n            throw Service::LogicError(\"Shape: computing total size will cause overflow\");\n        }\n", ""),
            ("    void CheckNumDims(SizeType nDims) const noexcept(false)\n    {\n        if (nDims > NUM_MAX_DIMENSIONS) {\n            throw Service::LogicError(\"Shape: dims is greater than NUM_MAX_DIMENSIONS\");\n        }\n    }", "    void CheckNumDims(SizeType nDims) const noexcept(false) {}"),
        ],
        "types/kdnn_tensor_info.hpp": [
            ("throw Service::LogicError {\"Tensor Info: tensor dimensionality is incorrect\"};", "return Layout::UNDEFINED;"),
            ("throw Service::LogicError{\"Tensor Info: tensor dimensionality is incorrect\"};", "return Layout::UNDEFINED;"),
        ],
    }
    for relative, file_replacements in replacements.items():
        path = ctx.path("include").get_child(relative)
        content = ctx.read(path)
        for old, new in file_replacements:
            content = content.replace(old, new)
        ctx.file("include/" + relative, content)

    ctx.file("BUILD", """licenses(["restricted"])
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "kdnn",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
    includes = ["include"],
    srcs = ["lib/libkdnn.a"],
    linkopts = ["-lgomp"],
    alwayslink = 1,
)
""")

kdnn_repository = repository_rule(
    implementation = _kdnn_repository_impl,
    environ = ["KDNN_ROOT", "KDNN_ARCHIVE"],
    attrs = {"adapter_patch": attr.label(default = Label("//third_party/KDNN:tensorflow_kdnn_include_adapter.patch"))},
)
