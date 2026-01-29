load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
load("//third_party/hexagon:workspace.bzl", hexagon_nn = "repo")
load("//third_party/icu:workspace.bzl", icu = "repo")
load("//third_party/kissfft:workspace.bzl", kissfft = "repo")
load("//third_party/libwebp:workspace.bzl", libwebp = "repo")
load("//third_party/opencl_headers:workspace.bzl", opencl_headers = "repo")
load("//third_party/pasta:workspace.bzl", pasta = "repo")
load("//third_party/ruy:workspace.bzl", ruy = "repo")
load("//third_party/sobol_data:workspace.bzl", sobol_data = "repo")
load("//third_party/vulkan_headers:workspace.bzl", vulkan_headers = "repo")
load("//third_party/jpeg:workspace.bzl", jpeg = "repo")
load("@xla//third_party/nasm:workspace.bzl", nasm = "repo")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/tf_runtime:workspace.bzl", tf_runtime = "repo")

def _third_party_ext_impl(mctx):  # @unused
    flatbuffers()
    hexagon_nn()
    icu()
    kissfft()
    libwebp()
    opencl_headers()
    pasta()
    ruy()
    sobol_data()
    vulkan_headers()
    nasm()
    jpeg()

    tf_runtime()
    tf_http_archive(
        name = "org_xprof",
        sha256 = "d27bcd502a0843e463fc4eb7d3532d0d720ddd6af6e39942846f1aa769352625",
        strip_prefix = "xprof-c695e43eba127a74a67263775ab611bded7fba34",
        patch_file = ["//third_party/xprof:xprof.patch"],
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/c695e43eba127a74a67263775ab611bded7fba34.zip"),
    )

third_party_ext = module_extension(
    implementation = _third_party_ext_impl,
)
