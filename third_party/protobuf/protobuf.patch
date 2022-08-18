diff --git a/BUILD b/BUILD
index dbae719ff..4e276c854 100644
--- a/BUILD
+++ b/BUILD
@@ -23,7 +23,7 @@ config_setting(
 # ZLIB configuration
 ################################################################################
 
-ZLIB_DEPS = ["@zlib//:zlib"]
+ZLIB_DEPS = ["@zlib"]
 
 ################################################################################
 # Protobuf Runtime Library
@@ -100,6 +100,7 @@ LINK_OPTS = select({
 
 load(
     ":protobuf.bzl",
+    "adapt_proto_library",
     "cc_proto_library",
     "py_proto_library",
     "internal_copied_filegroup",
@@ -143,6 +144,7 @@ cc_library(
     copts = COPTS,
     includes = ["src/"],
     linkopts = LINK_OPTS,
+    alwayslink = 1,
     visibility = ["//visibility:public"],
 )
 
@@ -213,6 +215,7 @@ cc_library(
     copts = COPTS,
     includes = ["src/"],
     linkopts = LINK_OPTS,
+    alwayslink = 1,
     visibility = ["//visibility:public"],
     deps = [":protobuf_lite"] + PROTOBUF_DEPS,
 )
@@ -255,13 +258,15 @@ filegroup(
     visibility = ["//visibility:public"],
 )
 
-cc_proto_library(
+adapt_proto_library(
+    name = "cc_wkt_protos_genproto",
+    deps = [proto + "_proto" for proto in WELL_KNOWN_PROTO_MAP.keys()],
+    visibility = ["//visibility:public"],
+)
+
+cc_library(
     name = "cc_wkt_protos",
-    srcs = WELL_KNOWN_PROTOS,
-    include = "src",
-    default_runtime = ":protobuf",
-    internal_bootstrap_hack = 1,
-    protoc = ":protoc",
+    deprecation = "Only for backward compatibility. Do not use.",
     visibility = ["//visibility:public"],
 )
 
@@ -978,10 +983,10 @@ cc_library(
 
 proto_lang_toolchain(
     name = "cc_toolchain",
+    blacklisted_protos = [proto + "_proto" for proto in WELL_KNOWN_PROTO_MAP.keys()],
     command_line = "--cpp_out=$(OUT)",
     runtime = ":protobuf",
     visibility = ["//visibility:public"],
-    blacklisted_protos = [":_internal_wkt_protos_genrule"],
 )
 
 proto_lang_toolchain(
diff --git a/protobuf.bzl b/protobuf.bzl
index e0653321f..4156a1275 100644
--- a/protobuf.bzl
+++ b/protobuf.bzl
@@ -1,4 +1,5 @@
 load("@bazel_skylib//lib:versions.bzl", "versions")
+load("@rules_proto//proto:defs.bzl", "ProtoInfo")
 
 def _GetPath(ctx, path):
     if ctx.label.workspace_root:
@@ -85,6 +86,8 @@ def _proto_gen_impl(ctx):
     for dep in ctx.attr.deps:
         import_flags += dep.proto.import_flags
         deps += dep.proto.deps
+    import_flags = depset(import_flags).to_list()
+    deps = depset(deps).to_list()
 
     if not ctx.attr.gen_cc and not ctx.attr.gen_py and not ctx.executable.plugin:
         return struct(
@@ -222,6 +225,29 @@ Args:
   outs: a list of labels of the expected outputs from the protocol compiler.
 """
 
+def _adapt_proto_library_impl(ctx):
+    deps = [dep[ProtoInfo] for dep in ctx.attr.deps]
+
+    srcs = [src for dep in deps for src in dep.direct_sources]
+    return struct(
+        proto = struct(
+            srcs = srcs,
+            import_flags = ["-I{}".format(path) for dep in deps for path in dep.transitive_proto_path.to_list()],
+            deps = srcs,
+        ),
+    )
+
+adapt_proto_library = rule(
+    implementation = _adapt_proto_library_impl,
+    attrs = {
+        "deps": attr.label_list(
+            mandatory = True,
+            providers = [ProtoInfo],
+        ),
+    },
+    doc = "Adapts `proto_library` from `@rules_proto` to be used with `{cc,py}_proto_library` from this file.",
+)
+
 def cc_proto_library(
         name,
         srcs = [],
@@ -229,7 +255,6 @@ def cc_proto_library(
         cc_libs = [],
         include = None,
         protoc = "@com_google_protobuf//:protoc",
-        internal_bootstrap_hack = False,
         use_grpc_plugin = False,
         default_runtime = "@com_google_protobuf//:protobuf",
         **kargs):
@@ -247,41 +272,17 @@ def cc_proto_library(
           cc_library.
       include: a string indicating the include path of the .proto files.
       protoc: the label of the protocol compiler to generate the sources.
-      internal_bootstrap_hack: a flag indicate the cc_proto_library is used only
-          for bootstraping. When it is set to True, no files will be generated.
-          The rule will simply be a provider for .proto files, so that other
-          cc_proto_library can depend on it.
       use_grpc_plugin: a flag to indicate whether to call the grpc C++ plugin
           when processing the proto files.
       default_runtime: the implicitly default runtime which will be depended on by
           the generated cc_library target.
       **kargs: other keyword arguments that are passed to cc_library.
-
     """
 
     includes = []
     if include != None:
         includes = [include]
 
-    if internal_bootstrap_hack:
-        # For pre-checked-in generated files, we add the internal_bootstrap_hack
-        # which will skip the codegen action.
-        proto_gen(
-            name = name + "_genproto",
-            srcs = srcs,
-            deps = [s + "_genproto" for s in deps],
-            includes = includes,
-            protoc = protoc,
-            visibility = ["//visibility:public"],
-        )
-
-        # An empty cc_library to make rule dependency consistent.
-        native.cc_library(
-            name = name,
-            **kargs
-        )
-        return
-
     grpc_cpp_plugin = None
     if use_grpc_plugin:
         grpc_cpp_plugin = "//external:grpc_cpp_plugin"
diff --git a/python/google/protobuf/pyext/message.cc b/python/google/protobuf/pyext/message.cc
index 3530a9b37..c31fa8fcc 100644
--- a/python/google/protobuf/pyext/message.cc
+++ b/python/google/protobuf/pyext/message.cc
@@ -2991,8 +2991,12 @@ bool InitProto2MessageModule(PyObject *m) {
         reinterpret_cast<PyObject*>(
             &RepeatedCompositeContainer_Type));
 
-    // Register them as collections.Sequence
+    // Register them as MutableSequence.
+#if PY_MAJOR_VERSION >= 3
+    ScopedPyObjectPtr collections(PyImport_ImportModule("collections.abc"));
+#else
     ScopedPyObjectPtr collections(PyImport_ImportModule("collections"));
+#endif
     if (collections == NULL) {
       return false;
     }
diff --git a/python/google/protobuf/pyext/unknown_fields.cc b/python/google/protobuf/pyext/unknown_fields.cc
index c3679c0d3..e80a1d97a 100755
--- a/python/google/protobuf/pyext/unknown_fields.cc
+++ b/python/google/protobuf/pyext/unknown_fields.cc
@@ -221,7 +221,7 @@ const UnknownField* GetUnknownField(PyUnknownFieldRef* self) {
                  "The parent message might be cleared.");
     return NULL;
   }
-  ssize_t total_size = fields->field_count();
+  Py_ssize_t total_size = fields->field_count();
   if (self->index >= total_size) {
     PyErr_Format(PyExc_ValueError,
                  "UnknownField does not exist. "
