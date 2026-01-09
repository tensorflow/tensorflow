"""Contains embed_files build rule."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

visibility(DEFAULT_LOAD_VISIBILITY)

def embed_files(name, srcs, cpp_namespace = "", compatible_with = None, **kwargs):
    """Compiles srcs into a cc_library with functions returning embedded file data.

    Example:
        embed_files(
            name = "embed_some_file",
            srcs = ["file1.txt", "file2.txt"],
            cpp_namespace = "my_namespace",
        )

    will generate a cc_library with the following functions:

        const std::string& get_file1();
        const std::string& get_file2();

    Args:
        name: name for the generated cc_library target
        srcs: files to embed
        cpp_namespace: If set, the generated code will be wrapped in this namespace
        compatible_with: The `compatible_with` attribute to pass to the generated targets.
        **kwargs: keyword arguments passed onto the generated cc_library() rule.
    """

    namespace_open = ""
    namespace_close = ""
    if cpp_namespace:
        namespace_open = "namespace " + cpp_namespace + " { "
        namespace_close = "}  // namespace " + cpp_namespace + "\n"

    native.genrule(
        name = name + "_gen",
        srcs = srcs,
        outs = [
            name + ".cc",
            name + ".h",
        ],
        cmd = """
            HDR_OUT=$(location {name}.h)
            CC_OUT=$(location {name}.cc)
            GUARD="{guard}"

            # 1. Start Header File
            echo "#ifndef $${{GUARD}}" > "$${{HDR_OUT}}"
            echo "#define $${{GUARD}}" >> "$${{HDR_OUT}}"
            echo "#include <string>" >> "$${{HDR_OUT}}"
            echo "" >> "$${{HDR_OUT}}"
            echo "{namespace_open}" >> "$${{HDR_OUT}}"

            # 2. Start CC File
            # Include standard headers FIRST to avoid namespace issues if header is malformed
            echo "#include <cstddef>" > "$${{CC_OUT}}"
            echo "#include <string>" >> "$${{CC_OUT}}"
            echo '#include "{name}.h"' >> "$${{CC_OUT}}"
            echo "" >> "$${{CC_OUT}}"
            echo "{namespace_open}" >> "$${{CC_OUT}}"

            # 3. Iterate over source files
            for src in $(SRCS); do
                # Extract filename without path
                FILENAME=$$(basename "$${{src}}")
                # Extract stem (filename without extension)
                STEM=$$(echo "$${{FILENAME}}" | sed 's/\\.[^.]*$$//')
                # Create C++ identifier safe names
                SAFE_STEM=$$(echo "$${{STEM}}" | sed 's/[^a-zA-Z0-9_]/_/g')
                FUNC_NAME="get_$${{SAFE_STEM}}"
                VAR_NAME="$${{SAFE_STEM}}_data"

                # Header: Add function declaration
                echo "const std::string& $${{FUNC_NAME}}();" >> "$${{HDR_OUT}}"

                # CC: Embed data using xxd
                xxd -i "$${{src}}" | \
                sed -e "s/^unsigned char [^[]*/static const unsigned char $${{VAR_NAME}}/" \
                    -e "s/^unsigned int .*_len/static const size_t $${{VAR_NAME}}_size/" \
                    >> "$${{CC_OUT}}"
                echo "" >> "$${{CC_OUT}}"

                # CC: Define the accessor function
                echo "const std::string& $${{FUNC_NAME}}() {{" >> "$${{CC_OUT}}"
                echo "  static const std::string* const kInstance = new std::string(" >> "$${{CC_OUT}}"
                echo "      reinterpret_cast<const char*>($${{VAR_NAME}}), $${{VAR_NAME}}_size);" >> "$${{CC_OUT}}"
                echo "  return *kInstance;" >> "$${{CC_OUT}}"
                echo "}}" >> "$${{CC_OUT}}"
                echo "" >> "$${{CC_OUT}}"
            done

            # 4. Finish Header File
            echo "{namespace_close}" >> "$${{HDR_OUT}}"
            echo "{namespace_close}" >> "$${{CC_OUT}}"
            echo "#endif  // $${{GUARD}}" >> "$${{HDR_OUT}}"
        """.format(
            name = name,
            guard = name.upper() + "_H_",
            namespace_open = namespace_open,
            namespace_close = namespace_close,
        ),
        compatible_with = compatible_with,
    )

    cc_library(
        name = name,
        srcs = [name + ".cc"],
        hdrs = [name + ".h"],
        compatible_with = compatible_with,
        **kwargs
    )
