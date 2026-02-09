"""Replacement for http_archive with `exclude` and 'files' parameters."""

def _glob_match(glob_expression, component):
    if "*" not in glob_expression:
        return glob_expression == component

    glob_components = glob_expression.split("*")

    # The first chunk must match the start of the component.
    glob_chunk = glob_components[0]
    if not component.startswith(glob_chunk):
        return False

    # For middle chunks, find the next occurrence.
    pos = len(glob_chunk)
    for i in range(1, len(glob_components) - 1):
        glob_chunk = glob_components[i]
        if glob_chunk:
            idx = component.find(glob_chunk, pos)
            if idx == -1:
                return False
            pos = idx + len(glob_chunk)

    # The last chunk must match the end of the component.
    if not component.endswith(glob_chunk):
        return False

    return True

# The implementation function for our custom rule
def _http_archive_impl(ctx):
    ctx.download_and_extract(
        url = ctx.attr.urls,
        sha256 = ctx.attr.sha256,
        stripPrefix = ctx.attr.strip_prefix,
    )

    # _recursive_delete_builds(ctx, ctx.path("."))

    for path_or_glob in ctx.attr.exclude:
        # Fast-path for specific file.
        if ctx.path(path_or_glob).exists:
            ctx.delete(path_or_glob)

        # Search for file components.
        components = path_or_glob.split("/")

        # Pseudo recursion to walk the directory tree
        # and match against the glob expression.
        parent_stack = [ctx.path(".")]
        component_idx_stack = [0]
        match_any_dir_stack = [False]

        for _ in range(2147483647):
            if not parent_stack:
                # Recursion complete.
                break

            parent = parent_stack.pop()
            component_idx = component_idx_stack.pop()
            match_any_dir = match_any_dir_stack.pop()

            if component_idx == len(components) - 1:
                # Final component, must exactly match child name.
                for child in parent.readdir():
                    if _glob_match(components[component_idx], child.basename):
                        # print("deleting " + str(child))
                        ctx.delete(child)
                    elif child.is_dir and match_any_dir:
                        # Recurse into children.
                        # print("Matching ** to " +  child.basename)
                        parent_stack.append(child)
                        component_idx_stack.append(component_idx)
                        match_any_dir_stack.append(True)

            else:
                # Check for wildcard.
                if components[component_idx] == "**":
                    # print("Found **, retrying " + parent.basename)
                    parent_stack.append(parent)
                    component_idx_stack.append(component_idx + 1)
                    match_any_dir_stack.append(True)

                else:
                    for child in parent.readdir():
                        # Only continue recursively down directories.
                        if child.is_dir:
                            if _glob_match(components[component_idx], child.basename):
                                # print("Matched " + components[component_idx] + " to " + child.basename)
                                parent_stack.append(child)
                                component_idx_stack.append(component_idx + 1)
                                match_any_dir_stack.append(False)
                            elif match_any_dir:
                                # print("Matching ** to " +  child.basename)
                                parent_stack.append(child)
                                component_idx_stack.append(component_idx)
                                match_any_dir_stack.append(True)
                            else:
                                # print(components[component_idx] + " did not match " + child.basename)
                                pass

    for path, label in ctx.attr.files.items():
        src_path = ctx.path(label)

        # On Windows `ctx.symlink` may be implemented as a copy, so the file MUST be watched
        ctx.watch(src_path)
        # ctx.read(src_path)

        if not src_path.exists:
            fail("Input %s does not exist" % label)
        if ctx.path(path).exists:
            ctx.delete(path)
        ctx.symlink(src_path, path)

    return None

# Define the new repository rule
custom_http_archive = repository_rule(
    implementation = _http_archive_impl,
    # Define the attributes for the rule
    attrs = {
        "urls": attr.string_list(
            mandatory = True,
            doc = "List of URLs for the archive. They are tried in order.",
        ),
        "sha256": attr.string(
            mandatory = False,
            doc = "SHA-256 checksum of the downloaded file.",
        ),
        "strip_prefix": attr.string(
            doc = "Directory prefix to strip from extracted files.",
        ),
        "files": attr.string_keyed_label_dict(
            mandatory = False,
            doc = "A map of destination paths (key) to local file labels (value) to overlay/replace.",
        ),
        "exclude": attr.string_list(
            mandatory = False,
            doc = "List of files (or directories) to exclude.",
        ),
    },
)
