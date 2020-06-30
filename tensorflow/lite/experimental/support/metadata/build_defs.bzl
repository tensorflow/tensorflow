"""Build rules to generate metadata schema versions."""

METADATA_SCHEMA_FILE = "//tensorflow/lite/experimental/support/metadata:metadata_schema.fbs"

def stamp_metadata_parser_version(
        name,
        srcs,
        outs):
    """Stamps the latest metadata parser version into the srcs files.

    Replaces all the occurrences of "{LATEST_METADATA_PARSER_VERSION}" in the
    srcs files with the metadata schema version extracted from
    METADATA_SCHEMA_FILE and then outputs the generated file into outs,
    respectively. The number of srcs files needs to match the number of outs
    files.

    Args:
        name: Rule name. (required)
        srcs: List of source files. (required)
        outs: List of output files. (required)
    """
    if len(srcs) != len(outs):
        fail(("The number of srcs files (%d) does not match that of the outs" +
              " files (%d).") %
             (len(srcs), len(outs)))

    for i in range(0, len(srcs)):
        native.genrule(
            name = "%s_file%d" % (name, i),
            srcs = [srcs[i]],
            outs = [outs[i]],
            tools = [METADATA_SCHEMA_FILE],
            # Gets the metadata schema version from the file, and stamps it
            # into the srcs file.
            cmd = "version=$$(sed -n -e '/Schema Semantic version/ s/.*\\: *//p' $(location %s));" %
                  METADATA_SCHEMA_FILE +
                  'sed "s/{LATEST_METADATA_PARSER_VERSION}/$$version/" $< > $@',
        )

    native.filegroup(
        name = name,
        srcs = outs,
    )
