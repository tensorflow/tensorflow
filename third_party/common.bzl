# Rule for simple expansion of template files. This performs a simple
# search over the template file for the keys in substitutions,
# and replaces them with the corresponding values.
#
# Typical usage:
#   load("/tools/build_rules/expand_header_template", "expand_header_template")
#   expand_header_template(
#       name = "ExpandMyTemplate",
#       template = "my.template",
#       out = "my.txt",
#       substitutions = {
#         "$VAR1": "foo",
#         "$VAR2": "bar",
#       }
#   )
#
# Args:
#   name: The name of the rule.
#   template: The template file to expand
#   out: The destination of the expanded file
#   substitutions: A dictionary mapping strings to their substitutions

def expand_header_template_impl(ctx):
  ctx.template_action(
      template = ctx.file.template,
      output = ctx.outputs.out,
      substitutions = ctx.attr.substitutions,
  )

expand_header_template = rule(
    implementation = expand_header_template_impl,
    attrs = {
        "template": attr.label(mandatory=True, allow_files=True, single_file=True),
        "substitutions": attr.string_dict(mandatory=True),
        "out": attr.output(mandatory=True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
)
