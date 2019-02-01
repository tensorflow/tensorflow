
# Reverses a space-separated list of words.
reverse = $(if $(1),$(call reverse,$(wordlist 2,$(words $(1)),$(1)))) $(firstword $(1))

# Look for platform or target-specific implementation files to replace reference
# implementations with, given a tag. These are expected to occur in subfolders
# of a directory where a reference implementation exists, and have the same
# interface and header file. For example,
# tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc
# defines a module for supplying audio data, but since no platform or OS can be
# presumed, it just always returns zeroes for its samples. The MacOS-specific
# tensorflow/lite/experimental/micro/examples/micro_speech/osx/audio_provider.cc
# has an implementation that relies on CoreAudio, and there are equivalent
# versions for other operating systems.
# The specific implementation yielded by the first tag in the list that produces
# a match is returned, else the reference version if none of the tags produce a
# match.
# All lists of source files are put through this substitution process with the
# tags of their target OS and architecture, so that implementations can be added
# by simply placing them in the file tree, with no changes to the build files
# needed.
# One confusing thing about this implementation is that we're using wildcard to
# act as a 'does file exist?' function, rather than expanding an expression.
# Wildcard will return an empty string if given a plain file path with no actual
# wildcards, if the file doesn't exist, so taking the first word of the list
# between that and the reference path will pick the specialized one if it's
# available.
substitute_specialized_implementation = \
  $(firstword $(wildcard $(dir $(1))$(2)/$(notdir $(1))) $(wildcard $(1)))
substitute_specialized_implementations = \
  $(foreach source,$(1),$(call substitute_specialized_implementation,$(source),$(2)))
# Here we're first looking for specialized implementations in ref_dir/$(TAG1)
# and then ref_dir/$(TAG2), etc, before falling back to ref_dir's
# implementation.
# The argument to this function should be a list of space-separated file paths,
# with any wildcards already expanded.
define specialize_on_tags
$(if $(2),$(call substitute_specialized_implementations,$(call specialize_on_tags,$(1),$(wordlist 2,$(words $(2)),$(2))),$(firstword $(2))),$(1))
endef
# The entry point that most targets should use to find implementation-specific
# versions of their source files. The only argument is a list of file paths.
specialize = $(call specialize_on_tags,$(1),$(strip $(call reverse,$(ALL_TAGS))))

# Creates a set of rules to build a standalone makefile project for an
# executable, including all of the source and header files required in a
# separate folder and a simple makefile.
# Arguments are:
# 1 - Project type (make, mbed, etc).
# 2 - Project file template name.
# 3 - Name of executable.
# 4 - List of C/C++ source files needed to build the target.
# 5 - List of C/C++ header files needed to build the target.
# 6 - Linker flags required.
# 7 - C++ compilation flags needed.
# Calling eval on the output will create a <Name>_makefile target that you
# can invoke to create the standalone project.
define generate_project
$(PRJDIR)$(3)/$(1)/%: %
	@mkdir -p $$(dir $$@)
	cp $$< $$@

$(PRJDIR)$(3)/$(1)/third_party/%: tensorflow/lite/experimental/micro/tools/make/downloads/%
	@mkdir -p $$(dir $$@)
	cp $$< $$@

$(PRJDIR)$(3)/$(1)/%: tensorflow/lite/experimental/micro/tools/make/templates/%.tpl
	@mkdir -p $$(dir $$@)
	sed -E 's#\%\{SRCS\}\%#$(4)#g' $$< | \
	sed -E 's#\%\{EXECUTABLE\}\%#$(3)#g' | \
	sed -E 's#\%\{LINKER_FLAGS\}\%#$(6)#g' | \
	sed -E 's#\%\{CXX_FLAGS\}\%#$(7)#g' > $$@

generate_$(3)_$(1)_project: $(addprefix $(PRJDIR)$(3)/$(1)/, $(4) $(5) $(2))
endef

# Specialized version of generate_project for TF Lite Micro test targets that
# automatically includes standard library files, so you just need to pass the
# test name and any extra source files required.
# Arguments are:
# 1 - Name of test.
# 2 - C/C++ source files implementing the test.
# 3 - C/C++ header files needed for the test.
# Calling eval on the output will create targets that you can invoke to
# generate the standalone project.
define generate_microlite_projects
$(call generate_project,make,$(MAKE_PROJECT_FILES),$(1),$(MICROLITE_CC_SRCS) $(THIRD_PARTY_CC_SRCS) $(2),$(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(MICROLITE_TEST_HDRS) $(3),$(MICROLITE_LIBS),$(CXXFLAGS))
$(call generate_project,mbed,$(MBED_PROJECT_FILES),$(1),$(MICROLITE_CC_SRCS) $(THIRD_PARTY_CC_SRCS) $(2),$(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(MICROLITE_TEST_HDRS) $(3),$(MICROLITE_LIBS),$(CXXFLAGS))
endef


# Handles the details of generating a binary target, including specializing
# for the current platform, and generating project file targets.
# Arguments are:
# 1 - Name of test.
# 2 - C/C++ source files implementing the test.
# 3 - C/C++ header files needed for the test.
# Calling eval on the output will create the targets that you need.
define microlite_test
$(1)_LOCAL_SRCS := $(2)
$(1)_LOCAL_SRCS := $$(call specialize,$$($(1)_LOCAL_SRCS))
ALL_SRCS += $$($(1)_LOCAL_SRCS)
$(1)_LOCAL_HDRS := $(3)
$(1)_LOCAL_OBJS := $$(addprefix $$(OBJDIR), \
$$(patsubst %.cc,%.o,$$(patsubst %.c,%.o,$$($(1)_LOCAL_SRCS))))
$(1)_BINARY := $$(BINDIR)$(1)
$$($(1)_BINARY): $$($(1)_LOCAL_OBJS) $$(MICROLITE_LIB_PATH)
	@mkdir -p $$(dir $$@)
	$$(CXX) $$(CXXFLAGS) $$(INCLUDES) \
	-o $$($(1)_BINARY) $$($(1)_LOCAL_OBJS) \
	$$(LIBFLAGS) $$(MICROLITE_LIB_PATH) $$(LDFLAGS) $$(MICROLITE_LIBS)
$(1): $$($(1)_BINARY)
$(1)_bin: $$($(1)_BINARY).bin
test_$(1): $$($(1)_BINARY)
	$$(TEST_SCRIPT) $$($(1)_BINARY) '~~~ALL TESTS PASSED~~~'
ifneq (,$(findstring _test,$(1)))
  MICROLITE_TEST_TARGETS += test_$(1)
endif
$(eval $(call generate_microlite_projects,$(1),$(call specialize,$(2)),$(3)))
endef
