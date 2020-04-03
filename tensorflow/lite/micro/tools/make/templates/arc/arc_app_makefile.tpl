#=============================================================
# OS-specific definitions
#=============================================================
COMMA=,
OPEN_PAREN=(
CLOSE_PAREN=)
BACKSLASH=\$(nullstring)
ifneq ($(ComSpec)$(COMSPEC),)
    O_SYS=Windows
    RM=del /F /Q
    MKDIR=mkdir 
    CP=copy /Y
    TYPE=type
    PS=$(BACKSLASH)
    Q=
    coQ=\$(nullstring)
    fix_platform_path = $(subst /,$(PS), $(1))
    DEV_NULL = nul
else
    O_SYS=Unix
    RM=rm -rf
    MKDIR=mkdir -p
    CP=cp 
    TYPE=cat
    PS=/
    Q=$(BACKSLASH)
    coQ=
    fix_platform_path=$(1)
    DEV_NULL=/dev/null
endif

# Note: Windows escaping rules is very combersome 
# initially I tried to use Q=^, but this depends on the context and (looks like) on Win version.
# Also expecially ugly thing is that in quoted strings the quotes the same are remain.
# Batch has special parameter expansion syntax to remove quotes,
# but many tools themselves remove quotes (unless escaped with backslash)
# So finally we've found that in our use cases we may not escaping any symbols but prepend backslashes before quotes.

quote=$(subst %,$(Q)%, \
      $(subst &,$(Q)&, \
      $(subst <,$(Q)<, \
      $(subst >,$(Q)>, \
      $(subst |,$(Q)|, \
      $(subst ',$(Q)', \
      $(subst $(COMMA),$(Q)$(COMMA), \
      $(subst =,$(Q)=, \
      $(subst $(OPEN_PAREN),$(Q)$(OPEN_PAREN), \
      $(subst $(CLOSE_PAREN),$(Q)$(CLOSE_PAREN), \
      $(subst !,$(Q)!, \
      $(subst ",$(BACKSLASH)", \
      $(subst $(Q),$(Q)$(Q), \
      $(1) )))))))))))))

#=============================================================
# Toolchain definitions
#=============================================================
CC = %{CC}%
CXX = %{CXX}%
LD = %{LD}%


#=============================================================
# Applications settings
#=============================================================
OUT_NAME = %{EXECUTABLE}%

DBG_ARGS ?= 

RUN_ARGS ?= 

EXT_CFLAGS ?=

CXXFLAGS += %{CXX_FLAGS}%

CCFLAGS += %{CC_FLAGS}%

LDFLAGS += %{LINKER_FLAGS}%

%{EXTRA_APP_SETTINGS}%


#=============================================================
# Files and directories
#=============================================================
SRCS := \
%{SRCS}%

OBJS := \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SRCS)))


#=============================================================
# Common rules
#=============================================================
.PHONY: all app flash clean run debug

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(EXT_CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(EXT_CFLAGS) $(INCLUDES) -c $< -o $@

$(OUT_NAME): $(OBJS)
	$(LD) $(CXXFLAGS) -o $@ -Ccrossref $(OBJS) $(LDFLAGS)

%{EXTRA_APP_RULES}%


#=================================================================
# Global rules
#=================================================================
all: $(OUT_NAME)

app: $(OUT_NAME)

flash: %{BIN_DEPEND}%
%{BIN_RULE}%

clean: 
	-@$(RM) $(call fix_platform_path,$(OBJS))
	-@$(RM) $(OUT_NAME) %{EXTRA_RM_TARGETS}%

#=================================================================
# Execution rules
#=================================================================

APP_RUN := %{APP_RUN_CMD}%
APP_DEBUG := %{APP_DEBUG_CMD}%

run: $(OUT_NAME)
	$(APP_RUN) $(OUT_NAME) $(RUN_ARGS)

debug: $(OUT_NAME)
	$(APP_DEBUG) $(OUT_NAME) $(RUN_ARGS)

%{EXTRA_EXECUTE_RULES}%
