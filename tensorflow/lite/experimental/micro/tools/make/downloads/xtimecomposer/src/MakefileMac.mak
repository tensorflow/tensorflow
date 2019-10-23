ifeq "$(CONFIG)" "Release"
else
ifeq "$(CONFIG)" "Debug"
else
CONFIG = Release
endif
endif

.PHONY: all build clean 

TOOLPATH    =
CFLAGS      = -fPIC
CPPFLAGS    = -fPIC
EXTRALIBS   = -lc -ldl -lstdc++

# Preprocessor defines
DDEFINES := $(foreach flag,$(DEFINES),-D $(flag))

CPP  = $(TOOLPATH)clang $(DDEFINES) $(FLAGS)
CC   = $(TOOLPATH)gcc $(DDEFINES) $(FLAGS)
CCPP = $(TOOLPATH)clang $(DDEFINES) $(FLAGS)

#*FLAGS_LOCAL is a mechanism so local makefile can enable extra flags
#CFLAGS      += -Wdeclaration-after-statement
ifeq "$(CONFIG)" "Release"
  CFLAGS    += -O3 -Wall -Wsign-compare -Wpointer-arith -Wno-unused-function -Wdeclaration-after-statement $(CFLAGS_LOCAL)
  CPPFLAGS  += -O3 -Wall -Wsign-compare -Wpointer-arith  $(CPPFLAGS_LOCAL)
else # Debug
  CFLAGS    += -g -DDEBUG -Wall -Wsign-compare -Wpointer-arith -Wno-unused-function -Wdeclaration-after-statement $(CFLAGS_LOCAL)
  CPPFLAGS  += -g -DDEBUG -Wall -Wsign-compare -Wpointer-arith $(CPPFLAGS_LOCAL)
endif

BINDIR      = $(TOOLS_ROOT)/bin
DLLDIR      = $(TOOLS_ROOT)/lib
