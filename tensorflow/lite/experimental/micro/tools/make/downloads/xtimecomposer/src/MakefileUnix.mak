ifeq "$(CONFIG)" "Release"
else
ifeq "$(CONFIG)" "Debug"
else
CONFIG = Release
endif
endif

.PHONY: all build clean 

BINDIR      = $(TOOLS_ROOT)/bin
DLLDIR      = $(TOOLS_ROOT)/lib

ifeq "$(HOST)" "Linux"
else
ifeq "$(HOST)" "Cygwin"
else
HOST = Linux
endif
endif

ifeq "$(HOST)" "Cygwin"
  TOOLPATH    =
  CFLAGS      =  
  CPPFLAGS    =  
  DLLEXT      = .dll
  EXTRALIBS   = -lc
else # Linux
  TOOLPATH    =
  CFLAGS      = -fPIC 
  CPPFLAGS    = -fPIC 
  DLLEXT      = .so
  EXTRALIBS   = -lc -ldl -lstdc++
endif

# Preprocessor defines
DDEFINES := $(foreach flag,$(DEFINES),-D $(flag))

CPP   = $(TOOLPATH)g++ -g $(DDEFINES) $(FLAGS)
CC    = $(TOOLPATH)gcc -g $(DDEFINES) $(FLAGS)
CCPP  = $(TOOLPATH)g++ $(DDEFINES) $(FLAGS)

#*FLAGS_LOCAL is a mechanism so local makefile can enable extra flags
#CFLAGS      += -Wdeclaration-after-statement
ifeq "$(CONFIG)" "Release"
  CFLAGS    += -O3 -Wall -Wsign-compare -Wpointer-arith -Wno-unused-function -Wdeclaration-after-statement $(CFLAGS_LOCAL)
  CPPFLAGS  += -O3 -Wall -Wsign-compare -Wpointer-arith  $(CPPFLAGS_LOCAL)
else # Debug
  CFLAGS    += -g -DDEBUG -Wall -Wsign-compare -Wpointer-arith -Wno-unused-function -Wdeclaration-after-statement $(CFLAGS_LOCAL)
  CPPFLAGS  += -g -DDEBUG -Wall -Wsign-compare -Wpointer-arith $(CPPFLAGS_LOCAL)
endif

# Removes the unitialised stl warnings from the Cygwin builds..
ifeq "$(HOST)" "Cygwin"
  CPPFLAGS += -Wno-uninitialized
endif

