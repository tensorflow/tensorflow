#
# Check the config has been defined
#
!IF "$(CONFIG)" == "Release"
# ok
!ELSEIF "$(CONFIG)" == "Debug"
# ok
!ELSE
CONFIG = Release
!ENDIF

CPP         = @cl.exe
LINK32      = @link.exe

.SUFFIXES: .cpp .obj .a .exe

BINDIR      = $(TOOLS_ROOT)/bin
DLLDIR      = $(TOOLS_ROOT)/lib

#
# Define the flags
#
SHAREDFLAGS = /nologo /Wall /wd4514 /wd4996 /wd4820 /wd4710 /wd4711 /EHac /D "WIN32" /D "_MBCS" /Fo"./" /Fd"./" /FD /c

!IF "$(CONFIG)" == "Release"
CFLAGS    = /MT /O2 /Ob2 $(SHAREDFLAGS)
LINKFLAGS = /incremental:no
!ELSE # Debug
CFLAGS    = /MTd /Gm /GS /Zi /RTC1 /RTCs /RTCu /D "DEBUG" /D "_DEBUG" /FR"./" $(SHAREDFLAGS)
LINKFLAGS = /debug /incremental:yes
!ENDIF

LINK32_LIBS = kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib \
            advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib \
            odbccp32.lib comctl32.lib wsock32.lib rpcrt4.lib

EXE32_FLAGS = $(LINKFLAGS) $(LINK32_LIBS) /nologo /subsystem:console /fixed:no
