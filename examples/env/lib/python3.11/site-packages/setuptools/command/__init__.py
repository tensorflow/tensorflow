from distutils.command.bdist import bdist
import sys

if 'egg' not in bdist.format_commands:
    try:
        bdist.format_commands['egg'] = ('bdist_egg', "Python .egg file")
    except TypeError:
        # For backward compatibility with older distutils (stdlib)
        bdist.format_command['egg'] = ('bdist_egg', "Python .egg file")
        bdist.format_commands.append('egg')

del bdist, sys
