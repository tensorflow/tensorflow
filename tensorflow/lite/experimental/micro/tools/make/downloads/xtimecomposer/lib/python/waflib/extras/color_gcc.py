#!/usr/bin/env python
# encoding: utf-8

# Replaces the default formatter by one which understands GCC output and colorizes it.

__author__ = __maintainer__ = "Jérôme Carretero <cJ-waf@zougloub.eu>"
__copyright__ = "Jérôme Carretero, 2012"

import sys
from waflib import Logs

class ColorGCCFormatter(Logs.formatter):
	def __init__(self, colors):
		self.colors = colors
		Logs.formatter.__init__(self)
	def format(self, rec):
		frame = sys._getframe()
		while frame:
			func = frame.f_code.co_name
			if func == 'exec_command':
				cmd = frame.f_locals.get('cmd')
				if isinstance(cmd, list) and ('gcc' in cmd[0] or 'g++' in cmd[0]):
					lines = []
					for line in rec.msg.splitlines():
						if 'warning: ' in line:
							lines.append(self.colors.YELLOW + line)
						elif 'error: ' in line:
							lines.append(self.colors.RED + line)
						elif 'note: ' in line:
							lines.append(self.colors.CYAN + line)
						else:
							lines.append(line)
					rec.msg = "\n".join(lines)
			frame = frame.f_back
		return Logs.formatter.format(self, rec)

def options(opt):
	Logs.log.handlers[0].setFormatter(ColorGCCFormatter(Logs.colors))

