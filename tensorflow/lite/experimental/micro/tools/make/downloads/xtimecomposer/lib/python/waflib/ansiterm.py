#!/usr/bin/env python
# encoding: utf-8

"""
Emulate a vt100 terminal in cmd.exe

By wrapping sys.stdout / sys.stderr with Ansiterm,
the vt100 escape characters will be interpreted and
the equivalent actions will be performed with Win32
console commands.

"""

import os, re, sys
from waflib import Utils

wlock = Utils.threading.Lock()

try:
	from ctypes import Structure, windll, c_short, c_ushort, c_ulong, c_int, byref, c_wchar, POINTER, c_long
except ImportError:

	class AnsiTerm(object):
		def __init__(self, stream):
			self.stream = stream
			try:
				self.errors = self.stream.errors
			except AttributeError:
				pass # python 2.5
			self.encoding = self.stream.encoding

		def write(self, txt):
			try:
				wlock.acquire()
				self.stream.write(txt)
				self.stream.flush()
			finally:
				wlock.release()

		def fileno(self):
			return self.stream.fileno()

		def flush(self):
			self.stream.flush()

		def isatty(self):
			return self.stream.isatty()
else:

	class COORD(Structure):
		_fields_ = [("X", c_short), ("Y", c_short)]

	class SMALL_RECT(Structure):
		_fields_ = [("Left", c_short), ("Top", c_short), ("Right", c_short), ("Bottom", c_short)]

	class CONSOLE_SCREEN_BUFFER_INFO(Structure):
		_fields_ = [("Size", COORD), ("CursorPosition", COORD), ("Attributes", c_ushort), ("Window", SMALL_RECT), ("MaximumWindowSize", COORD)]

	class CONSOLE_CURSOR_INFO(Structure):
		_fields_ = [('dwSize', c_ulong), ('bVisible', c_int)]

	try:
		_type = unicode
	except NameError:
		_type = str

	to_int = lambda number, default: number and int(number) or default

	STD_OUTPUT_HANDLE = -11
	STD_ERROR_HANDLE = -12

	windll.kernel32.GetStdHandle.argtypes = [c_ulong]
	windll.kernel32.GetStdHandle.restype = c_ulong
	windll.kernel32.GetConsoleScreenBufferInfo.argtypes = [c_ulong, POINTER(CONSOLE_SCREEN_BUFFER_INFO)]
	windll.kernel32.GetConsoleScreenBufferInfo.restype = c_long
	windll.kernel32.SetConsoleTextAttribute.argtypes = [c_ulong, c_ushort]
	windll.kernel32.SetConsoleTextAttribute.restype = c_long
	windll.kernel32.FillConsoleOutputCharacterW.argtypes = [c_ulong, c_wchar, c_ulong, POINTER(COORD), POINTER(c_ulong)]
	windll.kernel32.FillConsoleOutputCharacterW.restype = c_long
	windll.kernel32.FillConsoleOutputAttribute.argtypes = [c_ulong, c_ushort, c_ulong, POINTER(COORD), POINTER(c_ulong) ]
	windll.kernel32.FillConsoleOutputAttribute.restype = c_long
	windll.kernel32.SetConsoleCursorPosition.argtypes = [c_ulong, POINTER(COORD) ]
	windll.kernel32.SetConsoleCursorPosition.restype = c_long
	windll.kernel32.SetConsoleCursorInfo.argtypes = [c_ulong, POINTER(CONSOLE_CURSOR_INFO)]
	windll.kernel32.SetConsoleCursorInfo.restype = c_long

	class AnsiTerm(object):
		"""
		emulate a vt100 terminal in cmd.exe
		"""
		def __init__(self, s):
			self.stream = s
			try:
				self.errors = s.errors
			except AttributeError:
				pass # python2.5
			self.encoding = s.encoding
			self.cursor_history = []

			handle = (s.fileno() == 2) and STD_ERROR_HANDLE or STD_OUTPUT_HANDLE
			self.hconsole = windll.kernel32.GetStdHandle(handle)

			self._sbinfo = CONSOLE_SCREEN_BUFFER_INFO()

			self._csinfo = CONSOLE_CURSOR_INFO()
			windll.kernel32.GetConsoleCursorInfo(self.hconsole, byref(self._csinfo))

			# just to double check that the console is usable
			self._orig_sbinfo = CONSOLE_SCREEN_BUFFER_INFO()
			r = windll.kernel32.GetConsoleScreenBufferInfo(self.hconsole, byref(self._orig_sbinfo))
			self._isatty = r == 1

		def screen_buffer_info(self):
			"""
			Updates self._sbinfo and returns it
			"""
			windll.kernel32.GetConsoleScreenBufferInfo(self.hconsole, byref(self._sbinfo))
			return self._sbinfo

		def clear_line(self, param):
			mode = param and int(param) or 0
			sbinfo = self.screen_buffer_info()
			if mode == 1: # Clear from beginning of line to cursor position
				line_start = COORD(0, sbinfo.CursorPosition.Y)
				line_length = sbinfo.Size.X
			elif mode == 2: # Clear entire line
				line_start = COORD(sbinfo.CursorPosition.X, sbinfo.CursorPosition.Y)
				line_length = sbinfo.Size.X - sbinfo.CursorPosition.X
			else: # Clear from cursor position to end of line
				line_start = sbinfo.CursorPosition
				line_length = sbinfo.Size.X - sbinfo.CursorPosition.X
			chars_written = c_ulong()
			windll.kernel32.FillConsoleOutputCharacterW(self.hconsole, c_wchar(' '), line_length, line_start, byref(chars_written))
			windll.kernel32.FillConsoleOutputAttribute(self.hconsole, sbinfo.Attributes, line_length, line_start, byref(chars_written))

		def clear_screen(self, param):
			mode = to_int(param, 0)
			sbinfo = self.screen_buffer_info()
			if mode == 1: # Clear from beginning of screen to cursor position
				clear_start = COORD(0, 0)
				clear_length = sbinfo.CursorPosition.X * sbinfo.CursorPosition.Y
			elif mode == 2: # Clear entire screen and return cursor to home
				clear_start = COORD(0, 0)
				clear_length = sbinfo.Size.X * sbinfo.Size.Y
				windll.kernel32.SetConsoleCursorPosition(self.hconsole, clear_start)
			else: # Clear from cursor position to end of screen
				clear_start = sbinfo.CursorPosition
				clear_length = ((sbinfo.Size.X - sbinfo.CursorPosition.X) + sbinfo.Size.X * (sbinfo.Size.Y - sbinfo.CursorPosition.Y))
			chars_written = c_ulong()
			windll.kernel32.FillConsoleOutputCharacterW(self.hconsole, c_wchar(' '), clear_length, clear_start, byref(chars_written))
			windll.kernel32.FillConsoleOutputAttribute(self.hconsole, sbinfo.Attributes, clear_length, clear_start, byref(chars_written))

		def push_cursor(self, param):
			sbinfo = self.screen_buffer_info()
			self.cursor_history.append(sbinfo.CursorPosition)

		def pop_cursor(self, param):
			if self.cursor_history:
				old_pos = self.cursor_history.pop()
				windll.kernel32.SetConsoleCursorPosition(self.hconsole, old_pos)

		def set_cursor(self, param):
			y, sep, x = param.partition(';')
			x = to_int(x, 1) - 1
			y = to_int(y, 1) - 1
			sbinfo = self.screen_buffer_info()
			new_pos = COORD(
				min(max(0, x), sbinfo.Size.X),
				min(max(0, y), sbinfo.Size.Y)
			)
			windll.kernel32.SetConsoleCursorPosition(self.hconsole, new_pos)

		def set_column(self, param):
			x = to_int(param, 1) - 1
			sbinfo = self.screen_buffer_info()
			new_pos = COORD(
				min(max(0, x), sbinfo.Size.X),
				sbinfo.CursorPosition.Y
			)
			windll.kernel32.SetConsoleCursorPosition(self.hconsole, new_pos)

		def move_cursor(self, x_offset=0, y_offset=0):
			sbinfo = self.screen_buffer_info()
			new_pos = COORD(
				min(max(0, sbinfo.CursorPosition.X + x_offset), sbinfo.Size.X),
				min(max(0, sbinfo.CursorPosition.Y + y_offset), sbinfo.Size.Y)
			)
			windll.kernel32.SetConsoleCursorPosition(self.hconsole, new_pos)

		def move_up(self, param):
			self.move_cursor(y_offset = -to_int(param, 1))

		def move_down(self, param):
			self.move_cursor(y_offset = to_int(param, 1))

		def move_left(self, param):
			self.move_cursor(x_offset = -to_int(param, 1))

		def move_right(self, param):
			self.move_cursor(x_offset = to_int(param, 1))

		def next_line(self, param):
			sbinfo = self.screen_buffer_info()
			self.move_cursor(
				x_offset = -sbinfo.CursorPosition.X,
				y_offset = to_int(param, 1)
			)

		def prev_line(self, param):
			sbinfo = self.screen_buffer_info()
			self.move_cursor(
				x_offset = -sbinfo.CursorPosition.X,
				y_offset = -to_int(param, 1)
			)

		def rgb2bgr(self, c):
			return ((c&1) << 2) | (c&2) | ((c&4)>>2)

		def set_color(self, param):
			cols = param.split(';')
			sbinfo = self.screen_buffer_info()
			attr = sbinfo.Attributes
			for c in cols:
				c = to_int(c, 0)
				if 29 < c < 38: # fgcolor
					attr = (attr & 0xfff0) | self.rgb2bgr(c - 30)
				elif 39 < c < 48: # bgcolor
					attr = (attr & 0xff0f) | (self.rgb2bgr(c - 40) << 4)
				elif c == 0: # reset
					attr = self._orig_sbinfo.Attributes
				elif c == 1: # strong
					attr |= 0x08
				elif c == 4: # blink not available -> bg intensity
					attr |= 0x80
				elif c == 7: # negative
					attr = (attr & 0xff88) | ((attr & 0x70) >> 4) | ((attr & 0x07) << 4)

			windll.kernel32.SetConsoleTextAttribute(self.hconsole, attr)

		def show_cursor(self,param):
			self._csinfo.bVisible = 1
			windll.kernel32.SetConsoleCursorInfo(self.hconsole, byref(self._csinfo))

		def hide_cursor(self,param):
			self._csinfo.bVisible = 0
			windll.kernel32.SetConsoleCursorInfo(self.hconsole, byref(self._csinfo))

		ansi_command_table = {
			'A': move_up,
			'B': move_down,
			'C': move_right,
			'D': move_left,
			'E': next_line,
			'F': prev_line,
			'G': set_column,
			'H': set_cursor,
			'f': set_cursor,
			'J': clear_screen,
			'K': clear_line,
			'h': show_cursor,
			'l': hide_cursor,
			'm': set_color,
			's': push_cursor,
			'u': pop_cursor,
		}
		# Match either the escape sequence or text not containing escape sequence
		ansi_tokens = re.compile(r'(?:\x1b\[([0-9?;]*)([a-zA-Z])|([^\x1b]+))')
		def write(self, text):
			try:
				wlock.acquire()
				if self._isatty:
					for param, cmd, txt in self.ansi_tokens.findall(text):
						if cmd:
							cmd_func = self.ansi_command_table.get(cmd)
							if cmd_func:
								cmd_func(self, param)
						else:
							self.writeconsole(txt)
				else:
					# no support for colors in the console, just output the text:
					# eclipse or msys may be able to interpret the escape sequences
					self.stream.write(text)
			finally:
				wlock.release()

		def writeconsole(self, txt):
			chars_written = c_ulong()
			writeconsole = windll.kernel32.WriteConsoleA
			if isinstance(txt, _type):
				writeconsole = windll.kernel32.WriteConsoleW

			# MSDN says that there is a shared buffer of 64 KB for the console
			# writes. Attempt to not get ERROR_NOT_ENOUGH_MEMORY, see waf issue #746
			done = 0
			todo = len(txt)
			chunk = 32<<10
			while todo != 0:
				doing = min(chunk, todo)
				buf = txt[done:done+doing]
				r = writeconsole(self.hconsole, buf, doing, byref(chars_written), None)
				if r == 0:
					chunk >>= 1
					continue
				done += doing
				todo -= doing


		def fileno(self):
			return self.stream.fileno()

		def flush(self):
			pass

		def isatty(self):
			return self._isatty

	if sys.stdout.isatty() or sys.stderr.isatty():
		handle = sys.stdout.isatty() and STD_OUTPUT_HANDLE or STD_ERROR_HANDLE
		console = windll.kernel32.GetStdHandle(handle)
		sbinfo = CONSOLE_SCREEN_BUFFER_INFO()
		def get_term_cols():
			windll.kernel32.GetConsoleScreenBufferInfo(console, byref(sbinfo))
			# Issue 1401 - the progress bar cannot reach the last character
			return sbinfo.Size.X - 1

# just try and see
try:
	import struct, fcntl, termios
except ImportError:
	pass
else:
	if (sys.stdout.isatty() or sys.stderr.isatty()) and os.environ.get('TERM', '') not in ('dumb', 'emacs'):
		FD = sys.stdout.isatty() and sys.stdout.fileno() or sys.stderr.fileno()
		def fun():
			return struct.unpack("HHHH", fcntl.ioctl(FD, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0)))[1]
		try:
			fun()
		except Exception as e:
			pass
		else:
			get_term_cols = fun

