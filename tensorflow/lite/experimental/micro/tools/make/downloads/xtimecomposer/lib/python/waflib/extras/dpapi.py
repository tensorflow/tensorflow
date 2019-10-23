#! /usr/bin/env python
# encoding: utf-8
# Matt Clarkson, 2012

'''
DPAPI access library (http://msdn.microsoft.com/en-us/library/ms995355.aspx)
This file uses code originally created by Crusher Joe:
http://article.gmane.org/gmane.comp.python.ctypes/420
And modified by Wayne Koorts:
http://stackoverflow.com/questions/463832/using-dpapi-with-python
'''

from ctypes import windll, byref, cdll, Structure, POINTER, c_char, c_buffer
from ctypes.wintypes import DWORD
from waflib.Configure import conf

LocalFree = windll.kernel32.LocalFree
memcpy = cdll.msvcrt.memcpy
CryptProtectData = windll.crypt32.CryptProtectData
CryptUnprotectData = windll.crypt32.CryptUnprotectData
CRYPTPROTECT_UI_FORBIDDEN = 0x01
try:
	extra_entropy = 'cl;ad13 \0al;323kjd #(adl;k$#ajsd'.encode('ascii')
except AttributeError:
	extra_entropy = 'cl;ad13 \0al;323kjd #(adl;k$#ajsd'

class DATA_BLOB(Structure):
	_fields_ = [
		('cbData', DWORD),
		('pbData', POINTER(c_char))
	]

def get_data(blob_out):
	cbData = int(blob_out.cbData)
	pbData = blob_out.pbData
	buffer = c_buffer(cbData)
	memcpy(buffer, pbData, cbData)
	LocalFree(pbData)
	return buffer.raw

@conf
def dpapi_encrypt_data(self, input_bytes, entropy = extra_entropy):
	'''
	Encrypts data and returns byte string

	:param input_bytes: The data to be encrypted
	:type input_bytes: String or Bytes
	:param entropy: Extra entropy to add to the encryption process (optional)
	:type entropy: String or Bytes
	'''
	if not isinstance(input_bytes, bytes) or not isinstance(entropy, bytes):
		self.fatal('The inputs to dpapi must be bytes')
	buffer_in      = c_buffer(input_bytes, len(input_bytes))
	buffer_entropy = c_buffer(entropy, len(entropy))
	blob_in        = DATA_BLOB(len(input_bytes), buffer_in)
	blob_entropy   = DATA_BLOB(len(entropy), buffer_entropy)
	blob_out       = DATA_BLOB()

	if CryptProtectData(byref(blob_in), 'python_data', byref(blob_entropy), 
		None, None, CRYPTPROTECT_UI_FORBIDDEN, byref(blob_out)):
		return get_data(blob_out)
	else:
		self.fatal('Failed to decrypt data')

@conf
def dpapi_decrypt_data(self, encrypted_bytes, entropy = extra_entropy):
	'''
	Decrypts data and returns byte string

	:param encrypted_bytes: The encrypted data
	:type encrypted_bytes: Bytes
	:param entropy: Extra entropy to add to the encryption process (optional)
	:type entropy: String or Bytes
	'''
	if not isinstance(encrypted_bytes, bytes) or not isinstance(entropy, bytes):
		self.fatal('The inputs to dpapi must be bytes')
	buffer_in      = c_buffer(encrypted_bytes, len(encrypted_bytes))
	buffer_entropy = c_buffer(entropy, len(entropy))
	blob_in        = DATA_BLOB(len(encrypted_bytes), buffer_in)
	blob_entropy   = DATA_BLOB(len(entropy), buffer_entropy)
	blob_out       = DATA_BLOB()
	if CryptUnprotectData(byref(blob_in), None, byref(blob_entropy), None,
		None, CRYPTPROTECT_UI_FORBIDDEN, byref(blob_out)):
		return get_data(blob_out)
	else:
		self.fatal('Failed to decrypt data')

