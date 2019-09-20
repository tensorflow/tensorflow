set buildtype=Release
if "%1"=="-b" set buildtype=%2

..\%buildtype%\flatc.exe --lua -I include_test monster_test.fbs

lua53.exe luatest.lua