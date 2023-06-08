# Enable signature checking on pacman
Add-Content -Path C:\tools\msys64\etc\pacman.d\mirrorlist.mingw32 -Value 'SigLevel = Required'
Add-Content -Path C:\tools\msys64\etc\pacman.d\mirrorlist.mingw64 -Value 'SigLevel = Required'
Add-Content -Path C:\tools\msys64\etc\pacman.d\mirrorlist.msys -Value 'SigLevel = Required'

# Install pacman packages.
C:\tools\msys64\usr\bin\bash.exe -lc 'pacman-key --init && pacman-key --populate msys2 && pacman --noconfirm -Syy git curl zip unzip patch'
