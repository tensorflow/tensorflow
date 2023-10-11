FROM mcr.microsoft.com/dotnet/framework/sdk:4.8-windowsservercore-ltsc2019

# Set default powershell policy for this script (ProgressPreference='SilentlyContinue' makes
# downloads with Invoke-WebRequest not show the progress bar and is MUCH faster).
SHELL ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $VerbosePreference = 'Continue';"]

# Workaround for networking (b/112379377) was closed as won't fix for MTU setting.
# Remaining lines handle making the metadata server on the VM accessible inside docker.
RUN Get-NetAdapter | Where-Object Name -like "*Ethernet*" | ForEach-Object { \
        & netsh interface ipv4 set subinterface $_.InterfaceIndex mtu=1460 store=persistent }; \
    $gateway = (Get-NetRoute | Where { $_.DestinationPrefix -eq \"0.0.0.0/0\" } | Sort-Object RouteMetric \
        | Select NextHop).NextHop; \
    $ifIndex = (Get-NetAdapter -InterfaceDescription \"Hyper-V Virtual Ethernet*\" | Sort-Object \
        | Select ifIndex).ifIndex; \
    New-NetRoute -DestinationPrefix 169.254.169.254/32 -InterfaceIndex $ifIndex -NextHop $gateway

# Enable Long Paths for Win32 File/Folder APIs.
RUN New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem \
        -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force

# Install Visual C++ Redistributable for Visual Studio 2015-2022.
RUN New-Item -Path "C:/" -Name "TEMP" -ItemType "directory"; \
    Invoke-WebRequest "https://aka.ms/vs/17/release/vc_redist.x64.exe" \
        -OutFile C:/TEMP/vc_redist.x64.exe -UseBasicParsing; \
    Start-Process -filepath C:/TEMP/vc_redist.x64.exe -ArgumentList '/install', '/passive', '/norestart' -Wait; \
    Remove-Item C:/TEMP/vc_redist.x64.exe

# Install Visual Studio 2022 Build Tools. Install ManagedDesktopBuildTools separately to ensure all Optional workloads are installed too.
RUN Invoke-WebRequest "https://aka.ms/vs/17/release/vs_buildtools.exe" \
        -OutFile C:/TEMP/vs_buildtools.exe -UseBasicParsing; \
    Start-Process -FilePath C:/TEMP/vs_buildtools.exe -ArgumentList "--installPath", "C:/VS", \
        "--quiet", "--wait", "--nocache", \
        "--add", "Microsoft.VisualStudio.Workload.VCTools", \
        "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", \
        "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041" -Wait; \
    Start-Process -FilePath C:/TEMP/vs_buildtools.exe -ArgumentList "--installPath", "C:/VS", \
        "--quiet", "--wait", "--nocache", "--includeOptional", \
        "--add", "Microsoft.VisualStudio.Workload.ManagedDesktopBuildTools" -Wait; \
    Remove-Item C:/TEMP/vs_buildtools.exe; \
    [Environment]::SetEnvironmentVariable(\"BAZEL_VC\", \"C:\VS\VC\", \"Machine\"); \
    $old_path = [Environment]::GetEnvironmentVariable(\"PATH\", \"Machine\"); \
    [Environment]::SetEnvironmentVariable(\"PATH\", $old_path + \";C:\VS\VC\Tools\MSVC\14.33.31629\bin\Hostx64\x64;C:\VS\Common7\Tools;C:\VS\MSBuild\Current\Bin\", \"Machine\");

# Add signtool.exe to the PATH. Note this path may need to be edited if updates
# are made to the Windows 10 SDK.
RUN $old_path = [Environment]::GetEnvironmentVariable(\"PATH\", \"Machine\"); \
    [Environment]::SetEnvironmentVariable(\"PATH\", $old_path + \";C:\Program Files (x86)\Windows Kits\10\App Certification Kit\", \"Machine\");

# Install WiX toolset (v4) - Necessary for MSI Installer/Signing builds
RUN dotnet tool install --global wix

# Install msys2, packages and add to path.
RUN [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; \
    Invoke-WebRequest "https://repo.msys2.org/distrib/x86_64/msys2-base-x86_64-20220319.sfx.exe" \
        -OutFile msys2_install.exe -UseBasicParsing; \
    .\msys2_install.exe -y -oC:\; \
    Remove-Item msys2_install.exe; \
    function msys() { C:\msys64\usr\bin\bash.exe @('-lc') + @Args; } \
    msys ' '; \
    msys 'pacman --noconfirm -Syy bsdcpio bsdtar bzip2'; \
    msys 'pacman --noconfirm -Syy coreutils curl dash file filesystem findutils'; \
    msys 'pacman --noconfirm -Syy flex gawk gcc-libs grep gzip inetutils info'; \
    msys 'pacman --noconfirm -Syy less lndir mintty ncurses pactoys-git patch'; \
    msys 'pacman --noconfirm -Syy pax-git pkgfile rebase sed tar tftp-hpa time tzcode util-linux which'; \
    $old_path = [Environment]::GetEnvironmentVariable(\"PATH\", \"Machine\"); \
    [Environment]::SetEnvironmentVariable(\"PATH\", $old_path + \";C:\msys64;C:\msys64\usr\bin\", \"Machine\");

# Install Go 1.19.1
RUN Invoke-WebRequest "https://go.dev/dl/go1.19.1.windows-amd64.msi" \
        -OutFile C:/TEMP/go_install.msi -UseBasicParsing; \
    Start-Process C:/TEMP/go_install.msi -ArgumentList "/quiet", "/log", "C:/TEMP/go_install_log.txt", \
        "InstallAllUsers=1", "PrependPath=1" -wait; \
    Remove-Item C:/TEMP/go_install.msi; \
    Remove-Item C:/TEMP/go_install_log.txt

# Install Python 3.
RUN Invoke-WebRequest "https://www.python.org/ftp/python/3.10.4/python-3.10.4-amd64.exe" \
        -OutFile C:/TEMP/python_install.exe -UseBasicParsing; \
    Start-Process C:/TEMP/python_install.exe -ArgumentList "/quiet", "/log", "C:/TEMP/python_install_log.txt", \
        "InstallAllUsers=1", "PrependPath=1" -wait; \
    Remove-Item C:/TEMP/python_install.exe; \
    Remove-Item C:/TEMP/python_install_log.txt

# Install JDK 17
RUN Add-Type -AssemblyName "System.IO.Compression.FileSystem"; \
    $zulu_url = \"https://cdn.azul.com/zulu/bin/zulu17.32.13-ca-jdk17.0.2-win_x64.zip\"; \
    $zulu_zip = \"c:/temp/jdk_install.zip\"; \
    $zulu_extracted_path = \"c:/temp/\" + [IO.Path]::GetFileNameWithoutExtension($zulu_url); \
    $zulu_root = \"c:/openjdk\"; \
    (New-Object Net.WebClient).DownloadFile($zulu_url, $zulu_zip); \
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zulu_zip, \"c:/temp\"); \
    Move-Item $zulu_extracted_path -Destination $zulu_root; \
    Remove-Item $zulu_zip; \
    $old_path = [Environment]::GetEnvironmentVariable(\"PATH\", \"Machine\"); \
    [Environment]::SetEnvironmentVariable(\"PATH\", $old_path + \";${zulu_root}\bin\", \"Machine\"); \
    [Environment]::SetEnvironmentVariable(\"JAVA_HOME\", $zulu_root, \"Machine\")

# Install gcloud (install.bat installs directly into bin folder of extracted zip contents)
# Install needed gcloud components
RUN Add-Type -AssemblyName "System.IO.Compression.FileSystem"; \
    $pkg_url = \"https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-396.0.0-windows-x86_64.zip\"; \
    $pkg_zip = \"c:/temp/gcloud.zip\"; \
    $pkg_extracted_path = \"c:/google-cloud-sdk\"; \
    (New-Object Net.WebClient).DownloadFile($pkg_url, $pkg_zip); \
    [System.IO.Compression.ZipFile]::ExtractToDirectory($pkg_zip, \"c:/\"); \
    Start-Process cmd.exe -ArgumentList "/c", "/s", "$pkg_extracted_path/install.bat", "-q" -wait; \
    Remove-Item $pkg_zip; \
    $old_path = [Environment]::GetEnvironmentVariable(\"PATH\", \"Machine\"); \
    [Environment]::SetEnvironmentVariable(\"PATH\", $old_path + \";${pkg_extracted_path}\bin\", \"Machine\"); \
    $env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine'); \
    gcloud components install docker-credential-gcr kubectl gsutil;

# Install cygwin and packages
# Running a seperate ps1 file since when running inside a Dockerfile, it does
# not work.
COPY install/install_cygwin.ps1 c:/
RUN c:/install_cygwin.ps1; \
    $old_path = [Environment]::GetEnvironmentVariable(\"PATH\", \"Machine\"); \
    [Environment]::SetEnvironmentVariable(\"PATH\", $old_path + \";C:\Cygwin64\bin\", \"Machine\");
RUN Remove-Item c:/install_cygwin.ps1

# Install Chocolatey and packages
RUN Invoke-Expression ((New-Object Net.WebClient).DownloadString('https://chocolatey.org/install.ps1')); \
    $env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine'); \
    choco feature enable -n allowGlobalConfirmation; \
    choco install 7zip; \
    choco install 7zip.install; \
    choco install 7zip.portable; \
    choco install anaconda2 --version 5.0.1; \
    choco install anaconda3 --version 5.0.1; \
    choco install android-sdk --version 25.2.3.1; \
    choco install AndroidStudio --version 3.0.1.0; \
    choco install ant --version 1.10.1; \
    choco install ccleaner; \
    choco install chocolatey; \
    choco install chocolatey-core.extension; \
    choco install chocolatey-visualstudio.extension; \
    choco install chocolatey-windowsupdate.extension; \
    choco install cmake.install; \
    choco install dotnetcore-sdk; \
    choco install git; \
    choco install git.install; \
    choco install GoogleChrome; \
    choco install gradle --version 4.4.1; \
    choco install jdk8; \
    choco install KB2533623; \
    choco install KB2919355; \
    choco install KB2919442; \
    choco install KB2999226; \
    choco install KB3033929; \
    choco install KB3035131; \
    choco install maven; \
    choco install ninja; \
    choco install nodejs --version 9.3.0; \
    choco install nodejs.install --version 9.3.0; \
    choco install nuget.commandline; \
    choco install openjdk11; \
    choco install peazip; \
    choco install peazip.install; \
    choco install peazip.portable; \
    choco install php --version 7.2.0; \
    choco install protoc --version 3.2.0; \
    choco install ruby --version 2.5.0.1; \
    choco install swig --version 3.0.9; \
    choco install sysinternals; \
    choco install unrar; \
    choco install unzip; \
    choco install vcredist140; \
    choco install vcredist2015; \
    choco install vim; \
    choco install winrar; \
    choco install zip; \
    choco install Firefox; \
    choco install iisexpress;

RUN cmd /c 'mklink /J c:\Anaconda c:\tools\anaconda2';
RUN cmd /c 'mklink c:\programdata\chocolatey\bin\rar.exe \"c:\program files\winrar\rar.exe\"';

# Installing pip packages
RUN pip install --upgrade setuptools; \
    pip install altgraph appdirs cachetools certifi cffi chardet colorama \
    cryptography cycler Cython decorator google-api-python-client \
    google-auth google-auth-httplib2 grpcio httplib2 idna ipython-genutils \
    kiwisolver macholib matplotlib nose numpy packaging pandas pickleshare pip \
    prompt-toolkit protobuf psutil pyasn1 pyasn1-modules pycparser Pygments \
    pyparsing pyreadline python-dateutil pytz pywin32 requests rsa setuptools \
    simplegeneric six Tempita traitlets uritemplate urllib3 virtualenv wcwidth \
    wheel win-unicode-console;

# Hardcoding Android license since I did not find any solution on accepting it
# through the docker build command. If the licensing agreement changes, this 
# will need to be updated as well.
RUN New-Item -ItemType Directory -Path C:\Android\android-sdk\licenses; \
    Set-Content -Path .\Android\android-sdk\licenses\android-sdk-license -Value "`n24333f8a63b6825ea9c5514f83c2829b004d1fee" -NoNewLine;

# Add sdkmanager to PATH
RUN $old_path = [Environment]::GetEnvironmentVariable(\"PATH\", \"Machine\"); \
    [Environment]::SetEnvironmentVariable(\"PATH\", $old_path + \";C:\Android\android-sdk\tools\bin\", \"Machine\");

# Install android packages
RUN $env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine'); \
    New-Item C:\Users\ContainerAdministrator\.android\repositories.cfg; \
    sdkmanager 'ndk-bundle'; \
    sdkmanager 'platforms;android-33'; \
    sdkmanager 'add-ons;addon-google_apis-google-24'; \
    sdkmanager 'cmake;3.10.2.4988404'; \
    sdkmanager 'cmake;3.18.1'; \
    sdkmanager 'cmake;3.22.1'; \
    sdkmanager 'cmake;3.6.4111459'; \
    sdkmanager 'emulator'; \
    sdkmanager 'system-images;android-27;google_apis;x86'; \
    sdkmanager 'sources;android-27'; \
    sdkmanager 'extras;google;Android_Emulator_Hypervisor_Driver'; \
    sdkmanager 'extras;google;auto'; \
    sdkmanager 'extras;google;google_play_services'; \
    sdkmanager 'extras;google;instantapps'; \
    sdkmanager 'extras;google;m2repository'; \
    sdkmanager 'extras;google;market_apk_expansion'; \
    sdkmanager 'extras;google;market_licensing'; \
    sdkmanager 'extras;google;simulators'; \
    sdkmanager 'extras;google;usb_driver'; \
    sdkmanager 'extras;google;webdriver'; \
    sdkmanager 'extras;android;m2repository'; \
    sdkmanager 'extras;intel;Hardware_Accelerated_Execution_Manager'; \
    sdkmanager 'extras;m2repository;com;android;support;constraint;constraint-layout;1.0.0'; \
    sdkmanager 'extras;m2repository;com;android;support;constraint;constraint-layout-solver;1.0.2'; \
    sdkmanager 'patcher;v4'; \
    sdkmanager 'ndk;25.1.8937393'; \
    sdkmanager 'build-tools;27.0.3';

# Install Scoop and packages
RUN iex \"& {$(irm get.scoop.sh)} -RunAsAdmin\"; \
    scoop install perl; \
    scoop install bazel; \
    scoop install cuda; \
    scoop install azure-functions-core-tools; \
    scoop install azure-cli;

# Setting environment variables
RUN [Environment]::SetEnvironmentVariable('CYGWIN', 'winsymlinks:native', 'Machine'); \
    [Environment]::SetEnvironmentVariable('HOME', 'C:\Users\ContainerAdministrator\', 'Machine'); \
    [Environment]::SetEnvironmentVariable('HOMEDRIVE', 'C:', 'Machine'); \
    [Environment]::SetEnvironmentVariable('HOMEPATH', '\Users\ContainerAdministrator\', 'Machine'); \
    [Environment]::SetEnvironmentVariable('GOROOT', 'C:\Program Files\Go\', 'Machine'); \
    [Environment]::SetEnvironmentVariable('KOKORO_POSIX_ROOT', '/tmpfs', 'Machine'); \
    [Environment]::SetEnvironmentVariable('KOKORO_ROOT', 'T:\', 'Machine'); \
    [Environment]::SetEnvironmentVariable('SHELL', '/bin/bash', 'Machine'); \
    $old_path = [Environment]::GetEnvironmentVariable(\"PATH\", \"Machine\"); \
    [Environment]::SetEnvironmentVariable(\"PATH\", $old_path + \";C:\Program Files\CMake\bin\", \"Machine\");


# Restore default shell for Windows containers.
SHELL ["cmd.exe", "/s", "/c"]

# Default to PowerShell if no other command specified.
CMD ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]
