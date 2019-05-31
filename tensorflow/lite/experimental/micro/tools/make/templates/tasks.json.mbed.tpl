{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Mbed Config Root",
            "type": "shell",
            "command": "mbed config root .",
        },
        {
            "label": "Mbed Deploy",
            "type": "shell",
            "command": "mbed deploy",
        },
        {
            "label": "Mbed Patch C++11",
            "type": "shell",
            "command": "python",
            "args": [
                "-c",
                "import fileinput, glob;\nfor filename in glob.glob(\"mbed-os/tools/profiles/*.json\"):\n  for line in fileinput.input(filename, inplace=True):\n    print line.replace(\"\\\"-std=gnu++98\\\"\",\"\\\"-std=c++11\\\", \\\"-fpermissive\\\"\")"
            ]
        },
        {
            "label": "Mbed Init",
            "dependsOn": ["Mbed Config Root", "Mbed Deploy", "Mbed Patch C++11"]
        },
        {
            "label": "Mbed build",
            "type": "shell",
            "command": "mbed compile -m auto -t GCC_ARM",
            "group": {
                "kind": "build",
                "isDefault": true
                }
        }
    ]
}