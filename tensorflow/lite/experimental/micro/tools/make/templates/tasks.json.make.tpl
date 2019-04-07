{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Make Build",
            "type": "shell",
            "command": "make",
            "group": {
                "kind": "build",
                "isDefault": true
                }
        }
    ]
}