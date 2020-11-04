*** Settings ***
Suite Setup                   Prepare Tests
Suite Teardown                Teardown
Test Setup                    Reset Emulation
Test Teardown                 Teardown With Custom Message
Resource                      ${RENODEKEYWORDS}

*** Variables ***
${CREATE_SNAPSHOT_ON_FAIL}    False
${UART}                       sysbus.cpu.uartSemihosting

*** Keywords ***
Prepare Tests
    [Documentation]           Make environment variables avaiable in whole test suite and list files in ${BIN_DIR}
    Setup
    ${SCRIPT} =               Get Environment Variable    SCRIPT
    ${LOGFILE} =              Get Environment Variable    LOGFILE
    ${EXPECTED} =             Get Environment Variable    EXPECTED
    Set Suite Variable        ${SCRIPT}
    Set Suite Variable        ${EXPECTED}
    Set Suite Variable        ${LOGFILE}
    List All Test Binaries

Teardown With Custom Message
    [Documentation]           Replace robot fail message with shorter one to avoid duplicated UART output in log
    Set Test Message          ${file} - FAILED
    Test Teardown

List All Test Binaries
    [Documentation]           List all files in ${BIN_DIR} and make it available from test cases
    Setup
    ${BIN_DIR} =              Get Environment Variable    BIN_DIR
    @{binaries} =             List Files In Directory     ${BIN_DIR}   absolute=True
    Set Suite Variable        @{binaries}

Test Binary
    Remove File               ${LOGFILE}
    Execute Command           $logfile = @${LOGFILE}
    Execute Script            ${SCRIPT}

    Create Terminal Tester    ${UART}  timeout=2
    Start Emulation

    Wait For Line On Uart     ${EXPECTED}

*** Test Cases ***
Should Run All Bluepill Tests
    [Documentation]           Runs Bluepill tests and waits for a specific string on the semihosting UART
    [Tags]                    bluepill  uart  tensorflow  arm
    FOR  ${BIN}  IN  @{binaries}
        Execute Command       $bin = @${BIN}
        ${_}  ${file} =       Split Path  ${BIN}
        Set Test Variable     ${file}
        Test Binary
        Execute Command       Clear

        Log                   \t${file} - PASSED   console=True
    END
