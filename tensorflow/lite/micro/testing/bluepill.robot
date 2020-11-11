*** Settings ***
Suite Setup                   Prepare Tests
Suite Teardown                Teardown
Test Setup                    Reset Emulation
Test Teardown                 Teardown With Custom Message
Resource                      ${RENODEKEYWORDS}

*** Variables ***
${CREATE_SNAPSHOT_ON_FAIL}    False
${UART}                       sysbus.cpu.uartSemihosting
${RESC}                       undefined_RESC
${RENODE_LOG}                 /tmp/renode.log
${UART_LINE_ON_SUCCESS}       ~~~ALL TESTS PASSED~~~
${DIR_WITH_TESTS}             undefined_DIR_WTH_TESTS

*** Keywords ***
Prepare Tests
    [Documentation]           List all binaries with _test suffix and make available from test cases
    Setup
    @{tests} =                List Files In Directory  ${DIR_WITH_TESTS}  pattern=*_test  absolute=True
    Set Suite Variable        @{tests}

Teardown With Custom Message
    [Documentation]           Replace robot fail message with shorter one to avoid duplicated UART output in log
    Set Test Message          ${file} - FAILED
    Test Teardown

Test Binary
    Remove File               ${RENODE_LOG}
    Execute Command           $logfile = @${RENODE_LOG}
    Execute Script            ${RESC}
    Create Terminal Tester    ${UART}  timeout=2
    Start Emulation
    Wait For Line On Uart     ${UART_LINE_ON_SUCCESS}

*** Test Cases ***
Run All Bluepill Tests
    [Documentation]           Runs Bluepill tests and waits for a specific string on the semihosting UART
    FOR  ${TEST}  IN  @{tests}
        Execute Command       Clear
        Execute Command       $bin = @${TEST}
        ${_}  ${file} =       Split Path  ${TEST}
        Set Test Variable     ${file}
        Test Binary
        Log                   \t${file} - PASSED   console=True
    END
