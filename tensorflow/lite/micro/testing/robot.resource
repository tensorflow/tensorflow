*** Variables ***
${UART}                       sysbus.cpu.uartSemihosting

*** Keywords ***
Prepare Tests
    [Documentation]           Make environment variables avaiable in whole test suite and list files in ${BIN_DIR}
    Setup
    ${SCRIPT} =               Get Environment Variable    SCRIPT
    ${LOGFILE} =              Get Environment Variable    LOGFILE
    ${EXPECTED} =             Get Environment Variable    EXPECTED
    Set Global Variable       ${SCRIPT}
    Set Global Variable       ${EXPECTED}
    Set Global Variable       ${LOGFILE}

Teardown With Custom Message
    Test Teardown
    [Documentation]           Replace robot fail message with whole UART output
    ${UART_LOGS}              Get File    ${LOGFILE}
    Set Test Message          UART OUTPUT:\n${UART_LOGS}

Test Binary
    [Arguments]               ${BIN}
    Remove File               ${LOGFILE}
    Execute Command           $logfile = @${LOGFILE}
    Execute Command           $bin = ${BIN}
    Execute Script            ${SCRIPT}

    Create Terminal Tester    ${UART}  timeout=2
    Start Emulation

    Wait For Line On Uart     ${EXPECTED}

