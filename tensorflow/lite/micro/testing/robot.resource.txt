*** Variables ***
${UART}                       sysbus.cpu.uartSemihosting

*** Keywords ***
Teardown With Custom Message
    Test Teardown
    [Documentation]           Replace robot fail message with whole UART output
    ${UART_LOGS}              Get File    ${UART_LOG}
    Set Test Message          UART OUTPUT:\n\n${UART_LOGS}
    Remove File               ${UART_LOG}

Create Platform
    Execute Command           $logfile=@${UART_LOG}
    Execute Script            ${RESC}
    Provides                  ready-platform

Test Binary
    [Arguments]               ${BIN}
    Requires                  ready-platform
    Execute Command           sysbus LoadELF ${BIN}

    Create Terminal Tester    ${UART}  timeout=2
    Start Emulation

    Wait For Line On Uart     ${UART_LINE_ON_SUCCESS}

