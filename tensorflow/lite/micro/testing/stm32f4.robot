*** Settings ***
Suite Setup                   Setup
Suite Teardown                Teardown
Test Setup                    Reset Emulation
Resource                      ${RENODE_KEYWORDS}

*** Variables ***
${UART}                       sysbus.cpu.uartSemihosting

*** Test Cases ***
Should Run Stm32f4 Test
    [Documentation]           Runs a Stm32f4 test and waits for a specific string on the semihosting UART
    [Tags]                    stm32f4  uart  tensorflow  arm
    Execute Command           $bin = @${BIN}
    Execute Script            ${SCRIPT}

    Create Terminal Tester    ${UART}  timeout=${TIMEOUT}
    Start Emulation

    Wait For Line On Uart     ${EXPECTED}
