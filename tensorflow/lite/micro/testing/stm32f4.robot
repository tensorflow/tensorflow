*** Settings ***
Suite Setup                   Setup
Suite Teardown                Teardown
Test Setup                    Reset Emulation
Test Teardown                 Test Teardown
Resource                      ${RENODEKEYWORDS}

*** Variables ***
${UART}                       sysbus.cpu.uartSemihosting

*** Test Cases ***
Should Run Stm32f4 Test
    [Documentation]           Runs a Stm32f4 test and waits for a specific string on the semihosting UART
    [Tags]                    stm32f4  uart  tensorflow  arm
    ${BIN} =                  Get Environment Variable    BIN
    ${SCRIPT} =               Get Environment Variable    SCRIPT
    ${EXPECTED} =             Get Environment Variable    EXPECTED
    Execute Command           $bin = @${BIN}
    Execute Script            ${SCRIPT}

    Create Terminal Tester    ${UART}  timeout=15
    Start Emulation

    Wait For Line On Uart     ${EXPECTED}
