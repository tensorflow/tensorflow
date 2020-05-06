*** Settings ***
Suite Setup                   Setup
Suite Teardown                Teardown
Test Setup                    Reset Emulation
Resource                      /opt/renode/tests/renode-keywords.robot

*** Variables ***
${UART}                       sysbus.cpu.uartSemihosting

*** Test Cases ***
Should Run Bluepill Test
    [Documentation]           Runs a Bluepill test and waits for a specific string on the semihosting UART
    [Tags]                    bluepill  uart  tensorflow  arm
    ${BIN} =                  Get Environment Variable    BIN
    ${SCRIPT} =               Get Environment Variable    SCRIPT
    ${EXPECTED} =             Get Environment Variable    EXPECTED
    Execute Command           $bin = @${BIN}
    Execute Script            ${SCRIPT}

    Create Terminal Tester    ${UART}  timeout=30
    Start Emulation

    Wait For Line On Uart     ${EXPECTED}
