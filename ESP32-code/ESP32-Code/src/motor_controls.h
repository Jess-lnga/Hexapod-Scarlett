#pragma once

#include <Arduino.h>

void begin_motors(HardwareSerial& serialPort, int8_t rxPin, int8_t txPin, uint32_t baudRate);
bool motors_initialized();

bool set_leg_pos(float x, float y, float z, uint16_t moveTimeMs = 500);
