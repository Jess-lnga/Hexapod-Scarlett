#pragma once

#include <Arduino.h>

bool follow_line(float xStart,
                 float yStart,
                 float zStart,
                 float xEnd,
                 float yEnd,
                 float zEnd,
                 uint16_t steps);
