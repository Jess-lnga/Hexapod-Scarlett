#include "leg_control.h"

#include "motor_controls.h"

static const uint16_t kLineMoveTimeMs = 100;

bool follow_line(float xStart,
                 float yStart,
                 float zStart,
                 float xEnd,
                 float yEnd,
                 float zEnd,
                 uint16_t steps) {
  if (steps == 0) {
    return false;
  }

  if (steps == 1) {
    return set_leg_pos(xEnd, yEnd, zEnd, kLineMoveTimeMs);
  }

  bool success = true;
  const float stepCount = static_cast<float>(steps - 1);

  for (uint16_t i = 0; i < steps; ++i) {
    const float t = static_cast<float>(i) / stepCount;
    const float x = xStart + (xEnd - xStart) * t;
    const float y = yStart + (yEnd - yStart) * t;
    const float z = zStart + (zEnd - zStart) * t;

    success = set_leg_pos(x, y, z, kLineMoveTimeMs) && success;
    delay(kLineMoveTimeMs);
  }

  return success;
}
