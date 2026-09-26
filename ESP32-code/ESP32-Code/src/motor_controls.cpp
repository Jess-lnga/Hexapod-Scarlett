#include "motor_controls.h"

#include "kinematics.hpp"

static const uint8_t kServo0Id = 3;
static const uint8_t kServo1Id = 2;
static const uint8_t kServo2Id = 1;

static const uint16_t kServoMinPosition = 0;
static const uint16_t kServoMaxPosition = 1000;

///////////////////////////////////////////////////
static const float theor_angle_0_at_min = -63.0;
static const float theor_angle_0_at_max = 77.0;
static const float real_angle_0_at_theor_min = 55.0;
static const float real_angle_0_at_theor_max = 195.0;

/////////////////

static const float theor_angle_1_at_min = -35.0;
static const float theor_angle_1_at_max = 129.0;
static const float real_angle_1_at_theor_min = 164.0;
static const float real_angle_1_at_theor_max = 0.0;

/////////////////

static const float theor_angle_2_at_min = 90.0;
static const float theor_angle_2_at_max = 180.0;
static const float real_angle_2_at_theor_min = 31.0;
static const float real_angle_2_at_theor_max = 121.0;

///////////////////////////////////////////////////


static const uint8_t kServoMoveTimeWriteCommand = 1;
static const uint8_t kServoMoveTimeWriteLength = 7;
//////////////////////////////////////////////////
static float linearise( float angle, 
                        float theor_min_angle, 
                        float theor_max_angle, 
                        float real_min_angle, 
                        float real_max_angle){

  float a = (real_max_angle - real_min_angle) / (theor_max_angle - theor_min_angle);
  float b = real_min_angle - a * theor_min_angle;

  return a * angle + b;
}

static HardwareSerial* servoSerial = nullptr;

static bool is_valid_servo(uint8_t servoId) {
  return servoId == kServo0Id || servoId == kServo1Id || servoId == kServo2Id;
}

static float constrain_between(float angleDeg, float limitA, float limitB) {
  const float minLimit = min(limitA, limitB);
  const float maxLimit = max(limitA, limitB);
  return constrain(angleDeg, minLimit, maxLimit);
}

static float clamp_angle(uint8_t servoId, float angleDeg) {
  switch (servoId) {
    case kServo0Id:
      return constrain_between(angleDeg, real_angle_0_at_theor_min, real_angle_0_at_theor_max);
    case kServo1Id:
      return constrain_between(angleDeg, real_angle_1_at_theor_min, real_angle_1_at_theor_max);
    case kServo2Id:
      return constrain_between(angleDeg, real_angle_2_at_theor_min, real_angle_2_at_theor_max);
    default:
      return angleDeg;
  }
}

static uint16_t angle_to_position(float angleDeg){
  const float clampedAngle = constrain(angleDeg, 0.0f, 240.0f);
  const float ratio = clampedAngle / 240.0f;
  return static_cast<uint16_t>((ratio * kServoMaxPosition) + 0.5f);
}

static bool write_servo_position(uint8_t servoId, uint16_t position, uint16_t moveTimeMs) {
  if (servoSerial == nullptr || !is_valid_servo(servoId)) {
    return false;
  }

  position = constrain(position, kServoMinPosition, kServoMaxPosition);

  const uint8_t positionLow = position & 0xFF;
  const uint8_t positionHigh = (position >> 8) & 0xFF;
  const uint8_t timeLow = moveTimeMs & 0xFF;
  const uint8_t timeHigh = (moveTimeMs >> 8) & 0xFF;
  const uint8_t checksum = static_cast<uint8_t>(
      ~(servoId + kServoMoveTimeWriteLength + kServoMoveTimeWriteCommand +
        positionLow + positionHigh + timeLow + timeHigh));

  servoSerial->write(0x55);
  servoSerial->write(0x55);
  servoSerial->write(servoId);
  servoSerial->write(kServoMoveTimeWriteLength);
  servoSerial->write(kServoMoveTimeWriteCommand);
  servoSerial->write(positionLow);
  servoSerial->write(positionHigh);
  servoSerial->write(timeLow);
  servoSerial->write(timeHigh);
  servoSerial->write(checksum);
  servoSerial->flush();

  return true;
}

static float rad_to_deg(float angleRad) {
  return angleRad * 180.0f / PI;
}

static void compute_offset(float thetaZeroRad,
                           float thetaOneRad,
                           float thetaTwoRad,
                           float& servo0AngleDeg,
                           float& servo1AngleDeg,
                           float& servo2AngleDeg) {

  servo0AngleDeg = linearise(rad_to_deg(thetaZeroRad),
                             theor_angle_0_at_min,
                             theor_angle_0_at_max,
                             real_angle_0_at_theor_min,
                             real_angle_0_at_theor_max);
  servo1AngleDeg = linearise(rad_to_deg(thetaOneRad),
                             theor_angle_1_at_min,
                             theor_angle_1_at_max,
                             real_angle_1_at_theor_min,
                             real_angle_1_at_theor_max);
  servo2AngleDeg = linearise(rad_to_deg(thetaTwoRad + asin(c2/c)),
                             theor_angle_2_at_min,
                             theor_angle_2_at_max,
                             real_angle_2_at_theor_min,
                             real_angle_2_at_theor_max);
}

static bool set_angle(uint8_t servoId, float angleDeg, uint16_t moveTimeMs) {
  if (!is_valid_servo(servoId)) {
    return false;
  }

  const float clampedAngleDeg = clamp_angle(servoId, angleDeg);
  return write_servo_position(servoId, angle_to_position(clampedAngleDeg), moveTimeMs);
}

void begin_motors(HardwareSerial& serialPort, int8_t rxPin, int8_t txPin, uint32_t baudRate) {
  servoSerial = &serialPort;
  servoSerial->begin(baudRate, SERIAL_8N1, rxPin, txPin);
}

bool motors_initialized() {
  return servoSerial != nullptr;
}

bool set_leg_pos(float x, float y, float z, uint16_t moveTimeMs) {
  compute_joint_angles(x, y, z);
  
  if (!get_kinematics_valid()) {
    return false;
  }

  float servo0AngleDeg = 0.0f;
  float servo1AngleDeg = 0.0f;
  float servo2AngleDeg = 0.0f;
  
  compute_offset(get_Theta_zero(),
                 get_Theta_one(),
                 get_Theta_two(),
                 servo0AngleDeg,
                 servo1AngleDeg,
                 servo2AngleDeg);

  const bool servo0Ok = set_angle(kServo0Id, servo0AngleDeg, moveTimeMs);
  const bool servo1Ok = set_angle(kServo1Id, servo1AngleDeg, moveTimeMs);
  const bool servo2Ok = set_angle(kServo2Id, servo2AngleDeg, moveTimeMs);

  return servo0Ok && servo1Ok && servo2Ok;
}
