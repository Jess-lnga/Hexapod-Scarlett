#include <Arduino.h>
#include <ctype.h>
#include <stdlib.h>
#include "kinematics.hpp"
#include "motor_controls.h"
#include "leg_control.h"

static const size_t INPUT_BUFFER_SIZE = 64;
static const int8_t MOTOR_RX_PIN = 16;
static const int8_t MOTOR_TX_PIN = 17;
static const uint32_t MOTOR_BAUD_RATE = 115200;
static const uint16_t MOTOR_MOVE_TIME_MS = 500;

static bool follow_line_state = false;

enum CoordinateStep {
  STEP_X,
  STEP_Y,
  STEP_Z
};

static const char *coordinateName(CoordinateStep step) {
  switch (step) {
    case STEP_X:
      return "x";
    case STEP_Y:
      return "y";
    case STEP_Z:
      return "z";
  }

  return "?";
}

static void printCoordinatePrompt(CoordinateStep step) {
  Serial.println();
  Serial.printf("Enter %s coordinate:\r\n", coordinateName(step));
  Serial.print("> ");
}

static bool parseSingleCoordinate(char *line, float &value) {
  char *cursor = line;
  char *end = nullptr;

  value = strtof(cursor, &end);
  if (end == cursor) {
    return false;
  }

  while (*end != '\0') {
    if (!isspace(static_cast<unsigned char>(*end))) {
      return false;
    }
    ++end;
  }

  return true;
}

static CoordinateStep nextCoordinateStep(CoordinateStep step) {
  switch (step) {
    case STEP_X:
      return STEP_Y;
    case STEP_Y:
      return STEP_Z;
    case STEP_Z:
      return STEP_X;
  }

  return STEP_X;
}

static void printJointAngles() {
  if (!get_kinematics_valid()) {
    Serial.println("Kinematics result invalid.");
    return;
  }

  Serial.printf("Theta zero: %.2f deg\r\n", get_Theta_zero() * 180.0f / PI);
  Serial.printf("Theta one:  %.2f deg\r\n", get_Theta_one() * 180.0f / PI);
  Serial.printf("Theta two:  %.2f deg\r\n", get_Theta_two() * 180.0f / PI);
}

static float z = -7.4f;
void setup() {
  Serial.begin(115200);

  while (!Serial) {
    delay(10);
  }

  begin_motors(Serial2, MOTOR_RX_PIN, MOTOR_TX_PIN, MOTOR_BAUD_RATE);

  Serial.println("Hexapod leg position test");
  Serial.printf("Motor bus: %s\r\n", motors_initialized() ? "initialized" : "not initialized");
  printCoordinatePrompt(STEP_X);

  set_leg_pos(11.5f, 0.0f, z, MOTOR_MOVE_TIME_MS);
}

void loop() {

  follow_line_state = follow_line(11.5f,   0.0f, z,    14.5f,   0.0f, z,   10); //1
  follow_line_state = follow_line(14.5f,   0.0f, z,    14.5f,  12.5f, z,   10); //2
  follow_line_state = follow_line(14.5f,  12.5f, z,    11.5f,  12.5f, z,   10); //3
  follow_line_state = follow_line(11.5f,  12.5f, z,    11.5f,   0.0f, z,   10); //4

  follow_line_state = follow_line(11.5f,   0.0f, z,    14.5f,   0.0f, z,   10); //5
  follow_line_state = follow_line(14.5f,   0.0f, z,    14.5f, -12.5f, z,   10); //6
  follow_line_state = follow_line(14.5f, -12.5f, z,    11.5f, -12.5f, z,   10); //7
  follow_line_state = follow_line(11.5f, -12.5f, z,    11.5f,   0.0f, z,   10); //8
  


  /*
  static char inputBuffer[INPUT_BUFFER_SIZE];
  static size_t inputLength = 0;
  static CoordinateStep currentStep = STEP_X;
  static float x = 0.0f;
  static float y = 0.0f;
  static float z = 0.0f;

  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());

    if (ch == '\r' || ch == '\n') {
      if (inputLength == 0) {
        continue;
      }

      inputBuffer[inputLength] = '\0';
      inputLength = 0;

      float value = 0.0f;

      if (!parseSingleCoordinate(inputBuffer, value)) {
        Serial.printf("Invalid %s value. Enter one number only.\r\n", coordinateName(currentStep));
        printCoordinatePrompt(currentStep);
        continue;
      }

      switch (currentStep) {
        case STEP_X:
          x = value;
          break;
        case STEP_Y:
          y = value;
          break;
        case STEP_Z:
          z = value;
          break;
      }

      Serial.printf("%s coordinate received: %.2f\r\n", coordinateName(currentStep), value);

      if (currentStep == STEP_Z) {
        Serial.printf("Input: x=%.2f, y=%.2f, z=%.2f\r\n", x, y, z);
        compute_joint_angles(x, y, z);
        printJointAngles();

        if (get_kinematics_valid()) {
          const bool motorsOk = set_leg_pos(x, y, z, MOTOR_MOVE_TIME_MS);
          Serial.printf("Motor command: %s\r\n", motorsOk ? "sent" : "failed");
        }
      }

      currentStep = nextCoordinateStep(currentStep);
      printCoordinatePrompt(currentStep);
      continue;
    }

    if (inputLength < INPUT_BUFFER_SIZE - 1) {
      inputBuffer[inputLength++] = ch;
    } else {
      inputLength = 0;
      Serial.println("Input too long. Enter one number only.");
      printCoordinatePrompt(currentStep);
    }
  }
  */
}
