#pragma once

#include <Arduino.h>

// Segment parameters for the leg kinematics.
static const float a = 4.0f; // a in cm
static const float b = 7.0f;  // b in cm
static const float c = 13.46f;  // c in cm
static const float c1 = 3.5f;
static const float c2 = 13.0f;


void compute_joint_angles(float x, float y, float z);

float get_Theta_zero();
float get_Theta_one();
float get_Theta_two();

bool get_kinematics_valid();
