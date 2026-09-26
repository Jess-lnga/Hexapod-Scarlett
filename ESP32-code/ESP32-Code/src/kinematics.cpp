#include "kinematics.hpp"

static float Theta_zero = 0.0f;
static float Theta_one = 0.0f;
static float Theta_two = 0.0f;
static bool Kinematics_valid = false;

bool check_reachability(float x, float y, float z) {

  Kinematics_valid = true;
  if (a <= 0.0f ||
      b <= 0.0f ||
      c <= 0.0f) {
    Kinematics_valid = false;
  }

  return Kinematics_valid;
}
void compute_joint_angles(float x, float y, float z) {
  Kinematics_valid = check_reachability(x, y, z);

  if(Kinematics_valid) {
    float lambda_1 = (sqrt(x * x + y * y) - a)/c;
    float lambda_2 = z/c;

    float G = (-c/(2.0*b))*(1-lambda_1*lambda_1-lambda_2*lambda_2-(b/c)*(b/c));
    
    float Omega = acos(G/(sqrt(lambda_1*lambda_1+lambda_2*lambda_2)));
    float Lambda = atan2(z, sqrt(x * x + y * y) - a);

  
    Theta_zero = atan2(y, x);
    Theta_one = Lambda + Omega;
    Theta_two = acos((b/c) - lambda_1*cos(Theta_one) - lambda_2*sin(Theta_one));

    Kinematics_valid = true;
  } 
}

float get_Theta_zero() {
  return Theta_zero;
}

float get_Theta_one() {
  return Theta_one;
}

float get_Theta_two() {
  return Theta_two;
}

bool get_kinematics_valid() {
  return Kinematics_valid;
}
