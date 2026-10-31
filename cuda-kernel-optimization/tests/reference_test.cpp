#include "reference.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
int main() {
 try {
  const std::vector<float> x={1,2,3,4};
  const auto result=second_moment::reference(x,2,2);
  if(result != std::vector<double>{5,7,7,10}) throw std::runtime_error("hand-computed second moment failed");
  second_moment::validate({5,7,7,10},result);
  bool failed=false;
  try {second_moment::validate({0,0,0,0},result);}catch(const std::runtime_error&){failed=true;}
  if(!failed) throw std::runtime_error("mismatch accepted");
  const auto rejects = [](const std::vector<float>& got, const std::vector<double>& expected) {
   bool rejected = false;
   try { second_moment::validate(got, expected); }
   catch (const std::runtime_error&) { rejected = true; }
   if (!rejected) throw std::runtime_error("invalid validation data accepted");
  };
  rejects({}, {1.0});
  for (double value : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
   rejects({0.0f}, {value});
  }
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
   rejects({value}, {0.0});
  }
  failed=false;
  try {second_moment::input(0,1);}catch(const std::invalid_argument&){failed=true;}
  if(!failed) throw std::runtime_error("zero dimension accepted");
  std::cout<<"CPU reference, validation rejection and input checks passed\n";
 }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
