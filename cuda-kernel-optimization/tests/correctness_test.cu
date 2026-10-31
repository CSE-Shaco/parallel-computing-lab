#include "validation.cuh"
#include <iostream>
int main() {
 try {
  const int shapes[][2]={{1,1},{1,17},{17,1},{7,9},{16,16},{17,19},{33,31},{257,65},
                        {32,32},{33,33},{63,65},{65,63}};
  for (const auto& shape:shapes) {
   const int m=shape[0],n=shape[1];
   const auto x=second_moment::input(m,n); const auto expected=second_moment::reference(x,m,n);
   second_moment::DeviceBuffer dx(x.size()*sizeof(float)),out(expected.size()*sizeof(float));
   CUDA_CHECK(cudaMemcpy(dx.pointer,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
   for(const auto& variant:second_moment::kernel_variants)
    second_moment::check_output(variant.launch,x,expected,m,n,dx,out);
  }
  std::cout<<"12 shapes passed for baseline, tiled, coalesced_tiled and register_tiled kernels\n";
 } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
}
