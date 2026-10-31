#include "validation.cuh"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <limits>
int positive(const char* text) {
    std::size_t used = 0; const std::string value(text);
    const long result = std::stol(value, &used);
    if (used != value.size() || result < 1 || result > 1000000) throw std::invalid_argument("expected integer in [1,1000000]");
    return static_cast<int>(result);
}
int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 4) throw std::invalid_argument("usage: kernel_benchmark [samples features repeats]");
        const int m = argc == 4 ? positive(argv[1]) : 257;
        const int n = argc == 4 ? positive(argv[2]) : 65;
        const int repeats = argc == 4 ? positive(argv[3]) : 15;
        const auto x = second_moment::input(m,n);
        const auto expected = second_moment::reference(x,m,n);
        cudaDeviceProp prop{}; int device=0, runtime=0, driver=0;
        CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
        second_moment::DeviceBuffer dx(x.size()*sizeof(float)), dout(expected.size()*sizeof(float));
        CUDA_CHECK(cudaMemcpy(dx.pointer,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
        second_moment::Launcher launches[] = {second_moment::launch_baseline,second_moment::launch_tiled};
        double errors[2];
        for (int k=0;k<2;++k) {
            errors[k] = second_moment::check_output(launches[k],x,expected,m,n,dx,dout);
            for(int w=0;w<3;++w) launches[k](dx.pointer,dout.pointer,m,n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        second_moment::Event start, stop;
        std::vector<float> samples[2];
        for(int r=0;r<repeats;++r) for(int order=0;order<2;++order) {
            const int k=(r+order)%2;
            CUDA_CHECK(cudaEventRecord(start.value));
            launches[k](dx.pointer,dout.pointer,m,n);
            CUDA_CHECK(cudaEventRecord(stop.value)); CUDA_CHECK(cudaEventSynchronize(stop.value));
            float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,stop.value));
            if(!std::isfinite(ms) || ms<=0) throw std::runtime_error("invalid elapsed time");
            samples[k].push_back(ms);
        }
        // Validate again after measurements, outside timed regions.
        for(int k=0;k<2;++k) second_moment::check_output(launches[k],x,expected,m,n,dx,dout);
        std::cout << std::setprecision(9) << "{\"schema_version\":1,\"operation\":\"uncentered_second_moment\",\"gpu\":\"" << prop.name
                  << "\",\"compute_capability\":\""<<prop.major<<"."<<prop.minor<<"\",\"runtime_version\":"<<runtime
                  <<",\"driver_version\":"<<driver<<",\"cuda_compiler_version\":\""<<__CUDACC_VER_MAJOR__<<"."<<__CUDACC_VER_MINOR__<<"."<<__CUDACC_VER_BUILD__
                  <<"\",\"samples\":"<<m<<",\"features\":"<<n<<",\"warmup\":3,\"block\":[16,16],\"validated\":true,\"results\":[";
        for(int k=0;k<2;++k) {
            if(k) std::cout<<",";
            std::cout<<"{\"kernel\":\""<<(k==0?"baseline":"tiled")<<"\",\"max_abs_error\":"<<errors[k]<<",\"milliseconds\":[";
            for(std::size_t i=0;i<samples[k].size();++i) {if(i) std::cout<<","; std::cout<<samples[k][i];}
            std::cout<<"]}";
        }
        std::cout<<"]}\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<"\n"; return 1;}
}
