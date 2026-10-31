#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static unsigned number(const char* value) {
    std::string s(value);
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected unsigned integer");
    auto n = std::stoull(s);
    if (n > 10000000) throw std::invalid_argument("integer exceeds limit 10000000");
    return static_cast<unsigned>(n);
}
// Independent integer workload; return value prevents dead-code elimination.
static std::uint64_t task(unsigned index, unsigned iterations) {
    std::uint64_t x = static_cast<std::uint64_t>(index) + 1;
    for (unsigned j = 0; j < iterations; ++j) {
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        x *= UINT64_C(2685821657736338717);
    }
    return x;
}
int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::invalid_argument("usage: scheduling static|dynamic|guided|auto threads chunk items light heavy uniform|tail");
        const std::string policy(argv[1]);
        const unsigned threads=number(argv[2]), chunk=number(argv[3]), n=number(argv[4]), light=number(argv[5]), heavy=number(argv[6]);
        // Final argument chooses uniform or tail-heavy work.
        const std::string layout(argv[7]);
        if (!threads || threads > 256 || !n || !light || !heavy || (layout!="uniform" && layout!="tail"))
            throw std::invalid_argument("invalid threads/items/work/layout");
        omp_sched_t kind;
        if (policy=="static") kind=omp_sched_static;
        else if (policy=="dynamic") kind=omp_sched_dynamic;
        else if (policy=="guided") kind=omp_sched_guided;
        else if (policy=="auto") kind=omp_sched_auto;
        else throw std::invalid_argument("unknown policy");
        if ((policy=="auto" && chunk!=0) || (policy!="auto" && !chunk))
            throw std::invalid_argument("auto requires chunk=0; other policies require chunk>0");
        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(threads));
        omp_set_schedule(kind, static_cast<int>(chunk));
        std::vector<std::uint64_t> output(n), work(threads,0), counts(threads,0);
        std::vector<double> active_ms(threads,0);
        unsigned actual=0;
        auto cost = [&](unsigned i) { return layout=="tail" && i >= n - std::max(1u,n/10) ? heavy : light; };
        const auto start=std::chrono::steady_clock::now();
        #pragma omp parallel
        {
            const int tid=omp_get_thread_num();
            #pragma omp single
            actual=static_cast<unsigned>(omp_get_num_threads());
            std::uint64_t local_work=0, local_count=0;
            const double begin=omp_get_wtime();
            #pragma omp for schedule(runtime) nowait
            for (unsigned i=0;i<n;++i) {
                const unsigned iterations=cost(i);
                output[i]=task(i,iterations);
                local_work+=iterations;
                ++local_count;
            }
            const double elapsed=(omp_get_wtime()-begin)*1000; // Excludes final barrier waiting.
            work[tid]=local_work;
            counts[tid]=local_count;
            active_ms[tid]=elapsed;
        }
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::uint64_t checksum=0;
        for (unsigned i=0;i<n;++i) {
            if (output[i]!=task(i,cost(i))) throw std::runtime_error("serial reference mismatch");
            checksum+=output[i]; // Defined unsigned wraparound.
        }
        std::cout << std::setprecision(15) << "{\"policy\":\"" << policy << "\",\"threads\":" << threads
                  << ",\"actual_threads\":" << actual << ",\"chunk\":" << chunk << ",\"items\":" << n
                  << ",\"light\":" << light << ",\"heavy\":" << heavy << ",\"layout\":\"" << layout
                  << "\",\"checksum\":" << checksum << ",\"verified\":true,\"ms\":" << ms << ",\"workers\":[";
        for (unsigned t=0;t<actual;++t) {
            if (t) std::cout << ',';
            std::cout << "{\"items\":" << counts[t] << ",\"iterations\":" << work[t] << ",\"active_ms\":" << active_ms[t] << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
