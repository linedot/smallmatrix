#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Can't be arsed to write my own
#include <boost/align/aligned_allocator.hpp>

#include "performance_counters.hpp"

constexpr std::size_t num_matrices = 4;
constexpr std::size_t matrix_size = 4;

typedef double scalar;

void naive_mm(scalar* A, scalar* B, scalar* C)
{
    for (std::size_t y = 0; y < matrix_size; y++)
    {
        for (std::size_t x = 0; x < matrix_size; x++)
        {
            for (std::size_t k = 0; k < matrix_size; k++)
            {
                C[y*matrix_size+x] += A[y*matrix_size+k]*B[k*matrix_size+x];
            }
        }
    }
}

void mm_avxfma_4x4_fp64_broadcast(double * __restrict__ A, double * __restrict__ B, double * __restrict__ C)
{
    asm volatile(

            "vmovapd 0x00(%[c]), %%ymm8\n\t"
            "vmovapd 0x20(%[c]), %%ymm9\n\t"
            "vmovapd 0x40(%[c]), %%ymm10\n\t"
            "vmovapd 0x60(%[c]), %%ymm11\n\t"

            "vbroadcastsd 0x00(%[a]), %%ymm0\n\t"
            "vfmadd231pd 0x00(%[b]), %%ymm0, %%ymm8\n\t"
            "vbroadcastsd 0x20(%[a]), %%ymm1\n\t"
            "vfmadd231pd 0x00(%[b]), %%ymm1, %%ymm9\n\t"
            "vbroadcastsd 0x40(%[a]), %%ymm2\n\t"
            "vfmadd231pd 0x00(%[b]), %%ymm2, %%ymm10\n\t"
            "vbroadcastsd 0x60(%[a]), %%ymm3\n\t"
            "vfmadd231pd 0x00(%[b]), %%ymm3, %%ymm11\n\t"

            "vbroadcastsd 0x08(%[a]), %%ymm4\n\t"
            "vfmadd231pd 0x20(%[b]), %%ymm4, %%ymm8\n\t"
            "vbroadcastsd 0x28(%[a]), %%ymm5\n\t"
            "vfmadd231pd 0x20(%[b]), %%ymm5, %%ymm9\n\t"
            "vbroadcastsd 0x48(%[a]), %%ymm6\n\t"
            "vfmadd231pd 0x20(%[b]), %%ymm6, %%ymm10\n\t"
            "vbroadcastsd 0x68(%[a]), %%ymm7\n\t"
            "vfmadd231pd 0x20(%[b]), %%ymm7, %%ymm11\n\t"

            "vbroadcastsd 0x10(%[a]), %%ymm0\n\t"
            "vfmadd231pd 0x40(%[b]), %%ymm0, %%ymm8\n\t"
            "vbroadcastsd 0x30(%[a]), %%ymm1\n\t"
            "vfmadd231pd 0x40(%[b]), %%ymm1, %%ymm9\n\t"
            "vbroadcastsd 0x50(%[a]), %%ymm2\n\t"
            "vfmadd231pd 0x40(%[b]), %%ymm2, %%ymm10\n\t"
            "vbroadcastsd 0x70(%[a]), %%ymm3\n\t"
            "vfmadd231pd 0x40(%[b]), %%ymm3, %%ymm11\n\t"

            "vbroadcastsd 0x18(%[a]), %%ymm4\n\t"
            "vfmadd231pd 0x60(%[b]), %%ymm4, %%ymm8\n\t"
            "vmovapd %%ymm8,  0x00(%[c])\n\t"
            "vbroadcastsd 0x38(%[a]), %%ymm5\n\t"
            "vfmadd231pd 0x60(%[b]), %%ymm5, %%ymm9\n\t"
            "vmovapd %%ymm9,  0x20(%[c])\n\t"
            "vbroadcastsd 0x58(%[a]), %%ymm6\n\t"
            "vfmadd231pd 0x60(%[b]), %%ymm6, %%ymm10\n\t"
            "vmovapd %%ymm10, 0x40(%[c])\n\t"
            "vbroadcastsd 0x78(%[a]), %%ymm7\n\t"
            "vfmadd231pd 0x60(%[b]), %%ymm7, %%ymm11\n\t"
            "vmovapd %%ymm11, 0x60(%[c])\n\t"
            :
            : [a] "r" (A),
              [b] "r" (B),
              [c] "r" (C)
            : "ymm0", "ymm1", "ymm2", "ymm3",
              "ymm4", "ymm5", "ymm6", "ymm7",
              "ymm8", "ymm9", "ymm10", "ymm11"
            );
}

void mm_avxfma_4x4x2_fp64_broadcast(double * __restrict__ A, double * __restrict__ B, double * __restrict__ C)
{
    asm volatile(

            "vbroadcastsd 0x00(%[a]), %%ymm0\n\t"
            "vmovapd 0x00(%[c]), %%ymm8\n\t"
            "vmovapd 0x20(%[c]), %%ymm9\n\t"
            "vmovapd 0x40(%[c]), %%ymm10\n\t"
            "vmovapd 0x60(%[c]), %%ymm11\n\t"
            "vbroadcastsd 0x20(%[a]), %%ymm1\n\t"

            "vfmadd231pd 0x00(%[b]), %%ymm0, %%ymm8\n\t"
            "vfmadd231pd 0x00(%[b]), %%ymm1, %%ymm9\n\t"
            "vbroadcastsd 0x40(%[a]), %%ymm2\n\t"
            "vfmadd231pd 0x00(%[b]), %%ymm2, %%ymm10\n\t"
            "vbroadcastsd 0x60(%[a]), %%ymm3\n\t"
            "vfmadd231pd 0x00(%[b]), %%ymm3, %%ymm11\n\t"

            "vmovapd 0x80(%[c]), %%ymm12\n\t"
            "vmovapd 0xA0(%[c]), %%ymm13\n\t"
            "vmovapd 0xC0(%[c]), %%ymm14\n\t"
            "vmovapd 0xE0(%[c]), %%ymm15\n\t"

            "vbroadcastsd 0x80(%[a]), %%ymm4\n\t"
            "vfmadd231pd 0x80(%[b]), %%ymm4, %%ymm12\n\t"
            "vbroadcastsd 0xA0(%[a]), %%ymm5\n\t"
            "vfmadd231pd 0x80(%[b]), %%ymm5, %%ymm13\n\t"
            "vbroadcastsd 0xC0(%[a]), %%ymm6\n\t"
            "vfmadd231pd 0x80(%[b]), %%ymm6, %%ymm14\n\t"
            "vbroadcastsd 0xE0(%[a]), %%ymm7\n\t"
            "vfmadd231pd 0x80(%[b]), %%ymm7, %%ymm15\n\t"

            "vbroadcastsd 0x08(%[a]), %%ymm0\n\t"
            "vfmadd231pd 0x20(%[b]), %%ymm0, %%ymm8\n\t"
            "vbroadcastsd 0x28(%[a]), %%ymm1\n\t"
            "vfmadd231pd 0x20(%[b]), %%ymm1, %%ymm9\n\t"
            "vbroadcastsd 0x48(%[a]), %%ymm2\n\t"
            "vfmadd231pd 0x20(%[b]), %%ymm2, %%ymm10\n\t"
            "vbroadcastsd 0x68(%[a]), %%ymm3\n\t"
            "vfmadd231pd 0x20(%[b]), %%ymm3, %%ymm11\n\t"

            "vbroadcastsd 0x88(%[a]), %%ymm4\n\t"
            "vfmadd231pd 0xA0(%[b]), %%ymm4, %%ymm12\n\t"
            "vbroadcastsd 0xA8(%[a]), %%ymm5\n\t"
            "vfmadd231pd 0xA0(%[b]), %%ymm5, %%ymm13\n\t"
            "vbroadcastsd 0xC8(%[a]), %%ymm6\n\t"
            "vfmadd231pd 0xA0(%[b]), %%ymm6, %%ymm14\n\t"
            "vbroadcastsd 0xE8(%[a]), %%ymm7\n\t"
            "vfmadd231pd 0xA0(%[b]), %%ymm7, %%ymm15\n\t"

            "vbroadcastsd 0x10(%[a]), %%ymm0\n\t"
            "vfmadd231pd 0x40(%[b]), %%ymm0, %%ymm8\n\t"
            "vbroadcastsd 0x30(%[a]), %%ymm1\n\t"
            "vfmadd231pd 0x40(%[b]), %%ymm1, %%ymm9\n\t"
            "vbroadcastsd 0x50(%[a]), %%ymm2\n\t"
            "vfmadd231pd 0x40(%[b]), %%ymm2, %%ymm10\n\t"
            "vbroadcastsd 0x70(%[a]), %%ymm3\n\t"
            "vfmadd231pd 0x40(%[b]), %%ymm3, %%ymm11\n\t"

            "vbroadcastsd 0x90(%[a]), %%ymm4\n\t"
            "vfmadd231pd 0xC0(%[b]), %%ymm4, %%ymm12\n\t"
            "vbroadcastsd 0xB0(%[a]), %%ymm5\n\t"
            "vfmadd231pd 0xC0(%[b]), %%ymm5, %%ymm13\n\t"
            "vbroadcastsd 0xD0(%[a]), %%ymm6\n\t"
            "vfmadd231pd 0xC0(%[b]), %%ymm6, %%ymm14\n\t"
            "vbroadcastsd 0xF0(%[a]), %%ymm7\n\t"
            "vfmadd231pd 0xC0(%[b]), %%ymm7, %%ymm15\n\t"

            "vbroadcastsd 0x18(%[a]), %%ymm0\n\t"
            "vfmadd231pd 0x60(%[b]), %%ymm0, %%ymm8\n\t"
            "vmovapd %%ymm8,  0x00(%[c])\n\t"
            "vbroadcastsd 0x38(%[a]), %%ymm1\n\t"
            "vfmadd231pd 0x60(%[b]), %%ymm1, %%ymm9\n\t"
            "vmovapd %%ymm9,  0x20(%[c])\n\t"
            "vbroadcastsd 0x58(%[a]), %%ymm2\n\t"
            "vfmadd231pd 0x60(%[b]), %%ymm2, %%ymm10\n\t"
            "vmovapd %%ymm10, 0x40(%[c])\n\t"
            "vbroadcastsd 0x78(%[a]), %%ymm3\n\t"
            "vfmadd231pd 0x60(%[b]), %%ymm3, %%ymm11\n\t"
            "vmovapd %%ymm11, 0x60(%[c])\n\t"

            "vbroadcastsd 0x98(%[a]), %%ymm0\n\t"
            "vfmadd231pd 0xE0(%[b]), %%ymm0, %%ymm12\n\t"
            "vmovapd %%ymm12, 0x80(%[c])\n\t"
            "vbroadcastsd 0xB8(%[a]), %%ymm1\n\t"
            "vfmadd231pd 0xE0(%[b]), %%ymm1, %%ymm13\n\t"
            "vmovapd %%ymm13, 0xA0(%[c])\n\t"
            "vbroadcastsd 0xD8(%[a]), %%ymm2\n\t"
            "vfmadd231pd 0xE0(%[b]), %%ymm2, %%ymm14\n\t"
            "vmovapd %%ymm14, 0xC0(%[c])\n\t"
            "vbroadcastsd 0xF8(%[a]), %%ymm3\n\t"
            "vfmadd231pd 0xE0(%[b]), %%ymm3, %%ymm15\n\t"
            "vmovapd %%ymm15, 0xE0(%[c])\n\t"
            :
            : [a] "r" (A),
              [b] "r" (B),
              [c] "r" (C)
            : "ymm0", "ymm1", "ymm2", "ymm3",
              "ymm4", "ymm5", "ymm6", "ymm7",
              "ymm8", "ymm9", "ymm10", "ymm11"
            );
}

inline void mm_avxfma_4x4_fp64_wrong(double * __restrict__ A, double * __restrict__ B, double * __restrict__ C)
{
    asm volatile(

            "vmovapd 0x00(%[c]), %%ymm8\n\t"
            "vmovapd 0x20(%[c]), %%ymm9\n\t"
            "vmovapd 0x40(%[c]), %%ymm10\n\t"
            "vmovapd 0x60(%[c]), %%ymm11\n\t"

            "vmovapd 0x00(%[b]), %%ymm4\n\t"
            "vmovapd 0x00(%[a]), %%ymm0\n\t"
            "vfmadd231pd %%ymm4, %%ymm0, %%ymm8\n\t"
            "vmovapd 0x20(%[a]), %%ymm1\n\t"
            "vfmadd231pd %%ymm4, %%ymm1, %%ymm9\n\t"
            "vmovapd 0x40(%[a]), %%ymm2\n\t"
            "vfmadd231pd %%ymm4, %%ymm2, %%ymm10\n\t"
            "vmovapd 0x60(%[a]), %%ymm3\n\t"
            "vfmadd231pd %%ymm4, %%ymm3, %%ymm11\n\t"

            "vmovapd 0x20(%[b]), %%ymm5\n\t"
            "vfmadd231pd %%ymm5, %%ymm0, %%ymm8\n\t"
            "vmovapd 0x40(%[b]), %%ymm6\n\t"
            "vfmadd231pd %%ymm5, %%ymm1, %%ymm9\n\t"
            "vmovapd 0x60(%[b]), %%ymm7\n\t"
            "vfmadd231pd %%ymm5, %%ymm2, %%ymm10\n\t"
            "vfmadd231pd %%ymm5, %%ymm3, %%ymm11\n\t"

            "vfmadd231pd %%ymm6, %%ymm0, %%ymm8\n\t"
            "vfmadd231pd %%ymm6, %%ymm1, %%ymm9\n\t"
            "vfmadd231pd %%ymm6, %%ymm2, %%ymm10\n\t"
            "vfmadd231pd %%ymm6, %%ymm3, %%ymm11\n\t"

            "vfmadd231pd %%ymm7, %%ymm0, %%ymm8\n\t"
            "vmovapd %%ymm8,  0x00(%[c])\n\t"
            "vfmadd231pd %%ymm7, %%ymm1, %%ymm9\n\t"
            "vmovapd %%ymm9,  0x20(%[c])\n\t"
            "vfmadd231pd %%ymm7, %%ymm2, %%ymm10\n\t"
            "vmovapd %%ymm10, 0x40(%[c])\n\t"
            "vfmadd231pd %%ymm7, %%ymm3, %%ymm11\n\t"
            "vmovapd %%ymm11, 0x60(%[c])\n\t"

            :
            : [a] "r" (A),
              [b] "r" (B),
              [c] "r" (C)
            : "ymm0", "ymm1", "ymm2", "ymm3",
              "ymm4", "ymm5", "ymm6", "ymm7",
              "ymm8", "ymm9", "ymm10", "ymm11"
            );
}

void print_matrix(scalar* m)
{
    for (std::size_t y = 0; y < matrix_size; y++)
    {
        std::cout << std::setprecision(2) << std::setw(4) << std::fixed  << m[y*matrix_size];
        for (std::size_t x = 1; x < matrix_size; x++)
        {
            std::cout << " " << std::setprecision(2) << std::setw(4) << std::fixed << m[y*matrix_size + x];
        }
        std::cout << "\n";
    }
}

bool compare_results(scalar* m1, scalar* m2, scalar epsilon = 1e-12)
{
    for (std::size_t i = 0; i < num_matrices; i++)
        for (std::size_t y = 0; y < matrix_size; y++)
            for (std::size_t x = 1; x < matrix_size; x++)
            {
                std::size_t index = i*matrix_size*matrix_size+y*matrix_size+x;
                if (epsilon < std::abs(m1[index] - m2[index]))
                    return false;
            }
    return true;
}

void print_usage(const char* app_name)
{
    std::cout << "Usage: " << app_name << " matrix_count iterations\n";
}

int main(int argc, char* argv[])
{

    performance_counters pc(std::vector<std::string>{
            "CYCLES",
            "DISPATCH_RESOURCE_STALL_CYCLES_1:LOAD_QUEUE_RSRC_STALL",
            "perf::L1-DCACHE-LOAD-MISSES"});
    // Allocate some space
    std::size_t element_count = num_matrices*matrix_size*matrix_size;
    std::vector<scalar, boost::alignment::aligned_allocator<scalar, 64>> A_storage(element_count);
    std::vector<scalar, boost::alignment::aligned_allocator<scalar, 64>> B_storage(element_count);
    std::vector<scalar, boost::alignment::aligned_allocator<scalar, 64>> C_storage(element_count);
    std::vector<scalar, boost::alignment::aligned_allocator<scalar, 64>> Cref_storage(element_count);

    // initialize random number generator
    std::random_device rnd{};
    std::vector<int> seeds(621);
    std::generate_n(seeds.begin(), 621, std::bind(std::ref(rnd)));
    std::seed_seq seed(seeds.begin(), seeds.end());
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<scalar> dist(0.0, 1.0);

    // Fill matrices
    std::generate_n(A_storage.begin(), element_count, [&](){return dist(rng);});
    std::generate_n(B_storage.begin(), element_count, [&](){return dist(rng);});
    std::generate_n(C_storage.begin(), element_count, [&](){return dist(rng);});
    std::copy(C_storage.begin(), C_storage.end(), Cref_storage.begin());

    std::cout << "A: \n";
    print_matrix(A_storage.data());
    std::cout << "B: \n";
    print_matrix(B_storage.data());

    std::cout << "C before: \n";
    print_matrix(C_storage.data());
    std::cout << "Cref before: \n";
    print_matrix(Cref_storage.data());

    for (std::size_t i = 0; i < num_matrices; i++)
    {
        naive_mm(A_storage.data()+i*matrix_size*matrix_size, 
                 B_storage.data()+i*matrix_size*matrix_size,
                 Cref_storage.data()+i*matrix_size*matrix_size);
    }
    for (std::size_t i = 0; i < num_matrices/2; i++)
    {
        mm_avxfma_4x4x2_fp64_broadcast(A_storage.data() + i*2*matrix_size*matrix_size,
                                     B_storage.data() + i*2*matrix_size*matrix_size,
                                     C_storage.data() + i*2*matrix_size*matrix_size);
    }

    std::cout << "C after: \n";
    print_matrix(C_storage.data());
    std::cout << "Cref after: \n";
    print_matrix(Cref_storage.data());

    if (!compare_results(C_storage.data(), Cref_storage.data()))
    {
        std::cout << "Calculation incorrect, skipping benchmark\n";
        return -1;
    }

    std::cout << "Calculation correct, starting benchmark\n";

    if (3 != argc)
    {
        std::cout << "Invalid parameters\n";
        print_usage(argv[0]);
        return -2;
    }

    int matrix_count = std::abs(std::atoi(argv[1]));
    int iterations = std::abs(std::atoi(argv[2]));

    std::cout << "Number of 4x4 matrices for benchmark: " << matrix_count << "\n";
    std::cout << "Number of iterations for benchmark:   " << iterations << "\n";

    element_count = matrix_size*matrix_size*matrix_count;

    pc.tic();
    
    C_storage.resize(element_count);
    Cref_storage.resize(element_count);
    A_storage.resize(element_count);
    B_storage.resize(element_count);
    
    auto cycles = pc.toc()[0];

    std::cout << "Allocated " << element_count*4*sizeof(scalar) << " bytes in " << cycles << " cycles\n";


    pc.tic();
    std::generate_n(A_storage.begin(), element_count, [&](){return dist(rng);});
    std::generate_n(B_storage.begin(), element_count, [&](){return dist(rng);});
    std::generate_n(C_storage.begin(), element_count, [&](){return dist(rng);});
    std::copy(C_storage.begin(), C_storage.end(), Cref_storage.begin());
    cycles = pc.toc()[0];

    std::cout << "Filled " << element_count*4*sizeof(scalar) << " bytes with random values in " << cycles << " cycles\n";

    std::size_t flops = matrix_count*2*matrix_size*matrix_size*matrix_size;
    pc.tic();
    for(std::size_t j = 0; j < iterations; j++)
    for(std::size_t i = 0; i < matrix_count; i++)
    {
        naive_mm(A_storage.data()+i*matrix_size*matrix_size, 
                 B_storage.data()+i*matrix_size*matrix_size,
                 Cref_storage.data()+i*matrix_size*matrix_size);
    }
    auto counters = pc.toc();
    cycles = counters[0];
    auto stalls = counters[1];
    auto misses = counters[2];
    std::cout << "naive C algorithm finished in " << cycles/iterations << " cycles\n";
    std::cout << "FLOPS/CYCLE:     " << static_cast<double>(flops)/static_cast<double>(cycles/iterations) << "\n";
    std::cout << "LD queue stalls: " << static_cast<double>(stalls/iterations) << "\n";
    std::cout << "L1D misses:      " << static_cast<double>(misses/iterations) << "\n";

    pc.tic();
    for(std::size_t j = 0; j < iterations; j++)
    for(std::size_t i = 0; i < matrix_count; i++)
    {
        mm_avxfma_4x4_fp64_broadcast(
                 A_storage.data()+i*matrix_size*matrix_size, 
                 B_storage.data()+i*matrix_size*matrix_size,
                 C_storage.data()+i*matrix_size*matrix_size);
    }
    counters = pc.toc();
    cycles = counters[0];
    stalls = counters[1];
    misses = counters[2];
    std::cout << "AVX/FMA algorithm finished in " << cycles/iterations << " cycles\n";
    std::cout << "FLOPS/CYCLE:     " << static_cast<double>(flops)/static_cast<double>(cycles/iterations) << "\n";
    std::cout << "LD queue stalls: " << static_cast<double>(stalls/iterations) << "\n";
    std::cout << "L1D misses:      " << static_cast<double>(misses/iterations) << "\n";


    return 0;
}
