#include <cuda_runtime.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <signal.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

constexpr unsigned long long pulse_ns = 2500000ULL;

__device__ __forceinline__ unsigned long long gpu_time() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void pulse() {
    const auto start = gpu_time();
    while (gpu_time() - start < pulse_ns) {
        __nanosleep(512);
    }
}

static void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    char* end;
    const long target = std::strtol(argv[1], &end, 10);
    if (*end || target < 1 || target > 100) return 2;
    // Retire the worker if the collector is terminated.
    const pid_t parent = getppid();
    if (parent == 1 || prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || getppid() != parent) {
        return 1;
    }
    const auto period = std::chrono::nanoseconds(100 * pulse_ns / target);
    check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    int lowest, highest;
    check(cudaDeviceGetStreamPriorityRange(&lowest, &highest));
    cudaStream_t stream;
    check(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, lowest));
    for (;;) {
        const auto start = std::chrono::steady_clock::now();
        pulse<<<1, 1, 0, stream>>>();
        check(cudaGetLastError());
        check(cudaStreamSynchronize(stream));
        std::this_thread::sleep_until(start + period);
    }
}
