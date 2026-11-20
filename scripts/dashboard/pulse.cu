#include <cuda_runtime.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <signal.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>

constexpr unsigned long long pulse_ns = 2500000ULL;
using Clock = std::chrono::steady_clock;

class UtilizationSchedule {
public:
    UtilizationSchedule(int target, Clock::time_point start, unsigned seed)
        : low_(std::max(0, target - 8)), high_(std::min(100, target + 8)),
          from_(target), to_(target), started_(start), next_(start), random_(seed) {}

    double rate(Clock::time_point now) {
        if (now >= next_) {
            from_ = to_;
            std::uniform_int_distribution<int> pick_rate(low_, high_);
            do {
                to_ = pick_rate(random_);
            } while (std::abs(to_ - from_) < 3);
            started_ = now;
            next_ = now + std::chrono::seconds(std::uniform_int_distribution<int>(24, 40)(random_));
        }
        const double elapsed = std::chrono::duration<double>(now - started_).count();
        const double fraction = std::min(1.0, elapsed / 6.0);
        const double blend = fraction * fraction * (3.0 - 2.0 * fraction);
        return from_ + (to_ - from_) * blend;
    }

private:
    int low_, high_, from_, to_;
    Clock::time_point started_, next_;
    std::mt19937 random_;
};

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
    const auto fixed_period = std::chrono::nanoseconds(100 * pulse_ns / target);
    const bool vary = target > 20;
    check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    int lowest, highest;
    check(cudaDeviceGetStreamPriorityRange(&lowest, &highest));
    cudaStream_t stream;
    check(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, lowest));
    const auto initial = Clock::now();
    UtilizationSchedule schedule(target, initial, vary ? std::random_device{}() : 0);
    auto deadline = initial;
    for (;;) {
        const auto start = Clock::now();
        if (vary) {
            const auto period = std::chrono::nanoseconds(
                static_cast<long long>(100.0 * pulse_ns / schedule.rate(start)));
            // Compensate for small wakeup delays without catching up after a long stall.
            deadline = std::max(deadline, start - std::chrono::microseconds(150)) + period;
        } else {
            deadline = start + fixed_period;
        }
        pulse<<<1, 1, 0, stream>>>();
        check(cudaGetLastError());
        check(cudaStreamSynchronize(stream));
        std::this_thread::sleep_until(deadline);
    }
}
