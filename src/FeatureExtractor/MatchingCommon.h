#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <immintrin.h>

namespace sfm {

/**
 * @brief Persistent worker pool shared by the matching routines. Threads are
 *        created once and reused across calls, and work is handed out in small
 *        chunks through an atomic counter, so sparse flag sets cannot leave
 *        some workers idle while others run long. A generation counter lets a
 *        worker that has drained the current batch sleep until the next batch
 *        is published; the gate mutex serializes concurrent callers.
 */
class ThreadPool {
public:
    static ThreadPool& Instance() {
        static ThreadPool pool;
        return pool;
    }

    int WorkerCount() const { return (int)workers_.size(); }

    /**
     * @brief Runs f(index, tid) for index in [0, count) in parallel.
     * @param tid Stable executor id in [0, WorkerCount()]; the calling thread
     *            participates with tid == WorkerCount().
     */
    template <typename F>
    void For(int count, F&& f) {
        if (count <= 0) {
            return;
        }
        std::lock_guard<std::mutex> gateLock(gate_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = std::function<void(int, int)>(std::forward<F>(f));
            next_ = 0;
            total_ = count;
            pending_ = (int)workers_.size();
            ++generation_;
        }
        cvWork_.notify_all();

        ClaimAndRun(job_, (int)workers_.size());

        std::unique_lock<std::mutex> lock(mutex_);
        cvDone_.wait(lock, [this] { return pending_ == 0; });
    }

private:
    ThreadPool() {
        const unsigned hw = std::thread::hardware_concurrency();
        const unsigned n = hw > 2 ? hw - 1 : 1;  // leave one core for the caller
        workers_.reserve(n);
        for (unsigned i = 0; i < n; ++i) {
            workers_.emplace_back([this, i] { WorkerLoop((int)i); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cvWork_.notify_all();
        for (auto& t : workers_) {
            t.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void ClaimAndRun(const std::function<void(int, int)>& job, int tid) {
        const int chunk = 64;
        while (true) {
            const int i = next_.fetch_add(chunk, std::memory_order_relaxed);
            if (i >= total_) {
                return;
            }
            const int end = std::min(i + chunk, total_);
            for (int k = i; k < end; ++k) {
                job(k, tid);
            }
        }
    }

    void WorkerLoop(int tid) {
        std::unique_lock<std::mutex> lock(mutex_);
        int gen = generation_;
        while (true) {
            cvWork_.wait(lock, [this, &gen] { return stop_ || generation_ != gen; });
            if (stop_) {
                return;
            }
            gen = generation_;
            const std::function<void(int, int)> job = job_;
            lock.unlock();
            ClaimAndRun(job, tid);
            lock.lock();
            if (--pending_ == 0) {
                cvDone_.notify_all();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::function<void(int, int)> job_;
    std::atomic<int> next_{0};
    int total_ = 0;
    int pending_ = 0;
    int generation_ = 0;
    std::mutex mutex_;
    std::mutex gate_;
    std::condition_variable cvWork_;
    std::condition_variable cvDone_;
    bool stop_ = false;
};

/**
 * @brief Squared L2 distance between two 64-D descriptors, vectorized with
 *        AVX2 (8-wide FMA) when the target supports it.
 */
inline float L2Sqr64(const float* a, const float* b) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < 64; k += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k));
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
#else
    float s = 0.f;
    for (int k = 0; k < 64; ++k) {
        const float d = a[k] - b[k];
        s += d * d;
    }
    return s;
#endif
}

/**
 * @brief Squared L2 distance between two 128-D descriptors (SIFT), vectorized
 *        with AVX2 (8-wide FMA) when the target supports it.
 */
inline float L2Sqr128(const float* a, const float* b) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < 128; k += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k));
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
#else
    float s = 0.f;
    for (int k = 0; k < 128; ++k) {
        const float d = a[k] - b[k];
        s += d * d;
    }
    return s;
#endif
}

}  // namespace sfm
