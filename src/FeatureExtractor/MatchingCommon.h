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

class ThreadPool {
public:
    static ThreadPool& Instance() {
        static ThreadPool pool;
        return pool;
    }

    int WorkerCount() const { return nThreads_; }

    template <typename F>
    void For(int count, F&& f) {
        if (count <= 0) return;

        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = std::function<void(int, int)>(std::forward<F>(f));
            total_ = count;
            next_.store(0, std::memory_order_relaxed);
            done_.store(false, std::memory_order_relaxed);
            active_ = nThreads_ + 1;
            ++generation_;
        }
        cvWork_.notify_all();

        runWorker(nThreads_);

        std::unique_lock<std::mutex> lk(m_);
        cvDone_.wait(lk, [this] { return done_.load(std::memory_order_relaxed); });
    }

private:
    ThreadPool() {
        const unsigned hw = std::thread::hardware_concurrency();
        nThreads_ = hw > 2 ? (int)(hw - 1) : 1;
        for (int i = 0; i < nThreads_; ++i)
            threads_.emplace_back([this, i] { workerLoop(i); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cvWork_.notify_all();
        for (auto& t : threads_) t.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void runWorker(int tid) {
        while (true) {
            int i = next_.fetch_add(64, std::memory_order_relaxed);
            if (i >= total_) break;
            int end = (std::min)(i + 64, total_);
            for (int k = i; k < end; ++k)
                job_(k, tid);
        }
        if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            done_.store(true, std::memory_order_relaxed);
            cvDone_.notify_one();
        }
    }

    void workerLoop(int tid) {
        int gen = 0;
        while (true) {
            std::unique_lock<std::mutex> lk(m_);
            cvWork_.wait(lk, [this, &gen] { return stop_ || generation_ != gen; });
            if (stop_) return;
            gen = generation_;
            lk.unlock();
            runWorker(tid);
        }
    }

    int nThreads_ = 1;
    std::vector<std::thread> threads_;

    std::mutex m_;
    std::condition_variable cvWork_;
    std::condition_variable cvDone_;
    std::function<void(int, int)> job_;
    int total_ = 0;
    std::atomic<int> next_{0};
    std::atomic<bool> done_{false};
    std::atomic<int> active_{0};
    int generation_ = 0;
    bool stop_ = false;
};

/**
 * @brief Squared L2 distance between two 64-D descriptors, vectorized with
 *        AVX2 (8-wide FMA) when the target supports it.
 */
inline float L2Sqr64(const float* a, const float* b) {
#if defined(__AVX2__) || defined(_M_AVX2)
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
#if defined(__AVX2__) || defined(_M_AVX2)
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
