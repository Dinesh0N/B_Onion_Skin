#ifndef ONION_SKIN_THREAD_POOL_HPP
#define ONION_SKIN_THREAD_POOL_HPP

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace onion_internal {

// Persistent zero-allocation worker pool scaled to all available CPU threads
class FastThreadPool {
public:
    static FastThreadPool& instance() {
        static FastThreadPool pool;
        return pool;
    }

    int hardware_threads() const {
        unsigned int hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4;
        return std::clamp(static_cast<int>(hw), 1, 32);
    }

    void ensure_started(int desired_workers) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (running_ || desired_workers <= 0) return;
        running_ = true;
        stop_ = false;
        workers_.reserve(static_cast<size_t>(desired_workers));
        for (int i = 0; i < desired_workers; ++i) {
            workers_.emplace_back([this, i]() { worker_loop(i); });
        }
    }

    void shutdown() {
        std::vector<std::thread> to_join;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (!running_) return;
            stop_ = true;
            running_ = false;
            ++generation_;
            to_join.swap(workers_);
        }
        cv_work_.notify_all();
        for (auto& t : to_join) {
            if (t.joinable()) t.join();
        }
    }

    ~FastThreadPool() {
#if defined(_WIN32)
        // Avoid deadlock on Windows Loader Lock if process exits without explicit shutdown
        for (auto& t : workers_) {
            if (t.joinable()) t.detach();
        }
#else
        shutdown();
#endif
    }

    template <typename Fn>
    void parallel_for_chunks(int total_chunks, Fn&& fn) {
        if (total_chunks <= 1) {
            fn(0);
            return;
        }
        const int hw = hardware_threads();
        const int max_workers = std::clamp(hw - 1, 1, 31);
        const int num_workers = std::min(total_chunks - 1, max_workers);
        ensure_started(std::min(std::max(num_workers, max_workers), 31));

        std::unique_lock<std::mutex> dispatch_lock(dispatch_mtx_);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (!running_ || workers_.empty()) {
                dispatch_lock.unlock();
                for (int c = 0; c < total_chunks; ++c) fn(c);
                return;
            }
            active_tasks_ = std::min(num_workers, static_cast<int>(workers_.size()));
            pending_.store(active_tasks_, std::memory_order_release);
            task_fn_ = [&fn](int idx) { fn(idx + 1); };
            ++generation_;
        }
        const int spawned = active_tasks_;
        cv_work_.notify_all();

        // Main thread processes chunk 0 + any overflow chunks beyond worker count
        fn(0);
        for (int extra = spawned + 1; extra < total_chunks; ++extra) {
            fn(extra);
        }

        if (pending_.load(std::memory_order_acquire) > 0) {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_done_.wait(lock, [this]() { return pending_.load(std::memory_order_acquire) == 0; });
        }
    }

private:
    FastThreadPool() = default;

    void worker_loop(int worker_id) {
        uint64_t seen_gen = 0;
        while (true) {
            std::function<void(int)> local_fn;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_work_.wait(lock, [this, &seen_gen]() {
                    return stop_ || generation_ != seen_gen;
                });
                if (stop_) return;
                seen_gen = generation_;
                if (worker_id >= active_tasks_) continue;
                local_fn = task_fn_;
            }
            if (local_fn) {
                local_fn(worker_id);
            }
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(mtx_);
                cv_done_.notify_all();
            }
        }
    }

    std::mutex dispatch_mtx_;
    std::mutex mtx_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::vector<std::thread> workers_;
    std::function<void(int)> task_fn_;
    uint64_t generation_ = 0;
    int active_tasks_ = 0;
    std::atomic<int> pending_{0};
    bool running_ = false;
    bool stop_ = false;
};

}  // namespace onion_internal

#endif  // ONION_SKIN_THREAD_POOL_HPP
