#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>

namespace Ort {
    class Env;
    class Session;
    class MemoryInfo;
    class Value;
}

class RVMMatter {
public:
    struct InferenceResult {
        std::vector<float> alpha;
        std::vector<uint8_t> foreground;
        int width = 0;
        int height = 0;
        double inference_ms = 0.0;
    };

    enum class Backend {
        CPU,
        CUDA,
        DirectML,
        CoreML,
        Auto
    };

    RVMMatter();
    ~RVMMatter();

    bool load_model(const std::string &model_path, Backend preferred = Backend::Auto);
    bool is_loaded() const { return loaded_; }
    const std::string &last_error() const { return last_error_; }
    Backend active_backend() const { return active_backend_; }
    std::string backend_name() const;

    InferenceResult infer(const uint8_t *bgra_data,
                          int width,
                          int height,
                          float ratio = 0.25f,
                          float alpha_gamma = 1.0f);

    void reset_states();

    void get_model_dimensions(int &out_h, int &out_w) const;
    static constexpr int num_recurrent_states() { return 4; }
    static constexpr int rec_channels(int i) {
        constexpr int ch[] = {16, 20, 40, 64};
        return ch[i];
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool loaded_ = false;
    Backend active_backend_ = Backend::CPU;
    std::string last_error_;

    std::vector<Ort::Value> rec_states_;
    bool states_initialized_ = false;
    std::mutex infer_mutex_;

    bool try_create_session(const std::string &model_path, Backend backend);
    void initialize_states(int h, int w);
};
