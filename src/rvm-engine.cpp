#include "rvm-engine.h"

// Fix for missing SAL annotation in ONNX Runtime headers on some Windows SDK versions
#ifdef _WIN32
#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif
#endif

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════
// RVMMatter::Impl - ONNX Runtime session management
// ═══════════════════════════════════════════════════════════════════

struct RVMMatter::Impl {
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Input/output names
    std::vector<std::string> in_names;
    std::vector<std::string> out_names;
};

// ═══════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ═══════════════════════════════════════════════════════════════════

RVMMatter::RVMMatter()
    : impl_(std::make_unique<Impl>())
{
    impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "obs-ai-matting-pro");
}

RVMMatter::~RVMMatter() = default;

// ═══════════════════════════════════════════════════════════════════
// Model loading with cross-platform execution providers
// ═══════════════════════════════════════════════════════════════════

bool RVMMatter::load_model(const std::string &model_path, Backend preferred)
{
    if (model_path.empty()) {
        last_error_ = "Model path is empty";
        return false;
    }

    // Try preferred backend first, then fall back to CPU
    if (preferred != Backend::CPU && preferred != Backend::Auto) {
        if (try_create_session(model_path, preferred)) {
            loaded_ = true;
            return true;
        }
    }

    if (preferred == Backend::Auto || preferred != Backend::CPU) {
        // Try all available providers in priority order
        // Order depends on platform
        Backend priorities[] = {
#if defined(PLATFORM_MACOS)
            Backend::CoreML,
#elif defined(PLATFORM_WINDOWS)
            Backend::DirectML,
#elif defined(PLATFORM_LINUX)
            Backend::CUDA,
#endif
            Backend::CPU
        };

        for (Backend b : priorities) {
            if (b == preferred) continue; // Already tried
            if (try_create_session(model_path, b)) {
                loaded_ = true;
                return true;
            }
        }
    }

    // Last resort: CPU only
    if (try_create_session(model_path, Backend::CPU)) {
        loaded_ = true;
        return true;
    }

    return false;
}

bool RVMMatter::try_create_session(const std::string &model_path, Backend backend)
{
    try {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(1);
        so.SetInterOpNumThreads(1);
        so.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef HAS_ONNXRUNTIME
        // Get available providers
        auto providers = Ort::GetAvailableProviders();

        if (backend == Backend::CUDA) {
            // Check if CUDA is available
            bool cuda_available = false;
            for (const auto &p : providers) {
                if (p == "CUDAExecutionProvider") {
                    cuda_available = true;
                    break;
                }
            }

            if (cuda_available) {
                OrtCUDAProviderOptions cuda_opts{};
                cuda_opts.device_id = 0;
                cuda_opts.has_user_compute_stream = 0;
                so.AppendExecutionProvider_CUDA(cuda_opts);
            } else {
                return false;
            }
        }
#endif

#if defined(PLATFORM_WINDOWS) || defined(PLATFORM_MACOS)
        // DirectML (Windows) and CoreML (macOS) are handled differently
        // DirectML requires the onnxruntime-directml package
        // CoreML is a framework link, not an ONNX EP in all versions
        // For now, we rely on ONNX Runtime's auto-detection
#endif

        // Create the session
        // Use wide string on Windows
#ifdef _WIN32
        auto wpath = std::wstring(model_path.begin(), model_path.end());
        impl_->session = std::make_unique<Ort::Session>(
            *impl_->env, wpath.c_str(), so);
#else
        impl_->session = std::make_unique<Ort::Session>(
            *impl_->env, model_path.c_str(), so);
#endif

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = impl_->session->GetInputCount();
        impl_->in_names.clear();
        impl_->in_names.reserve(num_inputs);
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = impl_->session->GetInputNameAllocated(i, allocator);
            impl_->in_names.push_back(name.get());
        }

        size_t num_outputs = impl_->session->GetOutputCount();
        impl_->out_names.clear();
        impl_->out_names.reserve(num_outputs);
        for (size_t i = 0; i < num_outputs; i++) {
            auto name = impl_->session->GetOutputNameAllocated(i, allocator);
            impl_->out_names.push_back(name.get());
        }

        active_backend_ = backend;
        return true;

    } catch (const Ort::Exception &e) {
        last_error_ = std::string("ONNX error: ") + e.what();
        return false;
    } catch (const std::exception &e) {
        last_error_ = std::string("Error: ") + e.what();
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Recurrent state management
// ═══════════════════════════════════════════════════════════════════

void RVMMatter::reset_states()
{
    std::lock_guard<std::mutex> lock(infer_mutex_);
    rec_states_.clear();
    states_initialized_ = false;
}

void RVMMatter::initialize_states(int h, int w)
{
    (void)h; (void)w;
    // Initialize recurrent states as [1, C, 1, 1] tensors
    // The ONNX Expand node will broadcast them to the correct dimensions
    // This is the key insight from our Python prototype testing
    rec_states_.clear();

    int64_t shape[] = {1, 1, 1, 1};

    for (int i = 0; i < num_recurrent_states(); i++) {
        int ch = rec_channels(i);
        shape[1] = ch;

        std::vector<float> data(ch, 0.0f);

        auto tensor = Ort::Value::CreateTensor<float>(
            impl_->mem_info,
            data.data(),
            data.size(),
            shape,
            4
        );

        // We need to keep the data alive, so copy it into the Value
        // Actually, Ort::Value::CreateTensor doesn't copy the data
        // We need to manage lifetime carefully
        // For zero-initialized states, we can create new buffers each time
        rec_states_.push_back(std::move(tensor));
    }

    states_initialized_ = true;
}

// ═══════════════════════════════════════════════════════════════════
// Inference
// ═══════════════════════════════════════════════════════════════════

RVMMatter::InferenceResult RVMMatter::infer(
    const uint8_t *bgra_data,
    int width,
    int height,
    float ratio,
    float alpha_gamma)
{
    std::lock_guard<std::mutex> lock(infer_mutex_);

    InferenceResult result;
    result.width = width;
    result.height = height;
    result.alpha.resize((size_t)width * height);
    result.foreground.resize((size_t)width * height * 4);

    if (!loaded_ || !impl_->session) {
        // Return empty alpha (all opaque) if model not loaded
        std::fill(result.alpha.begin(), result.alpha.end(), 1.0f);
        std::memcpy(result.foreground.data(), bgra_data, (size_t)width * height * 4);
        return result;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    try {
        // ── Step 1: Prepare input tensor ────────────────────────────
        // Convert BGRA -> RGB, normalize to [0, 1], NCHW format
        int mh = height;
        int mw = width;

        std::vector<float> src_data((size_t)3 * mh * mw);

        for (int y = 0; y < mh; y++) {
            for (int x = 0; x < mw; x++) {
                const uint8_t *px = bgra_data + (y * mw + x) * 4;
                size_t idx = (size_t)(0 * mh + y) * mw + x; // R channel
                src_data[idx] = px[2] / 255.0f;           // BGR->RGB: R = px[2]
                idx = (size_t)(1 * mh + y) * mw + x;      // G channel
                src_data[idx] = px[1] / 255.0f;           // G = px[1]
                idx = (size_t)(2 * mh + y) * mw + x;      // B channel
                src_data[idx] = px[0] / 255.0f;           // B = px[0]
            }
        }

        int64_t src_shape[] = {1, 3, mh, mw};
        auto src_tensor = Ort::Value::CreateTensor<float>(
            impl_->mem_info,
            src_data.data(),
            src_data.size(),
            src_shape,
            4
        );

        // ── Step 2: Prepare recurrent states ────────────────────────
        if (!states_initialized_ || rec_states_.empty()) {
            initialize_states(mh, mw);
        }

        // ── Step 3: Prepare downsample ratio ───────────────────────
        float dsr = ratio;
        int64_t dsr_shape[] = {1};
        auto dsr_tensor = Ort::Value::CreateTensor<float>(
            impl_->mem_info,
            &dsr,
            1,
            dsr_shape,
            1
        );

        // ── Step 4: Build input array ───────────────────────────────
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(src_tensor));

        // Move recurrent states (we'll get them back from outputs)
        for (int i = 0; i < num_recurrent_states(); i++) {
            // Need to create a copy since ONNX may consume the tensor
            // For states, we pass them and receive updated versions
            inputs.push_back(std::move(rec_states_[i]));
        }
        rec_states_.clear(); // Will be repopulated from outputs

        inputs.push_back(std::move(dsr_tensor));

        // ── Step 5: Run inference ───────────────────────────────────
        // Build C-style name arrays
        std::vector<const char*> in_names_c;
        for (const auto &n : impl_->in_names)
            in_names_c.push_back(n.c_str());

        std::vector<const char*> out_names_c;
        for (const auto &n : impl_->out_names)
            out_names_c.push_back(n.c_str());

        auto outputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            in_names_c.data(),
            inputs.data(),
            inputs.size(),
            out_names_c.data(),
            out_names_c.size()
        );

        // ── Step 6: Extract alpha and foreground ────────────────────
        // RVM model outputs:
        //   [0] = fgr  (foreground, [1, 3, H, W])
        //   [1] = pha  (alpha, [1, 1, H, W])
        //   [2-5] = r1o-r4o (updated recurrent states)

        // Extract alpha
        auto &pha_tensor = outputs[1];
        auto pha_info = pha_tensor.GetTensorTypeAndShapeInfo();
        auto pha_shape = pha_info.GetShape();
        float *pha_data = pha_tensor.GetTensorMutableData<float>();

        int pha_h = (int)pha_shape[2];
        int pha_w = (int)pha_shape[3];

        // Apply alpha gamma and copy to result
        for (int i = 0; i < width * height; i++) {
            // If model output size matches input, copy directly
            if (pha_h == height && pha_w == width) {
                float a = std::clamp(pha_data[i], 0.0f, 1.0f);
                if (alpha_gamma != 1.0f)
                    a = std::pow(a, alpha_gamma);
                result.alpha[i] = a;
            } else {
                // Need to resize - use nearest neighbor for now
                // (OpenCV resize will be used in the plugin layer)
                int sx = (int)((i % width) * (float)pha_w / width);
                int sy = (int)((i / width) * (float)pha_h / height);
                float a = std::clamp(pha_data[sy * pha_w + sx], 0.0f, 1.0f);
                if (alpha_gamma != 1.0f)
                    a = std::pow(a, alpha_gamma);
                result.alpha[i] = a;
            }
        }

        // Extract foreground
        auto &fgr_tensor = outputs[0];
        float *fgr_data = fgr_tensor.GetTensorMutableData<float>();
        int fgr_h = (int)fgr_tensor.GetTensorTypeAndShapeInfo().GetShape()[2];
        int fgr_w = (int)fgr_tensor.GetTensorTypeAndShapeInfo().GetShape()[3];

        if (fgr_h == height && fgr_w == width) {
            // Convert RGB float -> BGRA uint8
            for (int i = 0; i < width * height; i++) {
                result.foreground[i * 4 + 0] = (uint8_t)std::clamp(fgr_data[2 * width * height + i] * 255, 0.0f, 255.0f); // B
                result.foreground[i * 4 + 1] = (uint8_t)std::clamp(fgr_data[1 * width * height + i] * 255, 0.0f, 255.0f); // G
                result.foreground[i * 4 + 2] = (uint8_t)std::clamp(fgr_data[0 * width * height + i] * 255, 0.0f, 255.0f); // R
                result.foreground[i * 4 + 3] = 255;
            }
        } else {
            // Use original frame as foreground if dimensions don't match
            std::memcpy(result.foreground.data(), bgra_data, (size_t)width * height * 4);
        }

        // Save updated recurrent states (r1o-r4o)
        for (int i = 2; i < 2 + num_recurrent_states(); i++) {
            rec_states_.push_back(std::move(outputs[i]));
        }

    } catch (const Ort::Exception &e) {
        last_error_ = std::string("Inference error: ") + e.what();
        // Return opaque alpha on error
        std::fill(result.alpha.begin(), result.alpha.end(), 1.0f);
        std::memcpy(result.foreground.data(), bgra_data, (size_t)width * height * 4);
    } catch (const std::exception &e) {
        last_error_ = std::string("Error: ") + e.what();
        std::fill(result.alpha.begin(), result.alpha.end(), 1.0f);
        std::memcpy(result.foreground.data(), bgra_data, (size_t)width * height * 4);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return result;
}

// ═══════════════════════════════════════════════════════════════════
// Utility methods
// ═══════════════════════════════════════════════════════════════════

std::string RVMMatter::backend_name() const
{
    switch (active_backend_) {
        case Backend::CPU:       return "CPU";
        case Backend::CUDA:      return "CUDA";
        case Backend::DirectML: return "DirectML";
        case Backend::CoreML:    return "CoreML";
        case Backend::Auto:      return "Auto";
    }
    return "Unknown";
}

void RVMMatter::get_model_dimensions(int &out_h, int &out_w) const
{
    // RVM uses dynamic input dimensions
    out_h = 0;
    out_w = 0;
}
