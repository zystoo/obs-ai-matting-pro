#pragma once

#include <obs-module.h>
#include <obs-source.h>
#include <graphics/graphics.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

#include "rvm-engine.h"

struct am_filter {
    obs_source_t *context = nullptr;
    gs_texrender_t *texrender = nullptr;
    gs_stagesurf_t *stage = nullptr;
    gs_texture_t *out_tex = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;

    std::atomic<int> mode{0};
    std::atomic<double> blur_strength{25.0};
    std::atomic<double> alpha_gamma{1.0};
    std::atomic<int> quality{512};
    std::atomic<int> detail_ratio{75};
    std::atomic<double> temporal_smooth{0.85};
    std::atomic<double> feather_radius{3.0};
    std::atomic<bool> auto_match{false};
    std::atomic<double> match_strength{0.7};
    std::string model_path;
    std::string bg_name;
    uint32_t solid_color = 0xFF1E1E2E;

    std::unique_ptr<RVMMatter> matter;

    std::vector<uint8_t> bgra_frame;
    std::vector<uint8_t> worker_input;
    std::vector<float> worker_alpha;
    std::vector<uint8_t> worker_fg;
    int worker_w = 0;
    int worker_h = 0;

    std::vector<float> prev_alpha;
    bool prev_alpha_valid = false;

    std::thread worker;
    std::mutex in_mtx;
    std::condition_variable in_cv;
    std::mutex out_mtx;
    std::atomic<bool> stop{false};
    std::atomic<bool> has_new_input{false};
    std::atomic<bool> has_output{false};
    double last_inference_ms = 0.0;

    obs_weak_source_t *bg_weak = nullptr;
    gs_texrender_t *bg_texrender = nullptr;
    gs_stagesurf_t *bg_stage = nullptr;
    uint32_t bg_stage_width = 0;
    uint32_t bg_stage_height = 0;
    float bg_mean[3] = {0, 0, 0};
    float auto_gain[3] = {1, 1, 1};
    uint8_t auto_lut[3][256];
    bool rendering_bg = false;

    std::string backend_name;

    ~am_filter()
    {
        stop = true;
        in_cv.notify_all();
        if (worker.joinable())
            worker.join();
    }
};

bool obs_module_load(void);
void obs_module_unload(void);
const char *obs_module_name(void);

const char *am_get_name(void *type_data);
void *am_create(obs_data_t *settings, obs_source_t *source);
void am_destroy(void *data);
void am_update(void *data, obs_data_t *settings);
void am_render(void *data, gs_effect_t *effect);
obs_properties_t *am_get_properties(void *data);
void am_get_defaults(obs_data_t *settings);

void am_worker_loop(am_filter *f);

void am_run_inference(am_filter *f);
void am_composite(am_filter *f, const uint8_t *bgra,
                  const float *alpha, const uint8_t *fg,
                  int w, int h, int mode);

void am_feather_alpha(float *alpha, int w, int h, int radius);
void am_temporal_smooth(float *alpha, int w, int h,
                        std::vector<float> &prev, float coeff);
void am_apply_lut(uint8_t *bgra, int w, int h, const uint8_t lut[3][256]);

void am_sample_background(am_filter *f);
void am_update_auto_lut(am_filter *f);
