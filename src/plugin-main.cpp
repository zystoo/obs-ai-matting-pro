#include "plugin-main.h"
#include "rvm-engine.h"

#include <obs-module.h>
#include <obs-source.h>
#include <graphics/graphics.h>
#include <util/platform.h>

#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-ai-matting-pro", "en-US")

#define blog_info(...) blog(LOG_INFO, "[ai-matting-pro] " __VA_ARGS__)
#define blog_warn(...)  blog(LOG_WARNING, "[ai-matting-pro] " __VA_ARGS__)
#define blog_error(...) blog(LOG_ERROR, "[ai-matting-pro] " __VA_ARGS__)

// Store module data path at load time for later use
static std::string g_module_data_path;

// ═══════════════════════════════════════════════════════════════════
// Plugin registration
// ═══════════════════════════════════════════════════════════════════

static struct obs_source_info am_info = {};

bool obs_module_load(void)
{
    // Store module data path for default model loading
    obs_module_t *mod = obs_current_module();
    if (mod) {
        const char *dp = obs_get_module_data_path(mod);
        if (dp) {
            g_module_data_path = dp;
            blog_info("module data path: %s", dp);
        }
    }

    am_info.id = "obs_ai_matting_pro";
    am_info.type = OBS_SOURCE_TYPE_FILTER;
    am_info.output_flags = OBS_SOURCE_VIDEO;
    am_info.get_name = am_get_name;
    am_info.create = am_create;
    am_info.destroy = am_destroy;
    am_info.video_render = am_render;
    am_info.update = am_update;
    am_info.get_properties = am_get_properties;
    am_info.get_defaults = am_get_defaults;
    obs_register_source(&am_info);

    blog_info("plugin loaded successfully");
    return true;
}

void obs_module_unload(void)
{
    blog_info("plugin unloaded");
}

const char *obs_module_name(void)
{
    return "AI Matting Pro";
}

// ═══════════════════════════════════════════════════════════════════
// Filter name
// ═══════════════════════════════════════════════════════════════════

const char *am_get_name(void *type_data)
{
    (void)type_data;
    return obs_module_text("FilterName");
}

// ═══════════════════════════════════════════════════════════════════
// Create filter instance
// ═══════════════════════════════════════════════════════════════════

void *am_create(obs_data_t *settings, obs_source_t *source)
{
    auto *f = new am_filter();
    f->context = source;

    // Initialize GPU resources
    obs_enter_graphics();
    f->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    f->bg_texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    obs_leave_graphics();

    // Initialize RVM engine
    f->matter = std::make_unique<RVMMatter>();

    // Initialize auto LUT
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < 256; i++)
            f->auto_lut[c][i] = (uint8_t)i;

    // Start worker thread
    f->worker = std::thread(am_worker_loop, f);

    // Apply initial settings
    am_update(f, settings);

    blog_info("filter instance created");
    return f;
}

// ═══════════════════════════════════════════════════════════════════
// Destroy filter instance
// ═══════════════════════════════════════════════════════════════════

void am_destroy(void *data)
{
    auto *f = static_cast<am_filter *>(data);
    if (!f) return;

    f->stop = true;
    f->in_cv.notify_all();
    if (f->worker.joinable())
        f->worker.join();

    obs_enter_graphics();
    if (f->texrender)    gs_texrender_destroy(f->texrender);
    if (f->bg_texrender) gs_texrender_destroy(f->bg_texrender);
    if (f->stage)        gs_stagesurface_destroy(f->stage);
    if (f->bg_stage)     gs_stagesurface_destroy(f->bg_stage);
    if (f->out_tex)      gs_texture_destroy(f->out_tex);
    if (f->bg_weak)      obs_weak_source_release(f->bg_weak);
    obs_leave_graphics();

    delete f;
    blog_info("filter instance destroyed");
}

// ═══════════════════════════════════════════════════════════════════
// Update settings
// ═══════════════════════════════════════════════════════════════════

void am_update(void *data, obs_data_t *settings)
{
    auto *f = static_cast<am_filter *>(data);
    if (!f) return;

    f->mode.store(obs_data_get_int(settings, "mode"));
    f->blur_strength.store(obs_data_get_double(settings, "blur"));
    f->alpha_gamma.store(obs_data_get_double(settings, "alpha_gamma"));
    f->quality.store((int)obs_data_get_int(settings, "quality"));
    f->detail_ratio.store((int)obs_data_get_int(settings, "detail_ratio"));
    f->temporal_smooth.store(obs_data_get_double(settings, "temporal_smooth"));
    f->feather_radius.store(obs_data_get_double(settings, "feather_radius"));
    f->auto_match.store(obs_data_get_bool(settings, "auto_match"));
    f->match_strength.store(obs_data_get_double(settings, "match_strength"));
    f->solid_color = (uint32_t)obs_data_get_int(settings, "solid_color");

    // Model path - use default bundled model if not set
    const char *mp = obs_data_get_string(settings, "model_path");
    std::string model_path_str;
    if (mp && *mp) {
        model_path_str = mp;
    } else {
        // Try default model from plugin data directory (stored at module load)
        if (!g_module_data_path.empty()) {
            model_path_str = g_module_data_path + "/models/rvm_mobilenetv3_fp32.onnx";
        }
    }

    if (!model_path_str.empty() && model_path_str != f->model_path) {
        f->model_path = model_path_str;
        if (f->matter) {
            bool ok = f->matter->load_model(f->model_path);
            f->backend_name = ok ? f->matter->backend_name() : "failed";
            if (ok) {
                blog_info("model loaded: %s (backend: %s)",
                          f->model_path.c_str(),
                          f->backend_name.c_str());
            } else {
                blog_error("model load failed: %s",
                           f->matter->last_error().c_str());
            }
        }
    }

    // Background source for auto-match
    const char *bn = obs_data_get_string(settings, "bg_source");
    if (bn) {
        std::string new_bg = bn;
        if (new_bg != f->bg_name) {
            f->bg_name = new_bg;
            // Will be resolved in render
            if (f->bg_weak) {
                obs_weak_source_release(f->bg_weak);
                f->bg_weak = nullptr;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Video render (main processing pipeline)
// ═══════════════════════════════════════════════════════════════════

void am_render(void *data, gs_effect_t *effect)
{
    (void)effect;
    auto *f = static_cast<am_filter *>(data);
    if (!f || !f->context) return;

    obs_source_t *target = obs_filter_get_target(f->context);
    if (!target) {
        obs_source_skip_video_filter(f->context);
        return;
    }

    uint32_t w = obs_source_get_base_width(target);
    uint32_t h = obs_source_get_base_height(target);
    if (w == 0 || h == 0) {
        obs_source_skip_video_filter(f->context);
        return;
    }

    // Resize GPU resources if dimensions changed
    if (w != f->width || h != f->height) {
        f->width = w;
        f->height = h;
        obs_enter_graphics();
        if (f->stage) gs_stagesurface_destroy(f->stage);
        f->stage = gs_stagesurface_create(w, h, GS_BGRA);
        if (f->out_tex) gs_texture_destroy(f->out_tex);
        f->out_tex = gs_texture_create(w, h, GS_BGRA, 1, nullptr, GS_DYNAMIC);
        obs_leave_graphics();
        f->bgra_frame.assign((size_t)w * h * 4, 0);
        f->prev_alpha_valid = false; // Reset temporal smoothing
    }

    // ── Step 1: Render target source to texrender ─────────────────
    gs_texrender_reset(f->texrender);
    if (!gs_texrender_begin(f->texrender, w, h)) {
        obs_source_skip_video_filter(f->context);
        return;
    }

    struct vec4 clear_color;
    vec4_zero(&clear_color);
    gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
    gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    obs_source_video_render(target);
    gs_blend_state_pop();
    gs_texrender_end(f->texrender);

    // ── Step 2: Copy GPU texture to CPU ──────────────────────────
    gs_stage_texture(f->stage, gs_texrender_get_texture(f->texrender));
    uint8_t *mapped = nullptr;
    uint32_t linesize = 0;
    if (!gs_stagesurface_map(f->stage, &mapped, &linesize)) {
        obs_source_skip_video_filter(f->context);
        return;
    }

    // Copy to our buffer
    for (uint32_t y = 0; y < h; y++)
        std::memcpy(f->bgra_frame.data() + (size_t)y * w * 4,
                    mapped + (size_t)y * linesize, (size_t)w * 4);
    gs_stagesurface_unmap(f->stage);

    // ── Step 3: Send frame to worker thread ──────────────────────
    {
        std::lock_guard<std::mutex> lk(f->in_mtx);
        f->worker_input = f->bgra_frame;
        f->worker_w = (int)w;
        f->worker_h = (int)h;
        f->has_new_input = true;
    }
    f->in_cv.notify_one();

    // ── Step 4: Get latest alpha from worker (non-blocking) ──────
    std::vector<float> alpha;
    std::vector<uint8_t> foreground;
    int aw = 0, ah = 0;
    bool have_output = false;

    {
        std::lock_guard<std::mutex> lk(f->out_mtx);
        if (f->has_output) {
            alpha = f->worker_alpha;
            foreground = f->worker_fg;
            aw = f->worker_w;
            ah = f->worker_h;
            have_output = true;
        }
    }

    // ── Step 5: Composite output ─────────────────────────────────
    if (!have_output || alpha.empty()) {
        // No output yet - pass through original frame
        gs_texture_set_image(f->out_tex, f->bgra_frame.data(), w * 4, false);
    } else {
        // Apply post-processing
        // Resize alpha if dimensions don't match
        if (aw != (int)w || ah != (int)h) {
            // Nearest-neighbor resize for now
            std::vector<float> resized((size_t)w * h);
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    int sx = (int)(x * aw / w);
                    int sy = (int)(y * ah / h);
                    sx = std::clamp(sx, 0, aw - 1);
                    sy = std::clamp(sy, 0, ah - 1);
                    resized[y * w + x] = alpha[sy * aw + sx];
                }
            }
            alpha = std::move(resized);
        }

        // Temporal smoothing
        float ts = (float)f->temporal_smooth.load();
        if (ts > 0.0 && f->prev_alpha_valid && f->prev_alpha.size() == alpha.size()) {
            for (size_t i = 0; i < alpha.size(); i++)
                alpha[i] = ts * f->prev_alpha[i] + (1.0f - ts) * alpha[i];
        }
        f->prev_alpha = alpha;
        f->prev_alpha_valid = true;

        // Edge feathering
        int feather = (int)f->feather_radius.load();
        if (feather > 0)
            am_feather_alpha(alpha.data(), (int)w, (int)h, feather);

        // Composite
        am_composite(f, f->bgra_frame.data(), alpha.data(),
                     foreground.data(), (int)w, (int)h,
                     f->mode.load());
        gs_texture_set_image(f->out_tex, f->bgra_frame.data(), w * 4, false);
    }

    // ── Step 6: Draw output texture ───────────────────────────────
    int mode = f->mode.load();
    gs_effect_t *def = obs_get_base_effect(
        mode == 0 ? OBS_EFFECT_PREMULTIPLIED_ALPHA : OBS_EFFECT_DEFAULT);

    gs_eparam_t *image = gs_effect_get_param_by_name(def, "image");
    gs_effect_set_texture(image, f->out_tex);

    while (gs_effect_loop(def, "Draw"))
        gs_draw_sprite(f->out_tex, 0, w, h);

    // ── Auto light match: sample background ───────────────────────
    if (f->auto_match.load() && mode == 0) {
        am_sample_background(f);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Worker thread - runs RVM inference asynchronously
// ═══════════════════════════════════════════════════════════════════

void am_worker_loop(am_filter *f)
{
    while (!f->stop) {
        // Wait for input
        std::vector<uint8_t> input;
        int w = 0, h = 0;

        {
            std::unique_lock<std::mutex> lk(f->in_mtx);
            f->in_cv.wait(lk, [f] { return f->has_new_input || f->stop; });
            if (f->stop) break;

            if (f->has_new_input) {
                input = std::move(f->worker_input);
                w = f->worker_w;
                h = f->worker_h;
                f->has_new_input = false;
            } else {
                continue;
            }
        }

        if (!f->matter || !f->matter->is_loaded()) {
            // Model not loaded - output transparent (all opaque)
            std::lock_guard<std::mutex> lk(f->out_mtx);
            f->worker_alpha.assign((size_t)w * h, 1.0f);
            f->worker_fg = input;
            f->worker_w = w;
            f->worker_h = h;
            f->has_output = true;
            continue;
        }

        // Run inference
        float ratio = (float)f->detail_ratio.load() / 100.0f;
        float ag = (float)f->alpha_gamma.load();

        auto result = f->matter->infer(input.data(), w, h, ratio, ag);

        // Store output
        {
            std::lock_guard<std::mutex> lk(f->out_mtx);
            f->worker_alpha = std::move(result.alpha);
            f->worker_fg = std::move(result.foreground);
            f->worker_w = result.width;
            f->worker_h = result.height;
            f->last_inference_ms = result.inference_ms;
            f->has_output = true;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Compositing: blend foreground with background based on mode
// ═══════════════════════════════════════════════════════════════════

void am_composite(am_filter *f, const uint8_t *bgra,
                  const float *alpha, const uint8_t *fg,
                  int w, int h, int mode)
{
    // Apply auto LUT if auto-match is enabled
    bool use_lut = f->auto_match.load() && mode == 0;

    if (mode == 0) {
        // Transparent mode: output premultiplied alpha
        for (int i = 0; i < w * h; i++) {
            float a = alpha[i];
            uint8_t alpha_byte = (uint8_t)(a * 255 + 0.5f);

            // Use model foreground if available, otherwise use original
            uint8_t b = fg ? fg[i * 4 + 0] : bgra[i * 4 + 0];
            uint8_t g = fg ? fg[i * 4 + 1] : bgra[i * 4 + 1];
            uint8_t r = fg ? fg[i * 4 + 2] : bgra[i * 4 + 2];

            if (use_lut) {
                b = f->auto_lut[0][b];
                g = f->auto_lut[1][g];
                r = f->auto_lut[2][r];
            }

            // Premultiply alpha
            const_cast<uint8_t*>(bgra)[i * 4 + 0] = (uint8_t)(b * a + 0.5f);
            const_cast<uint8_t*>(bgra)[i * 4 + 1] = (uint8_t)(g * a + 0.5f);
            const_cast<uint8_t*>(bgra)[i * 4 + 2] = (uint8_t)(r * a + 0.5f);
            const_cast<uint8_t*>(bgra)[i * 4 + 3] = alpha_byte;
        }
    } else if (mode == 1) {
        // Blur mode: blend original with blurred background
        // (In production, use Gaussian or bilateral filter)
        for (int i = 0; i < w * h; i++) {
            float a = alpha[i];
            float bg_b = bgra[i * 4 + 0];
            float bg_g = bgra[i * 4 + 1];
            float bg_r = bgra[i * 4 + 2];

            // Simple darkening for blur effect
            // (Full Gaussian blur would be done on GPU in production)
            bg_b *= 0.5f;
            bg_g *= 0.5f;
            bg_r *= 0.5f;

            uint8_t fg_b = fg ? fg[i * 4 + 0] : bgra[i * 4 + 0];
            uint8_t fg_g = fg ? fg[i * 4 + 1] : bgra[i * 4 + 1];
            uint8_t fg_r = fg ? fg[i * 4 + 2] : bgra[i * 4 + 2];

            const_cast<uint8_t*>(bgra)[i * 4 + 0] =
                (uint8_t)(fg_b * a + bg_b * (1 - a) + 0.5f);
            const_cast<uint8_t*>(bgra)[i * 4 + 1] =
                (uint8_t)(fg_g * a + bg_g * (1 - a) + 0.5f);
            const_cast<uint8_t*>(bgra)[i * 4 + 2] =
                (uint8_t)(fg_r * a + bg_r * (1 - a) + 0.5f);
            const_cast<uint8_t*>(bgra)[i * 4 + 3] = 255;
        }
    } else {
        // Solid color mode: composite over solid color
        uint32_t sc = f->solid_color;
        uint8_t sc_b = (sc >> 16) & 0xFF;
        uint8_t sc_g = (sc >> 8) & 0xFF;
        uint8_t sc_r = sc & 0xFF;

        for (int i = 0; i < w * h; i++) {
            float a = alpha[i];
            uint8_t fg_b = fg ? fg[i * 4 + 0] : bgra[i * 4 + 0];
            uint8_t fg_g = fg ? fg[i * 4 + 1] : bgra[i * 4 + 1];
            uint8_t fg_r = fg ? fg[i * 4 + 2] : bgra[i * 4 + 2];

            const_cast<uint8_t*>(bgra)[i * 4 + 0] =
                (uint8_t)(fg_b * a + sc_b * (1 - a) + 0.5f);
            const_cast<uint8_t*>(bgra)[i * 4 + 1] =
                (uint8_t)(fg_g * a + sc_g * (1 - a) + 0.5f);
            const_cast<uint8_t*>(bgra)[i * 4 + 2] =
                (uint8_t)(fg_r * a + sc_r * (1 - a) + 0.5f);
            const_cast<uint8_t*>(bgra)[i * 4 + 3] = 255;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Post-processing: alpha feathering (edge smoothing)
// ═══════════════════════════════════════════════════════════════════

void am_feather_alpha(float *alpha, int w, int h, int radius)
{
    if (radius <= 0 || w <= 0 || h <= 0) return;

    // Create a copy for box blur
    std::vector<float> blurred(alpha, alpha + (size_t)w * h);

    // Simple separable box blur
    std::vector<float> temp((size_t)w * h);

    // Horizontal pass
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0;
            int count = 0;
            for (int k = -radius; k <= radius; k++) {
                int xx = x + k;
                if (xx >= 0 && xx < w) {
                    sum += blurred[y * w + xx];
                    count++;
                }
            }
            temp[y * w + x] = sum / count;
        }
    }

    // Vertical pass
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0;
            int count = 0;
            for (int k = -radius; k <= radius; k++) {
                int yy = y + k;
                if (yy >= 0 && yy < h) {
                    sum += temp[yy * w + x];
                    count++;
                }
            }
            blurred[y * w + x] = sum / count;
        }
    }

    // Blend: only feather edge regions (0.1 < alpha < 0.9)
    for (int i = 0; i < w * h; i++) {
        float a = alpha[i];
        if (a > 0.1f && a < 0.9f) {
            alpha[i] = 0.5f * a + 0.5f * blurred[i];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Auto light match: sample background source
// ═══════════════════════════════════════════════════════════════════

void am_sample_background(am_filter *f)
{
    if (f->bg_name.empty()) return;

    // Resolve background source
    if (!f->bg_weak) {
        obs_source_t *bg = obs_get_source_by_name(f->bg_name.c_str());
        if (bg) {
            f->bg_weak = obs_source_get_weak_source(bg);
            obs_source_release(bg);
        }
    }

    if (!f->bg_weak) return;

    obs_source_t *bg = obs_weak_source_get_source(f->bg_weak);
    if (!bg) return;

    // Render background at small size for sampling
    const int SW = 64, SH = 36;
    if (!f->bg_stage || f->bg_stage_width != SW) {
        obs_enter_graphics();
        if (f->bg_stage) gs_stagesurface_destroy(f->bg_stage);
        f->bg_stage = gs_stagesurface_create(SW, SH, GS_BGRA);
        f->bg_stage_width = SW;
        f->bg_stage_height = SH;
        obs_leave_graphics();
    }

    f->rendering_bg = true;
    gs_texrender_reset(f->bg_texrender);
    if (gs_texrender_begin(f->bg_texrender, SW, SH)) {
        struct vec4 cc;
        vec4_zero(&cc);
        gs_clear(GS_CLEAR_COLOR, &cc, 0, 0);
        gs_ortho(0, SW, 0, SH, -100, 100);
        obs_source_video_render(bg);
        gs_texrender_end(f->bg_texrender);
    }
    f->rendering_bg = false;

    obs_enter_graphics();
    gs_stage_texture(f->bg_stage, gs_texrender_get_texture(f->bg_texrender));
    uint8_t *map = nullptr;
    uint32_t linesize = 0;
    if (gs_stagesurface_map(f->bg_stage, &map, &linesize)) {
        // Compute mean BGR
        double sum[3] = {0, 0, 0};
        int count = 0;
        for (int y = 0; y < SH; y++) {
            for (int x = 0; x < SW; x++) {
                sum[0] += map[y * linesize + x * 4 + 0]; // B
                sum[1] += map[y * linesize + x * 4 + 1]; // G
                sum[2] += map[y * linesize + x * 4 + 2]; // R
                count++;
            }
        }
        gs_stagesurface_unmap(f->bg_stage);

        float bg_m[3] = {
            (float)(sum[0] / count),
            (float)(sum[1] / count),
            (float)(sum[2] / count)
        };

        // EMA smoothing
        float ema = 0.1f;
        f->bg_mean[0] = ema * bg_m[0] + (1 - ema) * f->bg_mean[0];
        f->bg_mean[1] = ema * bg_m[1] + (1 - ema) * f->bg_mean[1];
        f->bg_mean[2] = ema * bg_m[2] + (1 - ema) * f->bg_mean[2];

        am_update_auto_lut(f);
    }
    obs_leave_graphics();

    obs_source_release(bg);
}

void am_update_auto_lut(am_filter *f)
{
    // Compute gain to match foreground to background lighting
    // This is a simplified version of the obs-ai-matting approach
    float strength = (float)f->match_strength.load();

    for (int c = 0; c < 3; c++) {
        float target = f->bg_mean[c];
        if (target < 1.0f) target = 128.0f; // Default to mid-gray

        float gain = 1.0f + strength * ((target - 128.0f) / 128.0f);
        gain = std::clamp(gain, 0.5f, 2.0f);

        for (int i = 0; i < 256; i++) {
            float val = i * gain;
            f->auto_lut[c][i] = (uint8_t)std::clamp(val, 0.0f, 255.0f);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// UI: Properties panel
// ═══════════════════════════════════════════════════════════════════

static bool am_mode_changed(obs_properties_t *props,
                            obs_property_t *p,
                            obs_data_t *settings)
{
    (void)p;
    int mode = (int)obs_data_get_int(settings, "mode");

    // Show auto-match only in transparent mode
    obs_property_set_visible(
        obs_properties_get(props, "auto_match"), mode == 0);
    obs_property_set_visible(
        obs_properties_get(props, "bg_source"), mode == 0);
    obs_property_set_visible(
        obs_properties_get(props, "match_strength"), mode == 0);

    // Show solid color only in solid mode
    obs_property_set_visible(
        obs_properties_get(props, "solid_color"), mode == 2);

    // Show blur only in blur mode
    obs_property_set_visible(
        obs_properties_get(props, "blur"), mode == 1);

    return true;
}

obs_properties_t *am_get_properties(void *data)
{
    auto *f = static_cast<am_filter *>(data);
    obs_properties_t *p = obs_properties_create();

    // ── Mode selection ─────────────────────────────────────────────
    obs_property_t *m = obs_properties_add_list(p, "mode",
        obs_module_text("Mode"),
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(m, obs_module_text("Mode.Transparent"), 0);
    obs_property_list_add_int(m, obs_module_text("Mode.Blur"), 1);
    obs_property_list_add_int(m, obs_module_text("Mode.SolidColor"), 2);
    obs_property_set_modified_callback(m, am_mode_changed);

    // ── Blur strength ─────────────────────────────────────────────
    obs_properties_add_float_slider(p, "blur",
        obs_module_text("BlurStrength"), 4, 60, 1);

    // ── Solid color ───────────────────────────────────────────────
    obs_properties_add_color(p, "solid_color",
        obs_module_text("SolidColor"));

    // ── Auto light match ──────────────────────────────────────────
    obs_properties_add_bool(p, "auto_match",
        obs_module_text("AutoMatch"));

    // Background source for auto-match
    obs_property_t *bl = obs_properties_add_list(p, "bg_source",
        obs_module_text("BgSource"),
        OBS_COMBO_TYPE_EDITABLE, OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(bl,
        obs_module_text("BgSource.None"), "");

    // Enumerate sources for background
    if (f && f->context) {
        // Would enumerate scenes and sources here
        // (omitted for brevity - same as obs-ai-matting)
    }

    obs_properties_add_float_slider(p, "match_strength",
        obs_module_text("MatchStrength"), 0.0, 1.0, 0.05);

    // ── Alpha / quality settings ──────────────────────────────────
    obs_properties_add_float_slider(p, "alpha_gamma",
        obs_module_text("AlphaGamma"), 0.4, 1.5, 0.05);

    obs_property_t *q = obs_properties_add_list(p, "quality",
        obs_module_text("Quality"),
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(q, obs_module_text("Quality.Fast"), 384);
    obs_property_list_add_int(q, obs_module_text("Quality.Medium"), 512);
    obs_property_list_add_int(q, obs_module_text("Quality.High"), 720);

    obs_property_t *d = obs_properties_add_list(p, "detail_ratio",
        obs_module_text("Detail"),
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(d, obs_module_text("Detail.Fast"), 40);
    obs_property_list_add_int(d, obs_module_text("Detail.Balanced"), 75);
    obs_property_list_add_int(d, obs_module_text("Detail.Max"), 100);

    // ── Advanced settings ─────────────────────────────────────────
    obs_properties_t *adv = obs_properties_create();

    obs_properties_add_float_slider(adv, "temporal_smooth",
        obs_module_text("TemporalSmooth"), 0.0, 0.98, 0.01);

    obs_properties_add_float_slider(adv, "feather_radius",
        obs_module_text("FeatherRadius"), 0, 10, 1);

    obs_properties_add_group(p, "advanced",
        obs_module_text("AdvancedSettings"),
        OBS_GROUP_NORMAL, adv);

    // ── Model path ────────────────────────────────────────────────
    obs_properties_add_path(p, "model_path",
        obs_module_text("ModelPath"),
        OBS_PATH_FILE,
        obs_module_text("ModelFilter"),
        nullptr);

    // ── Backend info ───────────────────────────────────────────────
    if (f && f->matter && f->matter->is_loaded()) {
        std::string info = "Backend: " + f->backend_name +
                           " | Inference: " +
                           std::to_string(f->last_inference_ms) + "ms";
        obs_properties_add_text(p, "backend_info",
            obs_module_text("BackendInfo"),
            OBS_TEXT_DEFAULT);
    }

    return p;
}

// ═══════════════════════════════════════════════════════════════════
// Default settings
// ═══════════════════════════════════════════════════════════════════

void am_get_defaults(obs_data_t *settings)
{
    obs_data_set_default_int(settings, "mode", 0);
    obs_data_set_default_double(settings, "blur", 25.0);
    obs_data_set_default_double(settings, "alpha_gamma", 1.0);
    obs_data_set_default_int(settings, "quality", 512);
    obs_data_set_default_int(settings, "detail_ratio", 75);
    obs_data_set_default_double(settings, "temporal_smooth", 0.85);
    obs_data_set_default_double(settings, "feather_radius", 3.0);
    obs_data_set_default_bool(settings, "auto_match", false);
    obs_data_set_default_double(settings, "match_strength", 0.7);
    obs_data_set_default_int(settings, "solid_color", 0xFF1E1E2E);
}
