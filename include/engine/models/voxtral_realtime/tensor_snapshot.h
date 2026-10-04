#pragma once

// Speakrail patch (live peek): a device-side copy of a set of backend tensors. save() copies them into a
// shadow buffer on the same backend (allocated once, on the first save), restore() copies them back, so a
// peek can run throw-away stream steps on the live caches and put them back exactly as they were. The copies
// stay on the device (ggml_backend_tensor_copy between two buffers of the same backend).

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <stdexcept>
#include <vector>

namespace engine::models::voxtral_realtime {

class TensorSnapshot {
public:
    TensorSnapshot() = default;
    TensorSnapshot(const TensorSnapshot &) = delete;
    TensorSnapshot & operator=(const TensorSnapshot &) = delete;
    ~TensorSnapshot() { release(); }

    void save(ggml_backend_t backend, const std::vector<ggml_tensor *> & tensors) {
        if (ctx_ != nullptr && !matches(tensors)) {
            release();
        }
        if (ctx_ == nullptr) {
            allocate(backend, tensors);
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            ggml_backend_tensor_copy(tensors[i], shadow_[i]);
        }
        ggml_backend_synchronize(backend);
    }

    void restore(ggml_backend_t backend, const std::vector<ggml_tensor *> & tensors) const {
        if (ctx_ == nullptr || !matches(tensors)) {
            throw std::runtime_error("VoxTral peek: restore without a matching snapshot");
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            ggml_backend_tensor_copy(shadow_[i], tensors[i]);
        }
        ggml_backend_synchronize(backend);
    }

private:
    bool matches(const std::vector<ggml_tensor *> & tensors) const {
        if (tensors.size() != shadow_.size()) {
            return false;
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i]->type != shadow_[i]->type || ggml_nbytes(tensors[i]) != ggml_nbytes(shadow_[i])) {
                return false;
            }
        }
        return true;
    }

    void allocate(ggml_backend_t backend, const std::vector<ggml_tensor *> & tensors) {
        ggml_init_params params{ggml_tensor_overhead() * (tensors.size() + 8), nullptr, true};
        ctx_ = ggml_init(params);
        if (ctx_ == nullptr) {
            throw std::runtime_error("VoxTral peek: failed to create the snapshot context");
        }
        shadow_.clear();
        for (ggml_tensor * t : tensors) {
            shadow_.push_back(ggml_new_tensor(ctx_, t->type, GGML_MAX_DIMS, t->ne));
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
        if (buffer_ == nullptr) {
            release();
            throw std::runtime_error("VoxTral peek: failed to allocate the snapshot buffer");
        }
    }

    void release() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        shadow_.clear();
    }

    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<ggml_tensor *> shadow_;
};

}  // namespace engine::models::voxtral_realtime
