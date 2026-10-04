#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/voxtral_realtime/assets.h"
#include "engine/models/voxtral_realtime/audio_encoder.h"
#include "engine/models/voxtral_realtime/frontend.h"
#include "engine/models/voxtral_realtime/text_decoder.h"
#include "engine/models/voxtral_realtime/tokenizer_text.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::voxtral_realtime {

class VoxtralRealtimeSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    VoxtralRealtimeSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const VoxtralRealtimeAssets> assets);
    ~VoxtralRealtimeSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finish_stream() override;
    runtime::TaskResult finalize() override;

    // Speakrail patch (live peek): decode `n_frames` steps of silence after the
    // audio received so far and return their steps (token, text, frame, turn head) -- the words the live
    // stream WILL emit for audio it has already heard, the delay line drained early. Every piece of stream
    // state (encoder and decoder caches on the device, the session's text / step / audio bookkeeping) is
    // snapshotted first and restored afterwards, so the live stream continues exactly as if nothing happened.
    // Call it only between process_audio_chunk calls on the stream's own thread.
    runtime::StreamEvent peek_stream(int64_t n_frames);

private:
    VoxtralRealtimeRequest make_request(const runtime::TaskRequest & request, bool streaming) const;
    runtime::TaskResult run_single(const VoxtralRealtimeRequest & request, bool first_chunk);
    runtime::StreamEvent process_available_stream_chunks();
    runtime::StreamEvent process_one_stream_chunk(const runtime::AudioBuffer & audio);
    // Feeds a freshly decoded token back as the next stream input and, when it carries text,
    // appends it to the transcript. `frame` is the audio frame the token describes (Speakrail
    // patch: the prefill's token is frame 0, every stream step advances one 80 ms frame); the
    // step is queued for the next event with the turn head's probabilities, when a head is loaded.
    void record_stream_token(int32_t token, int64_t frame);
    // Moves the transcript decoded since the last partial onto the event. Called once a chunk
    // rather than once a token, so a chunk that decoded a batch of them reports all their text.
    void take_stream_delta(runtime::StreamEvent & event);

    runtime::TaskSpec task_;
    std::shared_ptr<const VoxtralRealtimeAssets> assets_;
    size_t audio_encoder_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t audio_encoder_weight_context_bytes_ = 128ull * 1024ull * 1024ull;
    size_t text_decoder_prefill_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t text_decoder_decode_graph_arena_bytes_ = 512ull * 1024ull * 1024ull;
    size_t text_decoder_weight_context_bytes_ = 128ull * 1024ull * 1024ull;
    assets::TensorStorageType audio_encoder_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType text_decoder_weight_storage_type_ = assets::TensorStorageType::Native;
    int64_t stream_decode_cache_steps_ = 1024;
    int64_t stream_batch_tokens_ = 1;
    // Speakrail patch (session options): `voxtral_realtime.turn_head` = exported head file;
    // `voxtral_realtime.stream_chunk_samples` = how many samples a live source is read in
    // (default: one steady chunk, 1680 = 105 ms; smaller values cut the wait for a full read).
    std::string turn_head_path_;
    int64_t stream_chunk_samples_ = 0;
    VoxtralRealtimeTokenizer tokenizer_;
    VoxtralRealtimeFrontend frontend_;
    VoxtralRealtimeAudioEncoderRuntime audio_encoder_;
    VoxtralRealtimeTextDecoderRuntime text_decoder_;
    runtime::TaskResult streaming_result_;
    runtime::AudioBuffer streaming_audio_;
    size_t streaming_audio_offset_values_ = 0;
    int64_t streaming_steps_processed_ = 0;
    // Per-stage wall time summed over every steady step, reported by finalize().
    double stream_frontend_ms_ = 0.0;
    double stream_encoder_ms_ = 0.0;
    double stream_decoder_ms_ = 0.0;
    VoxtralRealtimeGenerationOptions streaming_generation_;
    runtime::StreamEventCallback stream_event_sink_;
    VoxtralRealtimeFrontendStreamState frontend_stream_state_;
    VoxtralRealtimeAudioEncoderStreamState audio_stream_state_;
    // The transcript decoded so far, and how much of it has already gone out as a partial. Every
    // partial is the suffix between the two, so the deltas concatenate to exactly this string.
    std::string streaming_text_;
    size_t streaming_published_bytes_ = 0;
    int64_t streaming_token_count_ = 0;
    int64_t streaming_frame_ = 0;                        // frame of the last recorded token
    std::vector<runtime::DecodedStep> pending_steps_;    // recorded since the last event
    int32_t previous_stream_token_ = 0;
    bool stream_started_ = false;
    bool first_stream_chunk_ = true;
    bool have_previous_stream_token_ = false;
    std::chrono::steady_clock::time_point stream_wall_start_{};
};

}  // namespace engine::models::voxtral_realtime
