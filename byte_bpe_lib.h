#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "byte_bpe_config.h"

std::unordered_map<std::string, std::string> read_env_file(const std::string &path);
void apply_env_overrides(Config &cfg, const std::unordered_map<std::string, std::string> &env);

std::vector<std::string> glob_files(const std::string &pattern);

bool next_codepoint(const std::string &s, std::size_t &i, uint32_t &cp);
std::string truncate_utf8(const std::string &s, std::size_t max_chars);
std::vector<std::string> pretokenize(const std::string &text);
std::array<uint32_t, 256> build_byte_to_unicode_cp();
std::array<std::string, 256> build_byte_to_unicode_str(const std::array<uint32_t, 256> &cp_map);
std::string byte_level_encode(const std::string &token, const std::array<std::string, 256> &byte_to_unicode);

bool extract_json_field(const std::string &s, const std::string &field, std::string &out);

std::unordered_map<std::string, uint32_t> reduce_top_k(std::unordered_map<std::string, uint32_t> &local,
                                                       std::size_t top_k);

bool read_gz_lines(const std::string &path, const std::function<void(const std::string &)> &cb);
bool read_text_lines(const std::string &path, const std::function<void(const std::string &)> &cb);

bool write_vocab_json(const std::string &path, const std::vector<std::string> &id_to_token);
bool write_merges_txt(const std::string &path, const std::vector<std::string> &merges_out);
bool write_tokenizer_json(const std::string &path, const Config &cfg, const std::vector<std::string> &id_to_token,
                          const std::vector<std::string> &merges_out);

struct ChunkHeader
{
    uint32_t magic = 0x314B4243; // "CBK1"
    uint32_t version = 1;
    uint64_t doc_count = 0;
    uint64_t entry_count = 0;
};

bool write_chunk_file(const std::string &path, const std::unordered_map<std::string, uint32_t> &counts,
                      uint64_t doc_count);

bool read_chunk_header(const std::string &path, ChunkHeader &header);

bool merge_chunk_file(const std::string &path, std::unordered_map<std::string, uint64_t> &global_counts,
                      uint64_t *doc_count_out = nullptr);

class ProgressTracker
{
  public:
    ProgressTracker(uint64_t total_chunks, const std::string &label, uint64_t interval_ms);
    void add(uint64_t chunks, uint64_t docs);
    void finish();

  private:
    void maybe_print(bool force);

    std::string label_;
    uint64_t total_ = 0;
    uint64_t interval_ms_ = 1000;
    std::atomic<uint64_t> done_chunks_{0};
    std::atomic<uint64_t> done_docs_{0};
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_print_;
    std::mutex print_mu_;
};
