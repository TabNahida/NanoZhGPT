#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct Config
{
    std::string env_path = ".env";
    std::string data_glob;
    std::string text_field = "text";
    std::string output_json = "tokenizer.json";
    std::string output_vocab = "vocab.json";
    std::string output_merges = "merges.txt";
    std::string unk_token = "<unk>";
    std::vector<std::string> special_tokens = {"<s>", "</s>", "<pad>", "<unk>", "<mask>"};

    std::size_t vocab_size = 50000;
    std::size_t min_freq = 2;
    std::size_t min_pair_freq = 2;
    std::size_t chunk_files = 1;
    std::size_t chunk_docs = 20000; // in-chunk reduce cadence
    std::size_t top_k = 200000;
    std::size_t max_chars_per_doc = 20000;
    std::size_t threads = 0; // 0 -> auto
    std::size_t progress_interval_ms = 1000;

    std::string chunk_dir = "chunks";
    bool resume = true;
    bool write_vocab = true;
    bool write_merges = true;
};
