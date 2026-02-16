#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct Word
{
    std::vector<int> symbols;
    uint64_t freq = 0;
};

std::vector<Word> build_words(const std::unordered_map<std::string, uint64_t> &global_counts,
                              const std::unordered_map<uint32_t, int> &cp_to_id, std::size_t min_freq);

void train_bpe(std::vector<Word> &words, std::vector<std::string> &id_to_symbol, std::vector<std::string> &merges_out,
               std::size_t target_vocab, std::size_t min_pair_freq);
