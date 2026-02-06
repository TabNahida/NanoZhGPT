#include "byte_bpe_bpe.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "byte_bpe_lib.h"

static uint64_t pair_key(int a, int b)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

struct HeapItem
{
    uint64_t key;
    uint64_t count;
};

struct HeapCmp
{
    bool operator()(const HeapItem &a, const HeapItem &b) const
    {
        return a.count < b.count;
    }
};

std::vector<Word> build_words(const std::unordered_map<std::string, uint64_t> &global_counts,
                              const std::unordered_map<uint32_t, int> &cp_to_id, std::size_t min_freq)
{
    std::vector<Word> words;
    words.reserve(global_counts.size());
    for (const auto &kv : global_counts)
    {
        if (kv.second < min_freq)
        {
            continue;
        }
        std::vector<int> symbols;
        symbols.reserve(kv.first.size());
        std::size_t i = 0;
        while (i < kv.first.size())
        {
            uint32_t cp = 0;
            if (!next_codepoint(kv.first, i, cp))
            {
                break;
            }
            auto it = cp_to_id.find(cp);
            if (it == cp_to_id.end())
            {
                symbols.clear();
                break;
            }
            symbols.push_back(it->second);
        }
        if (!symbols.empty())
        {
            words.push_back({std::move(symbols), kv.second});
        }
    }
    return words;
}

void train_bpe(std::vector<Word> &words, std::vector<std::string> &id_to_symbol, std::vector<std::string> &merges_out,
               std::size_t target_vocab, std::size_t min_pair_freq)
{
    std::unordered_map<uint64_t, uint64_t> pair_counts;
    pair_counts.reserve(words.size() * 2 + 1024);
    std::unordered_map<uint64_t, std::vector<uint32_t>> pair_words;
    pair_words.reserve(words.size() * 2 + 1024);

    for (uint32_t idx = 0; idx < words.size(); ++idx)
    {
        const auto &syms = words[idx].symbols;
        if (syms.size() < 2)
        {
            continue;
        }
        std::vector<uint64_t> keys;
        keys.reserve(syms.size() - 1);
        for (std::size_t i = 0; i + 1 < syms.size(); ++i)
        {
            uint64_t key = pair_key(syms[i], syms[i + 1]);
            pair_counts[key] += words[idx].freq;
            keys.push_back(key);
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        for (auto key : keys)
        {
            pair_words[key].push_back(idx);
        }
    }

    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapCmp> heap;
    for (const auto &kv : pair_counts)
    {
        heap.push({kv.first, kv.second});
    }

    const std::size_t initial_vocab = id_to_symbol.size();
    std::size_t max_merges = 0;
    if (target_vocab > initial_vocab)
    {
        max_merges = target_vocab - initial_vocab;
    }

    std::size_t merges_done = 0;
    while (merges_done < max_merges && !heap.empty())
    {
        HeapItem top = heap.top();
        heap.pop();
        auto it = pair_counts.find(top.key);
        if (it == pair_counts.end())
        {
            continue;
        }
        if (it->second != top.count)
        {
            continue;
        }
        if (top.count < min_pair_freq)
        {
            break;
        }
        int a = static_cast<int>(top.key >> 32);
        int b = static_cast<int>(top.key & 0xFFFFFFFFu);

        std::string merged_sym = id_to_symbol[a] + id_to_symbol[b];
        int new_id = static_cast<int>(id_to_symbol.size());
        id_to_symbol.push_back(std::move(merged_sym));
        merges_out.push_back(id_to_symbol[a] + " " + id_to_symbol[b]);

        auto words_it = pair_words.find(top.key);
        if (words_it != pair_words.end())
        {
            const auto &wlist = words_it->second;
            for (uint32_t widx : wlist)
            {
                auto &w = words[widx];
                if (w.symbols.size() < 2)
                {
                    continue;
                }
                std::vector<int> new_syms;
                new_syms.reserve(w.symbols.size());
                bool changed = false;
                for (std::size_t i = 0; i < w.symbols.size(); ++i)
                {
                    if (i + 1 < w.symbols.size() && w.symbols[i] == a && w.symbols[i + 1] == b)
                    {
                        int prev = new_syms.empty() ? -1 : new_syms.back();
                        int next = (i + 2 < w.symbols.size()) ? w.symbols[i + 2] : -1;

                        if (prev != -1)
                        {
                            uint64_t k_prev = pair_key(prev, a);
                            auto &cnt = pair_counts[k_prev];
                            if (cnt >= w.freq)
                            {
                                cnt -= w.freq;
                            }
                            else
                            {
                                cnt = 0;
                            }
                            heap.push({k_prev, cnt});
                        }
                        if (next != -1)
                        {
                            uint64_t k_next = pair_key(b, next);
                            auto &cnt = pair_counts[k_next];
                            if (cnt >= w.freq)
                            {
                                cnt -= w.freq;
                            }
                            else
                            {
                                cnt = 0;
                            }
                            heap.push({k_next, cnt});
                        }

                        if (prev != -1)
                        {
                            uint64_t k_new_prev = pair_key(prev, new_id);
                            auto &cnt = pair_counts[k_new_prev];
                            cnt += w.freq;
                            heap.push({k_new_prev, cnt});
                            auto &vec = pair_words[k_new_prev];
                            if (vec.empty() || vec.back() != widx)
                            {
                                vec.push_back(widx);
                            }
                        }
                        if (next != -1)
                        {
                            uint64_t k_new_next = pair_key(new_id, next);
                            auto &cnt = pair_counts[k_new_next];
                            cnt += w.freq;
                            heap.push({k_new_next, cnt});
                            auto &vec = pair_words[k_new_next];
                            if (vec.empty() || vec.back() != widx)
                            {
                                vec.push_back(widx);
                            }
                        }

                        new_syms.push_back(new_id);
                        ++i;
                        changed = true;
                    }
                    else
                    {
                        new_syms.push_back(w.symbols[i]);
                    }
                }
                if (changed)
                {
                    w.symbols.swap(new_syms);
                }
            }
        }
        pair_counts[top.key] = 0;
        ++merges_done;
        if (merges_done % 1000 == 0)
        {
            std::cerr << "Merges: " << merges_done << "/" << max_merges << "\n";
        }
    }
}
