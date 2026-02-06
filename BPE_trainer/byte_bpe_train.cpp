#include "byte_bpe_bpe.h"
#include "byte_bpe_config.h"
#include "byte_bpe_lib.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

static void print_usage()
{
    std::cerr << "Byte-level BPE trainer (HF-compatible tokenizer.json)\n"
              << "Usage:\n"
              << "  byte_bpe_train [options]\n\n"
              << "Options:\n"
              << "  --env <path>                Path to .env (default: .env)\n"
              << "  --data <glob>               Data path glob (overrides DATA_PATH)\n"
              << "  --text-field <name>         JSON field name (default: text)\n"
              << "  --vocab-size <n>            Total vocab size incl. specials (default: 50000)\n"
              << "  --min-freq <n>              Min word frequency (default: 2)\n"
              << "  --min-pair-freq <n>         Min pair frequency (default: 2)\n"
              << "  --chunk-files <n>           Files per chunk (default: 1)\n"
              << "  --chunk-docs <n>            Reduce map every N docs within a chunk (default: 20000)\n"
              << "  --top-k <n>                 Keep top-k words per chunk (default: 200000)\n"
              << "  --chunk-dir <path>          Chunk output directory (default: chunks)\n"
              << "  --resume / --no-resume      Resume from existing chunk files (default: on)\n"
              << "  --progress-interval <ms>    Progress update interval (default: 1000)\n"
              << "  --max-chars <n>             Truncate docs to N chars (default: 20000)\n"
              << "  --threads <n>               Worker threads (0=auto)\n"
              << "  --output <path>             tokenizer.json output (default: tokenizer.json)\n"
              << "  --vocab <path>              vocab.json output (default: vocab.json)\n"
              << "  --merges <path>             merges.txt output (default: merges.txt)\n"
              << "  --no-vocab                  Do not write vocab.json\n"
              << "  --no-merges                 Do not write merges.txt\n"
              << "  --special-tokens <csv>      Comma-separated specials\n"
              << "  --help                      Show this help\n";
}

static bool parse_args(int argc, char **argv, Config &cfg)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage();
            return false;
        }
        else if (arg == "--env" && i + 1 < argc)
        {
            cfg.env_path = argv[++i];
        }
        else if (arg == "--data" && i + 1 < argc)
        {
            cfg.data_glob = argv[++i];
        }
        else if (arg == "--text-field" && i + 1 < argc)
        {
            cfg.text_field = argv[++i];
        }
        else if (arg == "--vocab-size" && i + 1 < argc)
        {
            cfg.vocab_size = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--min-freq" && i + 1 < argc)
        {
            cfg.min_freq = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--min-pair-freq" && i + 1 < argc)
        {
            cfg.min_pair_freq = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--chunk-files" && i + 1 < argc)
        {
            cfg.chunk_files = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--chunk-docs" && i + 1 < argc)
        {
            cfg.chunk_docs = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--top-k" && i + 1 < argc)
        {
            cfg.top_k = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--chunk-dir" && i + 1 < argc)
        {
            cfg.chunk_dir = argv[++i];
        }
        else if (arg == "--resume")
        {
            cfg.resume = true;
        }
        else if (arg == "--no-resume")
        {
            cfg.resume = false;
        }
        else if (arg == "--progress-interval" && i + 1 < argc)
        {
            cfg.progress_interval_ms = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--max-chars" && i + 1 < argc)
        {
            cfg.max_chars_per_doc = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            cfg.threads = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            cfg.output_json = argv[++i];
        }
        else if (arg == "--vocab" && i + 1 < argc)
        {
            cfg.output_vocab = argv[++i];
        }
        else if (arg == "--merges" && i + 1 < argc)
        {
            cfg.output_merges = argv[++i];
        }
        else if (arg == "--no-vocab")
        {
            cfg.write_vocab = false;
        }
        else if (arg == "--no-merges")
        {
            cfg.write_merges = false;
        }
        else if (arg == "--special-tokens" && i + 1 < argc)
        {
            auto tokens = std::vector<std::string>();
            std::string csv = argv[++i];
            std::string cur;
            for (char c : csv)
            {
                if (c == ',')
                {
                    if (!cur.empty())
                    {
                        tokens.push_back(cur);
                    }
                    cur.clear();
                }
                else
                {
                    cur.push_back(c);
                }
            }
            if (!cur.empty())
            {
                tokens.push_back(cur);
            }
            if (!tokens.empty())
            {
                cfg.special_tokens = std::move(tokens);
            }
        }
        else
        {
            std::cerr << "Unknown arg: " << arg << "\n";
            print_usage();
            return false;
        }
    }
    return true;
}

static std::string chunk_path(const Config &cfg, std::size_t chunk_id)
{
    std::ostringstream oss;
    oss << cfg.chunk_dir << "/chunk_" << std::setw(8) << std::setfill('0') << chunk_id << ".bin";
    return oss.str();
}

static void process_text(const std::string &text, const Config &cfg,
                         const std::array<std::string, 256> &byte_to_unicode,
                         std::unordered_map<std::string, uint32_t> &local_counts, uint64_t &doc_count,
                         std::size_t &reduce_counter)
{
    std::string trimmed = text;
    if (cfg.max_chars_per_doc > 0)
    {
        trimmed = truncate_utf8(trimmed, cfg.max_chars_per_doc);
    }
    auto tokens = pretokenize(trimmed);
    for (const auto &tok : tokens)
    {
        if (tok.empty())
        {
            continue;
        }
        std::string encoded = byte_level_encode(tok, byte_to_unicode);
        if (encoded.empty())
        {
            continue;
        }
        ++local_counts[encoded];
    }
    ++doc_count;
    ++reduce_counter;
    if (cfg.chunk_docs > 0 && reduce_counter >= cfg.chunk_docs)
    {
        local_counts = reduce_top_k(local_counts, cfg.top_k);
        reduce_counter = 0;
    }
}

static void process_file(const std::string &path, const Config &cfg,
                         const std::array<std::string, 256> &byte_to_unicode,
                         std::unordered_map<std::string, uint32_t> &local_counts, uint64_t &doc_count,
                         std::size_t &reduce_counter)
{
    auto handle_line = [&](const std::string &line) {
        if (line.empty())
        {
            return;
        }
        std::string text;
        if (!extract_json_field(line, cfg.text_field, text))
        {
            return;
        }
        if (text.empty())
        {
            return;
        }
        process_text(text, cfg, byte_to_unicode, local_counts, doc_count, reduce_counter);
    };

    if (path.size() >= 3 && path.substr(path.size() - 3) == ".gz")
    {
        read_gz_lines(path, handle_line);
    }
    else
    {
        read_text_lines(path, handle_line);
    }
}

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Config cfg;
    if (!parse_args(argc, argv, cfg))
    {
        return 0;
    }

    auto env = read_env_file(cfg.env_path);
    apply_env_overrides(cfg, env);
    if (cfg.data_glob.empty())
    {
        std::cerr << "DATA_PATH not set in .env and --data not provided.\n";
        return 1;
    }

    auto files = glob_files(cfg.data_glob);
    if (files.empty())
    {
        if (cfg.data_glob.size() >= 5 && cfg.data_glob.substr(cfg.data_glob.size() - 5) == ".json")
        {
            std::string alt = cfg.data_glob + ".gz";
            files = glob_files(alt);
            if (files.empty())
            {
                std::string alt2 = cfg.data_glob.substr(0, cfg.data_glob.size() - 5) + ".json.gz";
                files = glob_files(alt2);
            }
        }
    }
    if (files.empty())
    {
        std::cerr << "No files matched: " << cfg.data_glob << "\n";
        return 1;
    }

    if (cfg.chunk_files == 0)
    {
        cfg.chunk_files = 1;
    }

    if (cfg.threads == 0)
    {
        cfg.threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }

    if (std::find(cfg.special_tokens.begin(), cfg.special_tokens.end(), cfg.unk_token) == cfg.special_tokens.end())
    {
        cfg.special_tokens.push_back(cfg.unk_token);
    }

    std::error_code ec;
    std::filesystem::create_directories(cfg.chunk_dir, ec);
    if (ec)
    {
        std::cerr << "Failed to create chunk dir: " << cfg.chunk_dir << "\n";
        return 1;
    }

    std::size_t total_chunks = (files.size() + cfg.chunk_files - 1) / cfg.chunk_files;

    std::cerr << "Files: " << files.size() << "\n";
    std::cerr << "Chunks: " << total_chunks << " (" << cfg.chunk_files << " files/chunk)\n";
    std::cerr << "Threads: " << cfg.threads << "\n";

    auto byte_to_unicode_cp = build_byte_to_unicode_cp();
    auto byte_to_unicode = build_byte_to_unicode_str(byte_to_unicode_cp);

    ProgressTracker progress(total_chunks, "processing", cfg.progress_interval_ms);
    std::atomic<std::size_t> next_chunk{0};
    std::atomic<bool> had_error{false};
    std::mutex err_mu;
    std::string err_msg;

    auto worker = [&]() {
        while (true)
        {
            std::size_t chunk_id = next_chunk.fetch_add(1);
            if (chunk_id >= total_chunks)
            {
                break;
            }
            std::size_t start = chunk_id * cfg.chunk_files;
            std::size_t end = std::min(start + cfg.chunk_files, files.size());
            std::string out_path = chunk_path(cfg, chunk_id);

            if (cfg.resume && std::filesystem::exists(out_path))
            {
                ChunkHeader header;
                uint64_t docs = 0;
                if (read_chunk_header(out_path, header))
                {
                    docs = header.doc_count;
                }
                progress.add(1, docs);
                continue;
            }

            std::unordered_map<std::string, uint32_t> local_counts;
            local_counts.reserve(cfg.top_k * 2 + 16);
            uint64_t docs = 0;
            std::size_t reduce_counter = 0;

            for (std::size_t i = start; i < end; ++i)
            {
                process_file(files[i], cfg, byte_to_unicode, local_counts, docs, reduce_counter);
            }
            if (cfg.top_k > 0)
            {
                local_counts = reduce_top_k(local_counts, cfg.top_k);
            }
            if (!write_chunk_file(out_path, local_counts, docs))
            {
                std::lock_guard<std::mutex> lock(err_mu);
                had_error.store(true);
                err_msg = "Failed to write chunk: " + out_path;
                progress.add(1, docs);
                continue;
            }

            progress.add(1, docs);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(cfg.threads);
    for (std::size_t t = 0; t < cfg.threads; ++t)
    {
        threads.emplace_back(worker);
    }
    for (auto &t : threads)
    {
        t.join();
    }
    progress.finish();

    if (had_error.load())
    {
        std::cerr << err_msg << "\n";
        return 1;
    }

    std::unordered_map<std::string, uint64_t> global_counts;
    global_counts.reserve(total_chunks * 1024);
    ProgressTracker merge_progress(total_chunks, "merging", cfg.progress_interval_ms);
    uint64_t total_docs = 0;
    for (std::size_t chunk_id = 0; chunk_id < total_chunks; ++chunk_id)
    {
        std::string path = chunk_path(cfg, chunk_id);
        if (!merge_chunk_file(path, global_counts, &total_docs))
        {
            std::cerr << "Failed to read chunk: " << path << "\n";
            return 1;
        }
        merge_progress.add(1, 0);
    }
    merge_progress.finish();

    std::cerr << "Docs: " << total_docs << "\n";
    std::cerr << "Unique words after chunk merge: " << global_counts.size() << "\n";

    std::unordered_map<uint32_t, int> cp_to_id;
    cp_to_id.reserve(256);
    std::vector<std::string> id_to_symbol;
    id_to_symbol.reserve(256 + cfg.vocab_size);
    for (std::size_t i = 0; i < 256; ++i)
    {
        const auto &sym = byte_to_unicode[i];
        id_to_symbol.push_back(sym);
        std::size_t j = 0;
        uint32_t cp = 0;
        next_codepoint(sym, j, cp);
        cp_to_id[cp] = static_cast<int>(i);
    }

    auto words = build_words(global_counts, cp_to_id, cfg.min_freq);
    global_counts.clear();
    global_counts.rehash(0);

    std::cerr << "Training words: " << words.size() << "\n";

    std::size_t target_vocab = cfg.vocab_size;
    if (cfg.special_tokens.size() < target_vocab)
    {
        target_vocab -= cfg.special_tokens.size();
    }
    else
    {
        target_vocab = id_to_symbol.size();
    }
    if (target_vocab < id_to_symbol.size())
    {
        target_vocab = id_to_symbol.size();
    }

    std::vector<std::string> merges_out;
    merges_out.reserve(target_vocab - id_to_symbol.size() + 16);

    train_bpe(words, id_to_symbol, merges_out, target_vocab, cfg.min_pair_freq);

    std::vector<std::string> id_to_token;
    id_to_token.reserve(cfg.special_tokens.size() + id_to_symbol.size());
    for (const auto &t : cfg.special_tokens)
    {
        id_to_token.push_back(t);
    }
    for (const auto &sym : id_to_symbol)
    {
        if (std::find(id_to_token.begin(), id_to_token.end(), sym) == id_to_token.end())
        {
            id_to_token.push_back(sym);
        }
    }

    if (!write_tokenizer_json(cfg.output_json, cfg, id_to_token, merges_out))
    {
        std::cerr << "Failed to write tokenizer.json: " << cfg.output_json << "\n";
        return 1;
    }
    std::cerr << "Saved tokenizer: " << cfg.output_json << "\n";

    if (cfg.write_vocab)
    {
        if (!write_vocab_json(cfg.output_vocab, id_to_token))
        {
            std::cerr << "Failed to write vocab.json: " << cfg.output_vocab << "\n";
            return 1;
        }
        std::cerr << "Saved vocab: " << cfg.output_vocab << "\n";
    }

    if (cfg.write_merges)
    {
        if (!write_merges_txt(cfg.output_merges, merges_out))
        {
            std::cerr << "Failed to write merges.txt: " << cfg.output_merges << "\n";
            return 1;
        }
        std::cerr << "Saved merges: " << cfg.output_merges << "\n";
    }

    return 0;
}
