#include "byte_bpe_lib.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
struct Args
{
    std::string env_file = ".env";
    std::string data_glob;
    std::string text_field = "text";
    std::string tokenizer_path = "tokenizer.json";
    std::string out_dir = "data/tokens";
    std::uint64_t max_tokens_per_shard = 50'000'000;
    std::size_t encode_batch_size = 256; // CLI compatibility
    std::size_t min_chars = 1;
    std::size_t max_chars = 20'000;
    std::uint64_t max_docs = 0; // CLI compatibility
    std::string eos_token = "</s>";
    std::string bos_token;
    std::uint64_t progress_every = 10'000; // CLI compatibility
    std::size_t threads = 0;
    std::size_t cache_max_entries = 50'000;
    bool resume = true;
};

struct TokenizerData
{
    std::unordered_map<std::string, std::uint32_t> vocab;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, std::uint32_t> added_tokens;
    std::string unk_token;
};

struct MergeRule
{
    std::uint32_t rank = 0;
    std::uint32_t merged_symbol = 0;
};

struct FileTask
{
    std::size_t index = 0;
    std::string path;
    std::uint64_t file_size = 0;
    std::int64_t file_mtime = 0;
    std::filesystem::path part_bin_path;
    std::filesystem::path part_meta_path;
};

struct PartResult
{
    std::uint64_t num_docs = 0;
    std::uint64_t num_skipped = 0;
    std::uint64_t num_tokens = 0;
    bool reused = false;
};

struct ShardInfo
{
    std::string file;
    std::uint64_t num_tokens = 0;
};

struct PartSignature
{
    std::string tokenizer_fingerprint;
    std::string text_field;
    std::size_t min_chars = 1;
    std::size_t max_chars = 20'000;
    std::int64_t bos_id = -1;
    std::int64_t eos_id = -1;
    std::uint32_t dtype_bytes = 2;
};

static bool starts_with(const std::string &s, const std::string &prefix)
{
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static bool ends_with(const std::string &s, const std::string &suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string normalize_path_for_compare(const std::string &path)
{
    std::string out = path;
    for (char &c : out)
    {
        if (c == '\\')
        {
            c = '/';
        }
#ifdef _WIN32
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#endif
    }
    return out;
}

static void print_usage()
{
    std::cerr << "Tokenize json/json.gz to train_*.bin shards (C++, multi-thread, resumable)\n"
              << "Usage:\n"
              << "  prepare_shards [options]\n\n"
              << "Options:\n"
              << "  --env-file <path>            Path to .env (default: .env)\n"
              << "  --data-glob <glob>           Override DATA_PATH from .env\n"
              << "  --text-field <name>          JSON field name (default: text)\n"
              << "  --tokenizer <path>           tokenizer.json path (default: tokenizer.json)\n"
              << "  --out-dir <path>             Output directory (default: data/tokens)\n"
              << "  --max-tokens-per-shard <n>   Tokens per train shard (default: 50000000)\n"
              << "  --encode-batch-size <n>      CLI compatibility only (unused, default: 256)\n"
              << "  --min-chars <n>              Min chars per doc (default: 1)\n"
              << "  --max-chars <n>              Max chars per doc (default: 20000)\n"
              << "  --max-docs <n>               CLI compatibility only (must be 0)\n"
              << "  --eos-token <tok>            EOS token (default: </s>)\n"
              << "  --bos-token <tok>            BOS token (default: none)\n"
              << "  --threads <n>                Worker threads (0=auto)\n"
              << "  --cache-max-entries <n>      Max cached token pieces per file (default: 50000, 0=disable)\n"
              << "  --resume / --no-resume       Reuse completed part files (default: on)\n"
              << "  --progress-every <n>         CLI compatibility only (unused)\n"
              << "  --help                       Show this help\n";
}

static bool parse_u64_arg(const std::string &s, std::uint64_t &out)
{
    try
    {
        std::size_t pos = 0;
        std::uint64_t v = static_cast<std::uint64_t>(std::stoull(s, &pos, 10));
        if (pos != s.size())
        {
            return false;
        }
        out = v;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static bool parse_size_arg(const std::string &s, std::size_t &out)
{
    std::uint64_t v = 0;
    if (!parse_u64_arg(s, v))
    {
        return false;
    }
    if (v > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        return false;
    }
    out = static_cast<std::size_t>(v);
    return true;
}

static bool parse_args(int argc, char **argv, Args &args)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto need_value = [&](const std::string &name) -> const char * {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h")
        {
            print_usage();
            return false;
        }
        else if (arg == "--")
        {
            continue;
        }
        else if (arg == "--env-file")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            args.env_file = v;
        }
        else if (arg == "--data-glob")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            args.data_glob = v;
        }
        else if (arg == "--text-field")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            args.text_field = v;
        }
        else if (arg == "--tokenizer")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            args.tokenizer_path = v;
        }
        else if (arg == "--out-dir")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            args.out_dir = v;
        }
        else if (arg == "--max-tokens-per-shard")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            std::uint64_t x = 0;
            if (!parse_u64_arg(v, x) || x == 0)
            {
                std::cerr << "Invalid --max-tokens-per-shard: " << v << "\n";
                return false;
            }
            args.max_tokens_per_shard = x;
        }
        else if (arg == "--encode-batch-size")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --encode-batch-size: " << v << "\n";
                return false;
            }
            args.encode_batch_size = x;
        }
        else if (arg == "--min-chars")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --min-chars: " << v << "\n";
                return false;
            }
            args.min_chars = x;
        }
        else if (arg == "--max-chars")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --max-chars: " << v << "\n";
                return false;
            }
            args.max_chars = x;
        }
        else if (arg == "--max-docs")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            std::uint64_t x = 0;
            if (!parse_u64_arg(v, x))
            {
                std::cerr << "Invalid --max-docs: " << v << "\n";
                return false;
            }
            args.max_docs = x;
        }
        else if (arg == "--eos-token")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            args.eos_token = v;
        }
        else if (arg == "--bos-token")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            args.bos_token = v;
        }
        else if (arg == "--threads")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --threads: " << v << "\n";
                return false;
            }
            args.threads = x;
        }
        else if (arg == "--cache-max-entries")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            std::size_t x = 0;
            if (!parse_size_arg(v, x))
            {
                std::cerr << "Invalid --cache-max-entries: " << v << "\n";
                return false;
            }
            args.cache_max_entries = x;
        }
        else if (arg == "--resume")
        {
            args.resume = true;
        }
        else if (arg == "--no-resume")
        {
            args.resume = false;
        }
        else if (arg == "--progress-every")
        {
            const char *v = need_value(arg);
            if (!v)
                return false;
            std::uint64_t x = 0;
            if (!parse_u64_arg(v, x))
            {
                std::cerr << "Invalid --progress-every: " << v << "\n";
                return false;
            }
            args.progress_every = x;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage();
            return false;
        }
    }
    return true;
}

static std::vector<std::string> expand_data_files(const std::string &data_glob)
{
    auto files = glob_files(data_glob);
    if (!files.empty())
    {
        return files;
    }
    if (ends_with(data_glob, ".json"))
    {
        std::string alt = data_glob.substr(0, data_glob.size() - 5) + ".json.gz";
        files = glob_files(alt);
        if (!files.empty())
        {
            return files;
        }
    }
    if (!ends_with(data_glob, ".gz"))
    {
        std::string alt = data_glob + ".gz";
        files = glob_files(alt);
    }
    return files;
}

static void append_utf8(std::uint32_t cp, std::string &out)
{
    if (cp <= 0x7F)
    {
        out.push_back(static_cast<char>(cp));
    }
    else if (cp <= 0x7FF)
    {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else if (cp <= 0xFFFF)
    {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else
    {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

static void skip_ws(const std::string &s, std::size_t &i)
{
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
    {
        ++i;
    }
}

static bool parse_hex4(const std::string &s, std::size_t i, std::uint32_t &out)
{
    if (i + 4 > s.size())
    {
        return false;
    }
    std::uint32_t val = 0;
    for (std::size_t k = 0; k < 4; ++k)
    {
        char c = s[i + k];
        std::uint32_t v = 0;
        if (c >= '0' && c <= '9')
        {
            v = static_cast<std::uint32_t>(c - '0');
        }
        else if (c >= 'a' && c <= 'f')
        {
            v = static_cast<std::uint32_t>(10 + c - 'a');
        }
        else if (c >= 'A' && c <= 'F')
        {
            v = static_cast<std::uint32_t>(10 + c - 'A');
        }
        else
        {
            return false;
        }
        val = (val << 4) | v;
    }
    out = val;
    return true;
}

static bool parse_json_string(const std::string &s, std::size_t &i, std::string &out)
{
    if (i >= s.size() || s[i] != '"')
    {
        return false;
    }
    ++i;
    out.clear();
    while (i < s.size())
    {
        char c = s[i++];
        if (c == '"')
        {
            return true;
        }
        if (c == '\\')
        {
            if (i >= s.size())
            {
                return false;
            }
            char esc = s[i++];
            switch (esc)
            {
            case '"':
                out.push_back('"');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case '/':
                out.push_back('/');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                std::uint32_t cp = 0;
                if (!parse_hex4(s, i, cp))
                {
                    return false;
                }
                i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF)
                {
                    if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u')
                    {
                        std::uint32_t low = 0;
                        if (parse_hex4(s, i + 2, low) && low >= 0xDC00 && low <= 0xDFFF)
                        {
                            i += 6;
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                        }
                    }
                }
                append_utf8(cp, out);
                break;
            }
            default:
                out.push_back(esc);
                break;
            }
        }
        else
        {
            out.push_back(c);
        }
    }
    return false;
}

static bool skip_json_value(const std::string &s, std::size_t &i);

static bool skip_json_number(const std::string &s, std::size_t &i)
{
    if (i < s.size() && (s[i] == '-' || s[i] == '+'))
    {
        ++i;
    }
    bool has_digit = false;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
    {
        has_digit = true;
        ++i;
    }
    if (i < s.size() && s[i] == '.')
    {
        ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        {
            has_digit = true;
            ++i;
        }
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E'))
    {
        ++i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+'))
        {
            ++i;
        }
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        {
            has_digit = true;
            ++i;
        }
    }
    return has_digit;
}

static bool skip_json_literal(const std::string &s, std::size_t &i, const char *lit)
{
    std::size_t n = std::char_traits<char>::length(lit);
    if (i + n > s.size())
    {
        return false;
    }
    if (s.compare(i, n, lit) != 0)
    {
        return false;
    }
    i += n;
    return true;
}

static bool skip_json_object(const std::string &s, std::size_t &i)
{
    if (i >= s.size() || s[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool skip_json_array(const std::string &s, std::size_t &i)
{
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool skip_json_value(const std::string &s, std::size_t &i)
{
    skip_ws(s, i);
    if (i >= s.size())
    {
        return false;
    }
    char c = s[i];
    if (c == '"')
    {
        std::string tmp;
        return parse_json_string(s, i, tmp);
    }
    if (c == '{')
    {
        return skip_json_object(s, i);
    }
    if (c == '[')
    {
        return skip_json_array(s, i);
    }
    if (c == 't')
    {
        return skip_json_literal(s, i, "true");
    }
    if (c == 'f')
    {
        return skip_json_literal(s, i, "false");
    }
    if (c == 'n')
    {
        return skip_json_literal(s, i, "null");
    }
    return skip_json_number(s, i);
}

static bool parse_json_uint(const std::string &s, std::size_t &i, std::uint32_t &out)
{
    skip_ws(s, i);
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
    {
        return false;
    }
    std::uint64_t val = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
    {
        val = val * 10 + static_cast<std::uint64_t>(s[i] - '0');
        if (val > std::numeric_limits<std::uint32_t>::max())
        {
            return false;
        }
        ++i;
    }
    out = static_cast<std::uint32_t>(val);
    return true;
}

static bool parse_vocab_object(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        std::uint32_t val = 0;
        if (!parse_json_uint(s, i, val))
        {
            return false;
        }
        out.vocab[std::move(key)] = val;
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_merge_line(const std::string &line, std::pair<std::string, std::string> &out)
{
    auto pos = line.find(' ');
    if (pos == std::string::npos || pos + 1 >= line.size())
    {
        return false;
    }
    out.first = line.substr(0, pos);
    out.second = line.substr(pos + 1);
    return true;
}

static bool parse_merges_array(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (s[i] == '"')
        {
            std::string merge_line;
            if (!parse_json_string(s, i, merge_line))
            {
                return false;
            }
            std::pair<std::string, std::string> m;
            if (parse_merge_line(merge_line, m))
            {
                out.merges.push_back(std::move(m));
            }
        }
        else if (s[i] == '[')
        {
            ++i;
            skip_ws(s, i);
            std::string left;
            std::string right;
            bool ok = parse_json_string(s, i, left);
            skip_ws(s, i);
            if (ok && i < s.size() && s[i] == ',')
            {
                ++i;
                skip_ws(s, i);
                ok = parse_json_string(s, i, right);
            }
            while (i < s.size())
            {
                skip_ws(s, i);
                if (i < s.size() && s[i] == ']')
                {
                    ++i;
                    break;
                }
                if (i < s.size() && s[i] == ',')
                {
                    ++i;
                    if (!skip_json_value(s, i))
                    {
                        return false;
                    }
                    continue;
                }
                return false;
            }
            if (ok && !left.empty() && !right.empty())
            {
                out.merges.push_back({std::move(left), std::move(right)});
            }
        }
        else
        {
            if (!skip_json_value(s, i))
            {
                return false;
            }
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_added_tokens_array(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (s[i] != '{')
        {
            if (!skip_json_value(s, i))
            {
                return false;
            }
        }
        else
        {
            ++i;
            std::string content;
            std::uint32_t id = 0;
            bool has_content = false;
            bool has_id = false;
            while (true)
            {
                skip_ws(s, i);
                if (i >= s.size())
                {
                    return false;
                }
                if (s[i] == '}')
                {
                    ++i;
                    break;
                }
                std::string key;
                if (!parse_json_string(s, i, key))
                {
                    return false;
                }
                skip_ws(s, i);
                if (i >= s.size() || s[i] != ':')
                {
                    return false;
                }
                ++i;
                if (key == "id")
                {
                    std::uint32_t x = 0;
                    if (!parse_json_uint(s, i, x))
                    {
                        return false;
                    }
                    id = x;
                    has_id = true;
                }
                else if (key == "content")
                {
                    std::string x;
                    if (!parse_json_string(s, i, x))
                    {
                        return false;
                    }
                    content = std::move(x);
                    has_content = true;
                }
                else
                {
                    if (!skip_json_value(s, i))
                    {
                        return false;
                    }
                }
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',')
                {
                    ++i;
                    continue;
                }
                if (i < s.size() && s[i] == '}')
                {
                    ++i;
                    break;
                }
                return false;
            }
            if (has_content && has_id)
            {
                out.added_tokens[content] = id;
            }
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_model_object(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        if (key == "vocab")
        {
            if (!parse_vocab_object(s, i, out))
            {
                return false;
            }
        }
        else if (key == "merges")
        {
            if (!parse_merges_array(s, i, out))
            {
                return false;
            }
        }
        else if (key == "unk_token")
        {
            std::string unk;
            if (!parse_json_string(s, i, unk))
            {
                return false;
            }
            out.unk_token = std::move(unk);
        }
        else
        {
            if (!skip_json_value(s, i))
            {
                return false;
            }
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_tokenizer_json(const std::string &content, TokenizerData &out)
{
    std::size_t i = 0;
    skip_ws(content, i);
    if (i >= content.size() || content[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(content, i);
        if (i >= content.size())
        {
            return false;
        }
        if (content[i] == '}')
        {
            ++i;
            break;
        }
        std::string key;
        if (!parse_json_string(content, i, key))
        {
            return false;
        }
        skip_ws(content, i);
        if (i >= content.size() || content[i] != ':')
        {
            return false;
        }
        ++i;
        if (key == "model")
        {
            if (!parse_model_object(content, i, out))
            {
                return false;
            }
        }
        else if (key == "added_tokens")
        {
            if (!parse_added_tokens_array(content, i, out))
            {
                return false;
            }
        }
        else
        {
            if (!skip_json_value(content, i))
            {
                return false;
            }
        }
        skip_ws(content, i);
        if (i < content.size() && content[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < content.size() && content[i] == '}')
        {
            ++i;
            break;
        }
        return false;
    }
    for (const auto &kv : out.added_tokens)
    {
        out.vocab[kv.first] = kv.second;
    }
    return !out.vocab.empty();
}

static std::string read_file_all(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

class BPETokenizer
{
  public:
    bool load(const std::string &path, std::string &err)
    {
        std::string content = read_file_all(path);
        if (content.empty())
        {
            err = "failed to read tokenizer file: " + path;
            return false;
        }
        TokenizerData data;
        data.vocab.reserve(65536);
        if (!parse_tokenizer_json(content, data))
        {
            err = "failed to parse tokenizer.json: " + path;
            return false;
        }
        vocab_ = std::move(data.vocab);
        unk_token_ = data.unk_token;
        has_unk_ = false;
        if (!unk_token_.empty())
        {
            auto it = vocab_.find(unk_token_);
            if (it != vocab_.end())
            {
                has_unk_ = true;
                unk_id_ = it->second;
            }
        }

        symbols_.clear();
        symbol_to_id_.clear();
        merge_rules_.clear();
        symbols_.reserve(vocab_.size() + data.merges.size() + 256);
        symbol_to_id_.reserve(vocab_.size() + data.merges.size() + 256);
        merge_rules_.reserve(data.merges.size() * 13 / 10 + 8);

        auto cp_map = build_byte_to_unicode_cp();
        byte_to_unicode_ = build_byte_to_unicode_str(cp_map);
        for (const auto &s : byte_to_unicode_)
        {
            ensure_symbol(s);
        }
        for (const auto &kv : vocab_)
        {
            ensure_symbol(kv.first);
        }
        for (std::size_t rank = 0; rank < data.merges.size(); ++rank)
        {
            const auto &m = data.merges[rank];
            std::uint32_t left = ensure_symbol(m.first);
            std::uint32_t right = ensure_symbol(m.second);
            std::uint32_t merged = ensure_symbol(m.first + m.second);
            std::uint64_t key = pair_key(left, right);
            if (merge_rules_.find(key) == merge_rules_.end())
            {
                merge_rules_[key] = MergeRule{static_cast<std::uint32_t>(rank), merged};
            }
        }
        return true;
    }

    std::size_t vocab_size() const
    {
        return vocab_.size();
    }

    bool token_to_id(const std::string &token, std::uint32_t &id) const
    {
        auto it = vocab_.find(token);
        if (it == vocab_.end())
        {
            return false;
        }
        id = it->second;
        return true;
    }

    void encode_text_append(const std::string &text,
                            std::unordered_map<std::string, std::vector<std::uint32_t>> &cache,
                            std::vector<std::uint32_t> &out_ids) const
    {
        auto pieces = pretokenize(text);
        for (const auto &piece : pieces)
        {
            if (piece.empty())
            {
                continue;
            }
            std::string encoded = byte_level_encode(piece, byte_to_unicode_);
            const auto &ids = encode_piece(encoded, cache);
            out_ids.insert(out_ids.end(), ids.begin(), ids.end());
        }
    }

  private:
    std::uint32_t ensure_symbol(const std::string &sym)
    {
        auto it = symbol_to_id_.find(sym);
        if (it != symbol_to_id_.end())
        {
            return it->second;
        }
        std::uint32_t id = static_cast<std::uint32_t>(symbols_.size());
        symbols_.push_back(sym);
        symbol_to_id_.emplace(symbols_.back(), id);
        return id;
    }

    static std::uint64_t pair_key(std::uint32_t a, std::uint32_t b)
    {
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    }

    const std::vector<std::uint32_t> &
    encode_piece(const std::string &encoded, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const
    {
        auto it_cache = cache.find(encoded);
        if (it_cache != cache.end())
        {
            return it_cache->second;
        }

        std::vector<std::uint32_t> symbols;
        symbols.reserve(encoded.size());
        std::size_t i = 0;
        while (i < encoded.size())
        {
            std::size_t prev = i;
            std::uint32_t cp = 0;
            if (!next_codepoint(encoded, i, cp))
            {
                break;
            }
            if (i <= prev)
            {
                break;
            }
            auto it_sym = symbol_to_id_.find(encoded.substr(prev, i - prev));
            if (it_sym == symbol_to_id_.end())
            {
                symbols.clear();
                break;
            }
            symbols.push_back(it_sym->second);
        }

        while (symbols.size() >= 2)
        {
            std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
            std::size_t best_pos = static_cast<std::size_t>(-1);
            std::uint32_t best_left = 0;
            std::uint32_t best_right = 0;
            std::uint32_t best_merged = 0;
            for (std::size_t pos = 0; pos + 1 < symbols.size(); ++pos)
            {
                std::uint64_t key = pair_key(symbols[pos], symbols[pos + 1]);
                auto it_rule = merge_rules_.find(key);
                if (it_rule == merge_rules_.end())
                {
                    continue;
                }
                if (it_rule->second.rank < best_rank)
                {
                    best_rank = it_rule->second.rank;
                    best_pos = pos;
                    best_left = symbols[pos];
                    best_right = symbols[pos + 1];
                    best_merged = it_rule->second.merged_symbol;
                }
            }
            if (best_pos == static_cast<std::size_t>(-1))
            {
                break;
            }

            std::vector<std::uint32_t> merged;
            merged.reserve(symbols.size());
            for (std::size_t p = 0; p < symbols.size();)
            {
                if (p + 1 < symbols.size() && symbols[p] == best_left && symbols[p + 1] == best_right)
                {
                    merged.push_back(best_merged);
                    p += 2;
                }
                else
                {
                    merged.push_back(symbols[p]);
                    ++p;
                }
            }
            symbols.swap(merged);
        }

        std::vector<std::uint32_t> ids;
        ids.reserve(symbols.size());
        for (std::uint32_t sid : symbols)
        {
            if (sid >= symbols_.size())
            {
                continue;
            }
            const auto &tok = symbols_[sid];
            auto it_vocab = vocab_.find(tok);
            if (it_vocab != vocab_.end())
            {
                ids.push_back(it_vocab->second);
            }
            else if (has_unk_)
            {
                ids.push_back(unk_id_);
            }
        }

        auto inserted = cache.emplace(encoded, std::move(ids));
        return inserted.first->second;
    }

    std::unordered_map<std::string, std::uint32_t> vocab_;
    std::string unk_token_;
    bool has_unk_ = false;
    std::uint32_t unk_id_ = 0;
    std::vector<std::string> symbols_;
    std::unordered_map<std::string, std::uint32_t> symbol_to_id_;
    std::unordered_map<std::uint64_t, MergeRule> merge_rules_;
    std::array<std::string, 256> byte_to_unicode_{};
};

class TokenWriter
{
  public:
    TokenWriter(const std::filesystem::path &path, std::uint32_t dtype_bytes) : dtype_bytes_(dtype_bytes)
    {
        out_.open(path, std::ios::binary | std::ios::trunc);
    }

    bool good() const
    {
        return static_cast<bool>(out_);
    }

    bool write_tokens(const std::vector<std::uint32_t> &tokens)
    {
        if (tokens.empty())
        {
            return true;
        }
        if (!out_)
        {
            return false;
        }
        if (dtype_bytes_ == 2)
        {
            tmp16_.clear();
            tmp16_.reserve(tokens.size());
            for (std::uint32_t v : tokens)
            {
                if (v > std::numeric_limits<std::uint16_t>::max())
                {
                    return false;
                }
                tmp16_.push_back(static_cast<std::uint16_t>(v));
            }
            out_.write(reinterpret_cast<const char *>(tmp16_.data()),
                       static_cast<std::streamsize>(tmp16_.size() * sizeof(std::uint16_t)));
        }
        else if (dtype_bytes_ == 4)
        {
            out_.write(reinterpret_cast<const char *>(tokens.data()),
                       static_cast<std::streamsize>(tokens.size() * sizeof(std::uint32_t)));
        }
        else
        {
            return false;
        }
        return static_cast<bool>(out_);
    }

    bool close()
    {
        if (!out_.is_open())
        {
            return true;
        }
        out_.flush();
        if (!out_)
        {
            return false;
        }
        out_.close();
        return true;
    }

  private:
    std::ofstream out_;
    std::uint32_t dtype_bytes_ = 2;
    std::vector<std::uint16_t> tmp16_;
};

static bool get_file_stat(const std::string &path, std::uint64_t &file_size, std::int64_t &mtime)
{
    std::error_code ec;
    file_size = std::filesystem::file_size(path, ec);
    if (ec)
    {
        return false;
    }
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec)
    {
        return false;
    }
    mtime = std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
    return true;
}

static std::string make_tokenizer_fingerprint(const std::string &path)
{
    std::error_code ec;
    std::filesystem::path abs_path = std::filesystem::absolute(path, ec);
    std::uint64_t sz = 0;
    std::int64_t mt = 0;
    if (!get_file_stat(path, sz, mt))
    {
        return {};
    }
    std::ostringstream oss;
    oss << normalize_path_for_compare(abs_path.string()) << "|" << sz << "|" << mt;
    return oss.str();
}

static bool parse_u64_env(const std::unordered_map<std::string, std::string> &m, const std::string &k,
                          std::uint64_t &out)
{
    auto it = m.find(k);
    if (it == m.end())
    {
        return false;
    }
    return parse_u64_arg(it->second, out);
}

static bool parse_i64_env(const std::unordered_map<std::string, std::string> &m, const std::string &k, std::int64_t &out)
{
    auto it = m.find(k);
    if (it == m.end())
    {
        return false;
    }
    try
    {
        std::size_t pos = 0;
        long long v = std::stoll(it->second, &pos, 10);
        if (pos != it->second.size())
        {
            return false;
        }
        out = static_cast<std::int64_t>(v);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static bool part_is_reusable(const FileTask &task, const PartSignature &sig, PartResult &result)
{
    if (!std::filesystem::exists(task.part_bin_path) || !std::filesystem::exists(task.part_meta_path))
    {
        return false;
    }
    auto meta = read_env_file(task.part_meta_path.string());
    if (meta.empty())
    {
        return false;
    }

    auto get = [&](const std::string &k) -> std::string {
        auto it = meta.find(k);
        if (it == meta.end())
        {
            return {};
        }
        return it->second;
    };

    if (normalize_path_for_compare(get("source_path")) != normalize_path_for_compare(task.path))
    {
        return false;
    }
    std::uint64_t source_size = 0;
    if (!parse_u64_env(meta, "source_size", source_size) || source_size != task.file_size)
    {
        return false;
    }
    std::int64_t source_mtime = 0;
    if (!parse_i64_env(meta, "source_mtime", source_mtime) || source_mtime != task.file_mtime)
    {
        return false;
    }
    if (get("tokenizer_fingerprint") != sig.tokenizer_fingerprint)
    {
        return false;
    }
    if (get("text_field") != sig.text_field)
    {
        return false;
    }

    std::uint64_t min_chars = 0;
    std::uint64_t max_chars = 0;
    std::uint64_t bos_id = 0;
    std::uint64_t eos_id = 0;
    std::uint64_t dtype_bytes = 0;
    if (!parse_u64_env(meta, "min_chars", min_chars) || min_chars != sig.min_chars)
    {
        return false;
    }
    if (!parse_u64_env(meta, "max_chars", max_chars) || max_chars != sig.max_chars)
    {
        return false;
    }
    if (!parse_u64_env(meta, "bos_id", bos_id) || static_cast<std::int64_t>(bos_id) != sig.bos_id)
    {
        return false;
    }
    if (!parse_u64_env(meta, "eos_id", eos_id) || static_cast<std::int64_t>(eos_id) != sig.eos_id)
    {
        return false;
    }
    if (!parse_u64_env(meta, "dtype_bytes", dtype_bytes) || dtype_bytes != sig.dtype_bytes)
    {
        return false;
    }

    if (!parse_u64_env(meta, "num_docs", result.num_docs))
    {
        return false;
    }
    if (!parse_u64_env(meta, "num_skipped", result.num_skipped))
    {
        return false;
    }
    if (!parse_u64_env(meta, "num_tokens", result.num_tokens))
    {
        return false;
    }

    std::error_code ec;
    std::uint64_t expected_size = result.num_tokens * static_cast<std::uint64_t>(sig.dtype_bytes);
    std::uint64_t actual_size = std::filesystem::file_size(task.part_bin_path, ec);
    if (ec || actual_size != expected_size)
    {
        return false;
    }

    result.reused = true;
    return true;
}

static std::string now_string()
{
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

static bool write_part_meta(const FileTask &task, const PartSignature &sig, const PartResult &result, std::string &err)
{
    std::ofstream out(task.part_meta_path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        err = "failed to write part meta: " + task.part_meta_path.string();
        return false;
    }
    out << "created_at=" << now_string() << "\n";
    out << "source_path=" << task.path << "\n";
    out << "source_size=" << task.file_size << "\n";
    out << "source_mtime=" << task.file_mtime << "\n";
    out << "tokenizer_fingerprint=" << sig.tokenizer_fingerprint << "\n";
    out << "text_field=" << sig.text_field << "\n";
    out << "min_chars=" << sig.min_chars << "\n";
    out << "max_chars=" << sig.max_chars << "\n";
    out << "bos_id=" << sig.bos_id << "\n";
    out << "eos_id=" << sig.eos_id << "\n";
    out << "dtype_bytes=" << sig.dtype_bytes << "\n";
    out << "num_docs=" << result.num_docs << "\n";
    out << "num_skipped=" << result.num_skipped << "\n";
    out << "num_tokens=" << result.num_tokens << "\n";
    out.flush();
    if (!out)
    {
        err = "failed to flush part meta: " + task.part_meta_path.string();
        return false;
    }
    return true;
}

static std::size_t utf8_char_count(const std::string &s)
{
    std::size_t i = 0;
    std::size_t n = 0;
    while (i < s.size())
    {
        std::size_t prev = i;
        std::uint32_t cp = 0;
        if (!next_codepoint(s, i, cp))
        {
            break;
        }
        if (i <= prev)
        {
            break;
        }
        ++n;
    }
    return n;
}

static bool process_file_to_part(const FileTask &task, const Args &args, const BPETokenizer &tokenizer,
                                 const PartSignature &sig, PartResult &result, std::string &err)
{
    if (args.resume && part_is_reusable(task, sig, result))
    {
        return true;
    }

    std::error_code ec;
    std::filesystem::create_directories(task.part_bin_path.parent_path(), ec);
    if (ec)
    {
        err = "failed to create parts dir: " + task.part_bin_path.parent_path().string();
        return false;
    }

    TokenWriter writer(task.part_bin_path, sig.dtype_bytes);
    if (!writer.good())
    {
        err = "failed to open part bin for write: " + task.part_bin_path.string();
        return false;
    }

    std::vector<std::uint32_t> write_buffer;
    write_buffer.reserve(1 << 20);
    const std::size_t flush_threshold_tokens = 1 << 20;
    std::unordered_map<std::string, std::vector<std::uint32_t>> cache;
    if (args.cache_max_entries > 0)
    {
        const std::size_t reserve_n = std::min<std::size_t>(args.cache_max_entries, 1 << 16);
        cache.reserve(reserve_n);
    }

    auto flush_buffer = [&]() -> bool {
        if (write_buffer.empty())
        {
            return true;
        }
        if (!writer.write_tokens(write_buffer))
        {
            err = "failed to write part tokens: " + task.part_bin_path.string();
            return false;
        }
        write_buffer.clear();
        return true;
    };

    bool read_ok = true;
    bool callback_ok = true;
    auto on_line = [&](const std::string &line) {
        if (!callback_ok || line.empty())
        {
            return;
        }
        std::string text;
        if (!extract_json_field(line, args.text_field, text) || text.empty())
        {
            return;
        }
        std::size_t chars = utf8_char_count(text);
        if (chars < args.min_chars)
        {
            ++result.num_skipped;
            return;
        }
        if (args.max_chars > 0 && chars > args.max_chars)
        {
            text = truncate_utf8(text, args.max_chars);
        }

        std::size_t before = write_buffer.size();
        if (sig.bos_id >= 0)
        {
            write_buffer.push_back(static_cast<std::uint32_t>(sig.bos_id));
        }
        tokenizer.encode_text_append(text, cache, write_buffer);
        if (sig.eos_id >= 0)
        {
            write_buffer.push_back(static_cast<std::uint32_t>(sig.eos_id));
        }
        if (args.cache_max_entries == 0 || cache.size() > args.cache_max_entries)
        {
            cache.clear();
            cache.rehash(0);
        }
        result.num_docs += 1;
        result.num_tokens += static_cast<std::uint64_t>(write_buffer.size() - before);
        if (write_buffer.size() >= flush_threshold_tokens)
        {
            if (!flush_buffer())
            {
                callback_ok = false;
            }
        }
    };

    if (ends_with(task.path, ".gz"))
    {
        read_ok = read_gz_lines(task.path, on_line);
    }
    else
    {
        read_ok = read_text_lines(task.path, on_line);
    }
    if (!read_ok)
    {
        err = "failed to read input file: " + task.path;
        return false;
    }
    if (!callback_ok)
    {
        return false;
    }
    if (!flush_buffer())
    {
        return false;
    }
    if (!writer.close())
    {
        err = "failed to close part bin: " + task.part_bin_path.string();
        return false;
    }

    result.reused = false;
    return write_part_meta(task, sig, result, err);
}

static std::string json_escape(const std::string &s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            }
            else
            {
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    return out;
}

static std::string shard_name(std::size_t idx)
{
    std::ostringstream oss;
    oss << "train_" << std::setw(6) << std::setfill('0') << idx << ".bin";
    return oss.str();
}

static bool remove_old_shards(const std::filesystem::path &out_dir)
{
    std::error_code ec;
    for (std::filesystem::directory_iterator it(out_dir, ec); !ec && it != std::filesystem::directory_iterator();
         it.increment(ec))
    {
        if (!it->is_regular_file())
        {
            continue;
        }
        std::string name = it->path().filename().string();
        if (starts_with(name, "train_") && ends_with(name, ".bin"))
        {
            std::filesystem::remove(it->path(), ec);
            if (ec)
            {
                return false;
            }
        }
    }
    return !ec;
}

static bool build_train_shards(const std::filesystem::path &out_dir, const std::vector<FileTask> &tasks,
                               std::uint32_t dtype_bytes, std::uint64_t max_tokens_per_shard,
                               std::vector<ShardInfo> &shards, std::uint64_t &total_tokens, std::string &err)
{
    // Best-effort cleanup; stale shard files are harmless because meta.json lists active shards.
    remove_old_shards(out_dir);

    std::size_t shard_idx = 0;
    std::uint64_t current_tokens = 0;
    total_tokens = 0;
    std::ofstream out;
    std::filesystem::path current_path;

    auto open_new_shard = [&]() -> bool {
        current_path = out_dir / shard_name(shard_idx);
        out.open(current_path, std::ios::binary | std::ios::trunc);
        current_tokens = 0;
        return static_cast<bool>(out);
    };
    auto close_current_shard = [&]() {
        if (out.is_open())
        {
            out.close();
            shards.push_back({current_path.filename().string(), current_tokens});
        }
    };

    if (!open_new_shard())
    {
        err = "failed to open shard for write: " + (out_dir / shard_name(shard_idx)).string();
        return false;
    }

    const std::size_t io_chunk_bytes = 4 * 1024 * 1024;
    std::vector<char> buffer(io_chunk_bytes);

    ProgressTracker progress(tasks.size(), "sharding", 1000);
    for (std::size_t idx = 0; idx < tasks.size(); ++idx)
    {
        const auto &task = tasks[idx];
        std::ifstream in(task.part_bin_path, std::ios::binary);
        if (!in)
        {
            err = "failed to open part bin: " + task.part_bin_path.string();
            return false;
        }
        while (in)
        {
            in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            std::streamsize got = in.gcount();
            if (got <= 0)
            {
                break;
            }
            if (got % static_cast<std::streamsize>(dtype_bytes) != 0)
            {
                err = "corrupt part file (size not aligned to dtype): " + task.part_bin_path.string();
                return false;
            }
            std::size_t tokens_in_buf = static_cast<std::size_t>(got / static_cast<std::streamsize>(dtype_bytes));
            std::size_t offset_tokens = 0;
            while (offset_tokens < tokens_in_buf)
            {
                if (current_tokens >= max_tokens_per_shard)
                {
                    close_current_shard();
                    ++shard_idx;
                    if (!open_new_shard())
                    {
                        err = "failed to open shard for write: " + (out_dir / shard_name(shard_idx)).string();
                        return false;
                    }
                }
                std::uint64_t remaining = max_tokens_per_shard - current_tokens;
                std::size_t take = static_cast<std::size_t>(
                    std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(tokens_in_buf - offset_tokens)));
                const char *ptr = buffer.data() + offset_tokens * dtype_bytes;
                out.write(ptr, static_cast<std::streamsize>(take * dtype_bytes));
                if (!out)
                {
                    err = "failed to write shard: " + current_path.string();
                    return false;
                }
                current_tokens += static_cast<std::uint64_t>(take);
                total_tokens += static_cast<std::uint64_t>(take);
                offset_tokens += take;
            }
        }
        progress.add(1, 0);
    }
    progress.finish();
    close_current_shard();

    if (shards.empty())
    {
        err = "no shards written";
        return false;
    }
    return true;
}

static bool write_meta_json(const std::filesystem::path &meta_path, const Args &args, const std::string &data_glob,
                            const std::vector<std::string> &input_files, std::size_t vocab_size,
                            const std::string &dtype_name, std::int64_t eos_id, std::int64_t bos_id,
                            std::uint64_t num_docs, std::uint64_t num_skipped, std::uint64_t total_tokens,
                            const std::vector<ShardInfo> &shards, std::size_t reused_files, std::string &err)
{
    std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        err = "failed to write meta.json: " + meta_path.string();
        return false;
    }
    out << "{\n";
    out << "  \"created_at\": \"" << json_escape(now_string()) << "\",\n";
    out << "  \"tokenizer_path\": \"" << json_escape(args.tokenizer_path) << "\",\n";
    out << "  \"text_field\": \"" << json_escape(args.text_field) << "\",\n";
    out << "  \"data_glob\": \"" << json_escape(data_glob) << "\",\n";
    out << "  \"num_input_files\": " << input_files.size() << ",\n";
    out << "  \"input_files\": [\n";
    for (std::size_t i = 0; i < input_files.size(); ++i)
    {
        out << "    \"" << json_escape(input_files[i]) << "\"";
        if (i + 1 < input_files.size())
        {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"vocab_size\": " << vocab_size << ",\n";
    out << "  \"dtype\": \"" << dtype_name << "\",\n";
    if (!args.eos_token.empty())
    {
        out << "  \"eos_token\": \"" << json_escape(args.eos_token) << "\",\n";
        out << "  \"eos_id\": " << eos_id << ",\n";
    }
    else
    {
        out << "  \"eos_token\": null,\n";
        out << "  \"eos_id\": null,\n";
    }
    if (!args.bos_token.empty())
    {
        out << "  \"bos_token\": \"" << json_escape(args.bos_token) << "\",\n";
        out << "  \"bos_id\": " << bos_id << ",\n";
    }
    else
    {
        out << "  \"bos_token\": null,\n";
        out << "  \"bos_id\": null,\n";
    }
    out << "  \"max_tokens_per_shard\": " << args.max_tokens_per_shard << ",\n";
    out << "  \"num_docs\": " << num_docs << ",\n";
    out << "  \"num_skipped\": " << num_skipped << ",\n";
    out << "  \"total_tokens\": " << total_tokens << ",\n";
    out << "  \"shards\": [\n";
    for (std::size_t i = 0; i < shards.size(); ++i)
    {
        out << "    {\"file\": \"" << json_escape(shards[i].file) << "\", \"num_tokens\": " << shards[i].num_tokens << "}";
        if (i + 1 < shards.size())
        {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"num_reused_files\": " << reused_files << "\n";
    out << "}\n";
    if (!out)
    {
        err = "failed to flush meta.json: " + meta_path.string();
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Args args;
    if (!parse_args(argc, argv, args))
    {
        return 0;
    }
    if (args.max_docs > 0)
    {
        std::cerr << "--max-docs is not supported in this C++ path yet. Please keep --max-docs 0.\n";
        return 1;
    }

    auto env = read_env_file(args.env_file);
    std::string data_glob = args.data_glob;
    if (data_glob.empty())
    {
        auto it = env.find("DATA_PATH");
        if (it != env.end())
        {
            data_glob = it->second;
        }
    }
    if (data_glob.empty())
    {
        std::cerr << "DATA_PATH is missing. Set it in .env or pass --data-glob.\n";
        return 1;
    }

    auto files = expand_data_files(data_glob);
    if (files.empty())
    {
        std::cerr << "No files matched: " << data_glob << "\n";
        return 1;
    }

    BPETokenizer tokenizer;
    std::string err;
    if (!tokenizer.load(args.tokenizer_path, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    std::uint32_t eos_id_u32 = 0;
    std::int64_t eos_id = -1;
    if (!args.eos_token.empty())
    {
        if (!tokenizer.token_to_id(args.eos_token, eos_id_u32))
        {
            std::cerr << "eos token not found in tokenizer: " << args.eos_token << "\n";
            return 1;
        }
        eos_id = static_cast<std::int64_t>(eos_id_u32);
    }

    std::uint32_t bos_id_u32 = 0;
    std::int64_t bos_id = -1;
    if (!args.bos_token.empty())
    {
        if (!tokenizer.token_to_id(args.bos_token, bos_id_u32))
        {
            std::cerr << "bos token not found in tokenizer: " << args.bos_token << "\n";
            return 1;
        }
        bos_id = static_cast<std::int64_t>(bos_id_u32);
    }

    std::uint32_t dtype_bytes = tokenizer.vocab_size() <= std::numeric_limits<std::uint16_t>::max() ? 2u : 4u;
    std::string dtype_name = dtype_bytes == 2 ? "uint16" : "uint32";

    std::filesystem::path out_dir = args.out_dir;
    std::filesystem::path parts_dir = out_dir / ".parts";
    std::error_code ec;
    std::filesystem::create_directories(parts_dir, ec);
    if (ec)
    {
        std::cerr << "failed to create output dir: " << out_dir.string() << "\n";
        return 1;
    }

    std::string tokenizer_fp = make_tokenizer_fingerprint(args.tokenizer_path);
    if (tokenizer_fp.empty())
    {
        std::cerr << "failed to fingerprint tokenizer file: " << args.tokenizer_path << "\n";
        return 1;
    }

    std::vector<FileTask> tasks;
    tasks.reserve(files.size());
    for (std::size_t i = 0; i < files.size(); ++i)
    {
        std::uint64_t sz = 0;
        std::int64_t mt = 0;
        if (!get_file_stat(files[i], sz, mt))
        {
            std::cerr << "failed to stat input file: " << files[i] << "\n";
            return 1;
        }
        std::ostringstream base;
        base << "part_" << std::setw(6) << std::setfill('0') << i;
        FileTask t;
        t.index = i;
        t.path = files[i];
        t.file_size = sz;
        t.file_mtime = mt;
        t.part_bin_path = parts_dir / (base.str() + ".bin");
        t.part_meta_path = parts_dir / (base.str() + ".meta");
        tasks.push_back(std::move(t));
    }

    PartSignature sig;
    sig.tokenizer_fingerprint = tokenizer_fp;
    sig.text_field = args.text_field;
    sig.min_chars = args.min_chars;
    sig.max_chars = args.max_chars;
    sig.bos_id = bos_id;
    sig.eos_id = eos_id;
    sig.dtype_bytes = dtype_bytes;

    std::size_t worker_threads = args.threads;
    if (worker_threads == 0)
    {
        worker_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }

    std::cerr << "Files: " << files.size() << "\n";
    std::cerr << "Threads: " << worker_threads << "\n";
    std::cerr << "Cache max entries/file: " << args.cache_max_entries << "\n";
    std::cerr << "Tokenizer vocab: " << tokenizer.vocab_size() << " (dtype=" << dtype_name << ")\n";
    std::cerr << "Output dir: " << out_dir.string() << "\n";

    auto start_time = std::chrono::steady_clock::now();
    std::vector<PartResult> results(tasks.size());
    std::atomic<std::size_t> next_idx{0};
    std::atomic<bool> had_error{false};
    std::mutex err_mu;
    std::string shared_err;
    ProgressTracker progress(tasks.size(), "tokenizing", 1000);

    auto worker = [&]() {
        while (true)
        {
            std::size_t idx = next_idx.fetch_add(1);
            if (idx >= tasks.size())
            {
                break;
            }
            PartResult r;
            std::string local_err;
            if (!process_file_to_part(tasks[idx], args, tokenizer, sig, r, local_err))
            {
                had_error.store(true);
                std::lock_guard<std::mutex> lock(err_mu);
                if (shared_err.empty())
                {
                    shared_err = local_err;
                }
            }
            results[idx] = r;
            progress.add(1, r.num_docs);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(worker_threads);
    for (std::size_t t = 0; t < worker_threads; ++t)
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
        std::cerr << (shared_err.empty() ? "tokenization failed" : shared_err) << "\n";
        return 1;
    }

    std::uint64_t num_docs = 0;
    std::uint64_t num_skipped = 0;
    std::uint64_t expected_tokens = 0;
    std::size_t reused_files = 0;
    for (const auto &r : results)
    {
        num_docs += r.num_docs;
        num_skipped += r.num_skipped;
        expected_tokens += r.num_tokens;
        if (r.reused)
        {
            ++reused_files;
        }
    }

    std::vector<ShardInfo> shards;
    std::uint64_t total_tokens = 0;
    err.clear();
    if (!build_train_shards(out_dir, tasks, dtype_bytes, args.max_tokens_per_shard, shards, total_tokens, err))
    {
        std::cerr << err << "\n";
        return 1;
    }
    if (expected_tokens != total_tokens)
    {
        std::cerr << "token count mismatch: expected " << expected_tokens << " from parts, got " << total_tokens
                  << " in shards\n";
        return 1;
    }

    std::filesystem::path meta_path = out_dir / "meta.json";
    if (!write_meta_json(meta_path, args, data_glob, files, tokenizer.vocab_size(), dtype_name, eos_id, bos_id, num_docs,
                         num_skipped, total_tokens, shards, reused_files, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    double elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start_time).count();
    if (elapsed < 1e-9)
    {
        elapsed = 1e-9;
    }
    std::cerr << "done. docs=" << num_docs << " skipped=" << num_skipped << " total_tokens=" << total_tokens << "\n";
    std::cerr << "shards=" << shards.size() << " dtype=" << dtype_name << " out=" << out_dir.string() << "\n";
    std::cerr << "reused_files=" << reused_files << "/" << files.size() << "\n";
    std::cerr << "throughput docs/s=" << static_cast<double>(num_docs) / elapsed
              << " tok/s=" << static_cast<double>(total_tokens) / elapsed << "\n";
    return 0;
}
