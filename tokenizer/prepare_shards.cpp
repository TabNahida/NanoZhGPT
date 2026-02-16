#include "byte_bpe_lib.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
struct Options
{
    std::string env_file = ".env";
    std::string data_glob;
    std::string text_field = "text";
    std::string tokenizer_path = "tokenizer.json";
    std::string out_dir = "data/tokens";
    std::uint64_t max_tokens_per_shard = 50'000'000;
    std::size_t encode_batch_size = 256;
    std::size_t min_chars = 1;
    std::size_t max_chars = 20'000;
    std::uint64_t max_docs = 0; // 0 means no limit
    std::optional<std::string> eos_token = std::string("</s>");
    std::optional<std::string> bos_token = std::nullopt;
    std::uint64_t progress_every = 10'000;
};

enum class ParseStatus
{
    Ok = 0,
    Help = 1,
    Error = 2
};

struct MergePair
{
    std::string left;
    std::string right;
};

struct TokenizerJsonModel
{
    std::unordered_map<std::string, std::uint32_t> vocab;
    std::vector<MergePair> merges;
    std::string unk_token;
};

struct MergeRule
{
    std::uint32_t rank = 0;
    std::uint32_t merged_id = 0;
};

struct ShardRecord
{
    std::string file;
    std::uint64_t num_tokens = 0;
};

static std::string trim(const std::string &s)
{
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
    {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
    {
        --end;
    }
    return s.substr(start, end - start);
}

static bool ends_with(const std::string &s, const std::string &suffix)
{
    if (suffix.size() > s.size())
    {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

static std::string format_with_commas(std::uint64_t value)
{
    std::string s = std::to_string(value);
    std::string out;
    out.reserve(s.size() + s.size() / 3);
    const std::size_t n = s.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        out.push_back(s[i]);
        std::size_t rem = n - i - 1;
        if (rem > 0 && rem % 3 == 0)
        {
            out.push_back(',');
        }
    }
    return out;
}

static std::string json_escape(const std::string &s)
{
    std::ostringstream oss;
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '"':
            oss << "\\\"";
            break;
        case '\\':
            oss << "\\\\";
            break;
        case '\b':
            oss << "\\b";
            break;
        case '\f':
            oss << "\\f";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            if (c < 0x20)
            {
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec
                    << std::setfill(' ');
            }
            else
            {
                oss << static_cast<char>(c);
            }
            break;
        }
    }
    return oss.str();
}

static bool parse_u64(const std::string &s, std::uint64_t &out)
{
    try
    {
        std::size_t idx = 0;
        std::uint64_t v = std::stoull(s, &idx);
        if (idx != s.size())
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

static std::string current_time_string()
{
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static void print_usage()
{
    std::cerr << "Prepare token shards (json/json.gz -> binary)\n"
              << "Usage:\n"
              << "  xmake run prepare_shards -- [options]\n\n"
              << "Options:\n"
              << "  --env-file <path>            Path to .env (default: .env)\n"
              << "  --data-glob <glob>           Override DATA_PATH from .env\n"
              << "  --text-field <name>          JSON text field (default: text)\n"
              << "  --tokenizer <path>           tokenizer.json path (default: tokenizer.json)\n"
              << "  --out-dir <path>             Output directory (default: data/tokens)\n"
              << "  --max-tokens-per-shard <n>   Max tokens per shard (default: 50000000)\n"
              << "  --encode-batch-size <n>      Docs per encode batch (default: 256)\n"
              << "  --min-chars <n>              Minimum chars per doc (default: 1)\n"
              << "  --max-chars <n>              Truncate doc to N chars, 0=disable (default: 20000)\n"
              << "  --max-docs <n>               Max docs to process, 0=all (default: 0)\n"
              << "  --eos-token <token>          EOS token text (default: </s>)\n"
              << "  --bos-token <token>          BOS token text (default: disabled)\n"
              << "  --progress-every <n>         Progress print cadence in docs (default: 10000)\n"
              << "  --help                       Show this help\n";
}

static ParseStatus parse_args(int argc, char **argv, Options &opt)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage();
            return ParseStatus::Help;
        }
        auto need_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--env-file")
        {
            const char *v = need_value("--env-file");
            if (!v)
            {
                return ParseStatus::Error;
            }
            opt.env_file = v;
        }
        else if (arg == "--data-glob")
        {
            const char *v = need_value("--data-glob");
            if (!v)
            {
                return ParseStatus::Error;
            }
            opt.data_glob = v;
        }
        else if (arg == "--text-field")
        {
            const char *v = need_value("--text-field");
            if (!v)
            {
                return ParseStatus::Error;
            }
            opt.text_field = v;
        }
        else if (arg == "--tokenizer")
        {
            const char *v = need_value("--tokenizer");
            if (!v)
            {
                return ParseStatus::Error;
            }
            opt.tokenizer_path = v;
        }
        else if (arg == "--out-dir")
        {
            const char *v = need_value("--out-dir");
            if (!v)
            {
                return ParseStatus::Error;
            }
            opt.out_dir = v;
        }
        else if (arg == "--max-tokens-per-shard")
        {
            const char *v = need_value("--max-tokens-per-shard");
            if (!v)
            {
                return ParseStatus::Error;
            }
            std::uint64_t parsed = 0;
            if (!parse_u64(v, parsed))
            {
                std::cerr << "Invalid integer for --max-tokens-per-shard: " << v << "\n";
                return ParseStatus::Error;
            }
            opt.max_tokens_per_shard = parsed;
        }
        else if (arg == "--encode-batch-size")
        {
            const char *v = need_value("--encode-batch-size");
            if (!v)
            {
                return ParseStatus::Error;
            }
            std::uint64_t parsed = 0;
            if (!parse_u64(v, parsed))
            {
                std::cerr << "Invalid integer for --encode-batch-size: " << v << "\n";
                return ParseStatus::Error;
            }
            opt.encode_batch_size = static_cast<std::size_t>(parsed);
        }
        else if (arg == "--min-chars")
        {
            const char *v = need_value("--min-chars");
            if (!v)
            {
                return ParseStatus::Error;
            }
            std::uint64_t parsed = 0;
            if (!parse_u64(v, parsed))
            {
                std::cerr << "Invalid integer for --min-chars: " << v << "\n";
                return ParseStatus::Error;
            }
            opt.min_chars = static_cast<std::size_t>(parsed);
        }
        else if (arg == "--max-chars")
        {
            const char *v = need_value("--max-chars");
            if (!v)
            {
                return ParseStatus::Error;
            }
            std::uint64_t parsed = 0;
            if (!parse_u64(v, parsed))
            {
                std::cerr << "Invalid integer for --max-chars: " << v << "\n";
                return ParseStatus::Error;
            }
            opt.max_chars = static_cast<std::size_t>(parsed);
        }
        else if (arg == "--max-docs")
        {
            const char *v = need_value("--max-docs");
            if (!v)
            {
                return ParseStatus::Error;
            }
            std::uint64_t parsed = 0;
            if (!parse_u64(v, parsed))
            {
                std::cerr << "Invalid integer for --max-docs: " << v << "\n";
                return ParseStatus::Error;
            }
            opt.max_docs = parsed;
        }
        else if (arg == "--eos-token")
        {
            const char *v = need_value("--eos-token");
            if (!v)
            {
                return ParseStatus::Error;
            }
            opt.eos_token = std::string(v);
        }
        else if (arg == "--bos-token")
        {
            const char *v = need_value("--bos-token");
            if (!v)
            {
                return ParseStatus::Error;
            }
            opt.bos_token = std::string(v);
        }
        else if (arg == "--progress-every")
        {
            const char *v = need_value("--progress-every");
            if (!v)
            {
                return ParseStatus::Error;
            }
            std::uint64_t parsed = 0;
            if (!parse_u64(v, parsed))
            {
                std::cerr << "Invalid integer for --progress-every: " << v << "\n";
                return ParseStatus::Error;
            }
            opt.progress_every = parsed;
        }
        else
        {
            std::cerr << "Unknown arg: " << arg << "\n";
            return ParseStatus::Error;
        }
    }

    if (opt.encode_batch_size == 0)
    {
        opt.encode_batch_size = 1;
    }
    return ParseStatus::Ok;
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

static bool has_min_chars(const std::string &s, std::size_t min_chars)
{
    if (min_chars == 0)
    {
        return true;
    }
    std::size_t i = 0;
    std::size_t count = 0;
    while (i < s.size() && count < min_chars)
    {
        std::size_t before = i;
        std::uint32_t cp = 0;
        if (!next_codepoint(s, i, cp))
        {
            break;
        }
        if (i <= before)
        {
            break;
        }
        ++count;
    }
    return count >= min_chars;
}

template <typename Fn>
bool iterate_texts(const std::string &path, const std::string &text_field, Fn &&cb)
{
    auto on_line = [&](const std::string &raw_line) {
        std::string line = trim(raw_line);
        if (line.empty())
        {
            return;
        }
        std::string text;
        if (!extract_json_field(line, text_field, text))
        {
            return;
        }
        if (text.empty())
        {
            return;
        }
        cb(text);
    };
    if (ends_with(path, ".gz"))
    {
        return read_gz_lines(path, on_line);
    }
    return read_text_lines(path, on_line);
}

static bool fail_parse(std::string &err, const std::string &msg)
{
    if (err.empty())
    {
        err = msg;
    }
    return false;
}

static void skip_ws(const std::string &s, std::size_t &i)
{
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
    {
        ++i;
    }
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
        if (c != '\\')
        {
            out.push_back(c);
            continue;
        }
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
    return false;
}

static bool parse_json_uint(const std::string &s, std::size_t &i, std::uint32_t &out)
{
    skip_ws(s, i);
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
    {
        return false;
    }
    std::size_t start = i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
    {
        ++i;
    }
    std::uint64_t val = 0;
    if (!parse_u64(s.substr(start, i - start), val))
    {
        return false;
    }
    if (val > std::numeric_limits<std::uint32_t>::max())
    {
        return false;
    }
    out = static_cast<std::uint32_t>(val);
    return true;
}

static bool skip_json_value(const std::string &s, std::size_t &i)
{
    skip_ws(s, i);
    if (i >= s.size())
    {
        return false;
    }
    if (s[i] == '"')
    {
        std::string tmp;
        return parse_json_string(s, i, tmp);
    }
    if (s[i] == '{')
    {
        ++i;
        skip_ws(s, i);
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        while (i < s.size())
        {
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
        return false;
    }
    if (s[i] == '[')
    {
        ++i;
        skip_ws(s, i);
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        while (i < s.size())
        {
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
        return false;
    }
    while (i < s.size())
    {
        char c = s[i];
        if (c == ',' || c == '}' || c == ']' || std::isspace(static_cast<unsigned char>(c)))
        {
            return true;
        }
        ++i;
    }
    return true;
}

static bool parse_merge_string(const std::string &s, MergePair &out)
{
    std::size_t pos = s.find(' ');
    if (pos == std::string::npos || pos == 0)
    {
        return false;
    }
    std::size_t rhs_start = pos + 1;
    while (rhs_start < s.size() && s[rhs_start] == ' ')
    {
        ++rhs_start;
    }
    if (rhs_start >= s.size())
    {
        return false;
    }
    out.left = s.substr(0, pos);
    out.right = s.substr(rhs_start);
    return true;
}

static bool parse_vocab_object(const std::string &s, std::size_t &i,
                               std::unordered_map<std::string, std::uint32_t> &vocab, std::string &err)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{')
    {
        return fail_parse(err, "vocab must be a JSON object");
    }
    ++i;
    skip_ws(s, i);
    if (i < s.size() && s[i] == '}')
    {
        ++i;
        return true;
    }
    while (i < s.size())
    {
        std::string token;
        if (!parse_json_string(s, i, token))
        {
            return fail_parse(err, "failed parsing vocab key");
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return fail_parse(err, "expected ':' after vocab key");
        }
        ++i;
        std::uint32_t id = 0;
        if (!parse_json_uint(s, i, id))
        {
            return fail_parse(err, "failed parsing vocab id");
        }
        vocab[token] = id;
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
        return fail_parse(err, "expected ',' or '}' in vocab object");
    }
    return fail_parse(err, "unexpected EOF in vocab object");
}

static bool parse_merge_pair_array(const std::string &s, std::size_t &i, MergePair &pair, std::string &err)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return fail_parse(err, "merge pair must be array");
    }
    ++i;
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"')
    {
        return fail_parse(err, "merge pair[0] must be string");
    }
    if (!parse_json_string(s, i, pair.left))
    {
        return fail_parse(err, "failed parsing merge pair[0]");
    }
    skip_ws(s, i);
    if (i >= s.size() || s[i] != ',')
    {
        return fail_parse(err, "expected ',' in merge pair");
    }
    ++i;
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"')
    {
        return fail_parse(err, "merge pair[1] must be string");
    }
    if (!parse_json_string(s, i, pair.right))
    {
        return fail_parse(err, "failed parsing merge pair[1]");
    }
    skip_ws(s, i);
    while (i < s.size() && s[i] == ',')
    {
        ++i;
        if (!skip_json_value(s, i))
        {
            return fail_parse(err, "failed skipping extra merge pair value");
        }
        skip_ws(s, i);
    }
    if (i >= s.size() || s[i] != ']')
    {
        return fail_parse(err, "expected ']' at end of merge pair");
    }
    ++i;
    return true;
}

static bool parse_merges_array(const std::string &s, std::size_t &i, std::vector<MergePair> &merges, std::string &err)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return fail_parse(err, "merges must be a JSON array");
    }
    ++i;
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']')
    {
        ++i;
        return true;
    }
    while (i < s.size())
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return fail_parse(err, "unexpected EOF in merges array");
        }
        if (s[i] == '"')
        {
            std::string merge_str;
            if (!parse_json_string(s, i, merge_str))
            {
                return fail_parse(err, "failed parsing merge string");
            }
            MergePair pair;
            if (parse_merge_string(merge_str, pair))
            {
                merges.push_back(std::move(pair));
            }
        }
        else if (s[i] == '[')
        {
            MergePair pair;
            if (!parse_merge_pair_array(s, i, pair, err))
            {
                return false;
            }
            merges.push_back(std::move(pair));
        }
        else
        {
            if (!skip_json_value(s, i))
            {
                return fail_parse(err, "failed skipping non-string merge entry");
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
        return fail_parse(err, "expected ',' or ']' in merges array");
    }
    return fail_parse(err, "unexpected EOF in merges array");
}

static bool parse_model_object(const std::string &s, std::size_t &i, TokenizerJsonModel &model, std::string &err)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{')
    {
        return fail_parse(err, "model must be a JSON object");
    }
    ++i;
    skip_ws(s, i);
    if (i < s.size() && s[i] == '}')
    {
        ++i;
        return true;
    }
    while (i < s.size())
    {
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return fail_parse(err, "failed parsing model key");
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return fail_parse(err, "expected ':' after model key");
        }
        ++i;
        if (key == "vocab")
        {
            if (!parse_vocab_object(s, i, model.vocab, err))
            {
                return false;
            }
        }
        else if (key == "merges")
        {
            if (!parse_merges_array(s, i, model.merges, err))
            {
                return false;
            }
        }
        else if (key == "unk_token")
        {
            skip_ws(s, i);
            if (i < s.size() && s[i] == '"')
            {
                if (!parse_json_string(s, i, model.unk_token))
                {
                    return fail_parse(err, "failed parsing model.unk_token");
                }
            }
            else
            {
                if (!skip_json_value(s, i))
                {
                    return fail_parse(err, "failed skipping model.unk_token value");
                }
            }
        }
        else
        {
            if (!skip_json_value(s, i))
            {
                return fail_parse(err, "failed skipping model value");
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
        return fail_parse(err, "expected ',' or '}' in model object");
    }
    return fail_parse(err, "unexpected EOF in model object");
}

static bool parse_tokenizer_json(const std::string &s, TokenizerJsonModel &model, std::string &err)
{
    std::size_t i = 0;
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{')
    {
        return fail_parse(err, "tokenizer root must be object");
    }
    ++i;

    bool found_model = false;
    skip_ws(s, i);
    if (i < s.size() && s[i] == '}')
    {
        return fail_parse(err, "empty tokenizer JSON");
    }

    while (i < s.size())
    {
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return fail_parse(err, "failed parsing tokenizer root key");
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return fail_parse(err, "expected ':' after root key");
        }
        ++i;
        if (key == "model")
        {
            if (!parse_model_object(s, i, model, err))
            {
                return false;
            }
            found_model = true;
        }
        else
        {
            if (!skip_json_value(s, i))
            {
                return fail_parse(err, "failed skipping root value");
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
        return fail_parse(err, "expected ',' or '}' in tokenizer root");
    }

    if (!found_model)
    {
        return fail_parse(err, "missing model object in tokenizer");
    }
    if (model.vocab.empty())
    {
        return fail_parse(err, "model.vocab is empty");
    }
    return true;
}

class BpeTokenizer
{
  public:
    bool load_from_file(const std::string &path, std::string &err)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            err = "failed to open tokenizer file: " + path;
            return false;
        }
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

        TokenizerJsonModel model;
        if (!parse_tokenizer_json(content, model, err))
        {
            err = "failed to parse tokenizer JSON: " + err;
            return false;
        }

        vocab_ = std::move(model.vocab);
        max_token_id_ = 0;
        for (const auto &kv : vocab_)
        {
            if (kv.second > max_token_id_)
            {
                max_token_id_ = kv.second;
            }
        }
        merge_rules_.clear();
        merge_rules_.reserve(model.merges.size() * 13 / 10 + 8);

        if (!model.unk_token.empty())
        {
            auto it = vocab_.find(model.unk_token);
            if (it != vocab_.end())
            {
                unk_id_ = static_cast<std::int32_t>(it->second);
            }
        }
        if (unk_id_ < 0)
        {
            auto it = vocab_.find("<unk>");
            if (it != vocab_.end())
            {
                unk_id_ = static_cast<std::int32_t>(it->second);
            }
        }

        for (std::size_t rank = 0; rank < model.merges.size(); ++rank)
        {
            if (rank > std::numeric_limits<std::uint32_t>::max())
            {
                break;
            }
            const auto &m = model.merges[rank];
            auto it_left = vocab_.find(m.left);
            auto it_right = vocab_.find(m.right);
            if (it_left == vocab_.end() || it_right == vocab_.end())
            {
                continue;
            }
            std::string merged = m.left + m.right;
            auto it_merged = vocab_.find(merged);
            if (it_merged == vocab_.end())
            {
                continue;
            }
            std::uint64_t key = pair_key(it_left->second, it_right->second);
            MergeRule rule{static_cast<std::uint32_t>(rank), it_merged->second};
            auto it = merge_rules_.find(key);
            if (it == merge_rules_.end() || rule.rank < it->second.rank)
            {
                merge_rules_[key] = rule;
            }
        }

        auto cp_map = build_byte_to_unicode_cp();
        byte_to_unicode_ = build_byte_to_unicode_str(cp_map);
        return true;
    }

    std::size_t vocab_size() const
    {
        return vocab_.size();
    }

    std::uint32_t max_token_id() const
    {
        return max_token_id_;
    }

    std::optional<std::uint32_t> token_to_id(const std::string &token) const
    {
        auto it = vocab_.find(token);
        if (it == vocab_.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    std::vector<std::uint32_t> encode(const std::string &text) const
    {
        std::vector<std::uint32_t> out_ids;
        auto words = pretokenize(text);
        out_ids.reserve(text.size());
        for (const auto &word : words)
        {
            if (word.empty())
            {
                continue;
            }
            std::string encoded = byte_level_encode(word, byte_to_unicode_);
            auto symbols = split_encoded_symbols_to_ids(encoded);
            if (symbols.empty())
            {
                continue;
            }
            apply_bpe(symbols);
            out_ids.insert(out_ids.end(), symbols.begin(), symbols.end());
        }
        return out_ids;
    }

  private:
    static std::uint64_t pair_key(std::uint32_t left, std::uint32_t right)
    {
        return (static_cast<std::uint64_t>(left) << 32) | static_cast<std::uint64_t>(right);
    }

    std::vector<std::uint32_t> split_encoded_symbols_to_ids(const std::string &encoded) const
    {
        std::vector<std::uint32_t> ids;
        ids.reserve(encoded.size());
        std::size_t i = 0;
        while (i < encoded.size())
        {
            std::size_t start = i;
            std::uint32_t cp = 0;
            if (!next_codepoint(encoded, i, cp))
            {
                break;
            }
            if (i <= start)
            {
                break;
            }
            (void)cp;
            std::string symbol = encoded.substr(start, i - start);
            auto it = vocab_.find(symbol);
            if (it != vocab_.end())
            {
                ids.push_back(it->second);
            }
            else if (unk_id_ >= 0)
            {
                ids.push_back(static_cast<std::uint32_t>(unk_id_));
            }
        }
        return ids;
    }

    void apply_bpe(std::vector<std::uint32_t> &symbols) const
    {
        if (symbols.size() < 2 || merge_rules_.empty())
        {
            return;
        }
        while (symbols.size() >= 2)
        {
            std::size_t best_idx = symbols.size();
            std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
            std::uint32_t best_merged_id = 0;

            for (std::size_t i = 0; i + 1 < symbols.size(); ++i)
            {
                auto it = merge_rules_.find(pair_key(symbols[i], symbols[i + 1]));
                if (it == merge_rules_.end())
                {
                    continue;
                }
                const MergeRule &rule = it->second;
                if (rule.rank < best_rank)
                {
                    best_rank = rule.rank;
                    best_merged_id = rule.merged_id;
                    best_idx = i;
                }
            }

            if (best_idx == symbols.size())
            {
                break;
            }

            symbols[best_idx] = best_merged_id;
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
        }
    }

    std::unordered_map<std::string, std::uint32_t> vocab_;
    std::unordered_map<std::uint64_t, MergeRule> merge_rules_;
    std::array<std::string, 256> byte_to_unicode_{};
    std::int32_t unk_id_ = -1;
    std::uint32_t max_token_id_ = 0;
};

class ShardWriter
{
  public:
    ShardWriter(std::filesystem::path out_dir, bool use_u16, std::uint64_t max_tokens_per_shard, std::string prefix)
        : out_dir_(std::move(out_dir)), use_u16_(use_u16), max_tokens_per_shard_(max_tokens_per_shard),
          prefix_(std::move(prefix))
    {
        open_new_shard();
    }

    ~ShardWriter()
    {
        try
        {
            close();
        }
        catch (...)
        {
        }
    }

    void add_tokens(const std::vector<std::uint32_t> &token_ids)
    {
        if (token_ids.empty())
        {
            return;
        }
        std::size_t idx = 0;
        while (idx < token_ids.size())
        {
            std::uint64_t remaining = max_tokens_per_shard_ - current_tokens_;
            if (remaining == 0)
            {
                close_current_shard();
                ++shard_index_;
                open_new_shard();
                remaining = max_tokens_per_shard_;
            }

            std::uint64_t left = static_cast<std::uint64_t>(token_ids.size() - idx);
            std::uint64_t take_u64 = std::min(remaining, left);
            std::size_t take = static_cast<std::size_t>(take_u64);
            write_slice(token_ids.data() + idx, take);
            current_tokens_ += take_u64;
            total_tokens_ += take_u64;
            idx += take;
        }
    }

    void close()
    {
        if (fp_.is_open())
        {
            close_current_shard();
        }
    }

    std::uint64_t total_tokens() const
    {
        return total_tokens_;
    }

    const std::vector<ShardRecord> &shards() const
    {
        return shards_;
    }

    const char *dtype_name() const
    {
        return use_u16_ ? "uint16" : "uint32";
    }

  private:
    void open_new_shard()
    {
        std::ostringstream oss;
        oss << prefix_ << "_" << std::setw(6) << std::setfill('0') << shard_index_ << ".bin";
        current_path_ = out_dir_ / oss.str();
        fp_.open(current_path_, std::ios::binary);
        if (!fp_)
        {
            throw std::runtime_error("failed to open shard file for write: " + current_path_.string());
        }
        current_tokens_ = 0;
    }

    void close_current_shard()
    {
        fp_.close();
        shards_.push_back(ShardRecord{current_path_.filename().string(), current_tokens_});
    }

    void write_slice(const std::uint32_t *ptr, std::size_t count)
    {
        if (count == 0)
        {
            return;
        }
        if (use_u16_)
        {
            scratch_u16_.resize(count);
            for (std::size_t i = 0; i < count; ++i)
            {
                scratch_u16_[i] = static_cast<std::uint16_t>(ptr[i]);
            }
            fp_.write(reinterpret_cast<const char *>(scratch_u16_.data()),
                      static_cast<std::streamsize>(count * sizeof(std::uint16_t)));
        }
        else
        {
            fp_.write(reinterpret_cast<const char *>(ptr), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
        }
        if (!fp_)
        {
            throw std::runtime_error("failed writing shard data: " + current_path_.string());
        }
    }

    std::filesystem::path out_dir_;
    bool use_u16_ = true;
    std::uint64_t max_tokens_per_shard_ = 0;
    std::string prefix_;

    std::uint64_t shard_index_ = 0;
    std::uint64_t current_tokens_ = 0;
    std::uint64_t total_tokens_ = 0;
    std::filesystem::path current_path_;
    std::ofstream fp_;
    std::vector<std::uint16_t> scratch_u16_;
    std::vector<ShardRecord> shards_;
};

static bool write_meta_json(const std::filesystem::path &meta_path, const Options &opt, const std::string &data_glob,
                            const std::vector<std::string> &input_files, std::size_t vocab_size,
                            const std::optional<std::uint32_t> &eos_id, const std::optional<std::uint32_t> &bos_id,
                            std::uint64_t num_docs, std::uint64_t num_skipped, std::uint64_t total_tokens,
                            const ShardWriter &writer, std::string &err)
{
    std::ofstream out(meta_path, std::ios::binary);
    if (!out)
    {
        err = "failed to open meta file: " + meta_path.string();
        return false;
    }

    auto write_opt_string = [&](const std::optional<std::string> &v) {
        if (!v.has_value())
        {
            out << "null";
            return;
        }
        out << "\"" << json_escape(*v) << "\"";
    };
    auto write_opt_u32 = [&](const std::optional<std::uint32_t> &v) {
        if (!v.has_value())
        {
            out << "null";
            return;
        }
        out << *v;
    };

    out << "{\n";
    out << "  \"created_at\": \"" << json_escape(current_time_string()) << "\",\n";
    out << "  \"tokenizer_path\": \"" << json_escape(opt.tokenizer_path) << "\",\n";
    out << "  \"text_field\": \"" << json_escape(opt.text_field) << "\",\n";
    out << "  \"data_glob\": \"" << json_escape(data_glob) << "\",\n";
    out << "  \"num_input_files\": " << input_files.size() << ",\n";
    out << "  \"input_files\": [\n";
    for (std::size_t i = 0; i < input_files.size(); ++i)
    {
        out << "    \"" << json_escape(input_files[i]) << "\"";
        if (i + 1 != input_files.size())
        {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"vocab_size\": " << vocab_size << ",\n";
    out << "  \"dtype\": \"" << writer.dtype_name() << "\",\n";
    out << "  \"eos_token\": ";
    write_opt_string(opt.eos_token);
    out << ",\n";
    out << "  \"eos_id\": ";
    write_opt_u32(eos_id);
    out << ",\n";
    out << "  \"bos_token\": ";
    write_opt_string(opt.bos_token);
    out << ",\n";
    out << "  \"bos_id\": ";
    write_opt_u32(bos_id);
    out << ",\n";
    out << "  \"max_tokens_per_shard\": " << opt.max_tokens_per_shard << ",\n";
    out << "  \"num_docs\": " << num_docs << ",\n";
    out << "  \"num_skipped\": " << num_skipped << ",\n";
    out << "  \"total_tokens\": " << total_tokens << ",\n";
    out << "  \"shards\": [\n";
    const auto &shards = writer.shards();
    for (std::size_t i = 0; i < shards.size(); ++i)
    {
        out << "    {\"file\": \"" << json_escape(shards[i].file) << "\", \"num_tokens\": " << shards[i].num_tokens
            << "}";
        if (i + 1 != shards.size())
        {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    if (!out)
    {
        err = "failed writing meta file: " + meta_path.string();
        return false;
    }
    return true;
}
} // namespace

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Options opt;
    ParseStatus status = parse_args(argc, argv, opt);
    if (status == ParseStatus::Help)
    {
        return 0;
    }
    if (status == ParseStatus::Error)
    {
        print_usage();
        return 1;
    }

    if (opt.max_tokens_per_shard == 0)
    {
        std::cerr << "--max-tokens-per-shard must be > 0\n";
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(opt.out_dir, ec);
    if (ec)
    {
        std::cerr << "failed to create output directory: " << opt.out_dir << "\n";
        return 1;
    }

    auto env = read_env_file(opt.env_file);
    std::string data_glob = opt.data_glob;
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

    BpeTokenizer tokenizer;
    std::string tok_err;
    if (!tokenizer.load_from_file(opt.tokenizer_path, tok_err))
    {
        std::cerr << tok_err << "\n";
        return 1;
    }

    std::optional<std::uint32_t> eos_id = std::nullopt;
    if (opt.eos_token.has_value() && !opt.eos_token->empty())
    {
        eos_id = tokenizer.token_to_id(*opt.eos_token);
        if (!eos_id.has_value())
        {
            std::cerr << "eos token not found in tokenizer: " << *opt.eos_token << "\n";
            return 1;
        }
    }

    std::optional<std::uint32_t> bos_id = std::nullopt;
    if (opt.bos_token.has_value() && !opt.bos_token->empty())
    {
        bos_id = tokenizer.token_to_id(*opt.bos_token);
        if (!bos_id.has_value())
        {
            std::cerr << "bos token not found in tokenizer: " << *opt.bos_token << "\n";
            return 1;
        }
    }

    bool use_u16 = tokenizer.max_token_id() <= std::numeric_limits<std::uint16_t>::max();
    ShardWriter writer(std::filesystem::path(opt.out_dir), use_u16, opt.max_tokens_per_shard, "train");

    std::uint64_t num_docs = 0;
    std::uint64_t num_skipped = 0;
    std::vector<std::string> batch;
    batch.reserve(opt.encode_batch_size);
    auto t0 = std::chrono::steady_clock::now();

    auto print_progress = [&]() {
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed < 1e-6)
        {
            elapsed = 1e-6;
        }
        double docs_per_s = static_cast<double>(num_docs) / elapsed;
        double tok_per_s = static_cast<double>(writer.total_tokens()) / elapsed;
        std::cout << "[progress] docs=" << format_with_commas(num_docs)
                  << " tokens=" << format_with_commas(writer.total_tokens()) << " docs/s=" << std::fixed
                  << std::setprecision(1) << docs_per_s << " tok/s=" << std::setprecision(0) << tok_per_s << "\n";
    };

    auto flush_batch = [&]() {
        if (batch.empty())
        {
            return;
        }
        for (const auto &doc : batch)
        {
            if (opt.max_docs > 0 && num_docs >= opt.max_docs)
            {
                break;
            }

            auto ids = tokenizer.encode(doc);
            std::vector<std::uint32_t> final_ids;
            final_ids.reserve(ids.size() + (bos_id.has_value() ? 1 : 0) + (eos_id.has_value() ? 1 : 0));
            if (bos_id.has_value())
            {
                final_ids.push_back(*bos_id);
            }
            final_ids.insert(final_ids.end(), ids.begin(), ids.end());
            if (eos_id.has_value())
            {
                final_ids.push_back(*eos_id);
            }
            writer.add_tokens(final_ids);
            ++num_docs;

            if (opt.progress_every > 0 && num_docs > 0 && num_docs % opt.progress_every == 0)
            {
                print_progress();
            }
        }
        batch.clear();
    };

    bool stop_requested = false;
    for (const auto &path : files)
    {
        bool ok = iterate_texts(path, opt.text_field, [&](const std::string &text) {
            if (stop_requested)
            {
                return;
            }
            if (!has_min_chars(text, opt.min_chars))
            {
                ++num_skipped;
                return;
            }

            std::string clipped = text;
            if (opt.max_chars > 0)
            {
                clipped = truncate_utf8(clipped, opt.max_chars);
            }

            batch.push_back(std::move(clipped));
            if (batch.size() >= opt.encode_batch_size)
            {
                flush_batch();
            }
            if (opt.max_docs > 0 && num_docs + static_cast<std::uint64_t>(batch.size()) >= opt.max_docs)
            {
                flush_batch();
            }
            if (opt.max_docs > 0 && num_docs >= opt.max_docs)
            {
                stop_requested = true;
            }
        });

        if (!ok)
        {
            std::cerr << "warning: failed to read file: " << path << "\n";
        }

        if (stop_requested)
        {
            break;
        }
    }

    flush_batch();
    writer.close();

    std::filesystem::path meta_path = std::filesystem::path(opt.out_dir) / "meta.json";
    std::string meta_err;
    if (!write_meta_json(meta_path, opt, data_glob, files, tokenizer.vocab_size(), eos_id, bos_id, num_docs, num_skipped,
                         writer.total_tokens(), writer, meta_err))
    {
        std::cerr << meta_err << "\n";
        return 1;
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (elapsed < 1e-6)
    {
        elapsed = 1e-6;
    }
    double docs_per_s = static_cast<double>(num_docs) / elapsed;
    double tok_per_s = static_cast<double>(writer.total_tokens()) / elapsed;

    std::cout << "done. docs=" << format_with_commas(num_docs) << " skipped=" << format_with_commas(num_skipped)
              << " total_tokens=" << format_with_commas(writer.total_tokens()) << "\n";
    std::cout << "shards=" << writer.shards().size() << " dtype=" << writer.dtype_name() << " out=" << opt.out_dir
              << "\n";
    std::cout << "throughput docs/s=" << std::fixed << std::setprecision(1) << docs_per_s << " tok/s="
              << std::setprecision(0) << tok_per_s << "\n";

    return 0;
}
