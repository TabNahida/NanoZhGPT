#include "byte_bpe_lib.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <zlib.h>

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

static std::vector<std::string> split_csv(const std::string &s)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : s)
    {
        if (c == ',')
        {
            out.push_back(trim(cur));
            cur.clear();
        }
        else
        {
            cur.push_back(c);
        }
    }
    if (!cur.empty() || !out.empty())
    {
        out.push_back(trim(cur));
    }
    out.erase(std::remove_if(out.begin(), out.end(), [](const std::string &v) { return v.empty(); }), out.end());
    return out;
}

static std::size_t parse_size(const std::string &s, std::size_t def_val)
{
    try
    {
        return static_cast<std::size_t>(std::stoull(s));
    }
    catch (...)
    {
        return def_val;
    }
}

static bool parse_bool(const std::string &s, bool def_val)
{
    std::string v;
    v.reserve(s.size());
    for (char c : s)
    {
        v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (v == "1" || v == "true" || v == "yes" || v == "y" || v == "on")
    {
        return true;
    }
    if (v == "0" || v == "false" || v == "no" || v == "n" || v == "off")
    {
        return false;
    }
    return def_val;
}

std::unordered_map<std::string, std::string> read_env_file(const std::string &path)
{
    std::unordered_map<std::string, std::string> env;
    std::ifstream in(path);
    if (!in)
    {
        return env;
    }
    bool first_line = true;
    std::string line;
    while (std::getline(in, line))
    {
        if (first_line)
        {
            first_line = false;
            if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB && static_cast<unsigned char>(line[2]) == 0xBF)
            {
                line.erase(0, 3);
            }
        }
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        auto trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#')
        {
            continue;
        }
        auto eq = trimmed.find('=');
        if (eq == std::string::npos)
        {
            continue;
        }
        std::string key = trim(trimmed.substr(0, eq));
        std::string val = trim(trimmed.substr(eq + 1));
        if (val.size() >= 2 &&
            ((val.front() == '"' && val.back() == '"') || (val.front() == '\'' && val.back() == '\'')))
        {
            val = val.substr(1, val.size() - 2);
        }
        env[key] = val;
    }
    return env;
}

void apply_env_overrides(Config &cfg, const std::unordered_map<std::string, std::string> &env)
{
    auto get = [&](const std::string &key) -> const std::string * {
        auto it = env.find(key);
        if (it == env.end())
        {
            return nullptr;
        }
        return &it->second;
    };
    if (auto v = get("DATA_PATH"))
        cfg.data_glob = *v;
    if (auto v = get("DATA_GLOB"))
        cfg.data_glob = *v;
    if (auto v = get("TEXT_FIELD"))
        cfg.text_field = *v;
    if (auto v = get("VOCAB_SIZE"))
        cfg.vocab_size = parse_size(*v, cfg.vocab_size);
    if (auto v = get("MIN_FREQ"))
        cfg.min_freq = parse_size(*v, cfg.min_freq);
    if (auto v = get("MIN_PAIR_FREQ"))
        cfg.min_pair_freq = parse_size(*v, cfg.min_pair_freq);
    if (auto v = get("CHUNK_FILES"))
        cfg.chunk_files = parse_size(*v, cfg.chunk_files);
    if (auto v = get("CHUNK_DOCS"))
        cfg.chunk_docs = parse_size(*v, cfg.chunk_docs);
    if (auto v = get("TOP_K"))
        cfg.top_k = parse_size(*v, cfg.top_k);
    if (auto v = get("MAX_CHARS_PER_DOC"))
        cfg.max_chars_per_doc = parse_size(*v, cfg.max_chars_per_doc);
    if (auto v = get("THREADS"))
        cfg.threads = parse_size(*v, cfg.threads);
    if (auto v = get("OUTPUT_JSON"))
        cfg.output_json = *v;
    if (auto v = get("OUTPUT_VOCAB"))
        cfg.output_vocab = *v;
    if (auto v = get("OUTPUT_MERGES"))
        cfg.output_merges = *v;
    if (auto v = get("UNK_TOKEN"))
        cfg.unk_token = *v;
    if (auto v = get("CHUNK_DIR"))
        cfg.chunk_dir = *v;
    if (auto v = get("RESUME"))
        cfg.resume = parse_bool(*v, cfg.resume);
    if (auto v = get("PROGRESS_INTERVAL"))
        cfg.progress_interval_ms = parse_size(*v, cfg.progress_interval_ms);
    if (auto v = get("PROGRESS_INTERVAL_MS"))
        cfg.progress_interval_ms = parse_size(*v, cfg.progress_interval_ms);
    if (auto v = get("SPECIAL_TOKENS"))
    {
        auto tokens = split_csv(*v);
        if (!tokens.empty())
        {
            cfg.special_tokens = std::move(tokens);
        }
    }
}

static std::string normalize_path(const std::string &s)
{
    std::string out = s;
    for (auto &c : out)
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

static bool wildcard_match(const std::string &pattern, const std::string &str)
{
    std::size_t p = 0;
    std::size_t s = 0;
    std::size_t star = std::string::npos;
    std::size_t match = 0;
    while (s < str.size())
    {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == str[s]))
        {
            ++p;
            ++s;
        }
        else if (p < pattern.size() && pattern[p] == '*')
        {
            star = p++;
            match = s;
        }
        else if (star != std::string::npos)
        {
            p = star + 1;
            s = ++match;
        }
        else
        {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*')
    {
        ++p;
    }
    return p == pattern.size();
}

std::vector<std::string> glob_files(const std::string &pattern)
{
    std::vector<std::string> out;
    std::string norm_pattern = normalize_path(pattern);
    auto first_wild = norm_pattern.find_first_of("*?");
    if (first_wild == std::string::npos)
    {
        if (std::filesystem::exists(pattern))
        {
            out.push_back(pattern);
        }
        return out;
    }
    auto last_sep = norm_pattern.substr(0, first_wild).find_last_of('/');
    std::filesystem::path base_dir = ".";
    if (last_sep != std::string::npos)
    {
        base_dir = pattern.substr(0, last_sep);
    }
    std::error_code ec;
    if (!std::filesystem::exists(base_dir, ec))
    {
        return out;
    }
    for (std::filesystem::recursive_directory_iterator it(base_dir, ec), end; it != end; it.increment(ec))
    {
        if (ec)
        {
            continue;
        }
        if (!it->is_regular_file())
        {
            continue;
        }
        std::string cand = normalize_path(it->path().string());
        if (wildcard_match(norm_pattern, cand))
        {
            out.push_back(it->path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static void append_utf8(uint32_t cp, std::string &out)
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

bool next_codepoint(const std::string &s, std::size_t &i, uint32_t &cp)
{
    if (i >= s.size())
    {
        return false;
    }
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80)
    {
        cp = c;
        i += 1;
        return true;
    }
    if ((c >> 5) == 0x6 && i + 1 < s.size())
    {
        cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        i += 2;
        return true;
    }
    if ((c >> 4) == 0xE && i + 2 < s.size())
    {
        cp = ((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        i += 3;
        return true;
    }
    if ((c >> 3) == 0x1E && i + 3 < s.size())
    {
        cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
             ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) | (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        i += 4;
        return true;
    }
    i += 1;
    cp = 0xFFFD;
    return true;
}

static bool is_space(uint32_t cp)
{
    if (cp <= 0x20)
    {
        return cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D || cp == 0x20;
    }
    return cp == 0x85 || cp == 0xA0 || cp == 0x1680 || cp == 0x180E || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

std::string truncate_utf8(const std::string &s, std::size_t max_chars)
{
    if (max_chars == 0)
    {
        return s;
    }
    std::size_t i = 0;
    std::size_t count = 0;
    while (i < s.size() && count < max_chars)
    {
        uint32_t cp = 0;
        std::size_t prev = i;
        if (!next_codepoint(s, i, cp))
        {
            break;
        }
        if (i == prev)
        {
            break;
        }
        ++count;
    }
    if (i >= s.size())
    {
        return s;
    }
    return s.substr(0, i);
}

std::vector<std::string> pretokenize(const std::string &text)
{
    std::vector<std::string> tokens;
    std::string pending_space;
    std::string current;
    std::size_t i = 0;
    while (i < text.size())
    {
        uint32_t cp = 0;
        std::size_t before = i;
        if (!next_codepoint(text, i, cp))
        {
            break;
        }
        if (i == before)
        {
            break;
        }
        std::string cp_utf8;
        append_utf8(cp, cp_utf8);
        if (is_space(cp))
        {
            if (!current.empty())
            {
                tokens.push_back(std::move(current));
                current.clear();
            }
            pending_space += cp_utf8;
        }
        else
        {
            if (current.empty())
            {
                current = pending_space;
                pending_space.clear();
            }
            current += cp_utf8;
        }
    }
    if (!current.empty())
    {
        tokens.push_back(std::move(current));
    }
    else if (!pending_space.empty())
    {
        tokens.push_back(std::move(pending_space));
    }
    return tokens;
}

std::array<uint32_t, 256> build_byte_to_unicode_cp()
{
    std::vector<int> bs;
    bs.reserve(256);
    std::array<bool, 256> present{};
    for (int b = 33; b <= 126; ++b)
    {
        bs.push_back(b);
        present[b] = true;
    }
    for (int b = 161; b <= 172; ++b)
    {
        bs.push_back(b);
        present[b] = true;
    }
    for (int b = 174; b <= 255; ++b)
    {
        bs.push_back(b);
        present[b] = true;
    }
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b)
    {
        if (!present[b])
        {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::array<uint32_t, 256> map{};
    for (std::size_t i = 0; i < bs.size(); ++i)
    {
        map[static_cast<std::size_t>(bs[i])] = static_cast<uint32_t>(cs[i]);
    }
    return map;
}

std::array<std::string, 256> build_byte_to_unicode_str(const std::array<uint32_t, 256> &cp_map)
{
    std::array<std::string, 256> out;
    for (std::size_t i = 0; i < 256; ++i)
    {
        std::string s;
        append_utf8(cp_map[i], s);
        out[i] = std::move(s);
    }
    return out;
}

std::string byte_level_encode(const std::string &token, const std::array<std::string, 256> &byte_to_unicode)
{
    std::string out;
    out.reserve(token.size() * 2);
    for (unsigned char b : token)
    {
        out += byte_to_unicode[b];
    }
    return out;
}

static bool parse_hex4(const std::string &s, std::size_t i, uint32_t &out)
{
    if (i + 4 > s.size())
    {
        return false;
    }
    uint32_t val = 0;
    for (std::size_t k = 0; k < 4; ++k)
    {
        char c = s[i + k];
        uint32_t v = 0;
        if (c >= '0' && c <= '9')
        {
            v = static_cast<uint32_t>(c - '0');
        }
        else if (c >= 'a' && c <= 'f')
        {
            v = static_cast<uint32_t>(10 + c - 'a');
        }
        else if (c >= 'A' && c <= 'F')
        {
            v = static_cast<uint32_t>(10 + c - 'A');
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
                uint32_t cp = 0;
                if (!parse_hex4(s, i, cp))
                {
                    return false;
                }
                i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF)
                {
                    if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u')
                    {
                        uint32_t low = 0;
                        if (parse_hex4(s, i + 2, low))
                        {
                            if (low >= 0xDC00 && low <= 0xDFFF)
                            {
                                i += 6;
                                cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                            }
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

static void skip_ws(const std::string &s, std::size_t &i)
{
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
    {
        ++i;
    }
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
    if (s[i] == '{' || s[i] == '[')
    {
        int depth = 0;
        while (i < s.size())
        {
            char c = s[i];
            if (c == '"')
            {
                std::string tmp;
                if (!parse_json_string(s, i, tmp))
                {
                    return false;
                }
                continue;
            }
            if (c == '{' || c == '[')
            {
                ++depth;
            }
            else if (c == '}' || c == ']')
            {
                --depth;
                if (depth == 0)
                {
                    ++i;
                    return true;
                }
            }
            ++i;
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

bool extract_json_field(const std::string &s, const std::string &field, std::string &out)
{
    std::size_t i = 0;
    while (i < s.size())
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ',')
        {
            ++i;
            continue;
        }
        if (s[i] != '"')
        {
            ++i;
            continue;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            continue;
        }
        ++i;
        if (key == field)
        {
            skip_ws(s, i);
            if (i >= s.size() || s[i] != '"')
            {
                return false;
            }
            return parse_json_string(s, i, out);
        }
        if (!skip_json_value(s, i))
        {
            return false;
        }
    }
    return false;
}

std::unordered_map<std::string, uint32_t> reduce_top_k(std::unordered_map<std::string, uint32_t> &local,
                                                       std::size_t top_k)
{
    std::vector<std::pair<std::string, uint32_t>> vec;
    vec.reserve(local.size());
    for (auto &kv : local)
    {
        vec.emplace_back(std::move(kv.first), kv.second);
    }
    local.clear();
    if (vec.size() > top_k)
    {
        auto nth = vec.begin() + static_cast<std::ptrdiff_t>(top_k);
        std::nth_element(vec.begin(), nth, vec.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
        vec.resize(top_k);
    }
    std::unordered_map<std::string, uint32_t> reduced;
    reduced.reserve(vec.size() * 13 / 10 + 8);
    for (auto &kv : vec)
    {
        reduced.emplace(std::move(kv.first), kv.second);
    }
    return reduced;
}

bool read_gz_lines(const std::string &path, const std::function<void(const std::string &)> &cb)
{
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f)
    {
        return false;
    }
    const int buf_size = 1 << 20;
    std::string buf(buf_size, '\0');
    std::string line;
    while (true)
    {
        char *res = gzgets(f, buf.data(), buf_size);
        if (!res)
        {
            break;
        }
        line.assign(res);
        while (!line.empty() && line.back() != '\n' && !gzeof(f))
        {
            res = gzgets(f, buf.data(), buf_size);
            if (!res)
            {
                break;
            }
            line.append(res);
        }
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        {
            line.pop_back();
        }
        cb(line);
    }
    gzclose(f);
    return true;
}

bool read_text_lines(const std::string &path, const std::function<void(const std::string &)> &cb)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }
    std::string line;
    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        cb(line);
    }
    return true;
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

bool write_vocab_json(const std::string &path, const std::vector<std::string> &id_to_token)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }
    out << "{";
    for (std::size_t i = 0; i < id_to_token.size(); ++i)
    {
        if (i > 0)
        {
            out << ",";
        }
        out << "\"" << json_escape(id_to_token[i]) << "\":" << i;
    }
    out << "}";
    return true;
}

bool write_merges_txt(const std::string &path, const std::vector<std::string> &merges_out)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }
    out << "#version: 0.2\n";
    for (const auto &m : merges_out)
    {
        out << m << "\n";
    }
    return true;
}

bool write_tokenizer_json(const std::string &path, const Config &cfg, const std::vector<std::string> &id_to_token,
                          const std::vector<std::string> &merges_out)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }
    out << "{";
    out << "\"version\":\"1.0\",";
    out << "\"truncation\":null,";
    out << "\"padding\":null,";

    out << "\"added_tokens\":[";
    for (std::size_t i = 0; i < cfg.special_tokens.size(); ++i)
    {
        if (i > 0)
        {
            out << ",";
        }
        out << "{";
        out << "\"id\":" << i << ",";
        out << "\"content\":\"" << json_escape(cfg.special_tokens[i]) << "\",";
        out << "\"single_word\":false,";
        out << "\"lstrip\":false,";
        out << "\"rstrip\":false,";
        out << "\"normalized\":false,";
        out << "\"special\":true";
        out << "}";
    }
    out << "],";

    out << "\"normalizer\":null,";
    out << "\"pre_tokenizer\":{";
    out << "\"type\":\"ByteLevel\",";
    out << "\"add_prefix_space\":false,";
    out << "\"trim_offsets\":true";
    out << "},";
    out << "\"post_processor\":null,";
    out << "\"decoder\":{";
    out << "\"type\":\"ByteLevel\",";
    out << "\"add_prefix_space\":false,";
    out << "\"trim_offsets\":true";
    out << "},";

    out << "\"model\":{";
    out << "\"type\":\"BPE\",";
    out << "\"dropout\":null,";
    out << "\"unk_token\":\"" << json_escape(cfg.unk_token) << "\",";
    out << "\"continuing_subword_prefix\":\"\",";
    out << "\"end_of_word_suffix\":\"\",";
    out << "\"fuse_unk\":false,";

    out << "\"vocab\":{";
    for (std::size_t i = 0; i < id_to_token.size(); ++i)
    {
        if (i > 0)
        {
            out << ",";
        }
        out << "\"" << json_escape(id_to_token[i]) << "\":" << i;
    }
    out << "},";

    out << "\"merges\":[";
    for (std::size_t i = 0; i < merges_out.size(); ++i)
    {
        if (i > 0)
        {
            out << ",";
        }
        out << "\"" << json_escape(merges_out[i]) << "\"";
    }
    out << "]";
    out << "}";
    out << "}";
    return true;
}

bool write_chunk_file(const std::string &path, const std::unordered_map<std::string, uint32_t> &counts,
                      uint64_t doc_count)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }
    ChunkHeader header;
    header.doc_count = doc_count;
    header.entry_count = counts.size();
    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    if (!out)
    {
        return false;
    }
    for (const auto &kv : counts)
    {
        uint32_t len = static_cast<uint32_t>(kv.first.size());
        out.write(reinterpret_cast<const char *>(&len), sizeof(len));
        out.write(kv.first.data(), len);
        uint32_t count = kv.second;
        out.write(reinterpret_cast<const char *>(&count), sizeof(count));
        if (!out)
        {
            return false;
        }
    }
    return true;
}

bool read_chunk_header(const std::string &path, ChunkHeader &header)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in)
    {
        return false;
    }
    if (header.magic != 0x314B4243 || header.version != 1)
    {
        return false;
    }
    return true;
}

bool merge_chunk_file(const std::string &path, std::unordered_map<std::string, uint64_t> &global_counts,
                      uint64_t *doc_count_out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }
    ChunkHeader header;
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!in)
    {
        return false;
    }
    if (header.magic != 0x314B4243 || header.version != 1)
    {
        return false;
    }
    if (doc_count_out)
    {
        *doc_count_out += header.doc_count;
    }
    for (uint64_t i = 0; i < header.entry_count; ++i)
    {
        uint32_t len = 0;
        in.read(reinterpret_cast<char *>(&len), sizeof(len));
        if (!in)
        {
            return false;
        }
        std::string key(len, '\0');
        if (len > 0)
        {
            in.read(&key[0], len);
            if (!in)
            {
                return false;
            }
        }
        uint32_t count = 0;
        in.read(reinterpret_cast<char *>(&count), sizeof(count));
        if (!in)
        {
            return false;
        }
        global_counts[key] += count;
    }
    return true;
}

static std::string format_duration(double seconds)
{
    int sec = static_cast<int>(seconds + 0.5);
    int h = sec / 3600;
    int m = (sec % 3600) / 60;
    int s = sec % 60;
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s;
    return oss.str();
}

ProgressTracker::ProgressTracker(uint64_t total_chunks, const std::string &label, uint64_t interval_ms)
    : label_(label), total_(total_chunks), interval_ms_(interval_ms)
{
    start_ = std::chrono::steady_clock::now();
    last_print_ = start_;
}

void ProgressTracker::add(uint64_t chunks, uint64_t docs)
{
    done_chunks_.fetch_add(chunks, std::memory_order_relaxed);
    done_docs_.fetch_add(docs, std::memory_order_relaxed);
    maybe_print(false);
}

void ProgressTracker::finish()
{
    maybe_print(true);
}

void ProgressTracker::maybe_print(bool force)
{
    auto now = std::chrono::steady_clock::now();
    if (!force)
    {
        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_).count();
        if (delta < static_cast<long long>(interval_ms_))
        {
            return;
        }
    }
    std::lock_guard<std::mutex> lock(print_mu_);
    now = std::chrono::steady_clock::now();
    if (!force)
    {
        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_).count();
        if (delta < static_cast<long long>(interval_ms_))
        {
            return;
        }
    }
    last_print_ = now;
    uint64_t done_chunks = done_chunks_.load(std::memory_order_relaxed);
    uint64_t done_docs = done_docs_.load(std::memory_order_relaxed);
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_).count();
    double pct = total_ ? (100.0 * static_cast<double>(done_chunks) / static_cast<double>(total_)) : 100.0;
    double chunk_rate = elapsed > 0.0 ? static_cast<double>(done_chunks) / elapsed : 0.0;
    double doc_rate = elapsed > 0.0 ? static_cast<double>(done_docs) / elapsed : 0.0;
    double eta =
        (chunk_rate > 0.0 && total_ > done_chunks) ? static_cast<double>(total_ - done_chunks) / chunk_rate : 0.0;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << "[" << label_ << "] chunks " << done_chunks << "/" << total_ << " (" << std::setprecision(1) << pct << "%)";
    if (done_docs > 0)
    {
        oss << " docs " << done_docs << " (" << std::setprecision(1) << (static_cast<double>(done_docs) / 1e6) << "M)";
    }
    if (chunk_rate > 0.0)
    {
        oss << " rate " << std::setprecision(2) << chunk_rate << " c/s";
    }
    if (doc_rate > 0.0)
    {
        oss << " " << std::setprecision(2) << (doc_rate / 1000.0) << " kdoc/s";
    }
    if (eta > 0.0)
    {
        oss << " ETA " << format_duration(eta);
    }
    oss << "\n";
    std::cerr << oss.str();
}
