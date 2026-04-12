// nb_classifier.cpp — C++ Naive Bayes classifier matching sklearn exactly.
#include "nb_classifier.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kmer {

static const std::string EMPTY_CLASS = "<unknown>";

const std::string& NbClassifier::class_name(int idx) const {
    if (idx >= 0 && idx < n_classes_) return classes_[idx];
    return EMPTY_CLASS;
}

// ─────────────────────────────────────────────────────────────
// MurmurHash3 — exact match to sklearn's murmurhash3_bytes_s32
// From: https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
// sklearn uses the 32-bit x86 variant with seed=0
// ─────────────────────────────────────────────────────────────
static inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

static inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

int32_t NbClassifier::murmurhash3(const char* data, int len, uint32_t seed) {
    const uint8_t* key = reinterpret_cast<const uint8_t*>(data);
    const int nblocks = len / 4;
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    // body
    const uint32_t* blocks = reinterpret_cast<const uint32_t*>(key);
    for (int i = 0; i < nblocks; i++) {
        uint32_t k1;
        std::memcpy(&k1, &blocks[i], sizeof(uint32_t));
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    // tail
    const uint8_t* tail = key + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16; [[fallthrough]];
        case 2: k1 ^= tail[1] << 8;  [[fallthrough]];
        case 1: k1 ^= tail[0];
                k1 *= c1;
                k1 = rotl32(k1, 15);
                k1 *= c2;
                h1 ^= k1;
    }

    // finalization
    h1 ^= static_cast<uint32_t>(len);
    h1 = fmix32(h1);

    return static_cast<int32_t>(h1);
}

// ─────────────────────────────────────────────────────────────
// HashingVectorizer: char_wb, ngram_range=(7,7), n_features=8192, l2 norm
// sklearn's char_wb adds " " at word boundaries before extracting n-grams.
// For DNA sequences (single "word"), this means: " ACGT... "
// Then extract all 7-char substrings, hash each, accumulate counts, L2 normalize.
// ─────────────────────────────────────────────────────────────
void NbClassifier::hash_sequence(std::string_view seq, std::vector<float>& features) const {
    features.assign(n_features_, 0.0f);

    // sklearn char_wb prepends and appends space
    std::string padded;
    padded.reserve(seq.size() + 2);
    padded += ' ';
    // sklearn lowercases
    for (char c : seq) {
        padded += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    padded += ' ';

    if (static_cast<int>(padded.size()) < ngram_size_) return;

    // Extract all n-grams and hash to features
    for (size_t i = 0; i + ngram_size_ <= padded.size(); ++i) {
        int32_t h = murmurhash3(padded.data() + i, ngram_size_, 0);
        int idx = static_cast<int>(static_cast<uint32_t>(h < 0 ? -h : h) % n_features_);
        features[idx] += 1.0f;
    }

    // L2 normalization (sklearn norm='l2')
    float norm = 0.0f;
    for (float v : features) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& v : features) v /= norm;
    }
}

// ─────────────────────────────────────────────────────────────
// Load model parameters from numpy files
// ─────────────────────────────────────────────────────────────

// Minimal .npy loader for float64 arrays
static std::vector<double> load_npy_f64(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open: " + path);
    }

    // .npy format: 6-byte magic + 2-byte version + 2-byte header_len + header + data
    char magic[6];
    in.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        throw std::runtime_error("Not a .npy file: " + path);
    }

    uint8_t major, minor;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl;
        in.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, '\0');
    in.read(header.data(), header_len);

    // Parse shape from header: "'shape': (N,)" or "'shape': (N, M),"
    size_t total = 1;
    auto shape_pos = header.find("'shape':");
    if (shape_pos != std::string::npos) {
        auto paren_open = header.find('(', shape_pos);
        auto paren_close = header.find(')', paren_open);
        std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);
        // Parse comma-separated dims
        std::istringstream ss(shape_str);
        std::string dim;
        while (std::getline(ss, dim, ',')) {
            dim.erase(0, dim.find_first_not_of(" "));
            dim.erase(dim.find_last_not_of(" ") + 1);
            if (!dim.empty()) {
                total *= std::stoull(dim);
            }
        }
    }

    // Read data (assuming float64 little-endian)
    std::vector<double> data(total);
    in.read(reinterpret_cast<char*>(data.data()), total * sizeof(double));
    return data;
}

void NbClassifier::load(const std::string& params_dir) {
    std::string dir = params_dir;
    if (dir.back() != '/' && dir.back() != '\\') dir += '/';

    // Load nb_params.json for dimensions
    {
        std::ifstream f(dir + "nb_params.json");
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open nb_params.json in " + params_dir);
        }
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        // Minimal JSON parse for n_classes and n_features
        auto parse_int = [&](const std::string& key) -> int {
            auto pos = content.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0;
            pos = content.find(':', pos);
            return std::stoi(content.substr(pos + 1));
        };
        n_classes_ = parse_int("n_classes");
        n_features_ = parse_int("n_features");
    }

    std::cout << "Loading NB model: " << n_classes_ << " classes x "
              << n_features_ << " features" << std::endl;

    // Load class_log_prior (double .npy → float)
    {
        auto tmp = load_npy_f64(dir + "class_log_prior.npy");
        class_log_prior_.assign(tmp.begin(), tmp.end());
    }
    assert(static_cast<int>(class_log_prior_.size()) == n_classes_);

    // Load feature_log_prob (double .npy → float, halves memory: 4.9GB → 2.45GB)
    std::cout << "Loading feature_log_prob matrix ("
              << (static_cast<size_t>(n_classes_) * n_features_ * 4 / (1024*1024))
              << " MB as float32)..." << std::endl;
    {
        auto tmp = load_npy_f64(dir + "feature_log_prob.npy");
        feature_log_prob_.assign(tmp.begin(), tmp.end());
    }
    assert(static_cast<int>(feature_log_prob_.size()) == n_classes_ * n_features_);

    // Load class names
    classes_.clear();
    classes_.reserve(n_classes_);
    {
        std::ifstream f(dir + "classes.txt");
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            classes_.push_back(std::move(line));
        }
    }
    assert(static_cast<int>(classes_.size()) == n_classes_);

    // Precompute rank prefixes for hierarchical confidence
    precompute_rank_prefixes();

    std::cout << "NB model loaded: " << n_classes_ << " classes, "
              << n_features_ << " features, "
              << n_unique_prefixes_ << " unique rank prefixes" << std::endl;
}

void NbClassifier::precompute_rank_prefixes() {
    // For each class, extract taxonomy prefix at each of 7 ranks.
    // "k__Bacteria; p__Firmicutes; c__Bacilli; ..." at rank 2 → "k__Bacteria; p__Firmicutes; c__Bacilli"
    // This allows fast grouping during confidence computation.
    std::unordered_map<std::string, int32_t> prefix_to_id;
    rank_prefix_strings_.clear();
    class_rank_prefix_ids_.resize(static_cast<size_t>(n_classes_) * 7, -1);

    for (int c = 0; c < n_classes_; ++c) {
        const std::string& tax = classes_[c];
        // Split by ";"
        std::vector<size_t> semicolons;
        for (size_t i = 0; i < tax.size(); ++i) {
            if (tax[i] == ';') semicolons.push_back(i);
        }

        // rank 0 = first field (kingdom), rank 6 = 7th field (species)
        // rank k needs prefix covering first (k+1) fields = substr up to semicolons[k]
        for (int rank = 0; rank < 7; ++rank) {
            std::string prefix;
            if (rank == 0) {
                // First field only (before first semicolon)
                prefix = semicolons.empty() ? tax : tax.substr(0, semicolons[0]);
            } else if (rank < static_cast<int>(semicolons.size())) {
                // First (rank+1) fields = up to semicolons[rank]
                prefix = tax.substr(0, semicolons[rank]);
            } else {
                // Not enough ranks — use full string
                prefix = tax;
            }

            auto it = prefix_to_id.find(prefix);
            int32_t pid;
            if (it != prefix_to_id.end()) {
                pid = it->second;
            } else {
                pid = static_cast<int32_t>(rank_prefix_strings_.size());
                prefix_to_id[prefix] = pid;
                rank_prefix_strings_.push_back(prefix);
            }
            class_rank_prefix_ids_[static_cast<size_t>(c) * 7 + rank] = pid;
        }
    }
    n_unique_prefixes_ = static_cast<int>(rank_prefix_strings_.size());
}

// ─────────────────────────────────────────────────────────────
// Prediction
// ─────────────────────────────────────────────────────────────
NbResult NbClassifier::classify_one(std::string_view sequence, double confidence_threshold) const {
    NbResult result;
    if (!is_loaded() || sequence.empty()) return result;

    // Step 1: Hash sequence to dense feature vector (L2-normalized, float)
    std::vector<float> features;
    hash_sequence(sequence, features);

    // Step 2: Dense matrix-vector multiply (float): scores = FLP @ features + prior
    int best_class = 0;
    float best_score = -1e30f;
    std::vector<float> scores(n_classes_);

    for (int c = 0; c < n_classes_; ++c) {
        const float* row = feature_log_prob_.data() + static_cast<size_t>(c) * n_features_;
        float score = class_log_prior_[c];
        for (int f = 0; f < n_features_; ++f) {
            score += row[f] * features[f];
        }
        scores[c] = score;
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }

    // Step 3-4: softmax (in double for stability) + hierarchical confidence truncation
    return truncate_confidence(scores, best_class, best_score, confidence_threshold);
}

// ─────────────────────────────────────────────────────────────
// Shared: softmax + hierarchical confidence truncation
// ─────────────────────────────────────────────────────────────
NbResult NbClassifier::truncate_confidence(const std::vector<float>& scores,
                                            int best_class, float best_score,
                                            double confidence_threshold) const {
    NbResult result;

    // Softmax (double precision for numerical stability)
    std::vector<double> probs(n_classes_);
    double sum_exp = 0.0;
    for (int c = 0; c < n_classes_; ++c) {
        probs[c] = std::exp(static_cast<double>(scores[c] - best_score));
        sum_exp += probs[c];
    }
    for (int c = 0; c < n_classes_; ++c) {
        probs[c] /= sum_exp;
    }

    // Hierarchical confidence truncation (same as QIIME2)
    int deepest_confident_rank = -1;
    double deepest_confidence = 0.0;
    std::string deepest_taxonomy;

    std::unordered_map<int32_t, double> prefix_prob;
    prefix_prob.reserve(1024);

    for (int rank = 6; rank >= 0; --rank) {
        prefix_prob.clear();

        for (int c = 0; c < n_classes_; ++c) {
            if (probs[c] < 1e-15) continue;
            int32_t pid = class_rank_prefix_ids_[static_cast<size_t>(c) * 7 + rank];
            prefix_prob[pid] += probs[c];
        }

        double max_prob = 0.0;
        int32_t best_pid = -1;
        for (auto& [pid, p] : prefix_prob) {
            if (p > max_prob) {
                max_prob = p;
                best_pid = pid;
            }
        }

        if (max_prob >= confidence_threshold) {
            deepest_confident_rank = rank;
            deepest_confidence = max_prob;
            deepest_taxonomy = rank_prefix_strings_[best_pid];
            break;
        }
    }

    if (deepest_confident_rank >= 0) {
        result.taxonomy = deepest_taxonomy;
        result.confidence = deepest_confidence;
        result.deepest_rank = deepest_confident_rank;
    } else {
        result.taxonomy = "Unassigned";
        result.confidence = 0.0;
        result.deepest_rank = -1;
    }
    result.class_idx = best_class;
    result.log_posterior = best_score;
    return result;
}

// ─────────────────────────────────────────────────────────────
// Batch classify: single pass through feature_log_prob matrix
// ─────────────────────────────────────────────────────────────
std::vector<NbResult> NbClassifier::classify(
    const std::vector<std::pair<std::string, std::string>>& id_seq_pairs,
    const NbConfig& config
) const {
    const size_t N = id_seq_pairs.size();
    std::vector<NbResult> results(N);
    if (N == 0 || !is_loaded()) return results;

    #ifdef _OPENMP
    omp_set_num_threads(config.num_threads);
    #endif

    // Step 1: Hash ALL sequences → transposed feature matrix [n_features_ x N] (float)
    // Each column (one feature across all seqs) is N*4 bytes = 1.2KB for 300 seqs → L1 cache.
    std::vector<float> feat_T(static_cast<size_t>(n_features_) * N, 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < N; ++i) {
        std::vector<float> feat;
        hash_sequence(id_seq_pairs[i].second, feat);
        for (int f = 0; f < n_features_; ++f) {
            feat_T[static_cast<size_t>(f) * N + i] = feat[f];
        }
    }

    // Step 2: GEMM-style batch multiply (float32).
    // Reads 2.45GB matrix ONCE. Innermost loop: pure FMA on float (AVX2: 8 floats/cycle).
    std::vector<float> all_scores(static_cast<size_t>(n_classes_) * N);

    // Initialize with class_log_prior (broadcast to all sequences)
    for (int c = 0; c < n_classes_; ++c) {
        float prior = class_log_prior_[c];
        float* score_row = all_scores.data() + static_cast<size_t>(c) * N;
        for (size_t i = 0; i < N; ++i) {
            score_row[i] = prior;
        }
    }

    // Main GEMM loop — parallelized over classes
    #pragma omp parallel for schedule(dynamic, 256)
    for (int c = 0; c < n_classes_; ++c) {
        const float* flp_row = feature_log_prob_.data() + static_cast<size_t>(c) * n_features_;
        float* score_row = all_scores.data() + static_cast<size_t>(c) * N;
        for (int f = 0; f < n_features_; ++f) {
            float val = flp_row[f];
            if (val == 0.0f) continue;
            const float* feat_col = feat_T.data() + static_cast<size_t>(f) * N;
            for (size_t i = 0; i < N; ++i) {
                score_row[i] += val * feat_col[i];
            }
        }
    }

    // Find best class per sequence
    std::vector<int> best_classes(N, 0);
    std::vector<float> best_scores(N, -1e30f);
    for (int c = 0; c < n_classes_; ++c) {
        const float* score_row = all_scores.data() + static_cast<size_t>(c) * N;
        for (size_t i = 0; i < N; ++i) {
            if (score_row[i] > best_scores[i]) {
                best_scores[i] = score_row[i];
                best_classes[i] = c;
            }
        }
    }

    // Step 3: Softmax (double for stability) + hierarchical confidence truncation
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < N; ++i) {
        std::vector<float> scores(n_classes_);
        for (int c = 0; c < n_classes_; ++c) {
            scores[c] = all_scores[static_cast<size_t>(c) * N + i];
        }
        results[i] = truncate_confidence(scores, best_classes[i], best_scores[i],
                                          config.confidence_threshold);
    }

    return results;
}

}  // namespace kmer
