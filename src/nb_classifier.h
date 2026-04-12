#pragma once
// nb_classifier.h — Naive Bayes classifier matching sklearn's
// HashingVectorizer(analyzer='char_wb', ngram_range=(7,7), n_features=8192)
// + MultinomialNB(alpha=0.001)
//
// Loads pre-trained parameters exported from QIIME2 sklearn classifier.
// Prediction is identical to sklearn — same hashing, same math.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

namespace kmer {

// Result of NB classification for one sequence
struct NbResult {
    int class_idx = -1;         // Index into classes array
    std::string taxonomy;       // Taxonomy truncated at confident rank
    double confidence = 0.0;    // Confidence at the deepest confident rank
    double log_posterior = 0.0; // Raw log-posterior score
    int deepest_rank = -1;      // 0=kingdom .. 6=species, -1=unclassified
};

// Configuration for NB classifier
struct NbConfig {
    int num_threads = 1;
    double confidence_threshold = 0.7;  // QIIME2 default
};

// The Naive Bayes classifier
class NbClassifier {
public:
    NbClassifier() = default;

    // Load pre-trained model parameters from directory containing:
    //   class_log_prior.npy, feature_log_prob.npy, classes.txt, nb_params.json
    void load(const std::string& params_dir);

    // Classify a single DNA sequence (with hierarchical confidence truncation)
    NbResult classify_one(std::string_view sequence, double confidence_threshold = 0.7) const;

    // Classify a batch of sequences (parallel with OpenMP)
    std::vector<NbResult> classify(
        const std::vector<std::pair<std::string, std::string>>& id_seq_pairs,
        const NbConfig& config = {}
    ) const;

    // Accessors
    int n_classes() const { return n_classes_; }
    int n_features() const { return n_features_; }
    bool is_loaded() const { return !class_log_prior_.empty(); }

    // Get taxonomy string for a class index
    const std::string& class_name(int idx) const;

private:
    int n_classes_ = 0;
    int n_features_ = 0;
    int ngram_size_ = 7;

    // Model parameters (loaded from .npy files)
    // GEMM uses float for 2x bandwidth + 2x SIMD width (AVX2: 8 floats vs 4 doubles)
    std::vector<float> class_log_prior_;            // [n_classes]
    std::vector<float> feature_log_prob_;            // [n_classes * n_features] row-major
    std::vector<std::string> classes_;               // [n_classes] taxonomy strings

    // Precomputed: for each class, taxonomy prefix at each rank (7 levels)
    // class_rank_prefixes_[c * 7 + rank] = index into rank_prefix_strings_
    std::vector<int32_t> class_rank_prefix_ids_;  // [n_classes * 7]
    std::vector<std::string> rank_prefix_strings_; // unique prefix strings
    int n_unique_prefixes_ = 0;

    void precompute_rank_prefixes();

    // Feature extraction: HashingVectorizer(char_wb, ngram_range=(7,7), n_features=8192)
    void hash_sequence(std::string_view seq, std::vector<float>& features) const;

    // Hierarchical confidence truncation on precomputed scores
    NbResult truncate_confidence(const std::vector<float>& scores,
                                 int best_class, float best_score,
                                 double confidence_threshold) const;

    // MurmurHash3 32-bit (same as sklearn's murmurhash3_bytes_s32)
    static int32_t murmurhash3(const char* data, int len, uint32_t seed = 0);
};

}  // namespace kmer
