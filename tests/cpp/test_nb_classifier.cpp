#include "nb_classifier.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace kmer;

// Test model loading with missing directory
void test_load_missing_dir() {
    NbClassifier nb;
    bool threw = false;
    try {
        nb.load("/nonexistent/path");
    } catch (const std::exception& e) {
        threw = true;
    }
    assert(threw && "load() should throw for missing directory");
    printf("  test_load_missing_dir: PASS\n");
}

// Test empty sequence batch
void test_classify_empty() {
    NbClassifier nb;
    NbConfig conf;
    std::vector<std::pair<std::string, std::string>> empty_seqs;
    auto results = nb.classify(empty_seqs, conf);
    assert(results.empty());
    printf("  test_classify_empty: PASS\n");
}

// Test NbConfig defaults
void test_config_defaults() {
    NbConfig conf;
    assert(conf.num_threads == 1);
    assert(std::abs(conf.confidence_threshold - 0.7) < 1e-6);
    printf("  test_config_defaults: PASS\n");
}

// Test NbResult fields
void test_result_fields() {
    NbResult r;
    r.taxonomy = "k__Bacteria";
    r.confidence = 0.95;
    assert(r.taxonomy == "k__Bacteria");
    assert(std::abs(r.confidence - 0.95) < 1e-6);
    printf("  test_result_fields: PASS\n");
}

int main() {
    printf("=== fastnb C++ unit tests ===\n");
    test_load_missing_dir();
    test_classify_empty();
    test_config_defaults();
    test_result_fields();
    printf("=== All tests passed ===\n");
    return 0;
}
