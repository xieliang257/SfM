#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <random>

namespace sfm {

/// Hierarchical k-means vocabulary for 64-bit binary descriptors.
///
/// Builds a tree of depth L with branching factor k.  Leaf nodes are
/// visual words; each stores an IDF weight for BoW weighting.
///
/// Tree layout (flat array, breadth-first):
///   node 0 = root
///   node i's children are at  i*k+1  …  i*k+k
///   total nodes = (k^(L+1) − 1) / (k − 1)
class Vocabulary {
public:
	struct Node {
		uint64_t center = 0;
		double weight  = 0.0;   // IDF weight (0 for internal nodes)
	};

	Vocabulary() = default;

	/// Build the vocabulary.
	/// @param allDescriptors  Concatenated descriptors from all training frames.
	/// @param descPerFrame    Number of descriptors contributed by each frame.
	/// @param k               Branching factor (children per node).
	/// @param L               Tree depth (levels below root).
	/// @param maxIter         Max k-means iterations per level.
	void build(const std::vector<uint64_t>& allDescriptors,
	           const std::vector<int>& descPerFrame,
	           int k = 10, int L = 5, int maxIter = 20);

	/// Quantize a single descriptor to its leaf index.
	int quantize(uint64_t desc) const;

	/// Quantize all descriptors of one frame to a BoW histogram.
	/// Returns (leaf_index, tfidf_weight) pairs (sparse representation).
	std::vector<std::pair<int, double>>
	transform(const std::vector<uint64_t>& descriptors) const;

	void save(const std::string& path) const;
	bool load(const std::string& path);

	int k() const { return k_; }
	int L() const { return L_; }
	int numWords() const { return numWords_; }
	int numNodes() const { return static_cast<int>(nodes_.size()); }

	static int hammingDist(uint64_t a, uint64_t b) {
		return __builtin_popcountll(a ^ b);
	}

private:
	std::vector<Node> nodes_;
	int k_ = 0;
	int L_ = 0;
	int numWords_ = 0;
};

} // namespace sfm
