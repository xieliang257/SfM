#include "Vocabulary/Vocabulary.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>

namespace sfm {

// ─── helpers ────────────────────────────────────────────────────────────
static inline uint64_t bitMajority(const std::vector<uint64_t>& descs) {
	// For each bit position, count the number of 1-bits.
	// Store in 64 counters packed into two uint64_t accumulators.
	uint64_t posMask = 0, negMask = 0;
	// Use an array of 64 counters (simpler, fast enough).
	int cnt[64] = {};
	const size_t n = descs.size();
	for (uint64_t d : descs) {
		for (int b = 0; b < 64; ++b) {
			cnt[b] += (d >> b) & 1;
		}
	}
	uint64_t result = 0;
	const int half = static_cast<int>(n) / 2;
	for (int b = 0; b < 64; ++b) {
		if (cnt[b] > half) {
			result |= (1ULL << b);
		}
	}
	return result;
}

// Find the descriptor in `pool` closest to `center` by Hamming distance.
// Returns the index within `pool`.
static size_t nearestDesc(const std::vector<uint64_t>& pool, uint64_t center) {
	size_t best = 0;
	int bestDist = Vocabulary::hammingDist(pool[0], center);
	for (size_t i = 1; i < pool.size(); ++i) {
		int d = Vocabulary::hammingDist(pool[i], center);
		if (d < bestDist) {
			bestDist = d;
			best = i;
		}
	}
	return best;
}

// ─── k-means++ init (Hamming) ──────────────────────────────────────────
static void kmeansPPInit(const std::vector<uint64_t>& descs,
                         std::vector<uint64_t>& centroids, int k,
                         std::mt19937& rng) {
	const int n = static_cast<int>(descs.size());
	centroids.clear();
	centroids.reserve(k);

	// First centroid: random
	centroids.push_back(descs[std::uniform_int_distribution<int>(0, n - 1)(rng)]);

	std::vector<int> minDist(n, std::numeric_limits<int>::max());

	for (int c = 1; c < k; ++c) {
		// Update minDist to the nearest existing centroid.
		const uint64_t newCent = centroids.back();
		for (int i = 0; i < n; ++i) {
			int d = Vocabulary::hammingDist(descs[i], newCent);
			if (d < minDist[i]) minDist[i] = d;
		}

		// Compute total weight = sum of minDist^2.
		double totalWeight = 0.0;
		for (int i = 0; i < n; ++i) {
			totalWeight += static_cast<double>(minDist[i]) * minDist[i];
		}
		if (totalWeight < 1e-12) {
			// All remaining points are identical to existing centroids.
			centroids.push_back(descs[std::uniform_int_distribution<int>(0, n - 1)(rng)]);
			continue;
		}

		// Roulette wheel selection.
		double r = std::uniform_real_distribution<double>(0.0, totalWeight)(rng);
		double acc = 0.0;
		int chosen = n - 1;
		for (int i = 0; i < n; ++i) {
			acc += static_cast<double>(minDist[i]) * minDist[i];
			if (acc >= r) {
				chosen = i;
				break;
			}
		}
		centroids.push_back(descs[chosen]);
	}
}

// ─── single-level k-means (Hamming) ────────────────────────────────────
// Returns cluster assignments (one per descriptor).
static std::vector<int>
kmeansHamming(const std::vector<uint64_t>& descs, int k, int maxIter,
              std::vector<uint64_t>& outCentroids) {
	const int n = static_cast<int>(descs.size());
	assert(n >= k);

	std::mt19937 rng(42);
	kmeansPPInit(descs, outCentroids, k, rng);

	std::vector<int> assignment(n, 0);
	bool changed = true;
	int iter = 0;

	while (changed && iter < maxIter) {
		changed = false;
		++iter;

		// Assign each point to nearest centroid.
		for (int i = 0; i < n; ++i) {
			int bestC = 0;
			int bestD = Vocabulary::hammingDist(descs[i], outCentroids[0]);
			for (int c = 1; c < k; ++c) {
				int d = Vocabulary::hammingDist(descs[i], outCentroids[c]);
				if (d < bestD) {
					bestD = d;
					bestC = c;
				}
			}
			if (assignment[i] != bestC) {
				assignment[i] = bestC;
				changed = true;
			}
		}

		// Update centroids: bit-majority → nearest real descriptor.
		for (int c = 0; c < k; ++c) {
			std::vector<uint64_t> cluster;
			cluster.reserve(n / k + 1);
			for (int i = 0; i < n; ++i) {
				if (assignment[i] == c) {
					cluster.push_back(descs[i]);
				}
			}
			if (cluster.empty()) {
				// Reinitialize empty cluster to a random point.
				outCentroids[c] = descs[rng() % n];
				continue;
			}
			uint64_t maj = bitMajority(cluster);
			size_t idx = nearestDesc(cluster, maj);
			outCentroids[c] = cluster[idx];
		}
	}

	return assignment;
}

// ─── build ──────────────────────────────────────────────────────────────
void Vocabulary::build(const std::vector<uint64_t>& allDescriptors,
                       const std::vector<int>& descPerFrame,
                       int k, int L, int maxIter) {
	k_ = k;
	L_ = L;
	numWords_ = static_cast<int>(std::pow(k, L));

	// Total nodes in a complete k-ary tree of depth L.
	int totalNodes = 0;
	{
		long long kn = 1;
		for (int i = 0; i <= L; ++i) {
			totalNodes += static_cast<int>(kn);
			kn *= k;
		}
	}

	nodes_.clear();
	nodes_.resize(totalNodes);
	// Root (node 0) holds all descriptors initially.
	// nodes_[i]'s children are at i*k+1 … i*k+k.

	// BFS queue: list of (nodeIndex, vector of descriptor indices in this node).
	struct Task {
		int nodeIdx;
		int depth;
		std::vector<int> descs; // indices into allDescriptors
	};

	Task root;
	root.nodeIdx = 0;
	root.depth = 0;
	root.descs.resize(allDescriptors.size());
	std::iota(root.descs.begin(), root.descs.end(), 0);

	// Set root center to global bit-majority.
	{
		std::vector<uint64_t> allDescVals;
		allDescVals.reserve(allDescriptors.size());
		for (int idx : root.descs) {
			allDescVals.push_back(allDescriptors[idx]);
		}
		uint64_t maj = bitMajority(allDescVals);
		size_t nearest = nearestDesc(allDescVals, maj);
		nodes_[0].center = allDescVals[nearest];
	}

	std::vector<Task> queue;
	queue.push_back(std::move(root));

	int nodeCounter = 1; // next node index to assign (root=0 already)

	while (!queue.empty()) {
		Task cur = std::move(queue.back());
		queue.pop_back();

		int nodeIdx = cur.nodeIdx;
		int depth = cur.depth;
		int nDescs = static_cast<int>(cur.descs.size());

		if (depth >= L || nDescs < k * 2) {
			// Leaf node – store weight placeholder (computed later).
			continue;
		}

		// Extract descriptors for this node.
		std::vector<uint64_t> nodeDescs(nDescs);
		for (int i = 0; i < nDescs; ++i) {
			nodeDescs[i] = allDescriptors[cur.descs[i]];
		}

		// Run k-means.
		std::vector<uint64_t> centroids;
		std::vector<int> assignment =
		    kmeansHamming(nodeDescs, k, maxIter, centroids);

		// Create k child nodes and bucket descriptors.
		int childBase = nodeIdx * k + 1;
		std::vector<std::vector<int>> buckets(k);
		for (int i = 0; i < nDescs; ++i) {
			buckets[assignment[i]].push_back(cur.descs[i]);
		}

		for (int c = 0; c < k; ++c) {
			int childIdx = childBase + c;
			if (childIdx >= totalNodes) break;
			nodes_[childIdx].center = centroids[c];

			if (depth + 1 < L && static_cast<int>(buckets[c].size()) >= k * 2) {
				Task t;
				t.nodeIdx = childIdx;
				t.depth = depth + 1;
				t.descs = std::move(buckets[c]);
				queue.push_back(std::move(t));
			}
		}
	}

	// ─── compute IDF weights ──────────────────────────────────────────
	// For each frame, quantize all its descriptors and record which leaves
	// are hit.  IDF_i = log(N / n_i) where n_i = number of frames hitting
	// leaf i.
	const int N = static_cast<int>(descPerFrame.size());
	std::vector<int> docFreq(numWords_, 0);
	std::unordered_map<int, int> leafHit;

	int offset = 0;
	for (int f = 0; f < N; ++f) {
		leafHit.clear();
		for (int j = 0; j < descPerFrame[f]; ++j) {
			int leaf = quantize(allDescriptors[offset + j]);
			leafHit[leaf] = 1;
		}
		for (auto& kv : leafHit) {
			docFreq[kv.first]++;
		}
		offset += descPerFrame[f];
	}

	for (int i = 0; i < numWords_; ++i) {
		int leafIdx = static_cast<int>(nodes_.size()) - numWords_ + i;
		if (leafIdx < 0 || leafIdx >= static_cast<int>(nodes_.size())) continue;
		if (docFreq[i] > 0) {
			nodes_[leafIdx].weight = std::log(static_cast<double>(N) / docFreq[i]);
		}
	}

	std::cout << "Vocabulary built: k=" << k_ << " L=" << L_
	          << " nodes=" << nodes_.size()
	          << " words=" << numWords_ << "\n";
}

// ─── quantize ───────────────────────────────────────────────────────────
int Vocabulary::quantize(uint64_t desc) const {
	if (nodes_.empty()) return 0;
	int idx = 0;
	for (int d = 0; d < L_; ++d) {
		int childBase = idx * k_ + 1;
		int bestChild = 0;
		int bestDist = hammingDist(desc, nodes_[childBase].center);
		for (int c = 1; c < k_; ++c) {
			int childIdx = childBase + c;
			if (childIdx >= static_cast<int>(nodes_.size())) break;
			int dist = hammingDist(desc, nodes_[childIdx].center);
			if (dist < bestDist) {
				bestDist = dist;
				bestChild = c;
			}
		}
		idx = childBase + bestChild;
		if (idx >= static_cast<int>(nodes_.size())) break;
	}
	return idx;
}

// ─── transform (BoW) ───────────────────────────────────────────────────
std::vector<std::pair<int, double>>
Vocabulary::transform(const std::vector<uint64_t>& descriptors) const {
	// Count occurrences of each leaf word.
	std::unordered_map<int, int> hist;
	for (uint64_t d : descriptors) {
		hist[quantize(d)]++;
	}

	// Build sparse TF-IDF vector.
	std::vector<std::pair<int, double>> bow;
	bow.reserve(hist.size());
	const double invN = 1.0 / std::max(1, static_cast<int>(descriptors.size()));
	for (auto& kv : hist) {
		int leafIdx = kv.first;
		double tf = kv.second * invN;
		double idf = 0.0;
		if (leafIdx >= 0 && leafIdx < static_cast<int>(nodes_.size())) {
			idf = nodes_[leafIdx].weight;
		}
		bow.emplace_back(leafIdx, tf * idf);
	}
	return bow;
}

// ─── save / load ────────────────────────────────────────────────────────
void Vocabulary::save(const std::string& path) const {
	std::ofstream f(path, std::ios::binary);
	if (!f.is_open()) {
		std::cerr << "Cannot open " << path << " for writing\n";
		return;
	}
	// Header: k, L, numNodes
	int32_t header[3] = { k_, L_, static_cast<int32_t>(nodes_.size()) };
	f.write(reinterpret_cast<const char*>(header), sizeof(header));

	// Node data: center (uint64) + weight (double) per node.
	for (const auto& n : nodes_) {
		uint64_t c = n.center;
		double w = n.weight;
		f.write(reinterpret_cast<const char*>(&c), 8);
		f.write(reinterpret_cast<const char*>(&w), 8);
	}

	std::cout << "Vocabulary saved to " << path
	          << " (" << nodes_.size() << " nodes)\n";
}

bool Vocabulary::load(const std::string& path) {
	std::ifstream f(path, std::ios::binary);
	if (!f.is_open()) {
		std::cerr << "Cannot open " << path << "\n";
		return false;
	}
	int32_t header[3];
	f.read(reinterpret_cast<char*>(header), sizeof(header));
	k_ = header[0];
	L_ = header[1];
	int nNodes = header[2];
	numWords_ = static_cast<int>(std::pow(k_, L_));

	nodes_.resize(nNodes);
	for (auto& n : nodes_) {
		uint64_t c;
		double w;
		f.read(reinterpret_cast<char*>(&c), 8);
		f.read(reinterpret_cast<char*>(&w), 8);
		n.center = c;
		n.weight = w;
	}

	std::cout << "Vocabulary loaded from " << path
	          << " (k=" << k_ << " L=" << L_ << " nodes=" << nNodes << ")\n";
	return true;
}

} // namespace sfm
