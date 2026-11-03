"""
Image Sequencing and Overlap Graph Module
=========================================
Builds an image connectivity graph from pairwise inlier counts and determines
the optimal spatial stitching order (e.g., Left-to-Right panoramic chain)
even when images are uploaded in random or shuffled order.
"""

from typing import List, Tuple, Dict, Set
import numpy as np


class ImageSequencer:
    """Reconstructs spatial panoramic order from pairwise match topology."""

    def __init__(self, num_images: int):
        self.num_images = num_images
        # Map: (min(i,j), max(i,j)) -> inlier_count
        self.edge_weights: Dict[Tuple[int, int], int] = {}

    def add_edge(self, img_i: int, img_j: int, inliers: int) -> None:
        """Record pairwise verified match count."""
        key = (min(img_i, img_j), max(img_i, img_j))
        self.edge_weights[key] = inliers

    def get_weight(self, img_i: int, img_j: int) -> int:
        key = (min(img_i, img_j), max(img_i, img_j))
        return self.edge_weights.get(key, 0)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return N x N symmetric matrix of inlier counts."""
        mat = np.zeros((self.num_images, self.num_images), dtype=int)
        for (i, j), count in self.edge_weights.items():
            mat[i, j] = count
            mat[j, i] = count
        return mat

    def determine_stitching_order(self) -> List[int]:
        """
        Greedy bidirectional graph walk:
        1. Find pair with the absolute highest verified inlier count (seed).
        2. Iteratively attach the best available candidate to either the left
           or right end of the current ordered sequence.
        3. Append any disconnected islands at the end.
        """
        n = self.num_images
        if n <= 1:
            return list(range(n))

        # Find best starting seed pair
        best_pair = (0, 1)
        max_inliers = -1

        for i in range(n):
            for j in range(i + 1, n):
                w = self.get_weight(i, j)
                if w > max_inliers:
                    max_inliers = w
                    best_pair = (i, j)

        # Initialize sequence with best pair
        chain: List[int] = list(best_pair)
        remaining: Set[int] = set(range(n)) - set(chain)

        # Grow chain bidirectionally
        while remaining:
            left_head = chain[0]
            right_tail = chain[-1]

            best_left_candidate = None
            best_left_weight = -1

            best_right_candidate = None
            best_right_weight = -1

            for cand in remaining:
                w_left = self.get_weight(left_head, cand)
                if w_left > best_left_weight:
                    best_left_weight = w_left
                    best_left_candidate = cand

                w_right = self.get_weight(right_tail, cand)
                if w_right > best_right_weight:
                    best_right_weight = w_right
                    best_right_candidate = cand

            # Require minimum connectivity
            if max(best_left_weight, best_right_weight) <= 0:
                # No more connected nodes, append remaining in index order
                chain.extend(sorted(list(remaining)))
                break

            if best_right_weight >= best_left_weight and best_right_candidate is not None:
                chain.append(best_right_candidate)
                remaining.remove(best_right_candidate)
            elif best_left_candidate is not None:
                chain.insert(0, best_left_candidate)
                remaining.remove(best_left_candidate)
            else:
                chain.extend(sorted(list(remaining)))
                break

        return chain
