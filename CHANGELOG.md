# Changelog

## [0.3.0] - 2025-09-23
- BREAKING: parallelize build with Rayon
- BREAKING: add support for generic trait distance via anndists
- More examples and benchmarks using MNIST and SIFT datasets
- Fix bug that leads to low recall

## [0.2.0] - 2025-01-03
- BREAKING: Complete rewrite using proper Vamana graph algorithm
- BREAKING: Renamed SingleFileDiskANN to DiskANN
- Fixed critical memory efficiency issue (now uses 16% of file size instead of 100%)
- Implemented proper beam search with medoid entry points
- Added alpha-pruning for diverse neighbor selection in graph construction
- Achieved 6-10x memory efficiency improvement
- Fixed search to use graph traversal instead of linear scanning
- Updated examples to use correct API with user-provided vectors

## [0.1.0] - 2024-12-28
- Initial release
- Single-file storage implementation
- Support for Euclidean and Cosine distance
- Three-layer search structure
