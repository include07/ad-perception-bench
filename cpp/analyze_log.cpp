// Pure-STL analyzer for the CARLA campaign logs (no external dependencies).
//
// Parses runs/carla_log_<weather>.csv files, aggregates detections per
// weather and per class, and reports the degradation vs the best scenario.
//
// Build:  cmake -S cpp -B cpp/build && cmake --build cpp/build
// Run:    ./cpp/build/analyze_log runs/carla_log_*.csv

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct WeatherStats {
  std::string weather;
  int frames = 0;
  int detections = 0;
  std::unordered_map<std::string, int> per_class;
};

std::vector<std::string> split(const std::string& line, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, sep)) out.push_back(item);
  return out;
}

WeatherStats parse_file(const std::string& path) {
  WeatherStats stats;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);

  std::string line;
  std::getline(in, line);  // header: frame,n_detections,classes,weather
  while (std::getline(in, line)) {
    const auto cols = split(line, ',');
    if (cols.size() < 4) continue;
    stats.frames += 1;
    stats.detections += std::stoi(cols[1]);
    stats.weather = cols[3];
    for (const auto& cls : split(cols[2], '|')) {
      if (!cls.empty()) stats.per_class[cls] += 1;
    }
  }
  return stats;
}

std::vector<std::pair<std::string, int>> top_classes(
    const std::unordered_map<std::string, int>& counts, size_t k) {
  std::vector<std::pair<std::string, int>> sorted(counts.begin(), counts.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  if (sorted.size() > k) sorted.resize(k);
  return sorted;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: analyze_log <carla_log_*.csv...>\n";
    return 1;
  }

  std::vector<WeatherStats> runs;
  for (int i = 1; i < argc; ++i) runs.push_back(parse_file(argv[i]));

  const auto best = std::max_element(
      runs.begin(), runs.end(),
      [](const auto& a, const auto& b) { return a.detections < b.detections; });

  std::cout << std::left << std::setw(16) << "weather" << std::setw(8)
            << "frames" << std::setw(12) << "detections" << std::setw(10)
            << "vs best" << "top classes\n";

  for (const auto& run : runs) {
    const double delta =
        100.0 * (run.detections - best->detections) / best->detections;
    std::ostringstream tops;
    for (const auto& [cls, n] : top_classes(run.per_class, 3)) {
      tops << cls << " " << n << "  ";
    }
    std::cout << std::setw(16) << run.weather << std::setw(8) << run.frames
              << std::setw(12) << run.detections << std::setw(10)
              << (std::to_string(static_cast<int>(delta)) + "%")
              << tops.str() << "\n";
  }

  const int total = std::accumulate(
      runs.begin(), runs.end(), 0,
      [](int acc, const auto& r) { return acc + r.detections; });
  std::cout << "\ntotal: " << total << " detections over "
            << std::accumulate(runs.begin(), runs.end(), 0,
                               [](int acc, const auto& r) {
                                 return acc + r.frames;
                               })
            << " frames in " << runs.size() << " scenarios\n";
  return 0;
}
