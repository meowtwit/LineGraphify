#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct Contour {
    int id = 0;
    double score = 1.0;
    double length = 0.0;
    double area = 0.0;
    std::vector<Point> points;
};

struct CubicCoefficients {
    double xa = 0.0;
    double xb = 0.0;
    double xc = 0.0;
    double xd = 0.0;
    double ya = 0.0;
    double yb = 0.0;
    double yc = 0.0;
    double yd = 0.0;
};

struct Args {
    std::string inputContours;
    std::string output;
    std::string format = "csv";
    int width = 1;
    int height = 1;
    double xHalfRange = 10.0;
    int64_t maxFormulas = 10000;
};

static void usage() {
    std::cerr
        << "Usage:\n"
        << "  formula_generator --input-contours contours.csv --output formulas.csv --format csv "
        << "--width 640 --height 800 --x-half-range 10 --max-formulas 100000\n\n"
        << "Formats: csv, jsonl, json, txt\n";
}

static std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) cells.push_back(cell);
    return cells;
}

static Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
            return argv[++i];
        };
        if (key == "--input-contours") args.inputContours = value();
        else if (key == "--output") args.output = value();
        else if (key == "--format") args.format = value();
        else if (key == "--width") args.width = std::stoi(value());
        else if (key == "--height") args.height = std::stoi(value());
        else if (key == "--x-half-range") args.xHalfRange = std::stod(value());
        else if (key == "--max-formulas") args.maxFormulas = std::stoll(value());
        else if (key == "-h" || key == "--help") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }
    if (args.inputContours.empty()) throw std::runtime_error("--input-contours is required");
    if (args.output.empty()) throw std::runtime_error("--output is required");
    if (args.width <= 1 || args.height <= 1) throw std::runtime_error("--width/--height must be > 1");
    if (args.xHalfRange <= 0.0) throw std::runtime_error("--x-half-range must be positive");
    if (args.maxFormulas <= 0) throw std::runtime_error("--max-formulas must be positive");
    if (args.format != "csv" && args.format != "jsonl" && args.format != "json" && args.format != "txt") {
        throw std::runtime_error("--format must be csv, jsonl, json, or txt");
    }
    return args;
}

static std::vector<Contour> readContours(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open contour input: " + path);

    std::string line;
    std::getline(in, line);  // header
    std::vector<Contour> contours;
    std::unordered_map<int, size_t> indexById;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cells = splitCsvLine(line);
        if (cells.size() < 6) continue;
        int id = std::stoi(cells[0]);
        auto found = indexById.find(id);
        if (found == indexById.end()) {
            Contour contour;
            contour.id = id;
            contour.score = std::max(1.0, std::stod(cells[3]));
            contour.length = std::stod(cells[4]);
            contour.area = std::stod(cells[5]);
            indexById[id] = contours.size();
            contours.push_back(std::move(contour));
            found = indexById.find(id);
        }
        contours[found->second].points.push_back({std::stod(cells[1]), std::stod(cells[2])});
    }

    contours.erase(
        std::remove_if(contours.begin(), contours.end(), [](const Contour& c) { return c.points.size() < 2; }),
        contours.end());
    std::sort(contours.begin(), contours.end(), [](const Contour& a, const Contour& b) {
        return a.score > b.score;
    });
    return contours;
}

static double distance(Point a, Point b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static std::vector<int64_t> allocateCounts(const std::vector<Contour>& contours, int64_t budget) {
    size_t n = std::min<size_t>(contours.size(), static_cast<size_t>(budget));
    std::vector<int64_t> counts(n, 1);
    if (n == 0) return counts;
    int64_t remaining = budget - static_cast<int64_t>(n);

    long double totalWeight = 0.0;
    for (size_t i = 0; i < n; ++i) totalWeight += std::max(1.0, contours[i].score);

    std::vector<long double> fractional(n);
    int64_t assigned = 0;
    for (size_t i = 0; i < n; ++i) {
        long double exact = static_cast<long double>(remaining) * std::max(1.0, contours[i].score) / totalWeight;
        int64_t extra = static_cast<int64_t>(std::floor(exact));
        counts[i] += extra;
        assigned += extra;
        fractional[i] = exact - static_cast<long double>(extra);
    }

    int64_t leftover = remaining - assigned;
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return fractional[a] > fractional[b]; });
    for (int64_t i = 0; i < leftover; ++i) counts[order[static_cast<size_t>(i % n)]]++;
    return counts;
}

static std::vector<Point> resamplePolyline(const std::vector<Point>& points, int64_t targetCount) {
    targetCount = std::max<int64_t>(2, targetCount);
    if (points.size() < 2) return points;

    std::vector<double> cumulative(points.size(), 0.0);
    for (size_t i = 1; i < points.size(); ++i) {
        cumulative[i] = cumulative[i - 1] + distance(points[i - 1], points[i]);
    }
    double total = cumulative.back();
    if (total <= 1e-9) return {points.front(), points.back()};

    std::vector<Point> out;
    out.reserve(static_cast<size_t>(targetCount));
    size_t segment = 0;
    for (int64_t i = 0; i < targetCount; ++i) {
        double d = total * static_cast<double>(i) / static_cast<double>(targetCount - 1);
        while (segment + 1 < cumulative.size() && cumulative[segment + 1] < d) segment++;
        if (segment + 1 >= points.size()) {
            out.push_back(points.back());
            continue;
        }
        double span = std::max(1e-9, cumulative[segment + 1] - cumulative[segment]);
        double t = (d - cumulative[segment]) / span;
        out.push_back({
            points[segment].x * (1.0 - t) + points[segment + 1].x * t,
            points[segment].y * (1.0 - t) + points[segment + 1].y * t,
        });
    }
    return out;
}

static Point pixelToMath(Point p, int width, int height, double xHalfRange) {
    double scale = (2.0 * xHalfRange) / static_cast<double>(std::max(width - 1, 1));
    double yHalfRange = scale * static_cast<double>(height - 1) / 2.0;
    return {-xHalfRange + p.x * scale, yHalfRange - p.y * scale};
}

static CubicCoefficients lineSegmentToCubic(Point p0px, Point p1px, const Args& args) {
    Point p0 = pixelToMath(p0px, args.width, args.height, args.xHalfRange);
    Point p1 = pixelToMath(p1px, args.width, args.height, args.xHalfRange);
    CubicCoefficients c;
    c.xa = 0.0;
    c.xb = 0.0;
    c.xc = p1.x - p0.x;
    c.xd = p0.x;
    c.ya = 0.0;
    c.yb = 0.0;
    c.yc = p1.y - p0.y;
    c.yd = p0.y;
    return c;
}

static void writeCsvHeader(std::ofstream& out) {
    out << "contour_index,segment_index,x_a,x_b,x_c,x_d,y_a,y_b,y_c,y_d,domain\n";
}

static void writeCsvSegment(std::ofstream& out, int contourIndex, int64_t segmentIndex, const CubicCoefficients& c) {
    out << contourIndex << ',' << segmentIndex << ','
        << c.xa << ',' << c.xb << ',' << c.xc << ',' << c.xd << ','
        << c.ya << ',' << c.yb << ',' << c.yc << ',' << c.yd << ",0 <= t <= 1\n";
}

static void writeJsonSegment(std::ofstream& out, int contourIndex, int64_t segmentIndex, const CubicCoefficients& c) {
    out << "{\"type\":\"segment\",\"data\":{\"contour_index\":" << contourIndex
        << ",\"segment_index\":" << segmentIndex
        << ",\"x_coefficients\":[" << c.xa << ',' << c.xb << ',' << c.xc << ',' << c.xd << ']'
        << ",\"y_coefficients\":[" << c.ya << ',' << c.yb << ',' << c.yc << ',' << c.yd << ']'
        << ",\"domain\":\"0 <= t <= 1\"}}";
}

static int64_t generate(const std::vector<Contour>& contours, const Args& args) {
    std::ofstream out(args.output);
    if (!out) throw std::runtime_error("Failed to open output: " + args.output);
    out << std::fixed << std::setprecision(9);

    std::vector<int64_t> counts = allocateCounts(contours, args.maxFormulas);
    int64_t total = 0;
    bool firstJson = true;

    if (args.format == "csv") {
        writeCsvHeader(out);
    } else if (args.format == "json") {
        out << "{\"formula_data\":[\n";
    } else if (args.format == "txt") {
        out << "=== LineGraphify C++ Formula Export ===\n\n";
    }

    for (size_t i = 0; i < counts.size() && total < args.maxFormulas; ++i) {
        int64_t want = std::min<int64_t>(counts[i], args.maxFormulas - total);
        std::vector<Point> sampled = resamplePolyline(contours[i].points, want + 1);
        int64_t madeForContour = 0;
        for (size_t j = 0; j + 1 < sampled.size() && total < args.maxFormulas; ++j) {
            CubicCoefficients c = lineSegmentToCubic(sampled[j], sampled[j + 1], args);
            int contourIndex = static_cast<int>(i + 1);
            int64_t segmentIndex = ++madeForContour;
            if (args.format == "csv") {
                writeCsvSegment(out, contourIndex, segmentIndex, c);
            } else if (args.format == "jsonl") {
                writeJsonSegment(out, contourIndex, segmentIndex, c);
                out << '\n';
            } else if (args.format == "json") {
                if (!firstJson) out << ",\n";
                firstJson = false;
                writeJsonSegment(out, contourIndex, segmentIndex, c);
            } else {
                out << "Contour " << contourIndex << " Segment " << segmentIndex << "\n";
                out << "  x(t) = " << c.xa << "*t^3 + " << c.xb << "*t^2 + " << c.xc << "*t + " << c.xd << "\n";
                out << "  y(t) = " << c.ya << "*t^3 + " << c.yb << "*t^2 + " << c.yc << "*t + " << c.yd << "\n";
                out << "  domain: 0 <= t <= 1\n";
            }
            total++;
        }
    }

    if (args.format == "json") {
        out << "\n],\"total_formula_count\":" << total << "}\n";
    }
    return total;
}

int main(int argc, char** argv) {
    try {
        Args args = parseArgs(argc, argv);
        std::vector<Contour> contours = readContours(args.inputContours);
        if (contours.empty()) throw std::runtime_error("No contour points found");
        int64_t total = generate(contours, args);
        std::cout << "contours=" << contours.size() << "\n";
        std::cout << "formulas=" << total << "\n";
        std::cout << "output=" << args.output << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n\n";
        usage();
        return 1;
    }
}
