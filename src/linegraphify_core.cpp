#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Settings {
    int maxImageSize = 800;
    int blurAmount = 5;
    int cannyLower = 20;
    int cannyUpper = 80;
    int morphologyKernel = 3;
    double minContourLength = 35.0;
    double minContourArea = 0.0;
    double approxEpsilon = 0.8;
    int64_t maxTotalFormulas = 450;
    int samplesPerSegment = 18;
    int lineWidth = 2;
    int fillCloseKernel = 5;
    int fillBoundaryWidth = 4;
    bool invertLines = false;
    bool overlay = false;
    bool fillZones = false;
    bool majorOnly = false;
    double xHalfRange = 10.0;
};

struct ContourEntry {
    std::vector<cv::Point> contour;
    double length = 0.0;
    double area = 0.0;
    cv::Rect bbox;
    double score = 0.0;
};

struct Cubic {
    cv::Point2d b0;
    cv::Point2d b1;
    cv::Point2d b2;
    cv::Point2d b3;
};

struct CurveEntry {
    int sourceIndex = 0;
    double length = 0.0;
    double area = 0.0;
    std::vector<Cubic> segments;
};

struct ZoneEntry {
    int index = 0;
    int pixelCount = 0;
    cv::Rect bbox;
    cv::Vec3b bgr;
};

static void usage() {
    std::cerr
        << "Usage:\n"
        << "  linegraphify_core --input image.png --output-png out.png [options]\n\n"
        << "Options:\n"
        << "  --output-json path          Write formula JSON\n"
        << "  --output-txt path           Write readable formula TXT\n"
        << "  --max-image-size n          Default 800\n"
        << "  --max-formulas n            Default 450, supports large values\n"
        << "  --fill-zones 0|1            Default 0\n"
        << "  --line-width n              Default 2\n"
        << "  --fill-close-kernel n       Default 5\n"
        << "  --fill-boundary-width n     Default 4\n"
        << "  --blur n                    Default 5\n"
        << "  --canny-lower n             Default 20\n"
        << "  --canny-upper n             Default 80\n"
        << "  --morphology-kernel n       Default 3\n"
        << "  --min-contour-length n      Default 35\n"
        << "  --min-contour-area n        Default 0\n"
        << "  --approx-epsilon n          Percent, default 0.8\n"
        << "  --samples-per-segment n     Default 18\n"
        << "  --invert-lines 0|1          Default 0\n"
        << "  --overlay 0|1               Default 0\n"
        << "  --major-only 0|1            Default 0\n"
        << "  --x-half-range n            Default 10\n";
}

static int parseBool(const std::string& value) {
    if (value == "1" || value == "true" || value == "True" || value == "yes") return 1;
    if (value == "0" || value == "false" || value == "False" || value == "no") return 0;
    throw std::runtime_error("Invalid boolean value: " + value);
}

static int oddKernel(int value) {
    value = std::max(0, value);
    if (value == 0) return 0;
    return (value % 2 == 1) ? value : value + 1;
}

static std::string jsonEscape(const std::string& input) {
    std::ostringstream out;
    for (char c : input) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

static Settings parseArgs(int argc, char** argv, std::string& inputPath, std::string& pngPath,
                          std::string& jsonPath, std::string& txtPath) {
    Settings s;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto requireValue = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
            return argv[++i];
        };

        if (key == "--input") inputPath = requireValue();
        else if (key == "--output-png") pngPath = requireValue();
        else if (key == "--output-json") jsonPath = requireValue();
        else if (key == "--output-txt") txtPath = requireValue();
        else if (key == "--max-image-size") s.maxImageSize = std::stoi(requireValue());
        else if (key == "--max-formulas") s.maxTotalFormulas = std::stoll(requireValue());
        else if (key == "--fill-zones") s.fillZones = parseBool(requireValue());
        else if (key == "--line-width") s.lineWidth = std::stoi(requireValue());
        else if (key == "--fill-close-kernel") s.fillCloseKernel = std::stoi(requireValue());
        else if (key == "--fill-boundary-width") s.fillBoundaryWidth = std::stoi(requireValue());
        else if (key == "--blur") s.blurAmount = std::stoi(requireValue());
        else if (key == "--canny-lower") s.cannyLower = std::stoi(requireValue());
        else if (key == "--canny-upper") s.cannyUpper = std::stoi(requireValue());
        else if (key == "--morphology-kernel") s.morphologyKernel = std::stoi(requireValue());
        else if (key == "--min-contour-length") s.minContourLength = std::stod(requireValue());
        else if (key == "--min-contour-area") s.minContourArea = std::stod(requireValue());
        else if (key == "--approx-epsilon") s.approxEpsilon = std::stod(requireValue());
        else if (key == "--samples-per-segment") s.samplesPerSegment = std::stoi(requireValue());
        else if (key == "--invert-lines") s.invertLines = parseBool(requireValue());
        else if (key == "--overlay") s.overlay = parseBool(requireValue());
        else if (key == "--major-only") s.majorOnly = parseBool(requireValue());
        else if (key == "--x-half-range") s.xHalfRange = std::stod(requireValue());
        else if (key == "--help" || key == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }

    if (inputPath.empty()) throw std::runtime_error("--input is required");
    if (pngPath.empty() && jsonPath.empty() && txtPath.empty()) {
        throw std::runtime_error("At least one output path is required");
    }
    if (s.maxImageSize < 64) throw std::runtime_error("--max-image-size must be >= 64");
    if (s.maxTotalFormulas <= 0) throw std::runtime_error("--max-formulas must be positive");
    if (s.cannyUpper <= s.cannyLower) throw std::runtime_error("--canny-upper must be greater than lower");
    if (s.samplesPerSegment < 2) throw std::runtime_error("--samples-per-segment must be >= 2");
    if (s.lineWidth <= 0) throw std::runtime_error("--line-width must be positive");
    if (s.fillBoundaryWidth <= 0) throw std::runtime_error("--fill-boundary-width must be positive");
    if (s.xHalfRange <= 0) throw std::runtime_error("--x-half-range must be positive");
    return s;
}

static cv::Mat resizeToMaxSide(const cv::Mat& img, int maxSide, double& scale) {
    int h = img.rows;
    int w = img.cols;
    scale = std::min(1.0, static_cast<double>(maxSide) / static_cast<double>(std::max(h, w)));
    if (scale >= 1.0) return img.clone();
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(std::max(1, static_cast<int>(std::round(w * scale))),
                                      std::max(1, static_cast<int>(std::round(h * scale)))),
               0, 0, cv::INTER_AREA);
    return resized;
}

static cv::Mat removeSmallEdgeComponents(const cv::Mat& edges, const Settings& s) {
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(edges, labels, stats, centroids, 8, CV_32S);
    cv::Mat cleaned = cv::Mat::zeros(edges.size(), CV_8U);
    int minPixels = std::max(6, static_cast<int>(s.minContourLength * 0.20));
    int minSpan = std::max(6, static_cast<int>(s.minContourLength * 0.18));

    for (int label = 1; label < count; ++label) {
        int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < minPixels) continue;
        if (std::max(w, h) < minSpan) continue;
        cleaned.setTo(255, labels == label);
    }
    return cleaned;
}

static cv::Mat preprocessEdges(const cv::Mat& bgr, const Settings& s, cv::Mat& gray) {
    cv::Mat denoised;
    cv::bilateralFilter(bgr, denoised, 9, 60, 60);
    cv::cvtColor(denoised, gray, cv::COLOR_BGR2GRAY);

    int blur = oddKernel(s.blurAmount);
    if (blur > 0) cv::GaussianBlur(gray, gray, cv::Size(blur, blur), 0);

    cv::Mat edges;
    cv::Canny(gray, edges, s.cannyLower, s.cannyUpper, 3, true);

    int morph = std::max(0, s.morphologyKernel);
    if (morph > 0) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph, morph));
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);
    }
    return removeSmallEdgeComponents(edges, s);
}

static std::vector<ContourEntry> extractContours(const cv::Mat& edges, const Settings& s) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, s.majorOnly ? cv::RETR_EXTERNAL : cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    std::vector<ContourEntry> entries;
    entries.reserve(contours.size());
    for (const auto& contour : contours) {
        double length = cv::arcLength(contour, true);
        double area = std::abs(cv::contourArea(contour));
        cv::Rect bbox = cv::boundingRect(contour);
        double span = static_cast<double>(std::max(bbox.width, bbox.height));
        double diag = std::hypot(static_cast<double>(bbox.width), static_cast<double>(bbox.height));

        if (length < s.minContourLength) continue;
        if (s.minContourArea > 0.0 && area < s.minContourArea) continue;
        if (span < s.minContourLength * 0.25) continue;

        ContourEntry entry;
        entry.contour = contour;
        entry.length = length;
        entry.area = area;
        entry.bbox = bbox;
        entry.score = length + diag * 2.5 + std::sqrt(std::max(0.0, area)) * 4.0;
        entries.push_back(std::move(entry));
    }

    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });
    return entries;
}

static std::vector<cv::Point2d> toPoint2d(const std::vector<cv::Point>& pts) {
    std::vector<cv::Point2d> out;
    out.reserve(pts.size());
    for (const auto& p : pts) out.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y));
    return out;
}

static std::vector<cv::Point2d> resampleOpenPolyline(const std::vector<cv::Point2d>& pts, int64_t targetCount) {
    targetCount = std::max<int64_t>(2, targetCount);
    if (static_cast<int64_t>(pts.size()) == targetCount) return pts;
    if (pts.size() < 2) return pts;

    std::vector<double> cumulative(pts.size(), 0.0);
    for (size_t i = 1; i < pts.size(); ++i) {
        cumulative[i] = cumulative[i - 1] + cv::norm(pts[i] - pts[i - 1]);
    }
    double total = cumulative.back();
    if (total <= 0.0) return {pts.front(), pts.back()};

    std::vector<cv::Point2d> out;
    out.reserve(static_cast<size_t>(targetCount));
    size_t seg = 0;
    for (int64_t i = 0; i < targetCount; ++i) {
        double d = total * static_cast<double>(i) / static_cast<double>(targetCount - 1);
        while (seg + 1 < cumulative.size() && cumulative[seg + 1] < d) ++seg;
        if (seg + 1 >= pts.size()) {
            out.push_back(pts.back());
            continue;
        }
        double span = std::max(1e-9, cumulative[seg + 1] - cumulative[seg]);
        double t = (d - cumulative[seg]) / span;
        out.push_back(pts[seg] * (1.0 - t) + pts[seg + 1] * t);
    }
    return out;
}

static std::vector<Cubic> openPolylineToCubicSegments(const std::vector<cv::Point2d>& pts) {
    std::vector<Cubic> segments;
    if (pts.size() < 2) return segments;

    std::vector<double> lengths;
    lengths.reserve(pts.size() - 1);
    for (size_t i = 0; i + 1 < pts.size(); ++i) lengths.push_back(cv::norm(pts[i + 1] - pts[i]));
    std::nth_element(lengths.begin(), lengths.begin() + lengths.size() / 2, lengths.end());
    double median = lengths[lengths.size() / 2];
    double maxJump = std::min(std::max(35.0, median * 3.0), 90.0);

    segments.reserve(pts.size() - 1);
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        cv::Point2d p0 = pts[i];
        cv::Point2d p1 = pts[i + 1];
        if (cv::norm(p1 - p0) > maxJump) continue;
        Cubic c;
        c.b0 = p0;
        c.b1 = p0 + (p1 - p0) / 3.0;
        c.b2 = p0 + (p1 - p0) * (2.0 / 3.0);
        c.b3 = p1;
        segments.push_back(c);
    }
    return segments;
}

static std::vector<int64_t> allocateFormulaCounts(const std::vector<ContourEntry>& entries, int64_t budget) {
    size_t n = std::min<size_t>(entries.size(), static_cast<size_t>(budget));
    std::vector<int64_t> counts(n, 1);
    if (n == 0) return counts;

    int64_t remaining = budget - static_cast<int64_t>(n);
    long double totalWeight = 0.0;
    for (size_t i = 0; i < n; ++i) totalWeight += std::max(1.0, entries[i].score);

    std::vector<long double> fractional(n, 0.0);
    int64_t assigned = 0;
    for (size_t i = 0; i < n; ++i) {
        long double exact = static_cast<long double>(remaining) * std::max(1.0, entries[i].score) / totalWeight;
        int64_t extra = static_cast<int64_t>(std::floor(exact));
        counts[i] += extra;
        assigned += extra;
        fractional[i] = exact - static_cast<long double>(extra);
    }

    int64_t leftover = remaining - assigned;
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return fractional[a] > fractional[b];
    });
    for (int64_t i = 0; i < leftover; ++i) counts[order[static_cast<size_t>(i % n)]]++;
    return counts;
}

static std::vector<CurveEntry> buildSegments(const std::vector<ContourEntry>& entries, const Settings& s) {
    std::vector<CurveEntry> curves;
    if (entries.empty()) return curves;

    int64_t budget = std::max<int64_t>(1, s.maxTotalFormulas);
    std::vector<int64_t> counts = allocateFormulaCounts(entries, budget);
    curves.reserve(counts.size());

    int64_t used = 0;
    for (size_t i = 0; i < counts.size() && used < budget; ++i) {
        int64_t want = std::min<int64_t>(counts[i], budget - used);
        if (want <= 0) continue;

        std::vector<cv::Point2d> base;
        if (want < static_cast<int64_t>(entries[i].contour.size())) {
            double epsilon = entries[i].length * s.approxEpsilon / 100.0;
            std::vector<cv::Point> approx;
            cv::approxPolyDP(entries[i].contour, approx, epsilon, false);
            base = toPoint2d(approx.empty() ? entries[i].contour : approx);
        } else {
            base = toPoint2d(entries[i].contour);
        }

        std::vector<cv::Point2d> points = resampleOpenPolyline(base, want + 1);
        std::vector<Cubic> segments = openPolylineToCubicSegments(points);
        if (segments.empty()) continue;
        if (static_cast<int64_t>(segments.size()) > want) segments.resize(static_cast<size_t>(want));

        CurveEntry curve;
        curve.sourceIndex = static_cast<int>(i + 1);
        curve.length = entries[i].length;
        curve.area = entries[i].area;
        curve.segments = std::move(segments);
        used += static_cast<int64_t>(curve.segments.size());
        curves.push_back(std::move(curve));
    }
    return curves;
}

static cv::Point2d sampleCubic(const Cubic& c, double t) {
    double u = 1.0 - t;
    return c.b0 * (u * u * u) + c.b1 * (3.0 * u * u * t) + c.b2 * (3.0 * u * t * t) + c.b3 * (t * t * t);
}

static cv::Mat renderFormulaBoundaryMask(cv::Size size, const std::vector<CurveEntry>& curves, const Settings& s) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    int samples = std::max(2, s.samplesPerSegment);
    int width = std::max(1, s.lineWidth);

    for (const auto& curve : curves) {
        for (const Cubic& c : curve.segments) {
            std::vector<cv::Point> pts;
            pts.reserve(samples);
            for (int i = 0; i < samples; ++i) {
                double t = static_cast<double>(i) / static_cast<double>(samples - 1);
                cv::Point2d p = sampleCubic(c, t);
                pts.emplace_back(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
            }
            if (pts.size() >= 2) cv::polylines(mask, pts, false, cv::Scalar(255), width, cv::LINE_AA);
        }
    }
    return mask;
}

static cv::Mat thickenBoundaryMask(const cv::Mat& edges, int lineWidth) {
    cv::Mat mask = edges.clone();
    int width = std::max(1, lineWidth);
    if (width > 1) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(width, width));
        cv::dilate(mask, mask, kernel);
    }
    return mask;
}

static cv::Mat prepareFillBoundaryMask(const cv::Mat& boundaryInput, const Settings& s) {
    cv::Mat boundary = boundaryInput.clone();
    int closeKernel = oddKernel(s.fillCloseKernel);
    if (closeKernel > 1) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closeKernel, closeKernel));
        cv::morphologyEx(boundary, boundary, cv::MORPH_CLOSE, kernel);
    }
    boundary = thickenBoundaryMask(boundary, s.fillBoundaryWidth);
    boundary.row(0).setTo(255);
    boundary.row(boundary.rows - 1).setTo(255);
    boundary.col(0).setTo(255);
    boundary.col(boundary.cols - 1).setTo(255);
    return boundary;
}

static cv::Mat renderLineImage(const cv::Mat& boundary, const cv::Mat& bgr, const Settings& s) {
    cv::Mat lineMask = thickenBoundaryMask(boundary, s.lineWidth);
    cv::Mat out;
    if (s.overlay) {
        out = bgr.clone();
        cv::Vec3b color = s.invertLines ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);
        out.setTo(color, lineMask);
        return out;
    }

    cv::Vec3b bg = s.invertLines ? cv::Vec3b(0, 0, 0) : cv::Vec3b(255, 255, 255);
    cv::Vec3b fg = s.invertLines ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);
    out = cv::Mat(bgr.size(), CV_8UC3, cv::Scalar(bg[0], bg[1], bg[2]));
    out.setTo(fg, lineMask);
    return out;
}

static cv::Mat colorZonesFromBoundary(const cv::Mat& boundary, const cv::Mat& bgr, const Settings& s,
                                      std::vector<ZoneEntry>& zones) {
    cv::Mat freeMask;
    cv::threshold(boundary, freeMask, 0, 255, cv::THRESH_BINARY_INV);

    cv::Mat labels, stats, centroids;
    int labelCount = cv::connectedComponentsWithStats(freeMask, labels, stats, centroids, 4, CV_32S);
    std::vector<cv::Vec3d> sums(labelCount, cv::Vec3d(0, 0, 0));
    std::vector<int> counts(labelCount, 0);

    for (int y = 0; y < labels.rows; ++y) {
        const int* labelRow = labels.ptr<int>(y);
        const cv::Vec3b* imgRow = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < labels.cols; ++x) {
            int label = labelRow[x];
            if (label <= 0) continue;
            sums[label][0] += imgRow[x][0];
            sums[label][1] += imgRow[x][1];
            sums[label][2] += imgRow[x][2];
            counts[label]++;
        }
    }

    cv::Mat out = cv::Mat::zeros(bgr.size(), CV_8UC3);
    std::vector<cv::Vec3b> colors(labelCount, cv::Vec3b(0, 0, 0));
    for (int label = 1; label < labelCount; ++label) {
        if (counts[label] <= 0) continue;
        colors[label] = cv::Vec3b(
            static_cast<uchar>(std::round(sums[label][0] / counts[label])),
            static_cast<uchar>(std::round(sums[label][1] / counts[label])),
            static_cast<uchar>(std::round(sums[label][2] / counts[label])));

        ZoneEntry zone;
        zone.index = static_cast<int>(zones.size() + 1);
        zone.pixelCount = counts[label];
        zone.bbox = cv::Rect(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                             stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));
        zone.bgr = colors[label];
        zones.push_back(zone);
    }

    for (int y = 0; y < labels.rows; ++y) {
        const int* labelRow = labels.ptr<int>(y);
        cv::Vec3b* outRow = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < labels.cols; ++x) {
            int label = labelRow[x];
            if (label > 0) outRow[x] = colors[label];
        }
    }

    cv::Vec3b line = s.invertLines ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);
    out.setTo(line, boundary);
    return out;
}

static void overlayBoundary(cv::Mat& image, const cv::Mat& boundary, const Settings& s) {
    cv::Vec3b line = s.invertLines ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);
    image.setTo(line, boundary);
}

static cv::Point2d pixelToMath(const cv::Point2d& p, int width, int height, double xHalfRange) {
    double scale = (2.0 * xHalfRange) / std::max(width - 1, 1);
    double yHalfRange = scale * static_cast<double>(height - 1) / 2.0;
    return cv::Point2d(-xHalfRange + p.x * scale, yHalfRange - p.y * scale);
}

static void powerBasis(const Cubic& c, cv::Point2d& a, cv::Point2d& b, cv::Point2d& cc, cv::Point2d& d) {
    a = c.b3 - c.b2 * 3.0 + c.b1 * 3.0 - c.b0;
    b = c.b0 * 3.0 - c.b1 * 6.0 + c.b2 * 3.0;
    cc = c.b1 * 3.0 - c.b0 * 3.0;
    d = c.b0;
}

static int64_t formulaCount(const std::vector<CurveEntry>& curves) {
    int64_t total = 0;
    for (const auto& c : curves) total += static_cast<int64_t>(c.segments.size());
    return total;
}

static void writeJson(const std::string& path, const std::string& inputPath, const Settings& s,
                      const cv::Size& size, double scale, const std::vector<CurveEntry>& curves,
                      const std::vector<ZoneEntry>& zones) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open JSON output: " + path);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"project_meta\": {\n";
    out << "    \"app\": \"LineGraphify C++ Core\",\n";
    out << "    \"source_image_path\": \"" << jsonEscape(inputPath) << "\",\n";
    out << "    \"processed_width\": " << size.width << ",\n";
    out << "    \"processed_height\": " << size.height << ",\n";
    out << "    \"resize_scale_from_original\": " << scale << ",\n";
    out << "    \"total_formula_count\": " << formulaCount(curves) << ",\n";
    out << "    \"zone_count\": " << zones.size() << ",\n";
    out << "    \"settings\": {\n";
    out << "      \"max_image_size\": " << s.maxImageSize << ",\n";
    out << "      \"max_total_formulas\": " << s.maxTotalFormulas << ",\n";
    out << "      \"samples_per_segment\": " << s.samplesPerSegment << ",\n";
    out << "      \"fill_zones\": " << (s.fillZones ? "true" : "false") << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"formula_data\": [\n";

    bool firstContour = true;
    int contourIndex = 0;
    for (const auto& curve : curves) {
        if (!firstContour) out << ",\n";
        firstContour = false;
        contourIndex++;
        out << "    {\n";
        out << "      \"contour_index\": " << contourIndex << ",\n";
        out << "      \"source_length_px\": " << curve.length << ",\n";
        out << "      \"source_area_px\": " << curve.area << ",\n";
        out << "      \"allocated_formula_count\": " << curve.segments.size() << ",\n";
        out << "      \"segments\": [\n";
        for (size_t i = 0; i < curve.segments.size(); ++i) {
            const Cubic& px = curve.segments[i];
            Cubic m;
            m.b0 = pixelToMath(px.b0, size.width, size.height, s.xHalfRange);
            m.b1 = pixelToMath(px.b1, size.width, size.height, s.xHalfRange);
            m.b2 = pixelToMath(px.b2, size.width, size.height, s.xHalfRange);
            m.b3 = pixelToMath(px.b3, size.width, size.height, s.xHalfRange);
            cv::Point2d a, b, cc, d;
            powerBasis(m, a, b, cc, d);
            if (i > 0) out << ",\n";
            out << "        {\"segment_index\": " << (i + 1)
                << ", \"x_coefficients\": [" << a.x << ", " << b.x << ", " << cc.x << ", " << d.x
                << "], \"y_coefficients\": [" << a.y << ", " << b.y << ", " << cc.y << ", " << d.y
                << "], \"domain\": \"0 <= t <= 1\"}";
        }
        out << "\n      ]\n";
        out << "    }";
    }

    out << "\n  ],\n";
    out << "  \"zone_data\": [\n";
    for (size_t i = 0; i < zones.size(); ++i) {
        const ZoneEntry& z = zones[i];
        if (i > 0) out << ",\n";
        out << "    {\"zone_index\": " << z.index
            << ", \"pixel_count\": " << z.pixelCount
            << ", \"bbox\": [" << z.bbox.x << ", " << z.bbox.y << ", " << z.bbox.width << ", " << z.bbox.height
            << "], \"fill_rgb\": [" << static_cast<int>(z.bgr[2]) << ", " << static_cast<int>(z.bgr[1])
            << ", " << static_cast<int>(z.bgr[0]) << "]}";
    }
    out << "\n  ]\n";
    out << "}\n";
}

static void writeTxt(const std::string& path, const std::string& inputPath, const Settings& s,
                     const cv::Size& size, const std::vector<CurveEntry>& curves) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open TXT output: " + path);
    out << std::fixed << std::setprecision(6);
    out << "=== LineGraphify C++ Formula Export ===\n\n";
    out << "Source image: " << inputPath << "\n";
    out << "Processed size: " << size.width << " x " << size.height << "\n";
    out << "Total formulas: " << formulaCount(curves) << "\n\n";
    out << "Formula form:\n";
    out << "  x(t) = a*t^3 + b*t^2 + c*t + d\n";
    out << "  y(t) = e*t^3 + f*t^2 + g*t + h\n";
    out << "  domain: 0 <= t <= 1\n\n";

    int contourIndex = 0;
    for (const auto& curve : curves) {
        contourIndex++;
        out << "--------------------------------------------------\n";
        out << "Contour " << contourIndex << " | length_px=" << curve.length
            << " | area_px=" << curve.area << " | formulas=" << curve.segments.size() << "\n";
        for (size_t i = 0; i < curve.segments.size(); ++i) {
            const Cubic& px = curve.segments[i];
            Cubic m;
            m.b0 = pixelToMath(px.b0, size.width, size.height, s.xHalfRange);
            m.b1 = pixelToMath(px.b1, size.width, size.height, s.xHalfRange);
            m.b2 = pixelToMath(px.b2, size.width, size.height, s.xHalfRange);
            m.b3 = pixelToMath(px.b3, size.width, size.height, s.xHalfRange);
            cv::Point2d a, b, cc, d;
            powerBasis(m, a, b, cc, d);
            out << "  Segment " << (i + 1) << "\n";
            out << "    x(t) = " << a.x << "*t^3 + " << b.x << "*t^2 + " << cc.x << "*t + " << d.x << "\n";
            out << "    y(t) = " << a.y << "*t^3 + " << b.y << "*t^2 + " << cc.y << "*t + " << d.y << "\n";
            out << "    domain: 0 <= t <= 1\n";
        }
    }
}

int main(int argc, char** argv) {
    auto started = std::chrono::steady_clock::now();
    try {
        std::string inputPath, pngPath, jsonPath, txtPath;
        Settings settings = parseArgs(argc, argv, inputPath, pngPath, jsonPath, txtPath);

        cv::Mat original = cv::imread(inputPath, cv::IMREAD_COLOR);
        if (original.empty()) throw std::runtime_error("Failed to read image: " + inputPath);

        double scale = 1.0;
        cv::Mat image = resizeToMaxSide(original, settings.maxImageSize, scale);
        cv::Mat gray;
        cv::Mat edges = preprocessEdges(image, settings, gray);
        std::vector<ContourEntry> contours = extractContours(edges, settings);
        if (contours.empty()) throw std::runtime_error("No contours found");

        std::vector<CurveEntry> curves = buildSegments(contours, settings);
        cv::Mat formulaBoundary = renderFormulaBoundaryMask(image.size(), curves, settings);
        std::vector<ZoneEntry> zones;

        cv::Mat result;
        if (settings.fillZones) {
            cv::Mat fillBoundary = prepareFillBoundaryMask(formulaBoundary, settings);
            result = colorZonesFromBoundary(fillBoundary, image, settings, zones);
            overlayBoundary(result, formulaBoundary, settings);
        } else {
            result = renderLineImage(formulaBoundary, image, settings);
        }

        if (!pngPath.empty() && !cv::imwrite(pngPath, result)) {
            throw std::runtime_error("Failed to write PNG: " + pngPath);
        }
        if (!jsonPath.empty()) writeJson(jsonPath, inputPath, settings, image.size(), scale, curves, zones);
        if (!txtPath.empty()) writeTxt(txtPath, inputPath, settings, image.size(), curves);

        auto ended = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(ended - started).count();
        std::cout << "processed_width=" << image.cols << "\n";
        std::cout << "processed_height=" << image.rows << "\n";
        std::cout << "contours=" << contours.size() << "\n";
        std::cout << "formulas=" << formulaCount(curves) << "\n";
        std::cout << "zones=" << zones.size() << "\n";
        std::cout << "seconds=" << std::fixed << std::setprecision(3) << seconds << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n\n";
        usage();
        return 1;
    }
}
