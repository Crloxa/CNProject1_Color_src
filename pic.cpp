#include "pic.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace ImgParse {

    using namespace std;
    using namespace cv;

    static Mat lastValidTransform;
    static int lastValidL = 133;
    static int lastValidTargetSize = 266;
    static Vec3b currentPalette[8];

    const Vec3b stdColors[8] = {
        Vec3b(0, 0, 0),
        Vec3b(0, 0, 255),
        Vec3b(0, 255, 0),
        Vec3b(0, 255, 255),
        Vec3b(255, 0, 0),
        Vec3b(255, 0, 255),
        Vec3b(255, 255, 0),
        Vec3b(255, 255, 255)
    };

    struct Marker {
        Point2f center;
        double area;
    };

    bool isDataOrHeaderCell(int r, int c) {
        if (r >= 3 && r < 6 && c >= 21 && c < 37) return true;
        if (r == 6 && c >= 21 && c < 29) return true;
        if (r >= 3 && r < 6 && c >= 37 && c < 112) return true;
        if (r == 6 && c >= 29 && c < 112) return true;
        if (r >= 7 && r < 21 && c >= 21 && c < 112) return true;
        if (r >= 21 && r < 109 && c >= 3 && c < 130) return true;
        if (r >= 109 && r < 112 && c >= 3 && c < 130) return true;
        if (r >= 112 && r < 130 && c >= 21 && c < 112) return true;
        if (r >= 112 && r < 130 && c >= 112 && c < 130) {
            if (abs(r - 126) <= 5 && abs(c - 126) <= 5) return false;
            return true;
        }
        return false;
    }

    void extractPalette(const Mat& img) {
        if (img.empty()) return;
        double k = img.cols / 133.0;
        for (int i = 0; i < 8; ++i) {
            int cx = cvRound((21.5 + i) * k);
            int cy = cvRound(6.5 * k);
            if (cx >= 0 && cx < img.cols && cy >= 0 && cy < img.rows) {
                currentPalette[i] = img.at<Vec3b>(cy, cx);
            }
        }
    }

    int getClosestPaletteIndex(Vec3b px) {
        int bestIdx = 0;
        int minDist = 1e9;
        for (int i = 0; i < 8; ++i) {
            int db = px[0] - currentPalette[i][0];
            int dg = px[1] - currentPalette[i][1];
            int dr = px[2] - currentPalette[i][2];
            int dist = db * db + dg * dg + dr * dr;
            if (dist < minDist) {
                minDist = dist;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    void drawDataGridAndStars(Mat& img) {
        double k = img.cols / 133.0;
        int markerSize = std::max(4, cvRound(0.5 * k));
        int thickness = std::max(1, cvRound(0.1 * k));

        for (int r = 0; r < 133; ++r) {
            for (int c = 0; c < 133; ++c) {
                if (isDataOrHeaderCell(r, c)) {
                    int x1 = cvRound(c * k);
                    int y1 = cvRound(r * k);
                    int x2 = cvRound((c + 1) * k);
                    int y2 = cvRound((r + 1) * k);
                    rectangle(img, Point(x1, y1), Point(x2, y2), Scalar(255, 255, 255), 1);

                    int cx = cvRound((c + 0.5) * k);
                    int cy = cvRound((r + 0.5) * k);
                    if (cx >= 0 && cx < img.cols && cy >= 0 && cy < img.rows) {
                        Vec3b px = img.at<Vec3b>(cy, cx);
                        int bestIdx = getClosestPaletteIndex(px);
                        Vec3b stdColor = stdColors[bestIdx];
                        drawMarker(img, Point(cx, cy), stdColor, MARKER_STAR, markerSize, thickness);
                    }
                }
            }
        }
    }

    void drawPaletteMarkers(Mat& img) {
        double k = img.cols / 133.0;
        int markerSize = std::max(4, cvRound(0.5 * k));
        int thickness = std::max(1, cvRound(0.1 * k));

        for (int i = 0; i < 8; ++i) {
            int cx = cvRound((21.5 + i) * k);
            int cy = cvRound(6.5 * k);
            int half = cvRound(0.5 * k);

            rectangle(img, Point(cx - half, cy - half), Point(cx + half, cy + half), Scalar(255, 0, 0), 2);
            drawMarker(img, Point(cx, cy), stdColors[i], MARKER_STAR, markerSize, thickness);
        }
    }

    int getBlackAreaLocal(const Mat& corner) {
        Mat bin;
        threshold(corner, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
        return (corner.rows * corner.cols) - countNonZero(bin);
    }

    int findLargestChild(int parentIdx, const vector<vector<Point>>& contours, const vector<Vec4i>& hierarchy) {
        int max_idx = -1;
        double max_area = -1.0;
        int child = hierarchy[parentIdx][2];
        while (child >= 0) {
            double area = contourArea(contours[child]);
            if (area > max_area) {
                max_area = area;
                max_idx = child;
            }
            child = hierarchy[child][0];
        }
        return max_idx;
    }

    bool processV5(const Mat& srcImg, Mat& disImg) {
        Mat gray, small_img;

        if (srcImg.channels() == 3) {
            Mat hsv, satMask;
            cvtColor(srcImg, hsv, COLOR_BGR2HSV);
            vector<Mat> hsv_ch;
            split(hsv, hsv_ch);

            threshold(hsv_ch[1], satMask, 100, 255, THRESH_BINARY);
            cvtColor(srcImg, gray, COLOR_BGR2GRAY);
            gray.setTo(0, satMask);
        }
        else {
            gray = srcImg.clone();
        }

        float scale = 800.0f / std::max(srcImg.cols, srcImg.rows);
        if (scale > 1.0f) scale = 1.0f;
        resize(gray, small_img, Size(), scale, scale, INTER_AREA);

        Mat binaryForOuter;
        adaptiveThreshold(small_img, binaryForOuter, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 101, 0);

        Mat kernelOuter = getStructuringElement(MORPH_CROSS, Size(5, 5));
        Mat closedForOuter;
        morphologyEx(binaryForOuter, closedForOuter, MORPH_CLOSE, kernelOuter);

        vector<vector<Point>> outerContours;
        findContours(closedForOuter, outerContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (outerContours.empty()) return false;

        int max_idx = -1;
        double max_area = 0;
        for (size_t i = 0; i < outerContours.size(); i++) {
            double area = contourArea(outerContours[i]);
            if (area > max_area) { max_area = area; max_idx = i; }
        }
        if (max_area < 2000) return false;

        vector<Point> hull;
        convexHull(outerContours[max_idx], hull);

        vector<Point> approx;
        double epsilon = 0.05 * arcLength(hull, true);
        approxPolyDP(hull, approx, epsilon, true);

        if (approx.size() != 4) {
            RotatedRect minRect = minAreaRect(hull);
            Point2f rect_points[4];
            minRect.points(rect_points);
            approx.clear();
            for (int j = 0; j < 4; j++) approx.push_back(rect_points[j]);
        }

        vector<Point2f> srcPointsOuter(4);
        for (int i = 0; i < 4; i++) {
            srcPointsOuter[i] = Point2f(approx[i].x / scale, approx[i].y / scale);
        }

        Point2f centerOuter(0, 0);
        for (int i = 0; i < 4; i++) centerOuter += srcPointsOuter[i];
        centerOuter.x /= 4.0f;
        centerOuter.y /= 4.0f;

        std::sort(srcPointsOuter.begin(), srcPointsOuter.end(), [&centerOuter](const Point2f& a, const Point2f& b) {
            return atan2(a.y - centerOuter.y, a.x - centerOuter.x) < atan2(b.y - centerOuter.y, b.x - centerOuter.x);
            });

        vector<Point2f> dstPointsOuter = {
            Point2f(0.0f, 0.0f), Point2f(532.0f, 0.0f),
            Point2f(532.0f, 532.0f), Point2f(0.0f, 532.0f)
        };

        Mat M_Outer = getPerspectiveTransform(srcPointsOuter, dstPointsOuter);
        Mat warped532;
        warpPerspective(gray, warped532, M_Outer, Size(532, 532), INTER_LINEAR);

        int cornerSize = 84;
        Rect tl(0, 0, cornerSize, cornerSize);
        Rect tr(532 - cornerSize, 0, cornerSize, cornerSize);
        Rect br(532 - cornerSize, 532 - cornerSize, cornerSize, cornerSize);
        Rect bl(0, 532 - cornerSize, cornerSize, cornerSize);

        int areas[4] = {
            getBlackAreaLocal(warped532(tl)),
            getBlackAreaLocal(warped532(tr)),
            getBlackAreaLocal(warped532(br)),
            getBlackAreaLocal(warped532(bl))
        };

        int minArea = areas[0];
        int smallQrIdx = 0;
        for (int i = 1; i < 4; ++i) {
            if (areas[i] < minArea) {
                minArea = areas[i];
                smallQrIdx = i;
            }
        }

        int maxA = *max_element(areas, areas + 4);
        if (maxA < minArea * 1.5) {
            return false;
        }

        double est_L = std::max(norm(srcPointsOuter[1] - srcPointsOuter[0]), norm(srcPointsOuter[3] - srcPointsOuter[0]));
        int L_int = cvRound(est_L);

        int targetSize = 266;
        if (est_L >= 1330) targetSize = 1330;
        else if (est_L >= 532) targetSize = 532;

        vector<Point2f> finalDst;
        if (smallQrIdx == 0)      finalDst = { Point2f(L_int,L_int), Point2f(0.0f,L_int), Point2f(0.0f,0.0f), Point2f(L_int,0.0f) };
        else if (smallQrIdx == 1) finalDst = { Point2f(L_int,0.0f), Point2f(L_int,L_int), Point2f(0.0f,L_int), Point2f(0.0f,0.0f) };
        else if (smallQrIdx == 3) finalDst = { Point2f(0.0f,L_int), Point2f(0.0f,0.0f), Point2f(L_int,0.0f), Point2f(L_int,L_int) };
        else                      finalDst = { Point2f(0.0f,0.0f), Point2f(L_int,0.0f), Point2f(L_int,L_int), Point2f(0.0f,L_int) };

        Mat M_highres = getPerspectiveTransform(srcPointsOuter, finalDst);

        lastValidTransform = M_highres;
        lastValidL = L_int;
        lastValidTargetSize = targetSize;

        Mat highResWarped;
        warpPerspective(srcImg, highResWarped, M_highres, Size(L_int, L_int), INTER_LINEAR);
        resize(highResWarped, disImg, Size(targetSize, targetSize), 0, 0, INTER_AREA);

        extractPalette(disImg);
        drawDataGridAndStars(disImg);
        drawPaletteMarkers(disImg);

        return true;
    }

    bool processV15(const Mat& srcImg, Mat& gray, Mat& disImg, bool useHSV) {
        Mat small_img, blurred, binaryForContours;

        if (useHSV && srcImg.channels() == 3) {
            Mat hsv, binaryMask;
            cvtColor(srcImg, hsv, COLOR_BGR2HSV);
            vector<Mat> hsv_ch;
            split(hsv, hsv_ch);

            threshold(hsv_ch[1], binaryMask, 180, 255, THRESH_BINARY);
            gray.setTo(255, binaryMask);
        }

        float scale = 1920.0f / std::max(srcImg.cols, srcImg.rows);
        if (scale > 1.0f) scale = 1.0f;
        resize(gray, small_img, Size(), scale, scale, INTER_AREA);

        GaussianBlur(small_img, blurred, Size(5, 5), 0);
        adaptiveThreshold(blurred, binaryForContours, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 31, 10);

        Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));
        Mat closedBinary;
        morphologyEx(binaryForContours, closedBinary, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(closedBinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        vector<Marker> markers;

        for (size_t i = 0; i < contours.size(); ++i) {
            int c1 = findLargestChild(i, contours, hierarchy);
            if (c1 < 0) continue;
            int c2 = findLargestChild(c1, contours, hierarchy);
            if (c2 < 0) continue;

            double area0 = contourArea(contours[i]);
            double area1 = contourArea(contours[c1]);
            double area2 = contourArea(contours[c2]);

            if (area0 < 15.0 * scale * scale) continue;

            double r01 = area0 / max(area1, 1.0);
            double r12 = area1 / max(area2, 1.0);

            if (r01 > 1.2 && r01 < 8.0 && r12 > 1.2 && r12 < 8.0) {
                Moments M = moments(contours[i]);
                if (M.m00 != 0) {
                    markers.push_back({
                        Point2f((M.m10 / M.m00) / scale, (M.m01 / M.m00) / scale),
                        area0 / (scale * scale)
                        });
                }
            }
        }

        vector<Marker> uniqueMarkers;
        for (const auto& m : markers) {
            bool duplicate = false;
            for (auto& um : uniqueMarkers) {
                if (norm(m.center - um.center) < 15.0) {
                    if (m.area > um.area) {
                        um.area = m.area;
                        um.center = m.center;
                    }
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) uniqueMarkers.push_back(m);
        }
        markers = uniqueMarkers;

        if (markers.size() < 3) return false;

        std::sort(markers.begin(), markers.end(), [](const Marker& a, const Marker& b) {
            return a.area > b.area;
            });

        double maxDist = 0;
        int rightAngleIdx = -1;
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                double d = norm(markers[i].center - markers[j].center);
                if (d > maxDist) {
                    maxDist = d;
                    rightAngleIdx = 3 - i - j;
                }
            }
        }

        Point2f TL = markers[rightAngleIdx].center;
        Point2f pt1 = markers[(rightAngleIdx + 1) % 3].center;
        Point2f pt2 = markers[(rightAngleIdx + 2) % 3].center;

        Point2f v1 = pt1 - TL;
        Point2f v2 = pt2 - TL;
        double len1 = norm(v1);
        double len2 = norm(v2);

        double legRatio = len1 / max(len2, 1.0);
        if (legRatio < 0.4 || legRatio > 2.5) return false;

        double cosTheta = (v1.x * v2.x + v1.y * v2.y) / max(len1 * len2, 1.0);
        if (std::abs(cosTheta) > 0.75) return false;

        double cross = v1.x * v2.y - v1.y * v2.x;
        Point2f TR, BL;
        if (cross > 0) { TR = pt1; BL = pt2; }
        else { TR = pt2; BL = pt1; }

        Point2f BR;
        bool foundBR = false;
        Point2f expectedBR = TR + BL - TL;

        if (markers.size() > 3) {
            double minDist = 1e9;
            int bestIdx = -1;
            for (size_t i = 3; i < markers.size(); ++i) {
                double d = norm(markers[i].center - expectedBR);
                if (d < minDist) {
                    minDist = d;
                    bestIdx = i;
                }
            }
            if (minDist < max(len1, len2) * 0.5) {
                BR = markers[bestIdx].center;
                foundBR = true;
            }
        }
        if (!foundBR) BR = expectedBR;

        double est_L = std::max(len1, len2) / 112.0 * 133.0;
        int L_int = cvRound(est_L);
        double k_warp = L_int / 133.0;

        int targetSize = 266;
        if (est_L >= 1330) targetSize = 1330;
        else if (est_L >= 532) targetSize = 532;

        vector<Point2f> srcPoints = { TL, TR, BR, BL };
        vector<Point2f> dstPoints = {
            Point2f(10.0f * k_warp, 10.0f * k_warp),
            Point2f(122.0f * k_warp, 10.0f * k_warp),
            foundBR ? Point2f(126.0f * k_warp, 126.0f * k_warp) : Point2f(122.0f * k_warp, 122.0f * k_warp),
            Point2f(10.0f * k_warp, 122.0f * k_warp)
        };

        Mat M_highres = getPerspectiveTransform(srcPoints, dstPoints);

        lastValidTransform = M_highres;
        lastValidL = L_int;
        lastValidTargetSize = targetSize;

        Mat highResWarped;
        warpPerspective(srcImg, highResWarped, M_highres, Size(L_int, L_int), INTER_LINEAR);
        resize(highResWarped, disImg, Size(targetSize, targetSize), 0, 0, INTER_AREA);

        extractPalette(disImg);
        drawDataGridAndStars(disImg);
        drawPaletteMarkers(disImg);

        return true;
    }

    bool Main(const cv::Mat& srcImg, cv::Mat& disImg) {
        if (srcImg.empty()) return false;

        static int last_cols = 0;
        static int last_rows = 0;
        static int v5_frame_count = 0;

        if (srcImg.cols != last_cols || srcImg.rows != last_rows) {
            last_cols = srcImg.cols;
            last_rows = srcImg.rows;
            v5_frame_count = 0;
            lastValidTransform = Mat();
        }

        double aspect = (double)srcImg.cols / srcImg.rows;
        if (aspect > 0.95 && aspect < 1.05 && srcImg.cols >= 266) {
            int targetSize = 266;
            if (srcImg.cols >= 1330) targetSize = 1330;
            else if (srcImg.cols >= 532) targetSize = 532;

            resize(srcImg, disImg, Size(targetSize, targetSize), 0, 0, INTER_AREA);
            extractPalette(disImg);
            drawDataGridAndStars(disImg);
            drawPaletteMarkers(disImg);
            return true;
        }

        if (v5_frame_count < 3) {
            v5_frame_count++;
            if (processV5(srcImg, disImg)) {
                return true;
            }
        }

        Mat grayNormal;
        if (srcImg.channels() == 3) cvtColor(srcImg, grayNormal, COLOR_BGR2GRAY);
        else grayNormal = srcImg.clone();

        if (processV15(srcImg, grayNormal, disImg, false)) {
            return true;
        }

        if (srcImg.channels() == 3) {
            Mat grayHSV = grayNormal.clone();
            if (processV15(srcImg, grayHSV, disImg, true)) {
                return true;
            }
        }

        if (!lastValidTransform.empty()) {
            Mat highResWarped;
            warpPerspective(srcImg, highResWarped, lastValidTransform, Size(lastValidL, lastValidL), INTER_LINEAR);
            resize(highResWarped, disImg, Size(lastValidTargetSize, lastValidTargetSize), 0, 0, INTER_AREA);

            extractPalette(disImg);
            drawDataGridAndStars(disImg);
            drawPaletteMarkers(disImg);
            return true;
        }

        return false;
    }

}