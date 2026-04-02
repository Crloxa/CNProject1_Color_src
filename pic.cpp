#include "pic.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace ImgParse {

    using namespace std;
    using namespace cv;

    static Mat lastValidTransform;
    static double lastValidS = 1.0;
    static Vec3b currentPalette[8];

    struct Marker {
        Point2f center;
        double area;
    };

    bool isDataCell(int r, int c) {
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

    void extractPalette(const Mat& croppedImg) {
        if (croppedImg.empty()) return;
        double S = croppedImg.cols / 133.0;
        for (int i = 0; i < 8; ++i) {
            int cx = cvRound((21.0 + i) * S);
            int cy = cvRound(6.0 * S);
            if (cx >= 0 && cx < croppedImg.cols && cy >= 0 && cy < croppedImg.rows) {
                currentPalette[i] = croppedImg.at<Vec3b>(cy, cx);
            }
        }
    }

    void drawDebug(Mat& img, double S) {
        Point2f tl(10.0 * S, 10.0 * S);
        Point2f tr(122.0 * S, 10.0 * S);
        Point2f bl(10.0 * S, 122.0 * S);
        int r = cvRound(3.5 * S);

        circle(img, tl, r, Scalar(0, 0, 255), 2);
        circle(img, tr, r, Scalar(0, 0, 255), 2);
        circle(img, bl, r, Scalar(0, 0, 255), 2);

        for (int i = 0; i < 8; ++i) {
            Point2f pc((21.0 + i) * S, 6.0 * S);
            int half = cvRound(0.5 * S);
            rectangle(img, Point2f(pc.x - half, pc.y - half), Point2f(pc.x + half, pc.y + half), Scalar(255, 0, 0), 2.5);
            circle(img, pc, 2, Scalar(0, 255, 0), -1);
        }
    }

    void drawGrid(Mat& img, double S) {
        for (int r = 0; r < 133; ++r) {
            for (int c = 0; c < 133; ++c) {
                if (isDataCell(r, c)) {
                    int x1 = cvRound(c * S);
                    int y1 = cvRound(r * S);
                    int x2 = cvRound((c + 1) * S);
                    int y2 = cvRound((r + 1) * S);
                    rectangle(img, Point(x1, y1), Point(x2, y2), Scalar(200, 200, 200), 1);
                }
            }
        }
    }

    void resizeToTarget(Mat& img) {
        int L = img.cols;
        int targetSize = 266;
        if (L >= 1330) {
            targetSize = 1330;
        }
        else if (L >= 532) {
            targetSize = 532;
        }
        resize(img, img, Size(targetSize, targetSize), 0, 0, INTER_NEAREST);
    }

    int getBlackArea(const Mat& corner) {
        return (corner.rows * corner.cols) - countNonZero(corner);
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

        Mat binWarped532;
        threshold(warped532, binWarped532, 0, 255, THRESH_BINARY | THRESH_OTSU);

        int cornerSize = 84;
        Rect tl(0, 0, cornerSize, cornerSize);
        Rect tr(532 - cornerSize, 0, cornerSize, cornerSize);
        Rect br(532 - cornerSize, 532 - cornerSize, cornerSize, cornerSize);
        Rect bl(0, 532 - cornerSize, cornerSize, cornerSize);

        int areas[4] = {
            getBlackArea(binWarped532(tl)), getBlackArea(binWarped532(tr)),
            getBlackArea(binWarped532(br)), getBlackArea(binWarped532(bl))
        };

        int minArea = areas[0];
        int smallQrIdx = 0;
        int maxArea = areas[0];
        for (int i = 1; i < 4; ++i) {
            if (areas[i] < minArea) {
                minArea = areas[i];
                smallQrIdx = i;
            }
            if (areas[i] > maxArea) {
                maxArea = areas[i];
            }
        }

        if (maxArea == 0 || (double)minArea / maxArea > 0.5) {
            return false;
        }

        double len1 = norm(srcPointsOuter[1] - srcPointsOuter[0]);
        double len2 = norm(srcPointsOuter[3] - srcPointsOuter[0]);
        double S = std::max(len1, len2) / 133.0;
        int L = cvRound(133.0 * S);

        vector<Point2f> finalDst;
        if (smallQrIdx == 0)      finalDst = { Point2f(L,L), Point2f(0.0f,L), Point2f(0.0f,0.0f), Point2f(L,0.0f) };
        else if (smallQrIdx == 1) finalDst = { Point2f(0.0f,L), Point2f(0.0f,0.0f), Point2f(L,0.0f), Point2f(L,L) };
        else if (smallQrIdx == 3) finalDst = { Point2f(L,0.0f), Point2f(L,L), Point2f(0.0f,L), Point2f(0.0f,0.0f) };
        else                      finalDst = { Point2f(0.0f,0.0f), Point2f(L,0.0f), Point2f(L,L), Point2f(0.0f,L) };

        lastValidTransform = getPerspectiveTransform(srcPointsOuter, finalDst);
        lastValidS = S;

        warpPerspective(srcImg, disImg, lastValidTransform, Size(L, L), INTER_LINEAR);

        extractPalette(disImg);
        resizeToTarget(disImg);

        double finalS = disImg.cols / 133.0;
        drawDebug(disImg, finalS);
        drawGrid(disImg, finalS);

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

        Point2f centerTR = TR;
        Point2f centerBL = BL;
        Point2f centerTL = TL;

        Point2f pCenterOuter(0, 0);

        double S = std::max(norm(centerTR - centerTL), norm(centerBL - centerTL)) / 112.0;
        int L = cvRound(133.0 * S);

        vector<Point2f> srcPoints = { centerTL, centerTR, BR, centerBL };

        Point2f dstTL(10.0f * S, 10.0f * S);
        Point2f dstTR(122.0f * S, 10.0f * S);
        Point2f dstBL(10.0f * S, 122.0f * S);
        Point2f dstBR = foundBR ? Point2f(122.0f * S, 122.0f * S) : Point2f(122.0f * S, 122.0f * S);

        vector<Point2f> dstPoints = { dstTL, dstTR, dstBR, dstBL };

        Mat transformMatrix = getPerspectiveTransform(srcPoints, dstPoints);
        lastValidTransform = transformMatrix.clone();
        lastValidS = S;

        warpPerspective(srcImg, disImg, transformMatrix, Size(L, L), INTER_LINEAR);

        extractPalette(disImg);
        resizeToTarget(disImg);

        double finalS = disImg.cols / 133.0;
        drawDebug(disImg, finalS);
        drawGrid(disImg, finalS);

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
        if (aspect > 0.95 && aspect < 1.05 && srcImg.cols > 266) {
            disImg = srcImg.clone();

            extractPalette(disImg);
            resizeToTarget(disImg);

            double finalS = disImg.cols / 133.0;
            drawDebug(disImg, finalS);
            drawGrid(disImg, finalS);
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
            int L = cvRound(133.0 * lastValidS);
            warpPerspective(srcImg, disImg, lastValidTransform, Size(L, L), INTER_LINEAR);

            extractPalette(disImg);
            resizeToTarget(disImg);

            double finalS = disImg.cols / 133.0;
            drawDebug(disImg, finalS);
            drawGrid(disImg, finalS);
            return true;
        }

        return false;
    }

}