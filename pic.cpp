#include "pic.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace ImgParse {

    using namespace std;
    using namespace cv;

    struct Marker {
        Point2f center;
        double area;
    };

 
    int getBlackArea(const Mat& corner) {
        Mat binCorner;
        threshold(corner, binCorner, 0, 255, THRESH_BINARY | THRESH_OTSU);
        return (corner.rows * corner.cols) - countNonZero(binCorner);
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
            Point2f(0.0f, 0.0f), Point2f(133.0f, 0.0f),
            Point2f(133.0f, 133.0f), Point2f(0.0f, 133.0f)
        };

        Mat M_Outer = getPerspectiveTransform(srcPointsOuter, dstPointsOuter);
        Mat warped_gray;
        warpPerspective(gray, warped_gray, M_Outer, Size(133, 133), INTER_LINEAR);

        Mat binWarped;
        threshold(warped_gray, binWarped, 0, 255, THRESH_BINARY | THRESH_OTSU);

       
        int cornerSize = 21;
        Rect tl(0, 0, cornerSize, cornerSize);
        Rect tr(133 - cornerSize, 0, cornerSize, cornerSize);
        Rect br(133 - cornerSize, 133 - cornerSize, cornerSize, cornerSize);
        Rect bl(0, 133 - cornerSize, cornerSize, cornerSize);

        int areas[4] = {
            getBlackArea(binWarped(tl)), getBlackArea(binWarped(tr)),
            getBlackArea(binWarped(br)), getBlackArea(binWarped(bl))
        };

        int minArea = areas[0];
        int smallQrIdx = 0;
        for (int i = 1; i < 4; ++i) {
            if (areas[i] < minArea) {
                minArea = areas[i];
                smallQrIdx = i;
            }
        }

        Mat finalImg;
        
        if (smallQrIdx == 0) rotate(binWarped, finalImg, ROTATE_180);
        else if (smallQrIdx == 1) rotate(binWarped, finalImg, ROTATE_90_CLOCKWISE);
        else if (smallQrIdx == 3) rotate(binWarped, finalImg, ROTATE_90_COUNTERCLOCKWISE);
        else finalImg = binWarped.clone();

        // 保留彩色信息，不转换为灰度
        // 直接使用彩色图像，不进行灰度转换
        warpPerspective(srcImg, disImg, M_Outer, Size(133, 133), INTER_LINEAR);
        return true;
    }

    
    bool processV15(const Mat& srcImg, Mat& gray, Mat& disImg, bool useHSV) {
        Mat blurred, binaryForContours;

       
        if (useHSV && srcImg.channels() == 3) {
            Mat hsv, binaryMask;
            cvtColor(srcImg, hsv, COLOR_BGR2HSV);
            vector<Mat> hsv_ch;
            split(hsv, hsv_ch);

          
            threshold(hsv_ch[1], binaryMask, 180, 255, THRESH_BINARY);
            gray.setTo(255, binaryMask);
        }

        GaussianBlur(gray, blurred, Size(5, 5), 0);
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

            if (area0 < 15) continue;

            double r01 = area0 / max(area1, 1.0);
            double r12 = area1 / max(area2, 1.0);

            if (r01 > 1.2 && r01 < 8.0 && r12 > 1.2 && r12 < 8.0) {
                Moments M = moments(contours[i]);
                if (M.m00 != 0) {
                    markers.push_back({ Point2f(M.m10 / M.m00, M.m01 / M.m00), area0 });
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

        
        double len1 = norm(pt1 - TL);
        double len2 = norm(pt2 - TL);
        double legRatio = len1 / max(len2, 1.0);
        if (legRatio < 0.5 || legRatio > 2.0) return false;

        Point2f v1 = pt1 - TL;
        Point2f v2 = pt2 - TL;
        double cross = v1.x * v2.y - v1.y * v2.x;

        Point2f TR, BL;
        if (cross > 0) { TR = pt1; BL = pt2; }
        else { TR = pt2; BL = pt1; }

        Point2f BR;
        bool foundBR = false;
        Point2f expectedBR = TR + BL - TL;

        if (markers.size() > 3) {
            double minDist = 1e9;
            for (size_t i = 3; i < markers.size(); ++i) {
                double d = norm(markers[i].center - expectedBR);
                if (d < minDist) {
                    minDist = d;
                    BR = markers[i].center;
                }
            }
            if (minDist < norm(TR - TL) * 0.4) foundBR = true;
        }
        if (!foundBR) BR = expectedBR;

        vector<Point2f> srcPoints = { TL, TR, BR, BL };
        vector<Point2f> dstPoints = {
            Point2f(10.0f, 10.0f),
            Point2f(122.0f, 10.0f),
            foundBR ? Point2f(126.0f, 126.0f) : Point2f(122.0f, 122.0f),
            Point2f(10.0f, 122.0f)
        };

        Mat transformMatrix = getPerspectiveTransform(srcPoints, dstPoints);
        Mat grayWarped;
        warpPerspective(gray, grayWarped, transformMatrix, Size(133, 133), INTER_LINEAR);

        Mat binWarped;
        threshold(grayWarped, binWarped, 0, 255, THRESH_BINARY | THRESH_OTSU);

        // 保留彩色信息，不转换为灰度
        // 直接使用彩色图像，不进行灰度转换
        warpPerspective(srcImg, disImg, transformMatrix, Size(133, 133), INTER_LINEAR);
        return true;
    }

    bool Main(const cv::Mat& srcImg, cv::Mat& disImg) {
        if (srcImg.empty()) return false;

       
        static bool isFirstFrame = true;

        double aspect = (double)srcImg.cols / srcImg.rows;
        if (aspect > 0.95 && aspect < 1.05 && srcImg.cols > 266) {
            Mat grayForDigital;
            if (srcImg.channels() == 3) cvtColor(srcImg, grayForDigital, COLOR_BGR2GRAY);
            else grayForDigital = srcImg.clone();

            disImg.create(133, 133, CV_8UC3);
            Mat binRaw;
            threshold(grayForDigital, binRaw, 0, 255, THRESH_BINARY | THRESH_OTSU);

            float stepX = (float)srcImg.cols / 133.0f;
            float stepY = (float)srcImg.rows / 133.0f;
            for (int r = 0; r < 133; ++r) {
                for (int c = 0; c < 133; ++c) {
                    int px = std::min(static_cast<int>((c + 0.5f) * stepX), srcImg.cols - 1);
                    int py = std::min(static_cast<int>((r + 0.5f) * stepY), srcImg.rows - 1);
                    uint8_t val = binRaw.at<uint8_t>(py, px);
                    disImg.at<Vec3b>(r, c) = val ? Vec3b(255, 255, 255) : Vec3b(0, 0, 0);
                }
            }
            return true;
        }

        if (isFirstFrame) {
            isFirstFrame = false;
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

        return false;
    }

} // namespace ImgParse