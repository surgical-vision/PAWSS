#ifndef MUTILS_H
#define MUTILS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "Rect.h"

inline float getPtDist(const cv::Point& pt1, const cv::Point& pt2)
{   cv::Point rel(pt1.x-pt2.x, pt1.y-pt2.y);
    return sqrtf(rel.x*rel.x+rel.y*rel.y);}

void rectangle(const cv::Mat& rMat, const FloatRect& rRect, const cv::Scalar& rColour, int thickness=1);
void RadialSamples(const FloatRect& centre, int radius, int nr, int nt, std::vector<FloatRect>& rects);
void PixelSamples(const FloatRect& centre, int radius, bool half, std::vector<FloatRect>& rects);

void scaleFrame(const float rWidth, const float rHeight, float& scaleWidth, float& scaleHeight);
void deinterlace(cv::Mat& m);
float mod(float a, float b);
float RandomFloat(float a, float b);
void getUnionRect(const std::vector<FloatRect> &rects, FloatRect& union_r);
cv::Mat colorMap(const cv::Mat& img);

bool readFramesFile(const std::string& filePath, int& startFrame, int& endFrame);
bool readGtFile(const std::string& filePath, std::vector<FloatRect>& BBs);
bool writeResultBBFile(const std::string& filePath, const std::vector<FloatRect>& BBs);
bool writePrecisionFile(const std::string& filePath, const std::vector<float>& prec);

std::vector<float> estPrecision(const std::vector<FloatRect>& result, const std::vector<FloatRect>& gt);

void mkdir(const std::string& path);

#endif
