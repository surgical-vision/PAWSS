#ifndef GRAD_FEATURE_H
#define GRAD_FEATURE_H
#include <opencv2/opencv.hpp>
#include "Rect.h"

class GradFeature
{
public:
    GradFeature();

    inline int GetCount() const { return mBinNum; }
    int compBinIdx(const float orientation) const;
    void compGrad(const cv::Mat& img, const IntRect& rect,
                  cv::Mat& orientation, cv::Mat & mag) const;
private:
    int mBinNum;
    float mBinStep;
};

#endif
