#ifndef MOT_FEATURE_H
#define MOT_FEATURE_H
#include <opencv2/opencv.hpp>
#include "Rect.h"

class MotFeature
{
public:
    MotFeature();

    inline int GetCount() const {return mBinNum; }
    int compBinIdx(const float orientation) const;
    void compMotion(const cv::Mat& prev, const cv::Mat& curr, const IntRect& rect,
                    cv::Mat& orientation, cv::Mat& mag) const;

private:
    int mBinNum;
    float mBinStep;
};

#endif
