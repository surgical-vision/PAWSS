#ifndef GRAY_FEATURE_H
#define GRAY_FEATURE_H
#include <opencv2/opencv.hpp>
#include "Rect.h"

class GrayFeature
{
public:
    GrayFeature();
    inline int GetCount() const { return mBinNum; }
    inline void compBinIdx(const uchar& pixel, int& index) const {index = pixel/mStep; }
    inline int compBinIdx(const uchar &pixel) const {return pixel/mStep; }
    void getBinImg(const cv::Mat& img, const IntRect &rect, cv::Mat& binImg) const;

private:
    int mBinNum;
    float mStep;

};

#endif
