#ifndef RGB_FEATURE_H
#define RGB_FEATURE_H
#include <opencv2/opencv.hpp>
#include <PAWSS/Rect.h>

struct rgbIndice
{
    int r_idx;
    int g_idx;
    int b_idx;
};

class RgbFeature
{
public:
    RgbFeature();
    inline int GetRbinNum() const { return mRbinNum; }
    inline int GetGbinNum() const { return mGbinNum; }
    inline int GetBbinNum() const { return mBbinNum; }
    inline int GetCount() const { return mBinNum; }
    void compBinIdx(const cv::Vec3b& pixel, rgbIndice& index) const;
    void getBinImg(const cv::Mat& img, const IntRect& rect, cv::Mat& binImg) const;

private:
    int mBinNum;
    int mRbinNum;
    int mGbinNum;
    int mBbinNum;

    float mRstep;
    float mGstep;
    float mBstep;
};

#endif

