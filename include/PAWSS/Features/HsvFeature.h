#ifndef HSV_FEATURE_H
#define HSV_FEATURE_H
#include <opencv2/opencv.hpp>
#include <PAWSS/Rect.h>

struct hsvIndice{
    int h_idx;
    int s_idx;
    int v_idx;
};

class HsvFeature
{
public:
    HsvFeature();
    inline int GetHbinNum() const { return mHbinNum; }
    inline int GetSbinNum() const { return mSbinNum; }
    inline int GetVbinNum() const { return mVbinNum; }
    inline int GetCount() const { return mBinNum; }
    void compBinIdx(const cv::Vec3b& pixel, hsvIndice& index) const;
    void getBinImg(const cv::Mat& img, const IntRect &rect, cv::Mat& binImg) const;
private:
    int mBinNum;
    int mHbinNum;
    int mSbinNum;
    int mVbinNum;

    float mHstep;
    float mSstep;
    float mVstep;
};


#endif
