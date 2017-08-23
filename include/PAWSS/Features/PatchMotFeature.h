#ifndef PATCH_MOT_FEATURE_H
#define PATCH_MOT_FEATURE_H
#include <PAWSS/Features/PatchFeature.h>
#include <PAWSS/Features/MotFeature.h>
#include <PAWSS/Config.h>

class PatchMotFeature : public PatchFeature
{
public:
    PatchMotFeature(const Config& conf);
    void UpdateWeightModel(const Sample& s);
    void setPrevImg(ImageRep& img);

private:
    void PrepEval(const multiSample& samples);
    void UpdateFeatureVector(const Sample& s);

    int mBinNum;
    Config::kernelType mKernelType;

    MotFeature mMotFeature;
    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;

    cv::Mat mPrevImg;
    cv::Mat mCurrImg;

};

#endif
