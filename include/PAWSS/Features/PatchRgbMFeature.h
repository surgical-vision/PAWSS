#ifndef PATCH_RGBM_FEATURE_H
#define PATCH_RGBM_FEATURE_H
#include <PAWSS/Features/PatchFeature.h>
#include <PAWSS/Features/RgbFeature.h>
#include <PAWSS/Features/MotFeature.h>
#include <PAWSS/Config.h>
#include <PAWSS/segModel.h>


class PatchRgbMFeature : public PatchFeature
{
public:
    PatchRgbMFeature(const Config& conf);
    void UpdateWeightModel(const Sample &s);
    void setPrevImg(ImageRep &img);

private:
    void UpdateFeatureVector(const Sample& s);
    void PrepEval(const multiSample& samples);

    int mRgbBinNum;
    int mMotBinNum;
    int mBinNum;
    Config::kernelType mKernelType;

    RgbFeature mRgbFeature;
    MotFeature mMotFeature;

    segModel mWeightModel;

    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;

    double mColorWeight;
    double mMotWeight;

    bool mPatchWeightInitialized;
    cv::Mat mPrevImg;

};

#endif
