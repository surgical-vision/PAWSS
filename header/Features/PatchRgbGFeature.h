#ifndef PATCH_RGBG_FEATURE_H
#define PATCH_RGBG_FEATURE_H
#include "Features/PatchFeature.h"
#include "Features/RgbFeature.h"
#include "Features/GradFeature.h"
#include "segModel.h"
#include "Config.h"

class PatchRgbGFeature : public PatchFeature
{
public:
    PatchRgbGFeature(const Config& conf);
    void UpdateWeightModel(const Sample &s);

private:
    void UpdateFeatureVector(const Sample& s);
    void PrepEval(const multiSample& samples);

    int mRgbBinNum;
    int mGradBinNum;
    int mBinNum;
    Config::kernelType mKernelType;

    RgbFeature mRgbFeature;
    GradFeature mGradFeature;

    segModel mWeightModel;

    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;

    double mColorWeight;
    double mGradWeight;

};

#endif
