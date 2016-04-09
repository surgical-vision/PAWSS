#ifndef PATCH_GRAY_FEATURE_H
#define PATCH_GRAY_FEATURE_H
#include "Features/PatchFeature.h"
#include "Features/GrayFeature.h"
#include "segModel.h"
#include "Config.h"

class PatchGrayFeature : public PatchFeature
{
public:
    PatchGrayFeature(const Config& conf);
    void UpdateWeightModel(const Sample &s);

private:
    void PrepEval(const multiSample& samples);
    void UpdateFeatureVector(const Sample& s);

    int mBinNum;
    Config::kernelType mKernelType;

    GrayFeature mGrayFeature;
    segModel mWeightModel;

    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;
};

#endif
