#ifndef PATCH_GRAYG_FEATURE_H
#define PATCH_GRAYG_FEATURE_H
#include "Features/PatchFeature.h"
#include "Features/GrayFeature.h"
#include "Features/GradFeature.h"
#include "segModel.h"
#include "Config.h"

class PatchGrayGFeature : public PatchFeature
{
public:
    PatchGrayGFeature(const Config& conf);
    void UpdateWeightModel(const Sample &s);

private:
    void UpdateFeatureVector(const Sample& s);
    void PrepEval(const multiSample& samples);

    int mGrayBinNum;
    int mGradBinNum;
    int mBinNum;
    Config::kernelType mKernelType;

    GrayFeature mGrayFeature;
    GradFeature mGradFeature;
    segModel mWeightModel;

    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;

    double mGrayWeight;
    double mGradWeight;

};

#endif
