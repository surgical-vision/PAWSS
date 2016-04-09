#ifndef PATCH_HSV_FEATURE_H
#define PATCH_HSV_FEATURE_H
#include "Features/PatchFeature.h"
#include "Features/HsvFeature.h"
#include "segModel.h"
#include "Config.h"

class PatchHsvFeature : public PatchFeature
{
public:
    PatchHsvFeature(const Config& conf);
    void UpdateWeightModel(const Sample &s);

private:
    void PrepEval(const multiSample& samples);
    void UpdateFeatureVector(const Sample& s);

    int mBinNum;
    Config::kernelType mKernelType;

    HsvFeature mHsvFeature;
    segModel mWeightModel;

    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;

};

#endif
