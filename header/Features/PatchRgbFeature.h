#ifndef PATCH_RGB_FEATURE_H
#define PATCH_RGB_FEATURE_H
#include "Features/PatchFeature.h"
#include "Features/RgbFeature.h"
#include "segModel.h"
#include "Config.h"

class PatchRgbFeature : public PatchFeature
{
public:
    PatchRgbFeature(const Config& conf);
    void UpdateWeightModel(const Sample &s);
private:
    void UpdateFeatureVector(const Sample& s);
    void PrepEval(const multiSample& samples);

    int mBinNum;
    Config::kernelType mKernelType;

    RgbFeature mRgbFeature;
    segModel mWeightModel;

    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;

};

#endif
