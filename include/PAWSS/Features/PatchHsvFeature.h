#ifndef PATCH_HSV_FEATURE_H
#define PATCH_HSV_FEATURE_H
#include <PAWSS/Features/PatchFeature.h>
#include <PAWSS/Features/HsvFeature.h>
#include <PAWSS/Config.h>
#include <PAWSS/segModel.h>

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

    bool mPatchWeightInitialized;
};

#endif
