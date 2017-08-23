#ifndef PATCH_GRAD_FEATURE_H
#define PATCH_GRAD_FEATURE_H
#include <PAWSS/Features/PatchFeature.h>
#include <PAWSS/Features/GradFeature.h>
#include <PAWSS/Config.h>


class PatchGradFeature : public PatchFeature
{
public:
    PatchGradFeature(const Config& conf);
    void UpdateWeightModel(const Sample& s);

private:
    void PrepEval(const multiSample& samples);
    void UpdateFeatureVector(const Sample& s);

    int mBinNum;
    Config::kernelType mKernelType;

    GradFeature mGradFeature;

    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;

};

#endif
