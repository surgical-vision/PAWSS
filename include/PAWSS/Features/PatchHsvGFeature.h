#ifndef PATCH_HSVG_FEATURE_H
#define PATCH_HSVG_FEATURE_H
#include <PAWSS/Features/PatchFeature.h>
#include <PAWSS/Features/HsvFeature.h>
#include <PAWSS/Features/GradFeature.h>
#include <PAWSS/Config.h>
#include <PAWSS/segModel.h>

class PatchHsvGFeature : public PatchFeature
{
public:
    PatchHsvGFeature(const Config& conf);
    void UpdateWeightModel(const Sample& s);

private:
    void UpdateFeatureVector(const Sample& s);
    void PrepEval(const multiSample& samples);


    int mHsvBinNum;
    int mGradBinNum;
    int mBinNum;
    Config::kernelType mKernelType;

    HsvFeature mHsvFeature;
    GradFeature mGradFeature;
    segModel mWeightModel;

    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;

    double mColorWeight;
    double mGradWeight;

    bool mPatchWeightInitialized;

};

#endif
