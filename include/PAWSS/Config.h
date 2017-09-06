#ifndef CONFIG_H
#define CONFIG_H
#include <ostream>
#include <vector>

#define VERBOSE (0)

class Config
{
public:
    Config() {SetDefaults(); }
    Config(const std::string& path);
    bool check();

    enum ScaleType{
        kScaleTypeOne,
        kScaleTypeGradual,
        kScaleTypeAll
    };

    enum FeatureType
    {
        kFeatureTypeNone,
        kFeatureTypePatchGray,
        kFeatureTypePatchRgb,
        kFeatureTypePatchHsv,
        kFeatureTypePatchGrad,
        kFeatureTypePatchGrayG,
        kFeatureTypePatchRgbG,
        kFeatureTypePatchHsvG,
        kFeatureTypePatchMot,
        kFeatureTypePatchRgbM,
        kFeatureTypePatchHsvM
    };

    enum kernelType
    {
        kKernelTypeLinear,
        kKernelTypeGaussian,
        kKernelTypeIntersection,
        kKernelTypeChi2
    };

    struct FeatureKernelPair
    {
        FeatureType mFeature;
        kernelType mKernel;
        std::vector<double> mParams;
    };

    bool mQuietMode;
    bool mDebugMode;
    bool mUseCamera;
    bool mSaveFrame;
    bool mSaveBB;
    bool mPrecision;
    std::string mSeqBasePath;
    std::vector<std::string> mSeqNames;
    std::string mSeqName;
    std::string mBbFileName;
    std::string mBbFileNamePrefix;
    std::string mPrecFileName;

    int mSeed;
    double mSvmC;
    int mSvmBudgetSize;
    int mPatchNumX;
    int mPatchNumY;
    float mSearchRadius;
    FeatureKernelPair mFeatureKernelPair;
    ScaleType mScaleType;
    std::string rectFilePath;

    friend std::ostream& operator << (std::ostream& out, const Config& conf);

private:
    void SetDefaults();
    static std::string FeatureName(FeatureType f);
    static std::string KernelName(kernelType k);
    static std::string ScaleName(ScaleType s);

};

#endif
