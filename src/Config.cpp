#include <fstream>
#include <iostream>
#include <sstream>
#include "Config.h"

Config::Config(const std::string& path)
{
    SetDefaults();
    std::ifstream f(path.c_str());
    if(!f.is_open())
    {
        std::cout<< "error: could not load config file: "<< path << std::endl;
        return;
    }

    std::string line, name, tmp;
    while (getline(f, line))
    {
        std::istringstream iss(line);
        iss >> name >> tmp;

        // skip invalid lines and comments
        if (iss.fail() || tmp != "=" || name[0] == '#') continue;

        if      (name == "seed") iss >> mSeed;
        else if (name == "quietMode") iss >> mQuietMode;
        else if (name == "debugMode") iss >> mDebugMode;
        else if (name == "useCamera") iss >> mUseCamera;
        else if (name == "saveFrame") iss >> mSaveFrame;
        else if (name == "saveBoundingBox") iss >> mSaveBB;
        else if (name == "savePrecisionFile") iss >> mPrecision;
        else if (name == "sequenceBasePath") iss >> mSeqBasePath;
        else if (name == "svmC") iss >> mSvmC;
        else if (name == "svmBudgetSize") iss >> mSvmBudgetSize;
        else if (name == "patchNumX") iss >> mPatchNumX;
        else if (name == "patchNumY") iss >> mPatchNumY;
        else if (name == "seqName")
        {
            std::string sn;
            iss >> sn;
            mSeqNames.push_back(sn);
        }
        else if (name == "scaleType")
        {
            std::string scaleName;
            iss >> scaleName;
            if(scaleName == ScaleName(kScaleTypeOne)) mScaleType = kScaleTypeOne;
            else if(scaleName == ScaleName(kScaleTypeGradual)) mScaleType = kScaleTypeGradual;
            else if(scaleName == ScaleName(kScaleTypeAll)) mScaleType = kScaleTypeAll;
            else
            {
                std::cout << "error: unrecognised scale type: "<< scaleName << std::endl;
                continue;
            }
        }
        else if (name == "feature")
        {
            std::string featureName, kernelName;
            double param;
            iss >> featureName >> kernelName >> param;

            if      (featureName == FeatureName(kFeatureTypePatchGrad)) mFeatureKernelPair.mFeature = kFeatureTypePatchGrad;
            else if (featureName == FeatureName(kFeatureTypePatchGray)) mFeatureKernelPair.mFeature = kFeatureTypePatchGray;
            else if (featureName == FeatureName(kFeatureTypePatchGrayG)) mFeatureKernelPair.mFeature = kFeatureTypePatchGrayG;
            else if (featureName == FeatureName(kFeatureTypePatchHsv)) mFeatureKernelPair.mFeature = kFeatureTypePatchHsv;
            else if (featureName == FeatureName(kFeatureTypePatchHsvG)) mFeatureKernelPair.mFeature = kFeatureTypePatchHsvG;
            else if (featureName == FeatureName(kFeatureTypePatchRgb)) mFeatureKernelPair.mFeature = kFeatureTypePatchRgb;
            else if (featureName == FeatureName(kFeatureTypePatchRgbG)) mFeatureKernelPair.mFeature = kFeatureTypePatchRgbG;
            else if (featureName == FeatureName(kFeatureTypePatchMot)) mFeatureKernelPair.mFeature = kFeatureTypePatchMot;
            else if (featureName == FeatureName(kFeatureTypePatchRgbM)) mFeatureKernelPair.mFeature = kFeatureTypePatchRgbM;
            else if (featureName == FeatureName(kFeatureTypePatchHsvM)) mFeatureKernelPair.mFeature = kFeatureTypePatchHsvM;
            else
            {
                std::cout << "error: unrecognised feature: "<< featureName << std::endl;
                continue;
            }

            if      (kernelName == KernelName(kKernelTypeLinear)) mFeatureKernelPair.mKernel = kKernelTypeLinear;
            else if (kernelName == KernelName(kKernelTypeIntersection)) mFeatureKernelPair.mKernel = kKernelTypeIntersection;
            else if (kernelName == KernelName(kKernelTypeChi2)) mFeatureKernelPair.mKernel = kKernelTypeChi2;
            else if (kernelName == KernelName(kKernelTypeGaussian))
            {
                if (iss.fail())
                {
                    std::cout << "error: gaussian kernel requires a parameter (sigma)" << std::endl;
                    continue;
                }
                mFeatureKernelPair.mKernel = kKernelTypeGaussian;
                mFeatureKernelPair.mParams.push_back(param);
            }
            else
            {
                std::cout << "error: unrecognised kernel: " << kernelName << std::endl;
                continue;
            }
        }

    }

    mBbFileName = FeatureName(mFeatureKernelPair.mFeature) + "_" + std::to_string(mPatchNumX) + "_" + std::to_string(mPatchNumY) + "_" + ScaleName(mScaleType) + "_bb.txt";
    mPrecFileName = FeatureName(mFeatureKernelPair.mFeature) + "_" + std::to_string(mPatchNumX) + "_" + std::to_string(mPatchNumY) + "_" + ScaleName(mScaleType) + "_prec.txt";

    if(mUseCamera)
    {
        mSaveBB = false;
        mPrecision = false;
    }
}

bool Config::check()
{
    if(mFeatureKernelPair.mFeature == kFeatureTypeNone)
        return false;

    return true;
}

void Config::SetDefaults()
{
    mQuietMode = false;
    mDebugMode = false;
    mUseCamera = false;

    mSaveFrame = false;
    mSaveBB = false;
    mPrecision = false;

    mSeqBasePath = "sequences";
    mSeqName = "";
    mBbFileName = "_bb.txt";
    mPrecFileName = "_prec.txt";
    mSeqNames.clear();
    mSeed = 0;
    mSvmC = 1.0;
    mSvmBudgetSize = 0;
    mPatchNumX = 1;
    mPatchNumY = 1;
    mSearchRadius = 0;
    mFeatureKernelPair.mFeature = kFeatureTypeNone;
    mFeatureKernelPair.mKernel = kKernelTypeLinear;
    mScaleType = kScaleTypeAll;

}

std::string Config::FeatureName(FeatureType f)
{
    switch (f)
    {
    case kFeatureTypePatchGrad:
        return "patchGrad";
    case kFeatureTypePatchGray:
        return "patchGray";
    case kFeatureTypePatchGrayG:
        return "patchGrayGrad";
    case kFeatureTypePatchRgb:
        return "patchRgb";
    case kFeatureTypePatchRgbG:
        return "patchRgbGrad";
    case kFeatureTypePatchHsv:
        return "patchHsv";
    case kFeatureTypePatchHsvG:
        return "patchHsvGrad";
    case kFeatureTypePatchMot:
        return "patchMotion";
    case kFeatureTypePatchRgbM:
        return "patchRgbMotion";
    case kFeatureTypePatchHsvM:
        return "patchHsvMotion";
    default:
        return "";
    }
}

std::string Config::KernelName(kernelType k)
{
    switch (k)
    {
    case kKernelTypeLinear:
        return "linear";
    case kKernelTypeGaussian:
        return "gaussian";
    case kKernelTypeIntersection:
        return "intersection";
    case kKernelTypeChi2:
        return "chi2";
    default:
        return "";
    }
}

std::string Config::ScaleName(ScaleType s)
{
    switch (s) {
    case kScaleTypeOne:
        return "one";
    case kScaleTypeGradual:
        return "gradual";
    case kScaleTypeAll:
        return "all";
    default:
        return "";
    }
}

std::ostream& operator <<(std::ostream &out, const Config &conf)
{
    out << "Config:"<<std::endl;
    out << "    quiteMode           = " << conf.mQuietMode << std::endl;
    out << "    debugMode           = " << conf.mDebugMode << std::endl;
    out << "    useCamera           = " << conf.mUseCamera << std::endl;
    out << "    saveFrame           = " << conf.mSaveFrame <<std::endl;
    out << "    saveBoundingBox     = " << conf.mSaveBB << std::endl;
    out << "    savePrecisionFile   = " << conf.mPrecision << std::endl;
    out << "    sequenceBasePath    = " << conf.mSeqBasePath << std::endl;
    if(!conf.mUseCamera)
    {
        for (int i=0; i< (int) conf.mSeqNames.size(); ++i)
            out << "    sequenceName        = " << conf.mSeqNames[i] << std::endl;
    }
    out << "    seed                = " << conf.mSeed << std::endl;
    out << "    patchNumX           = " << conf.mPatchNumX << std::endl;
    out << "    patchNumY           = " << conf.mPatchNumY << std::endl;
    out << "    svmC                = " << conf.mSvmC << std::endl;
    out << "    svmBudgetSize       = " << conf.mSvmBudgetSize << std::endl;
    out << "    feature             = " << Config::FeatureName(conf.mFeatureKernelPair.mFeature) << std::endl;
    out << "    kernel              = " << Config::KernelName(conf.mFeatureKernelPair.mKernel) << std::endl;
    if(conf.mFeatureKernelPair.mParams.size() > 0)
    {
        out << "    params: ";
        for(int i=0; i<(int)conf.mFeatureKernelPair.mParams.size(); ++i)
            out << " " << conf.mFeatureKernelPair.mParams[i];
        out << std::endl;
    }
    out << "    scaleType           = " << Config::ScaleName(conf.mScaleType) << std::endl;

    return out;
}

