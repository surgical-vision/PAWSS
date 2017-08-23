#include <PAWSS/Features/PatchGradFeature.h>
#include <PAWSS/Sample.h>
#include <PAWSS/mUtils.h>
#include <PAWSS/ImageRep.h>

//static const int kMiniPatchRadius = 1;

PatchGradFeature::PatchGradFeature(const Config &conf) :
    mGradFeature()
{
    mKernelType = conf.mFeatureKernelPair.mKernel;
    mSampleSize = cv::Size(-1, -1);
    mPatchNumX = conf.mPatchNumX;
    mPatchNumY = conf.mPatchNumY;
    mBinNum = mGradFeature.GetCount();
    SetCount(mBinNum*mPatchNumX*mPatchNumY);

    mPatchWeights = Eigen::VectorXd::Ones(mPatchNumX*mPatchNumY);

#if VERBOSE
    std::cout<<"Patch grad histogram bins: "<< GetCount() << std::endl;
#endif
}

void PatchGradFeature::PrepEval(const multiSample &samples)
{
    const cv::Mat& imgGray = samples.getImage().GetGrayImage();
    const int imgW = imgGray.cols;
    const int imgH = imgGray.rows;
    // get the union rect
    FloatRect r;
    getUnionRect(samples.getRects(), r);
    // make it slightly larger
    float x_min, y_min, x_max, y_max;
    x_min = (r.XMin()-1<=0) ? 0 : int(r.XMin()-1);
    y_min = (r.YMin()-1<=0) ? 0 : int(r.YMin()-1);
    x_max = (r.XMax()+1>=imgW) ? imgW: int(r.XMax()+1);
    y_max = (r.YMax()+1>=imgH) ? imgH: int(r.YMax()+1);
    IntRect unionRect(x_min, y_min, x_max-x_min, y_max-y_min);

    mIntegs.clear();
    std::vector<cv::Mat> hists;
    for(int i=0; i<mBinNum; ++i)
    {
        hists.push_back(cv::Mat::zeros(imgH, imgW, CV_32FC1));
        mIntegs.push_back(cv::Mat::zeros(imgH+1, imgW+1, CV_32FC1));
    }

    // todo: compute gradient map
    cv::Mat oriImg = cv::Mat::zeros(imgH, imgW, CV_32FC1);
    cv::Mat magImg = cv::Mat::zeros(imgH, imgW, CV_64FC1);
    mGradFeature.compGrad(imgGray, unionRect, oriImg, magImg);

    float *op;
    for(int iy=unionRect.YMin(); iy<unionRect.YMax(); ++iy)
    {
        op = oriImg.ptr<float>(iy);
        for(int ix=unionRect.XMin(); ix<unionRect.XMax(); ++ix)
        {
            int bin = mGradFeature.compBinIdx(op[ix]);
//            hists[bin].at<float>(iy, ix) = 1;
            hists[bin].at<float>(iy, ix) = 1;
        }
    }

    for(int i=0; i<mBinNum; ++i)
        cv::integral(hists[i], mIntegs[i], CV_32F);

//    mWeightInteg = cv::Mat::zeros(imgH+1, imgW+1, CV_64FC1);
//    cv::integral(magImg, mWeightInteg, CV_64F);


    for(size_t i=0; i<hists.size(); ++i)
        hists[i].release();
    hists.clear();
}

void PatchGradFeature::UpdateFeatureVector(const Sample &s)
{
    const IntRect rect = s.getRect();
    mFeatVec.setZero();
    // compute Patch map
    setPatchRect(cv::Size(rect.Width(), rect.Height()));
    // compute feature vector
    double fea_sum = 0.0;
    IntRect r;
    // for each patch
    for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid)
    {
        r = mPatchRects[pid];
        // add offset
        r.SetXMin(r.XMin() + rect.XMin());
        r.SetYMin(r.YMin() + rect.YMin());
        float weight = mPatchWeights[pid];
        // for each bin
        for(int i=0; i<mBinNum; ++i)
        {
            double sum = mIntegs[i].at<float>(r.YMin(), r.XMin())
                       + mIntegs[i].at<float>(r.YMax(), r.XMax())
                       - mIntegs[i].at<float>(r.YMax(), r.XMin())
                       - mIntegs[i].at<float>(r.YMin(), r.XMax());
            mFeatVec[mBinNum * pid + i] = weight * sum / r.Area();
        }
        if(mKernelType == Config::kKernelTypeIntersection)
        {
            fea_sum += mFeatVec.segment(mBinNum*pid, mBinNum).sum();
        }
        else if(mKernelType == Config::kKernelTypeLinear)
        {
            fea_sum += mFeatVec.segment(mBinNum*pid, mBinNum).squaredNorm();
        }
    }

    // normalize
    if(mKernelType == Config::kKernelTypeLinear) {
        if(fea_sum != 0) {
            mFeatVec /= sqrt(fea_sum);} }
    else if(mKernelType == Config::kKernelTypeIntersection) {
        if(fea_sum != 0) {
            mFeatVec /= fea_sum;}}
}

void PatchGradFeature::UpdateWeightModel(const Sample &s)
{
    const cv::Mat& grayImg = s.getImage().GetGrayImage();
    const int imgH = grayImg.rows;
    const int imgW = grayImg.cols;
    const IntRect rect = s.getRect();

    cv::Mat oriImg = cv::Mat::zeros(imgH, imgW, CV_32FC1);
    cv::Mat magImg = cv::Mat::zeros(imgH, imgW, CV_64FC1);
    mGradFeature.compGrad(grayImg, rect, oriImg, magImg);
    mWeightInteg = cv::Mat::zeros(imgH+1, imgW+1, CV_64FC1);
    cv::integral(magImg, mWeightInteg, CV_64F);

    // update the patch weight
    Eigen::VectorXd patchWeights = Eigen::VectorXd::Zero(mPatchNumX*mPatchNumY);
    for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid)
    {
        IntRect r = mPatchRects[pid];
        // add offset
        r.SetXMin(r.XMin() + rect.XMin());
        r.SetYMin(r.YMin() + rect.YMin());
        double weight = ( mWeightInteg.at<double>(r.YMin(), r.XMin())
                        + mWeightInteg.at<double>(r.YMax(), r.XMax())
                        - mWeightInteg.at<double>(r.YMax(), r.XMin())
                        - mWeightInteg.at<double>(r.YMin(), r.XMax()))/r.Area();
        patchWeights[pid] = weight;
    }
    double wmax = patchWeights.maxCoeff();
    mPatchWeights = (1-kAlpha) * mPatchWeights + kAlpha * patchWeights / wmax;

}
