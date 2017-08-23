#include <PAWSS/Features/PatchGrayGFeature.h>
#include <PAWSS/mUtils.h>
#include <PAWSS/ImageRep.h>
#include <PAWSS/Sample.h>

static const int kMiniPatchRadius = 3;

PatchGrayGFeature::PatchGrayGFeature(const Config &conf) :
    mGrayFeature(), mGradFeature(), mWeightModel(mGrayFeature.GetCount())
{
    mKernelType = conf.mFeatureKernelPair.mKernel;
    mSampleSize = cv::Size(-1, -1);
    mPatchNumX = conf.mPatchNumX;
    mPatchNumY = conf.mPatchNumY;
    mGrayBinNum = mGrayFeature.GetCount();
    mGradBinNum = mGradFeature.GetCount();
    mBinNum = mGrayBinNum + mGradBinNum;
    SetCount(mBinNum*mPatchNumX*mPatchNumY);

    mPatchWeights = Eigen::VectorXd::Ones(mPatchNumX*mPatchNumY);
    // todo
    mGrayWeight = 0.5;
    mGradWeight = 1-mGrayWeight;

#if VERBOSE
    std::cout<<"Patch gray grad histogram bins: "<< GetCount() << std::endl;
#endif

}

void PatchGrayGFeature::PrepEval(const multiSample &samples)
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
//    cv::Mat weight_map = cv::Mat::zeros(imgH, imgW, CV_64FC1);
//    mWeightInteg = cv::Mat::zeros(imgH+1, imgW+1, CV_64FC1);

    // compute gray color bin index map
    const uchar *p;
//    double *wp;
    int idx;
    for(int iy=unionRect.YMin(); iy<unionRect.YMax(); ++iy)
    {
        p = imgGray.ptr<uchar>(iy);
//        wp = weight_map.ptr<double>(iy);
        for(int ix=unionRect.XMin(); ix<unionRect.XMax(); ++ix)
        {
            // get local mini patch rect
            x_min = (ix - kMiniPatchRadius <= 0) ? 0 : ix - kMiniPatchRadius;
            y_min = (iy - kMiniPatchRadius <= 0) ? 0 : iy - kMiniPatchRadius;
            x_max = (ix + kMiniPatchRadius + 1 >= imgW ) ? imgW  : ix + kMiniPatchRadius + 1;
            y_max = (iy + kMiniPatchRadius + 1 >= imgH) ? imgH : iy + kMiniPatchRadius + 1;
            IntRect local_rect = IntRect(x_min, y_min, x_max-x_min, y_max-y_min);

            int sum_v = 0;
            for(int y = local_rect.YMin(); y<local_rect.YMax(); ++y)
                for(int x = local_rect.XMin(); x<local_rect.XMax(); ++x)
                {
                    sum_v += imgGray.at<uchar>(y, x);
                }

            uchar pixel = sum_v / local_rect.Area();
//            uchar pixel = p[ix];
            mGrayFeature.compBinIdx(pixel, idx);
            hists[idx].at<float>(iy, ix) = 1;
//            wp[ix] = mWeightModel.getForeProb(idx);
        }
    }
    for(int i=0; i<mGrayBinNum; ++i)
        cv::integral(hists[i], mIntegs[i], CV_32F);
//    cv::integral(weight_map, mWeightInteg, CV_64F);

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
            hists[bin + mGrayBinNum].at<float>(iy, ix) = 1;
        }
    }

    for(int i=0; i<mGradBinNum; ++i)
        cv::integral(hists[i+mGrayBinNum], mIntegs[i+mGrayBinNum], CV_32F);


    for(size_t i=0; i<hists.size(); ++i)
        hists[i].release();
    hists.clear();
}

void PatchGrayGFeature::UpdateFeatureVector(const Sample &s)
{
    const IntRect rect = s.getRect();
    mFeatVec.setZero();
    // compute Patch map
    setPatchRect(cv::Size(rect.Width(), rect.Height()));
    // compute feature vector
    double fea_gray_sum = 0.0;
    double fea_grad_sum = 0.0;
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
            fea_gray_sum += mFeatVec.segment(mBinNum*pid, mGrayBinNum).sum();
            fea_grad_sum += mFeatVec.segment(mBinNum*pid+mGrayBinNum, mGradBinNum).sum();
        }
        else if(mKernelType == Config::kKernelTypeLinear)
        {
            fea_gray_sum += mFeatVec.segment(mBinNum*pid, mGrayBinNum).squaredNorm();
            fea_grad_sum += mFeatVec.segment(mBinNum*pid+mGrayBinNum, mGradBinNum).squaredNorm();
        }
    }

    // normalize
    if(mKernelType == Config::kKernelTypeLinear)
    {
        float weight_norm = sqrt(mGrayWeight*mGrayWeight + mGradWeight*mGradWeight);
        if(fea_gray_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum*pid, mGrayBinNum) *= mGrayWeight/(weight_norm * sqrt(fea_gray_sum)); }}
        if(fea_grad_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum*pid+mGrayBinNum, mGradBinNum) *= mGradWeight/(weight_norm * sqrt(fea_grad_sum)); }}

    }
    else if(mKernelType == Config::kKernelTypeIntersection)
    {
        if(fea_gray_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum * pid, mGrayBinNum) *= mGrayWeight/fea_gray_sum; }}
        if(fea_grad_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum*pid+mGrayBinNum, mGradBinNum) *= mGradWeight/fea_grad_sum; }}
    }

}

void PatchGrayGFeature::UpdateWeightModel(const Sample &s)
{
    const cv::Mat& grayImg = s.getImage().GetGrayImage();
    const int imgH = grayImg.rows;
    const int imgW = grayImg.cols;

    IntRect inner_rect, bound_rect, outer_rect;
    inner_rect =s.getRect();

    float x_min, x_max, y_min, y_max;
    x_min = std::max(0, int(inner_rect.XMin()-2));
    y_min = std::max(0, int(inner_rect.YMin()-2));
    x_max = std::min(imgW, int(inner_rect.XMax()+2));
    y_max = std::min(imgH, int(inner_rect.YMax()+2));
    bound_rect = IntRect(x_min, y_min, x_max-x_min, y_max-y_min);

    x_min = std::max(0, int(inner_rect.XMin()-30));
    y_min = std::max(0, int(inner_rect.YMin()-30));
    x_max = std::min(imgW, int(inner_rect.XMax()+30));
    y_max = std::min(imgH, int(inner_rect.YMax()+30));
    outer_rect = IntRect(x_min, y_min, x_max-x_min, y_max-y_min);

    cv::Mat binImg = cv::Mat::zeros(grayImg.size(), CV_32SC1);
    cv::Mat weightImg = cv::Mat::zeros(grayImg.size(), CV_32FC1);
    mGrayFeature.getBinImg(grayImg, outer_rect, binImg);
    mWeightModel.getProbImg(binImg, outer_rect, weightImg);
    mWeightInteg = cv::Mat::zeros(imgH+1, imgW+1, CV_64FC1);
    cv::integral(weightImg, mWeightInteg, CV_64F);

    std::vector<IntRect> patchImgRects;
    setPatchRect(cv::Size(inner_rect.Width(), inner_rect.Height()));
    for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid)
    {
        IntRect r = mPatchRects[pid];
        // add offset
        r.SetXMin(r.XMin() + inner_rect.XMin());
        r.SetYMin(r.YMin() + inner_rect.YMin());
        patchImgRects.push_back(r);
    }

    // update the weight model
    mWeightModel.updateWProb(binImg, mPatchWeights, patchImgRects, bound_rect, outer_rect);

    // update the patch weight
    Eigen::VectorXd patchWeights = Eigen::VectorXd::Zero(mPatchNumX*mPatchNumY);
    for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid)
    {
        const IntRect& r = patchImgRects[pid];
        double weight = ( mWeightInteg.at<double>(r.YMin(), r.XMin())
                        + mWeightInteg.at<double>(r.YMax(), r.XMax())
                        - mWeightInteg.at<double>(r.YMax(), r.XMin())
                        - mWeightInteg.at<double>(r.YMin(), r.XMax()))/r.Area();
        patchWeights[pid] = weight;
    }
    double wmax = patchWeights.maxCoeff();
    mPatchWeights = (1-kAlpha) * mPatchWeights + kAlpha * patchWeights / wmax;

    binImg.release();
    weightImg.release();
    patchImgRects.clear();

}
