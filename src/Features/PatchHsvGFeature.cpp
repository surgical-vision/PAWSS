#include "Features/PatchHsvGFeature.h"
#include "Sample.h"
#include "mUtils.h"
#include "ImageRep.h"

static const int kMiniPatchRadius = 1;

PatchHsvGFeature::PatchHsvGFeature(const Config &conf) :
    mHsvFeature(), mGradFeature(),
    mWeightModel(mHsvFeature.GetHbinNum()*mHsvFeature.GetSbinNum()+mHsvFeature.GetVbinNum())
{
    mKernelType = conf.mFeatureKernelPair.mKernel;
    mSampleSize = cv::Size(-1, -1);
    mPatchNumX = conf.mPatchNumX;
    mPatchNumY = conf.mPatchNumY;
    mHsvBinNum = mHsvFeature.GetCount();
    mGradBinNum = mGradFeature.GetCount();
    mBinNum = mHsvBinNum + mGradBinNum;
    SetCount(mBinNum*mPatchNumX*mPatchNumY);

    mPatchWeightInitialized = false;
    mPatchWeights = Eigen::VectorXd::Ones(mPatchNumX*mPatchNumY);

    // todo
    mColorWeight = 0.5;
    mGradWeight = 1-mColorWeight;

#if VERBOSE
    std::cout<<"Patch hsv grad histogram bins: "<< GetCount() << std::endl;
#endif
}

void PatchHsvGFeature::PrepEval(const multiSample &samples)
{
    const int imgW = samples.getImage().GetBaseImage().cols;
    const int imgH = samples.getImage().GetBaseImage().rows;
    // get the union rect
    FloatRect r;
    getUnionRect(samples.getRects(), r);
    // make it slightly larger
    float x_min, y_min, x_max, y_max;
    x_min = (r.XMin()-2<=0) ? 0 : int(r.XMin()-1);
    y_min = (r.YMin()-2<=0) ? 0 : int(r.YMin()-1);
    x_max = (r.XMax()+2>=imgW) ? imgW: int(r.XMax()+1);
    y_max = (r.YMax()+2>=imgH) ? imgH: int(r.YMax()+1);
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

    // compute hsv color bin index map
    const cv::Mat& imgColor = samples.getImage().GetColorImage();
    const uchar *p;
//    double *wp;
    hsvIndice idx;
    int hBinNum = mHsvFeature.GetHbinNum();
    int sBinNum = mHsvFeature.GetSbinNum();
    for(int iy=unionRect.YMin(); iy<unionRect.YMax(); ++iy)
    {
        p = imgColor.ptr<uchar>(iy);
//        wp = weight_map.ptr<double>(iy);
        for(int ix=unionRect.XMin(); ix<unionRect.XMax(); ++ix)
        {
            // get local mini patch rect
            x_min = (ix - kMiniPatchRadius <= 0) ? 0 : ix - kMiniPatchRadius;
            y_min = (iy - kMiniPatchRadius <= 0) ? 0 : iy - kMiniPatchRadius;
            x_max = (ix + kMiniPatchRadius + 1 >= imgW ) ? imgW  : ix + kMiniPatchRadius + 1;
            y_max = (iy + kMiniPatchRadius + 1 >= imgH) ? imgH : iy + kMiniPatchRadius + 1;
            IntRect local_rect = IntRect(x_min, y_min, x_max-x_min, y_max-y_min);
            cv::Vec3b pixel;
            for(int channel=0; channel<3; ++channel)
            {
                int sum_v = 0;
                for(int y = local_rect.YMin(); y<local_rect.YMax(); ++y)
                    for(int x = local_rect.XMin(); x<local_rect.XMax(); ++x)
                    {
                        sum_v += samples.getImage().GetImage(channel).at<uchar>(y, x);
                    }

                pixel[channel] = sum_v / local_rect.Area();
            }
//            cv::Vec3b pixel(p[3*ix+0], p[3*ix+1], p[3*ix+2]);
            mHsvFeature.compBinIdx(pixel, idx);
            if(idx.h_idx<hBinNum)
            {
                hists[idx.h_idx].at<float>(iy, ix) = 1;
                hists[idx.s_idx+hBinNum].at<float>(iy, ix) = 1;
            }
            else
            {
                hists[idx.v_idx+hBinNum+sBinNum].at<float>(iy, ix) = 1;
            }
//            int bin = idx.h_idx * sBinNum + idx.s_idx + idx.v_idx;;
//            wp[ix] = mWeightModel.getForeProb(bin);
        }
    }
    for(int i=0; i<mHsvBinNum; ++i)
        cv::integral(hists[i], mIntegs[i], CV_32F);
//    cv::integral(weight_map, mWeightInteg, CV_64F);

    // todo: compute gradient map
    cv::Mat oriImg = cv::Mat::zeros(imgH, imgW, CV_32FC1);
    cv::Mat magImg = cv::Mat::zeros(imgH, imgW, CV_64FC1);
    mGradFeature.compGrad(samples.getImage().GetGrayImage(), unionRect, oriImg, magImg);

    float *op;
    for(int iy=unionRect.YMin(); iy<unionRect.YMax(); ++iy)
    {
        op = oriImg.ptr<float>(iy);
        for(int ix=unionRect.XMin(); ix<unionRect.XMax(); ++ix)
        {
            int bin = mGradFeature.compBinIdx(op[ix]);
            hists[bin + mHsvBinNum].at<float>(iy, ix) = 1;
        }
    }

    for(int i=0; i<mGradBinNum; ++i)
        cv::integral(hists[i+mHsvBinNum], mIntegs[i+mHsvBinNum], CV_32F);


    for(int i=0; i<hists.size(); ++i)
        hists[i].release();
    hists.clear();
}

void PatchHsvGFeature::UpdateFeatureVector(const Sample &s)
{
    const IntRect rect = s.getRect();
    mFeatVec.setZero();
    // compute Patch map
    setPatchRect(cv::Size(rect.Width(), rect.Height()));
    // compute feature vector
    double fea_hsv_sum = 0.0;
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
            fea_hsv_sum += mFeatVec.segment(mBinNum*pid, mHsvBinNum).sum();
            fea_grad_sum += mFeatVec.segment(mBinNum*pid+mHsvBinNum, mGradBinNum).sum();
        }
        else if(mKernelType == Config::kKernelTypeLinear)
        {
            fea_hsv_sum += mFeatVec.segment(mBinNum*pid, mHsvBinNum).squaredNorm();
            fea_grad_sum += mFeatVec.segment(mBinNum*pid+mHsvBinNum, mGradBinNum).squaredNorm();
        }
    }

    // normalize
    if(mKernelType == Config::kKernelTypeLinear)
    {
        float weight_norm = sqrt(mColorWeight*mColorWeight + mGradWeight*mGradWeight);
        if(fea_hsv_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum*pid, mHsvBinNum) *= mColorWeight/(weight_norm * sqrt(fea_hsv_sum)); }}
        if(fea_grad_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum*pid+mHsvBinNum, mGradBinNum) *= mGradWeight/(weight_norm * sqrt(fea_grad_sum)); }}

    }
    else if(mKernelType == Config::kKernelTypeIntersection)
    {
        if(fea_hsv_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum * pid, mHsvBinNum) *= mColorWeight/fea_hsv_sum; }}
        if(fea_grad_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum*pid+mHsvBinNum, mGradBinNum) *= mGradWeight/fea_grad_sum; }}
    }
}

void PatchHsvGFeature::UpdateWeightModel(const Sample &s)
{
    const cv::Mat& hsvImg = s.getImage().GetColorImage();
    const int imgH = hsvImg.rows;
    const int imgW = hsvImg.cols;

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

    cv::Mat binImg = cv::Mat::zeros(hsvImg.size(), CV_32SC1);
    cv::Mat weightImg = cv::Mat::zeros(hsvImg.size(), CV_32FC1);
    mHsvFeature.getBinImg(hsvImg, outer_rect, binImg);
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

    if(!mPatchWeightInitialized) {
        mPatchWeights = patchWeights / wmax;
        mPatchWeightInitialized = true;
    }
    else {
        mPatchWeights = (1-kAlpha) * mPatchWeights + kAlpha * patchWeights / wmax;
    }

    binImg.release();
    weightImg.release();
    patchImgRects.clear();

}



