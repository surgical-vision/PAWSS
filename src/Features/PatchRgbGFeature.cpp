#include <PAWSS/Features/PatchRgbGFeature.h>
#include <PAWSS/mUtils.h>
#include <PAWSS/ImageRep.h>
#include <PAWSS/Sample.h>

static const int kMiniPatchRadius = 1;

PatchRgbGFeature::PatchRgbGFeature(const Config &conf) :
    mRgbFeature(), mGradFeature(),
    mWeightModel(mRgbFeature.GetRbinNum()*mRgbFeature.GetGbinNum()*mRgbFeature.GetBbinNum())
{
    mKernelType = conf.mFeatureKernelPair.mKernel;
    mSampleSize = cv::Size(-1, -1);
    mPatchNumX = conf.mPatchNumX;
    mPatchNumY = conf.mPatchNumY;
    mRgbBinNum = mRgbFeature.GetCount();
    mGradBinNum = mGradFeature.GetCount();
    mBinNum = mRgbBinNum + mGradBinNum;
    SetCount(mBinNum*mPatchNumX*mPatchNumY);

    mPatchWeightInitialized = false;
    mPatchWeights = Eigen::VectorXd::Ones(mPatchNumX*mPatchNumY);

    // todo
    mColorWeight = 0.5;
    mGradWeight = 1-mColorWeight;

#if VERBOSE
    std::cout<<"Patch rgb grad histogram bins: "<< GetCount() << std::endl;
#endif

}

void PatchRgbGFeature::PrepEval(const multiSample &samples)
{
    const int imgW = samples.getImage().GetBaseImage().cols;
    const int imgH = samples.getImage().GetBaseImage().rows;
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

    // compute rgb color bin index map
    const cv::Mat& imgColor = samples.getImage().GetColorImage();
    const uchar *p;
//    double *wp;
    rgbIndice idx;
    int rBinNum = mRgbFeature.GetRbinNum();
    int gBinNum = mRgbFeature.GetGbinNum();
//    int bBinNum = mRgbFeature.GetBbinNum();
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
            mRgbFeature.compBinIdx(pixel, idx);
            hists[idx.r_idx].at<float>(iy, ix) = 1;
            hists[idx.g_idx + rBinNum].at<float>(iy, ix) = 1;
            hists[idx.b_idx + rBinNum + gBinNum].at<float>(iy, ix) = 1;

//            int bin = idx.r_idx * gBinNum * bBinNum + idx.g_idx * bBinNum + idx.b_idx;
//            wp[ix] = mWeightModel.getForeProb(bin);
        }
    }
    for(int i=0; i<mRgbBinNum; ++i)
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
            hists[bin + mRgbBinNum].at<float>(iy, ix) = 1;
        }
    }

    for(int i=0; i<mGradBinNum; ++i)
        cv::integral(hists[i+mRgbBinNum], mIntegs[i+mRgbBinNum], CV_32F);


    for(size_t i=0; i<hists.size(); ++i)
        hists[i].release();
    hists.clear();

}

void PatchRgbGFeature::UpdateFeatureVector(const Sample &s)
{
    const IntRect rect = s.getRect();
    mFeatVec.setZero();
    // compute Patch map
    setPatchRect(cv::Size(rect.Width(), rect.Height()));
    // compute feature vector
    double fea_rgb_sum = 0.0;
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
            fea_rgb_sum += mFeatVec.segment(mBinNum*pid, mRgbBinNum).sum();
            fea_grad_sum += mFeatVec.segment(mBinNum*pid+mRgbBinNum, mGradBinNum).sum();
        }
        else if(mKernelType == Config::kKernelTypeLinear)
        {
            fea_rgb_sum += mFeatVec.segment(mBinNum*pid, mRgbBinNum).squaredNorm();
            fea_grad_sum += mFeatVec.segment(mBinNum*pid+mRgbBinNum, mGradBinNum).squaredNorm();
        }
    }

    // normalize
    if(mKernelType == Config::kKernelTypeLinear)
    {
        float weight_norm = sqrt(mColorWeight*mColorWeight + mGradWeight*mGradWeight);
        if(fea_rgb_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum*pid, mRgbBinNum) *= mColorWeight/(weight_norm * sqrt(fea_rgb_sum)); }}
        if(fea_grad_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum*pid+mRgbBinNum, mGradBinNum) *= mGradWeight/(weight_norm * sqrt(fea_grad_sum)); }}

    }
    else if(mKernelType == Config::kKernelTypeIntersection)
    {
        if(fea_rgb_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum * pid, mRgbBinNum) *= mColorWeight/fea_rgb_sum; }}
        if(fea_grad_sum != 0) {
            for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid) {
                mFeatVec.segment(mBinNum*pid+mRgbBinNum, mGradBinNum) *= mGradWeight/fea_grad_sum; }}
    }
}

//void PatchRgbGFeature::UpdateWeightModel(const Sample &s)
//{
//    const cv::Mat& rgbImg = s.getImage().GetColorImage();
//    const int imgH = rgbImg.rows;
//    const int imgW = rgbImg.cols;

//    IntRect inner_rect, bound_rect, outer_rect;
//    inner_rect =s.getRect();

//    float x_min, x_max, y_min, y_max;
//    x_min = std::max(0, int(inner_rect.XMin()-2));
//    y_min = std::max(0, int(inner_rect.YMin()-2));
//    x_max = std::min(imgW, int(inner_rect.XMax()+2));
//    y_max = std::min(imgH, int(inner_rect.YMax()+2));
//    bound_rect = IntRect(x_min, y_min, x_max-x_min, y_max-y_min);

//    x_min = std::max(0, int(inner_rect.XMin()-30));
//    y_min = std::max(0, int(inner_rect.YMin()-30));
//    x_max = std::min(imgW, int(inner_rect.XMax()+30));
//    y_max = std::min(imgH, int(inner_rect.YMax()+30));
//    outer_rect = IntRect(x_min, y_min, x_max-x_min, y_max-y_min);

//    cv::Mat binImg = cv::Mat(rgbImg.size(), CV_32SC1);
//    cv::Mat weightImg = cv::Mat(rgbImg.size(), CV_32FC1);
//    mRgbFeature.getBinImg(rgbImg, outer_rect, binImg);
//    mWeightModel.getProbImg(binImg, outer_rect, weightImg);
//    mWeightInteg = cv::Mat::zeros(imgH+1, imgW+1, CV_64FC1);
//    cv::integral(weightImg, mWeightInteg, CV_64F);

//    std::vector<IntRect> patchImgRects;
//    setPatchRect(cv::Size(inner_rect.Width(), inner_rect.Height()));
//    for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid)
//    {
//        IntRect r = mPatchRects[pid];
//        // add offset
//        r.SetXMin(r.XMin() + inner_rect.XMin());
//        r.SetYMin(r.YMin() + inner_rect.YMin());
//        patchImgRects.push_back(r);
//    }

//    // update the weight model
//    mWeightModel.updateWProb(binImg, mPatchWeights, patchImgRects, bound_rect, outer_rect);

//    // update the patch weight
//    Eigen::VectorXd patchWeights = Eigen::VectorXd::Zero(mPatchNumX*mPatchNumY);
//    for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid)
//    {
//        const IntRect& r = patchImgRects[pid];
//        double weight = ( mWeightInteg.at<double>(r.YMin(), r.XMin())
//                        + mWeightInteg.at<double>(r.YMax(), r.XMax())
//                        - mWeightInteg.at<double>(r.YMax(), r.XMin())
//                        - mWeightInteg.at<double>(r.YMin(), r.XMax()))/r.Area();
//        patchWeights[pid] = weight;
//    }
//    double wmax = patchWeights.maxCoeff();
//    mPatchWeights = (1-kAlpha) * mPatchWeights + kAlpha * patchWeights / wmax;

//    binImg.release();
//    weightImg.release();
//    patchImgRects.clear();
//}

void PatchRgbGFeature::UpdateWeightModel(const Sample &s)
{

    const cv::Mat& rgbImg = s.getImage().GetColorImage();
    const int imgH = rgbImg.rows;
    const int imgW = rgbImg.cols;

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

    cv::Mat binImg = cv::Mat::zeros(rgbImg.size(), CV_32SC1);
    cv::Mat weightImg = cv::Mat::zeros(rgbImg.size(), CV_32FC1);
    mRgbFeature.getBinImg(rgbImg, outer_rect, binImg);


    // update the weight model
    mWeightModel.updateWProb(binImg, mPatchWeights, patchImgRects, bound_rect, outer_rect);


    mWeightModel.getProbImg(binImg, outer_rect, weightImg);
    mWeightInteg = cv::Mat::zeros(imgH+1, imgW+1, CV_64FC1);
    cv::integral(weightImg, mWeightInteg, CV_64F);



//    // update the weight model
//    mWeightModel.updateWProb(binImg, mPatchWeights, patchImgRects, bound_rect, outer_rect);

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
//       // test: map weight to 2^(weight)-1
//        patchWeights[pid] = exp2(weight)-1;
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


cv::Mat PatchRgbGFeature::getWeightImg(const cv::Mat &frame, const FloatRect &bb)
{
    const cv::Mat& rgbImg = frame;
    const int imgH = rgbImg.rows;
    const int imgW = rgbImg.cols;

    IntRect outer_rect;
    float x_min, x_max, y_min, y_max;
    x_min = std::max(0, int(bb.XMin()-30));
    y_min = std::max(0, int(bb.YMin()-30));
    x_max = std::min(imgW, int(bb.XMax()+30));
    y_max = std::min(imgH, int(bb.YMax()+30));
    outer_rect = IntRect(x_min, y_min, x_max-x_min, y_max-y_min);

    cv::Mat binImg = cv::Mat::zeros(rgbImg.size(), CV_32SC1);
    cv::Mat weightImg = cv::Mat::zeros(rgbImg.size(), CV_32FC1);
    mRgbFeature.getBinImg(rgbImg, outer_rect, binImg);
    mWeightModel.getProbImg(binImg, outer_rect, weightImg);

    return weightImg;

}
