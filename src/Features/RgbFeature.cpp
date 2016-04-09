#include "Features/RgbFeature.h"

static const int kRbinNum = 8;
static const int kGbinNum = 8;
static const int kBbinNum = 8;

RgbFeature::RgbFeature()
{
    mRbinNum = kRbinNum;
    mGbinNum = kGbinNum;
    mBbinNum = kBbinNum;
    mBinNum = mRbinNum + mGbinNum + mBbinNum;
    mRstep = 256.0 / mRbinNum;
    mGstep = 256.0 / mGbinNum;
    mBstep = 256.0 / mBbinNum;
}

void RgbFeature::compBinIdx(const cv::Vec3b &pixel, rgbIndice &index) const
{
    index.r_idx = pixel[0] / mRstep;
    index.g_idx = pixel[1] / mGstep;
    index.b_idx = pixel[2] / mBstep;
}


void RgbFeature::getBinImg(const cv::Mat &img, const IntRect &rect, cv::Mat &binImg) const
{
    assert(img.rows == binImg.rows && img.cols == binImg.cols);

    const uchar *p;
    int *bp;
    rgbIndice idx;
    for(int iy=rect.YMin(); iy<rect.YMax(); ++iy)
    {
        p = img.ptr<uchar>(iy);
        bp = binImg.ptr<int>(iy);
        for(int ix=rect.XMin(); ix<rect.XMax(); ++ix)
        {
            cv::Vec3b pixel(p[3*ix+0], p[3*ix+1], p[3*ix+2]);
            compBinIdx(pixel, idx);
            // todo:
            bp[ix] = idx.r_idx * mGbinNum * mBbinNum + idx.g_idx * mBbinNum + idx.b_idx;
        }
    }
}
