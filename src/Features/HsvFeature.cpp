#include <PAWSS/Features/HsvFeature.h>

static const int kHbinNum = 8;
static const int kSbinNum = 8;
static const int kVbinNum = 4;

HsvFeature::HsvFeature()
{
    mHbinNum = kHbinNum;
    mSbinNum = kSbinNum;
    mVbinNum = kVbinNum;
    mBinNum = mHbinNum + mSbinNum + mVbinNum;
    mHstep = 180.0 / mHbinNum;
    mSstep = 256.0 / mSbinNum;
    mVstep = 256.0 / mVbinNum;
}

void HsvFeature::compBinIdx(const cv::Vec3b &pixel, hsvIndice &index) const
{
    uchar h = pixel[0];
    uchar s = pixel[1];
    uchar v = pixel[2];

    if(s < 15 || v < 30)
    {
        index.h_idx = mHbinNum;
        index.s_idx = 0;
        index.v_idx = int(v/mVstep);
    }
    else
    {
        index.h_idx = int(h/mHstep);
        index.s_idx = int(s/mSstep);
        index.v_idx = 0;
    }
}

void HsvFeature::getBinImg(const cv::Mat &img, const IntRect& rect,  cv::Mat &binImg) const
{
    assert(img.rows==binImg.rows && img.cols==binImg.cols);

    const uchar *p;
    int *bp;
    hsvIndice idx;
    for(int iy=rect.YMin(); iy<rect.YMax(); ++iy)
    {
        p = img.ptr<uchar>(iy);
        bp = binImg.ptr<int>(iy);
        for(int ix=rect.XMin(); ix<rect.XMax(); ++ix)
        {
            cv::Vec3b pixel(p[3*ix+0], p[3*ix+1], p[3*ix+2]);
            compBinIdx(pixel, idx);
            bp[ix] = idx.h_idx * mSbinNum + idx.s_idx + idx.v_idx;
        }
    }
}
