#include "Features/PatchFeature.h"
#include "Sample.h"
#include "mUtils.h"

void PatchFeature::Eval(const multiSample &samples, std::vector<Eigen::VectorXd> &featVecs)
{
    const_cast<PatchFeature*>(this)->PrepEval(samples);
    Feature::Eval(samples, featVecs);
}

cv::Mat PatchFeature::getPatchWeightImg(const FloatRect& bb)
{
    IntRect rect(bb);
    cv::Mat patchWeightImg = cv::Mat::zeros(rect.Height(), rect.Width(), CV_32FC1);
    // compute patch map
    setPatchRect(cv::Size(rect.Width(), rect.Height()));

    for(int pid=0; pid<mPatchNumX*mPatchNumY; ++pid)
    {
        const IntRect r = mPatchRects[pid];
        patchWeightImg(cv::Rect(r.XMin(), r.YMin(), r.Width(), r.Height())).setTo(mPatchWeights[pid]);
    }

    // colour the weight image
    cv::Mat patchWeightColorImg = colorMap(patchWeightImg);

    return patchWeightColorImg;
}

void PatchFeature::showPatchWeightFrame(const cv::Mat& frame, const int frame_idx, const FloatRect& bb)
{
    const cv::Mat& patchWeightFrame = getPatchWeightFrame(frame, bb);
    cv::imshow("patchWeightFrame", patchWeightFrame);
    // todo: save
}

void PatchFeature::showPatchWeightImg(const int frame_idx, const FloatRect &bb)
{
    const cv::Mat& patchWeightImg = getPatchWeightImg(bb);
    cv::imshow("patchWeightImage", patchWeightImg);
    // todo: save
}

cv::Mat PatchFeature::getPatchWeightFrame(const cv::Mat &frame, const FloatRect &bb)
{
    // get the patch weight image
    const cv::Mat& patchWeightImg = getPatchWeightImg(bb);

    // replace the bb of frame with patch weight image
    cv::Rect r(int(bb.XMin()), int(bb.YMin()), patchWeightImg.cols, patchWeightImg.rows);
    cv::Mat patchedFrame = frame.clone();
    cv::Mat subFrame = patchedFrame(r);
    patchWeightImg.copyTo(subFrame);

    return patchedFrame;

}

void PatchFeature::extractPatchPts(const FloatRect &bb, const int ptNumPerPatch, std::vector<cv::Point2f> &pts)
{
    setPatchRect(cv::Size(bb.Width(), bb.Height()));
    pts.clear();

    for(int i=0; i<mPatchRects.size(); ++i)
    {
        IntRect r = mPatchRects[i];
        // add offset
        r.SetXMin(r.XMin()+bb.XMin());
        r.SetYMin(r.YMin()+bb.YMin());
        // random extract points from each patch
        for(int p=0; p<ptNumPerPatch; ++p)
        {
            float pt_x =RandomFloat(r.XMin(), r.XMax());
            float pt_y =RandomFloat(r.YMin(), r.YMax());
            pts.push_back(cv::Point2f(pt_x, pt_y));
        }
    }

}


void PatchFeature::setPatchRect(const cv::Size &sample_size)
{
    if(mSampleSize == sample_size)
        return;

    mSampleSize = sample_size;
    mPatchRects.clear();
    // compute step
    int height = mSampleSize.height;
    int width = mSampleSize.width;
    float stepX = float(width) / mPatchNumX;
    float stepY = float(height) / mPatchNumY;
    int x_min, y_min, x_max, y_max;
    for(int iy=0; iy<mPatchNumY; ++iy)
        for(int ix=0; ix<mPatchNumX; ++ix)
        {
            x_min = stepX * ix;
            x_max = stepX * (ix+1);
            y_min = stepY * iy;
            y_max = stepY * (iy+1);
            mPatchRects.push_back(IntRect(x_min, y_min, x_max-x_min, y_max-y_min));
        }
}
