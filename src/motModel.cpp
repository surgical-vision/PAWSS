#include <numeric>
#include "motModel.h"



static const float kLambda = 0.5;
static const int kTrajLength = 5;

motModel::motModel() : mMotFeature()
{

}

motModel::~motModel()
{
}

void motModel::showMotPropMap(const cv::Mat &propMap)
{
    cv::Mat cPropMap = colorMap(propMap);
    cv::imshow("motProp", cPropMap);
}

void motModel::evalMotion(const cv::Mat &curr, const IntRect &rect, cv::Mat &flow)
{
        // get the image ready
        assert(mPrevImg.channels() == 1 || mPrevImg.channels() == 3 || curr.channels() == 1 || curr.channels() == 3);
        cv::Mat grayPrev, grayCurr;
        if(mPrevImg.channels() == 3)
        {
            grayPrev = cv::Mat(mPrevImg.rows, mPrevImg.cols, CV_8UC1);
            cv::cvtColor(mPrevImg, grayPrev, CV_RGB2GRAY);
        }
        else
        {
            grayPrev = mPrevImg;
        }
        if(curr.channels() == 3)
        {
            grayCurr = cv::Mat(curr.rows, curr.cols, CV_8UC1);
            cv::cvtColor(curr, grayCurr, CV_RGB2GRAY);
        }
        else
        {
            grayCurr = curr;
        }

        // eval dense motion
        double pyr_scale = 0.5;
        int levels = 3;
        int  winsize = 2;
        int iterations = 3;
        int poly_n = 5;
        double poly_sigma = 1.1;
        int flags = 0;
        cv::calcOpticalFlowFarneback(grayPrev, grayCurr, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);


}

void motModel::getValidMotion(const cv::Vec2f &mot, const IntRect &rect, const cv::Mat &flow, cv::Mat &propMap)
{
    // the object motion
    float trueAngle = mod(atan2(mot[1], mot[0]), 2*M_PI); // range [0, 2pi]
    float trueMag = sqrt(pow(mot[0], 2) + pow(mot[1], 2));

    std::cout << "true angle: "<<trueAngle * 180 / M_PI << " trueMag: " << trueMag << std::endl;
//    std::cout << "true mag: " << trueMag << std::endl;

    // compare the true motion with the dense motion
    assert(propMap.type() == CV_32FC1);
    propMap.setTo(0.f);
    float *pp;
    for(int iy=rect.YMin(); iy<rect.YMax(); ++iy)
    {
        pp = propMap.ptr<float>(iy);
        for(int ix=rect.XMin(); ix<rect.XMax(); ++ix)
        {
            const cv::Point2f& fxy = flow.at<cv::Point2f>(iy, ix);
            float denseAngle = mod(atan2(fxy.y, fxy.x), 2*M_PI);  // range [0, 2pi]
            float denseMag = sqrt(pow(fxy.x,2) + pow(fxy.y,2));

            if(ix == int(rect.XCentre()) && iy == int(rect.YCentre()))
            {
                std::cout << "center flowx: "<< fxy.x << " flowy: " << fxy.y << std::endl;
                std::cout << "center denseAngle: " << denseAngle * 180 / M_PI << " denseMag: " << denseMag << std::endl;
            }

            // NOTE: range [0, 2pi], needs to be convert to [0, pi]
            float rAngle = fabs(denseAngle - trueAngle);
            if(rAngle > M_PI)
            {
                rAngle = mod(rAngle-2*M_PI, 2*M_PI); // range [0, pi]
            }
            // todo: has not involved in mag...
            pp[ix] = exp(-kLambda*rAngle); // range (0, 1]
        }
    }

    // debug
    cv::Mat cflow;
    cv::cvtColor(mPrevImg, cflow, CV_GRAY2RGB);
    drawOptFlowMap(flow, cflow, 10);
    cv::imshow("flow", cflow);


    // copy setPatchRect
    float stepX = float(rect.Width()) / 8;
    float stepY = float(rect.Height()) / 8;
    std::vector<IntRect> patchRects;
    int x_min, y_min, x_max, y_max;
    for(int iy=0; iy<8; iy++)
        for(int ix=0; ix<8; ix++)
        {
            // add offset
            x_min = stepX * ix + rect.XMin();
            x_max = stepX * (ix+1) + rect.XMin();
            y_min = stepY * iy + rect.YMin();
            y_max = stepY * (iy+1) + rect.YMin();
            patchRects.push_back(IntRect(x_min, y_min, x_max-x_min, y_max-y_min));
        }
    // average patch prop
    for(int pid=0; pid<64; ++pid)
    {
        const IntRect r = patchRects[pid];
        cv::Scalar m = cv::mean(propMap(cv::Rect(r.XMin(), r.YMin(), r.Width(), r.Height())));
        float p = m[0];
        propMap(cv::Rect(r.XMin(), r.YMin(), r.Width(), r.Height())).setTo(p);
    }



//    double min, max;
//    cv::minMaxLoc(propMap, &min, &max);
//    float centerProp = propMap.at<float>(rect.YCentre(), rect.XCentre());
//    float rAngle = -log(centerProp)/kLambda * 180 / M_PI;
//    std::cout << "middle patch rAngle: "<< rAngle << std::endl;
//    std::cout << "min: "<< min << "; max: " << max << std::endl;
}

void motModel::getValidMotion1(const cv::Vec2f &mot, const IntRect &rect, const cv::Mat &flow, cv::Mat &propMap)
{
    assert(propMap.type() == CV_32FC1);

    // the object motion
    float trueDist = sqrt(pow(mot[0], 2) + pow(mot[1], 2));

    // compare the true motion with the dense motion
    propMap.setTo(0.f);
    float *pp;
    if(trueDist != 0)
    {
        for(int iy=rect.YMin(); iy<rect.YMax(); ++iy)
        {
            pp = propMap.ptr<float>(iy);
            for(int ix=rect.XMin(); ix<rect.XMax(); ++ix)
            {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(iy, ix);
                float rx = fxy.x - mot[0];
                float ry = fxy.y - mot[1];
                float rdist = sqrt(pow(rx,2)+pow(ry,2));
                // todo: has not involved in mag...
                pp[ix] = exp(-0.5*rdist/trueDist); // range (0, 1]
            }
        }
    }
    else
    {
        for(int iy=rect.YMin(); iy<rect.YMax(); ++iy)
        {
            pp = propMap.ptr<float>(iy);
            for(int ix=rect.XMin(); ix<rect.XMax(); ++ix)
            {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(iy, ix);
                float rx = fxy.x - mot[0];
                float ry = fxy.y - mot[1];
                float rdist = sqrt(pow(rx,2)+pow(ry,2));
                pp[ix] = rdist;
            }
        }

        double min, max;
        cv::minMaxLoc(propMap(cv::Rect(rect.XMin(), rect.YMin(), rect.Width(), rect.Height())), &min, &max);
        std::cout << "min: "<< min << "; max: " << max << std::endl;

        for(int iy=rect.YMin(); iy<rect.YMax(); ++iy)
        {
            pp = propMap.ptr<float>(iy);
            for(int ix=rect.XMin(); ix<rect.XMax(); ++ix)
            {
                pp[ix] = exp(-0.5*(pp[ix] - min));
            }
        }
    }

    // copy setPatchRect
    float stepX = float(rect.Width()) / 8;
    float stepY = float(rect.Height()) / 8;
    std::vector<IntRect> patchRects;
    int x_min, y_min, x_max, y_max;
    for(int iy=0; iy<8; iy++)
        for(int ix=0; ix<8; ix++)
        {
            // add offset
            x_min = stepX * ix + rect.XMin();
            x_max = stepX * (ix+1) + rect.XMin();
            y_min = stepY * iy + rect.YMin();
            y_max = stepY * (iy+1) + rect.YMin();
            patchRects.push_back(IntRect(x_min, y_min, x_max-x_min, y_max-y_min));
        }
    // average patch prop
    for(int pid=0; pid<64; ++pid)
    {
        const IntRect r = patchRects[pid];
        cv::Scalar m = cv::mean(propMap(cv::Rect(r.XMin(), r.YMin(), r.Width(), r.Height())));
        float p = m[0];
        propMap(cv::Rect(r.XMin(), r.YMin(), r.Width(), r.Height())).setTo(p);
    }


    // debug
    cv::Mat cflow;
    cv::cvtColor(mPrevImg, cflow, CV_GRAY2RGB);
    drawOptFlowMap(flow, cflow, 10);
    cv::imshow("flow", cflow);

//    double min, max;
//    cv::minMaxLoc(propMap, &min, &max);
//    std::cout << "min: "<< min << "; max: " << max << std::endl;

}

void motModel::drawOptFlowMap(const cv::Mat &flow, cv::Mat &cflowmap, int step)
{
    for(int iy=0; iy<cflowmap.rows; iy+=step)
        for(int ix=0; ix<cflowmap.cols; ix+=step)
        {
            const cv::Point2f& fxy = flow.at<cv::Point2f>(iy, ix);
            cv::line(cflowmap, cv::Point(ix, iy), cv::Point(int(ix+fxy.x), int(iy+fxy.y)), cv::Scalar(0, 255, 0));
            cv::circle(cflowmap, cv::Point(int(ix+fxy.x), int(iy+fxy.y)), 1, cv::Scalar(0, 255, 0), -1);
        }
}

void motModel::getHistogram(const cv::Mat &curr, const IntRect &rect)
{
    cv::Mat flow;
    std::vector<cv::Mat> flows;
    evalMotion(curr, rect, flow);
    for(int i=0; i<flow.channels(); ++i)
        flows.push_back(cv::Mat(flow.rows, flow.cols, CV_32FC1));
    cv::split(flow, flows);

    // transform notion to mag and angle
    cv::Mat mag(flow.rows, flow.cols, CV_32FC1);
    cv::Mat orientation(flow.rows, flow.cols, CV_32FC1);
    cv::cartToPolar(flows[0], flows[1], mag, orientation, false);

    cv::Mat objHist = cv::Mat::zeros(mMotFeature.GetCount(), 1, CV_32FC1);
    cv::Mat backHist = cv::Mat::zeros(mMotFeature.GetCount(), 1, CV_32FC1);

    const uchar* cp;
    float *op, *mp;
    for(int iy=0; iy<curr.rows; ++iy)
    {
        op = orientation.ptr<float>(iy);
        mp = mag.ptr<float>(iy);
        cp = curr.ptr<uchar>(iy);
        for(int ix=0; ix<curr.cols; ++ix)
        {
            int bin = mMotFeature.compBinIdx(op[ix]);
            // object
            if( iy>=rect.YMin() && iy<rect.YMax() && ix>=rect.XMin() && ix<rect.XMax() )
            {
//                objHist.at<float>(bin, 0) += 1;
                objHist.at<float>(bin, 0) += mp[ix];
            }
            else
            {
//                backHist.at<float>(bin, 0) += 1;
                backHist.at<float>(bin, 0) += mp[ix];
            }
        }
    }

    // draw the histogram
    int hist_w = 512; int hist_h = 400;
    cv::Mat histImg(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));
    int bin_w = int((double)hist_w/mMotFeature.GetCount());
    // normalize histogram
    cv::normalize(objHist, objHist, 0, histImg.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(backHist, backHist, 0, histImg.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // draw
    for(int i=1; i<mMotFeature.GetCount(); ++i)
    {
        cv::line(histImg, cv::Point(bin_w*(i-1), hist_h - int(objHist.at<float>(i-1, 0))),
                          cv::Point(bin_w*(i), hist_h - int(objHist.at<float>(i, 0))),
                          cv::Scalar(0, 255, 0), 2);
        cv::line(histImg, cv::Point(bin_w*(i-1), hist_h - int(backHist.at<float>(i-1, 0))),
                          cv::Point(bin_w*(i), hist_h - int(backHist.at<float>(i, 0))),
                          cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("obj/back hist", histImg);
}

// save the traj
// noted: the traj is reversed, so it can be traced back
void motModel::evalMotionT(const cv::Mat &curr)
{
    // get the image ready
    assert(mPrevImg.channels() == 1 || mPrevImg.channels() == 3 || curr.channels() == 1 || curr.channels() == 3);
    cv::Mat grayPrev, grayCurr;
    if(mPrevImg.channels() == 3)
    {
        grayPrev = cv::Mat(mPrevImg.rows, mPrevImg.cols, CV_8UC1);
        cv::cvtColor(mPrevImg, grayPrev, CV_RGB2GRAY);
    }
    else
    {
        grayPrev = mPrevImg;
    }
    if(curr.channels() == 3)
    {
        grayCurr = cv::Mat(curr.rows, curr.cols, CV_8UC1);
        cv::cvtColor(curr, grayCurr, CV_RGB2GRAY);
    }
    else
    {
        grayCurr = curr;
    }

    // eval dense motion
    double pyr_scale = 0.5;
    int levels = 3;
    int  winsize = 2;
    int iterations = 3;
    int poly_n = 5;
    double poly_sigma = 1.1;
    int flags = 0;
    cv::Mat bFlow;
    //noted: the tracking is reversed: from curr to prev frame
    cv::calcOpticalFlowFarneback(grayCurr, grayPrev, bFlow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);

    // save the reversed dense motion
    setDenseMotion(bFlow);

}

void motModel::setTrueMotion(const FloatRect &rect)
{
    mTrueBbs.push_back(rect);
    // needs to maintain at most kTrajLength + 1
    if(mTrueBbs.size() > kTrajLength+1)
        mTrueBbs.erase(mTrueBbs.begin());

    int bbLen = mTrueBbs.size();
    if(bbLen > 1)
    {
        // noted the motion is stored reversely.
        cv::Point2f m = cv::Point2f(mTrueBbs[bbLen-2].XCentre() - mTrueBbs[bbLen-1].XCentre(),
                                    mTrueBbs[bbLen-2].YCentre() - mTrueBbs[bbLen-1].YCentre());
        mTrueMots.push_back(m);
        // needs to maintain at most kTrajLength
        if(mTrueMots.size() > kTrajLength)
            mTrueMots.erase(mTrueMots.begin());
    }
}

void motModel::setDenseMotion(const cv::Mat &flow)
{
    // save the reversed dense motion
    mBFlows.push_back(flow);

    //  mBFlows needs to maintain at most kTrajLength
    if(mBFlows.size() > kTrajLength)
        mBFlows.erase(mBFlows.begin());
}

// todo
void motModel::getValidMotionT(const IntRect& rect, cv::Mat &propMap)
{
    assert(propMap.type() == CV_32FC1);

    assert(mTrueMots.size() == mBFlows.size());

    // the object motion (reversed)
    std::vector<float> trueDists;
    for(int i=0; i<mTrueMots.size(); ++i)
    {
        trueDists.push_back(sqrt(pow(mTrueMots[i].x, 2) + pow(mTrueMots[i].y, 2)));
    }

//    // method1: compare the true motion with the dense motion (two consecutive frames)
//    propMap.setTo(0.f);
//    float *pp;
//    int tx, ty;
//    for(int iy=rect.YMin(); iy<rect.YMax(); ++iy)
//    {
//        pp = propMap.ptr<float>(iy);
//        for(int ix=rect.XMin(); ix<rect.XMax(); ++ix)
//        {
//            tx = ix; ty = iy;
//            for(int id = mBFlows.size()-1; id>=0; --id)
//            {
//                const cv::Point2f& dxy = mBFlows[id].at<cv::Point2f>(ty, tx);
//                cv::Point2f rxy = dxy - mTrueMots[id];
//                float rdist = sqrt(pow(rxy.x, 2)+pow(rxy.y, 2));
//                pp[ix] += exp(-0.5*rdist/(trueDists[id]+FLT_EPSILON));
//                // get the traj for the next frame
//                tx += dxy.x;
//                ty += dxy.y;
//            }
//            pp[ix] /= mBFlows.size();
//        }
//    }

    // method2: the object motion (not consecutive frames)
    propMap.setTo(0.f);
    float *pp;
    int tx, ty;
    for(int iy=rect.YMin(); iy<rect.YMax(); ++iy)
    {
        pp = propMap.ptr<float>(iy);
        for(int ix=rect.XMin(); ix<rect.XMax(); ++ix)
        {
            tx = ix; ty = iy;
            cv::Point2f dxy(0, 0);
            cv::Point2f txy(0, 0);
            for(int id = mBFlows.size()-1; id>=0; --id)
            {
                // increment
                dxy += mBFlows[id].at<cv::Point2f>(ty, tx);
                txy += mTrueMots[id];
                cv::Point2f sxy = dxy - txy;
                float rdist = sqrt(pow(sxy.x, 2)+pow(sxy.y, 2));
                pp[ix] += exp(-0.5*rdist/(trueDists[id]+FLT_EPSILON));
                // get the traj for the next frame
                tx = ix + dxy.x;
                ty = iy + dxy.y;
                // test
                tx = fmin(fmax(0, tx), propMap.cols-1);
                ty = fmin(fmax(0, ty), propMap.rows-1);
            }
            pp[ix] /= mBFlows.size();
        }
    }

//    // copy setPatchRect
//    float stepX = float(rect.Width()) / 8;
//    float stepY = float(rect.Height()) / 8;
//    std::vector<IntRect> patchRects;
//    int x_min, y_min, x_max, y_max;
//    for(int iy=0; iy<8; iy++)
//        for(int ix=0; ix<8; ix++)
//        {
//            // add offset
//            x_min = stepX * ix + rect.XMin();
//            x_max = stepX * (ix+1) + rect.XMin();
//            y_min = stepY * iy + rect.YMin();
//            y_max = stepY * (iy+1) + rect.YMin();
//            patchRects.push_back(IntRect(x_min, y_min, x_max-x_min, y_max-y_min));
//        }
//    // average patch prop
//    for(int pid=0; pid<64; ++pid)
//    {
//        const IntRect r = patchRects[pid];
//        cv::Scalar m = cv::mean(propMap(cv::Rect(r.XMin(), r.YMin(), r.Width(), r.Height())));
//        float p = m[0];
//        propMap(cv::Rect(r.XMin(), r.YMin(), r.Width(), r.Height())).setTo(p);
//    }

//    // debug
//    cv::Mat cflow;
//    cv::cvtColor(mPrevImg, cflow, CV_GRAY2RGB);
//    drawOptFlowMap(flow, cflow, 10);
//    cv::imshow("flow", cflow);

//    double min, max;
//    cv::minMaxLoc(propMap, &min, &max);
//    std::cout << "min: "<< min << "; max: " << max << std::endl;
}






