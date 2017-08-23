#ifndef PIXEL_SIM
#define PIXEL_SIM
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <PAWSS/Rect.h>
#include <PAWSS/ImageRep.h>
#include <PAWSS/mUtils.h>
#include <PAWSS/Features/MotFeature.h>

class pixelSim{
public:
    pixelSim() {}
    ~pixelSim() {}

    void showSimPropMap(const cv::Mat& propMap)
    {
        cv::Mat cPropMap = colorMap(propMap);
        cv::imshow("simMap", cPropMap);
    }

    void evalSimMap(const cv::Mat& curr_img, const IntRect& rect, cv::Mat& propMap)
    {
        propMap.setTo(0);

        float zoom_fac = 0.2;
        float x_min = fmax(0, rect.XCentre() - zoom_fac * rect.Width()/2);
        float y_min = fmax(0, rect.YCentre() - zoom_fac * rect.Height()/2);
        float x_max = fmin(curr_img.cols, rect.XCentre() + zoom_fac * rect.Width()/2);
        float y_max = fmin(curr_img.rows, rect.YCentre() + zoom_fac * rect.Height()/2);
        FloatRect centre_rect(x_min, y_min, x_max-x_min, y_max-y_min);

        mCentrePixel = compAvgPix(centre_rect, curr_img);

        zoom_fac = 2;
        x_min = fmax(0, rect.XCentre() - zoom_fac * rect.Width()/2);
        y_min = fmax(0, rect.YCentre() - zoom_fac * rect.Height()/2);
        x_max = fmin(curr_img.cols, rect.XCentre() + zoom_fac * rect.Width()/2);
        y_max = fmin(curr_img.rows, rect.YCentre() + zoom_fac * rect.Height()/2);
        FloatRect biggerBB(x_min, y_min, x_max-x_min, y_max-y_min);
        const uchar *cp;

        float *mp;
        for(int iy= biggerBB.YMin(); iy<biggerBB.YMax(); ++iy)
        {
            cp = curr_img.ptr<uchar>(iy);
            mp = propMap.ptr<float>(iy);
            for(int ix=biggerBB.XMin(); ix<biggerBB.XMax(); ++ix)
            {
                // todo get local mini patch rect
                cv::Vec3b pixel(cp[3*ix+0], cp[3*ix+1], cp[3*ix+2]);
                float sim = compSim(pixel, mCentrePixel);
                mp[ix] = sim;
            }
        }

    }

    float compSim(const cv::Vec3b& p1, const cv::Vec3b& p2)
    {
        float s[3];
        float avg_sim = 0;
        for(int i=0; i<3; ++i)
        {
            s[i] = compSim(p1[i], p2[i]);
            avg_sim += s[i];
        }
        avg_sim /= 3;
        return avg_sim;
    }

    float compSim(const uchar p1, const uchar p2)
    {
//        float r = 1 - fabs(p1 - p2) / 255.0;
//        int p = (int)(p1 - p2);
        float r = exp(-4.0*std::fabs(p1-p2)/255.0);
        return r;
    }

    cv::Vec3b compAvgPix(const IntRect& rect, const cv::Mat& img)
    {
        cv::Vec3i sum_v(0, 0, 0);
        const uchar *p;
        for(int iy = rect.YMin(); iy< rect.YMax(); ++iy)
        {
            p = img.ptr<uchar>(iy);
            for(int ix=rect.XMin(); ix<rect.XMax(); ++ix)
            {
                for(int channel=0; channel<3; ++channel)
                {
                    sum_v[channel] += p[3*ix+channel];
                }
            }
        }
        sum_v /= rect.Area();
        cv::Vec3b avg_v(sum_v[0], sum_v[1], sum_v[2]);
        return avg_v;
    }

private:
    cv::Vec3b mCentrePixel;


};

#endif
