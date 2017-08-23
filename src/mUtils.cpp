#include <cmath>
#include <fstream>
#include <boost/filesystem.hpp>
#include <PAWSS/mUtils.h>

#define MAX_THRES 50


void rectangle(const cv::Mat& rMat, const FloatRect& rRect, const cv::Scalar& rColour, int thickness)
{
    IntRect r(rRect);
    cv::rectangle(rMat, cv::Point(r.XMin(), r.YMin()), cv::Point(r.XMax(), r.YMax()), rColour, thickness);
}

void RadialSamples(const FloatRect &centre, int radius, int nr, int nt, std::vector<FloatRect> &rects)
{
    FloatRect s(centre);
    float rstep = (float)radius/nr;
    float tstep = 2*(float)M_PI/nt;
    rects.push_back(s);

    for(int ir=1; ir<=nr; ++ir)
    {
        float phase = (ir % 2) * tstep / 2;
        for(int it = 0; it<nt; ++it)
        {
            float dx = ir*rstep*cosf(it*tstep+phase);
            float dy = ir*rstep*sinf(it*tstep+phase);
            s.SetXMin(centre.XMin()+dx);
            s.SetYMin(centre.YMin()+dy);
            rects.push_back(s);
        }
    }
}

//void PixelSamples(const FloatRect &centre, int radius, int step, std::vector<FloatRect> &rects)
//{
//    FloatRect s(centre);
//    rects.push_back(s);
//    int r2 = radius * radius;
//    for (int iy = -radius; iy <= radius; iy+=step)
//        for(int ix=-radius; ix <= radius; ix+=step)
//        {
//            if(ix*ix+iy*iy > r2) continue;
//            if(iy == 0 && ix == 0) continue;
//            if (ix % 2 == 0 || iy % 2 == 0) continue;
//            int x = (int)centre.XMin() + ix;
//            int y = (int)centre.YMin() + iy;

//            s.SetXMin(x);
//            s.SetYMin(y);
//            rects.push_back(s);
//        }
//}

void PixelSamples(const FloatRect &centre, int radius, bool half, std::vector<FloatRect> &rects)
{
    FloatRect s(centre);
    rects.push_back(s);
    int r2 = radius * radius;
    for (int iy = -radius; iy <= radius; iy++)
        for(int ix=-radius; ix <= radius; ix++)
        {
            if(ix*ix+iy*iy > r2) continue;
            if(iy == 0 && ix == 0) continue;
            if (half && (ix % 2 == 0 || iy % 2 == 0)) continue;
            int x = (int)centre.XMin() + ix;
            int y = (int)centre.YMin() + iy;

            s.SetXMin(x);
            s.SetYMin(y);
            rects.push_back(s);
        }
}


bool readFramesFile(const std::string &filePath, int &startFrame, int &endFrame)
{
    std::ifstream framesFile(filePath, std::ios::in);
    if(!framesFile)
    {
        std::cout<<"No sequence frames file: "<<filePath<<std::endl;
        return false;
    }
    else
    {
        std::string framesLine;
        getline(framesFile, framesLine);
        sscanf(framesLine.c_str(), "%d,%d", &startFrame, &endFrame);
        if(startFrame<0 || endFrame<0)
        {
            std::cout<<"could not parse sequence frames file: "<<filePath<<std::endl;
            return false;
        }
        else
            return true;
    }
}

bool readGtFile(const std::string &filePath, std::vector<FloatRect> &BBs)
{
    BBs.clear();
    std::ifstream gtFile(filePath, std::ios::in);
    if(!gtFile)
    {
        std::cout<<"could not open sequence ground truth file: "<<std::endl;
        return false;
    }

    std::string gtLine;
//    float xmin, ymin, width, height;
    while(getline(gtFile, gtLine))
    {
          std::vector<float> digits;
          std::stringstream ss(gtLine);
          float v;
          while(ss >> v)
          {
              digits.push_back(v);
              if(ss.peek() == ',' || ss.peek() == ' ')
                  ss.ignore();
          }
          if(digits.size() == 4)
          {
              FloatRect bb(digits[0], digits[1], digits[2], digits[3]);
              if(bb.Width() < 0 || bb.Height() < 0) {
                  return false; }
              else {
                  BBs.push_back(bb); }
          }
          else if(digits.size() == 8)
          {
              FloatRect bb = Polygon2Rect(digits);
              if(bb.Width() < 0 || bb.Height() < 0) {
                  return false;
              }
              else {
                  BBs.push_back(bb);
              }
          }
          else {
              return false;
          }

    }
//    while(getline(gtFile, gtLine))
//    {
//        // the gt is seperated by comma
//        sscanf(gtLine.c_str(), "%f,%f,%f,%f", &xmin, &ymin, &width, &height);
//        if(width<0 || height<0){
//            return false;}
//        else{
//            BBs.push_back(FloatRect(xmin, ymin, width, height));}
//    }
    return true;
}

bool writeResultBBFile(const std::string &filePath, const std::vector<FloatRect> &BBs)
{
    std::ofstream bbFile(filePath, std::ios::out);
    if(!bbFile)
    {
        std::cout<<"could not write to bounding box file: "<<filePath<<std::endl;
        return false;
    }

    for(auto it=BBs.begin(); it!=BBs.end(); it++)
    {
        bbFile << it->XMin() << "," << it->YMin() << "," << it->Width() << "," << it->Height() << std::endl;
    }

    if(bbFile.is_open()){
        bbFile.close(); }

    return true;
}

bool writePrecisionFile(const std::string& filePath, const std::vector<float>& prec)
{
    std::ofstream precFile(filePath, std::ios::out);
    if(!precFile)
    {
        std::cout<<"could not write to bounding box file: "<<filePath<<std::endl;
        return false;
    }

    for(auto it=prec.begin(); it!=prec.end(); it++) {
        precFile << *it << std::endl; }

    if(precFile.is_open()){
        precFile.close(); }

    return true;
}


void scaleFrame(const float rWidth, const float rHeight, float &scaleWidth, float &scaleHeight)
{
    float minLength = std::min(rWidth, rHeight);

    if(minLength <= 32.f)
    {
        scaleWidth = 32.f / minLength;
        scaleHeight = scaleWidth;
    }
    else
    {
        scaleWidth = 0.4;
        scaleHeight = scaleWidth;
    }
}

void deinterlace(cv::Mat& m)
{
    for(int i=0; i<m.rows; ++i)
        if(i%2)
            m.row(i-1).copyTo(m.row(i));
}

float mod(float a, float b)
{
    float ret =fmodf(a, b);
    if(ret < 0)
        ret += b;
    return ret;
}

float RandomFloat(float a, float b)
{
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}


void getUnionRect(const std::vector<FloatRect> &rects, FloatRect& union_r)
{
    float x_min = FLT_MAX;
    float y_min = FLT_MAX;
    float x_max = -FLT_MAX;
    float y_max = -FLT_MAX;

    FloatRect r;
    for(size_t i=0; i < rects.size(); ++i)
    {
        r = rects[i];
        if(x_min > r.XMin())
            x_min = r.XMin();
        if(x_max < r.XMax())
            x_max = r.XMax();
        if(y_min > r.YMin())
            y_min = r.YMin();
        if(y_max < r.YMax())
            y_max = r.YMax();
    }

    union_r = FloatRect(x_min, y_min, x_max-x_min, y_max-y_min);
}


void Polygon2Rect(const std::vector<float> &pts, FloatRect &rect)
{
    assert(int(pts.size()) == 8);
    float x_min = FLT_MAX;
    float y_min = FLT_MAX;
    float x_max = -FLT_MAX;
    float y_max = -FLT_MAX;
    for(size_t i=0; i< pts.size()/2; ++i)
    {
        x_min = std::min(x_min, pts[2*i]);
        y_min = std::min(y_min, pts[2*i+1]);
        x_max = std::max(x_max, pts[2*i]);
        y_max = std::max(y_max, pts[2*i+1]);
    }
    rect = FloatRect(x_min, y_min, x_max-x_min, y_max-y_min);
}



FloatRect Polygon2Rect(const std::vector<float> &pts)
{
    assert(int(pts.size()) == 8);
    float x_min = FLT_MAX;
    float y_min = FLT_MAX;
    float x_max = -FLT_MAX;
    float y_max = -FLT_MAX;
    for(size_t i=0; i< pts.size()/2; ++i)
    {
        x_min = std::min(x_min, pts[2*i]);
        y_min = std::min(y_min, pts[2*i+1]);
        x_max = std::max(x_max, pts[2*i]);
        y_max = std::max(y_max, pts[2*i+1]);
    }
    return FloatRect(x_min, y_min, x_max-x_min, y_max-y_min);
}

cv::Mat colorMap(const cv::Mat& img)
{
    double min_val, max_val;
    cv::minMaxIdx(img, &min_val, &max_val);
    float ratio = 255.0 / max_val;
    cv::Mat norm_img = cv::Mat(img.size(), CV_32F);
    norm_img = ratio * img;
    norm_img.convertTo(norm_img, CV_8UC1);

    cv::Mat col_img;
    cv::applyColorMap(norm_img, col_img, cv::COLORMAP_JET);
    return col_img;
}

void mkdir(const std::string& path)
{
    boost::filesystem::path bpath(path);
    if(!boost::filesystem::exists(bpath))
        boost::filesystem::create_directories(bpath);
}

std::vector<float> estPrecision(const std::vector<FloatRect>& result, const std::vector<FloatRect>& gt)
{
    std::vector<float> precision;
//    assert(result.size() == gt.size());
    if(result.size() != gt.size())
    {
        std::cout<<"Assertion failed: (result.size() == gt.size())."<<std::endl;
        return precision;
    }

    // calculate L2 error distance
    std::vector<float> error;
    for(size_t i=0; i<result.size(); ++i)
    {
        float err = std::sqrtf( std::pow(result[i].XCentre()-gt[i].XCentre(), 2) + std::pow(result[i].YCentre()-gt[i].YCentre(), 2) );
        error.push_back(err);
    }

    // compute presicions
    int pos_frame_num;
    std::vector<float>::iterator it;
    for(int thres=0; thres<=MAX_THRES; ++thres)
    {
        pos_frame_num = 0;
        for(it=error.begin(); it != error.end(); it++)
        {
            if(*it <= thres)
                pos_frame_num++;
        }
        precision.push_back(1.0 * pos_frame_num / error.size());
    }

    return precision;
}



