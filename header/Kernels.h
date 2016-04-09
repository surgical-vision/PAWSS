#ifndef KERNELS_H
#define KERNELS_H
#include <Eigen/Core>
#include <cmath>

class Kernel
{
public:
    virtual ~Kernel() {}
    virtual double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const = 0;
    virtual double Eval(const Eigen::VectorXd &x1) const = 0;
};

class LinearKernel : public Kernel
{
public:
    inline double Eval(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const
    {
        return x1.dot(x2);
    }
    inline double Eval(const Eigen::VectorXd &x) const
    {
//        return 1.0;
        return x.squaredNorm();
    }
};

class GaussianKernel : public Kernel
{
public:
    GaussianKernel(double sigma) : m_sigma(sigma) {}
    inline double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
    {
        return exp(-m_sigma*(x1-x2).squaredNorm());
    }

    inline double Eval(const Eigen::VectorXd& x) const
    {
        return 1.0;
    }

private:
    double m_sigma;
};

class IntersectionKernel : public Kernel
{
public:
    inline double Eval(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const
    {
        return x1.array().min(x2.array()).sum();
    }
    inline double Eval(const Eigen::VectorXd &x) const
    {
        return x.sum();
    }
};

class Chi2Kernel : public Kernel
{
public:
    inline double Eval(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
    {
        double result = 0.0;
        for (int i = 0; i < x1.size(); ++i)
        {
            double a = x1[i];
            double b = x2[i];
            result += (a-b)*(a-b)/(0.5*(a+b)+1e-8);
        }
        return 1.0 - result;
    }

    inline double Eval(const Eigen::VectorXd& x) const
    {
        return 1.0;
    }
};


#endif
