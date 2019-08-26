#ifndef CYL_TYPES_HPP
#define CYL_TYPES_HPP

#ifndef __cplusplus
#  error cyl_types.hpp header must be compiled as C++
#endif

#include <climits>
#include <cfloat>
#include <vector>
#include <limits>

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/matx.hpp"

#include "opencv2/core/types.hpp"



namespace cv
{
//////////////////////////////// Point4_ ////////////////////////////////

/** @brief Template class for 4D_LF points specified by its coordinates `u`, `v`, `x` and `y`.

The following Point4_\<\> aliases are available:
@code
    typedef Point4_<int> Point4i;
    typedef Point4_<float> Point4f;
    typedef Point4_<double> Point4d;
@endcode
@see cyl::Point4i, cyl::Point4f and cyl::Point4d
*/
template<typename _Tp> class Point4_
{
public:
    typedef _Tp value_type;

    // various constructors
    Point4_();
    Point4_(_Tp _u, _Tp _v, _Tp _x, _Tp _y);
    Point4_(const Point4_& pt);
    explicit Point4_(const cv::Point_<_Tp>& pt);
    Point4_(const cv::Vec<_Tp, 4>& v);

    Point4_& operator = (const Point4_& pt);
    //! conversion to another data type
    template<typename _Tp2> operator Point4_<_Tp2>() const;
    //! conversion to cv::Vec<>
#if OPENCV_ABI_COMPATIBILITY > 300
    template<typename _Tp2> operator Vec<_Tp2, 4>() const;
#else
    operator cv::Vec<_Tp, 4>() const;
#endif

    //! dot product
    _Tp dot(const Point4_& pt) const;
    //! dot product computed in double-precision arithmetics
    double ddot(const Point4_& pt) const;
    //! cross product of the 2 4D points
    Point4_ cross(const Point4_& pt) const;

    _Tp u, v, x, y; //< the point coordinates
};

typedef Point4_<int> Point4i;
typedef Point4_<float> Point4f;
typedef Point4_<double> Point4d;

template<typename _Tp> class DataType< Point4_<_Tp> >
{
public:
    typedef Point4_<_Tp>                               value_type;
    typedef Point4_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp                                        channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 4,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef cv::Vec<channel_type, channels> vec_type;
};
  

//////////////////////////////// 4D Point ///////////////////////////////

template<typename _Tp> inline
Point4_<_Tp>::Point4_()
    : u(0), v(0), x(0), y(0) {}

template<typename _Tp> inline
Point4_<_Tp>::Point4_(_Tp _u, _Tp _v, _Tp _x, _Tp _y)
    : u(_u), v(_v), x(_x), y(_y) {}

template<typename _Tp> inline
Point4_<_Tp>::Point4_(const Point4_& pt)
    : u(pt.u), v(pt.v), x(pt.x), y(pt.y) {}

template<typename _Tp> inline
Point4_<_Tp>::Point4_(const Point_<_Tp>& pt)
    : u(pt.u), v(pt.v), x(_Tp()), y(_Tp()) {}

template<typename _Tp> inline
Point4_<_Tp>::Point4_(const Vec<_Tp, 4>& v)
    : u(v[0]), v(v[1]), x(v[2]), y(v[3]) {}

template<typename _Tp> template<typename _Tp2> inline
Point4_<_Tp>::operator Point4_<_Tp2>() const
{
    return Point4_<_Tp2>(saturate_cast<_Tp2>(u), saturate_cast<_Tp2>(v), saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y));
}

#if OPENCV_ABI_COMPATIBILITY > 300
template<typename _Tp> template<typename _Tp2> inline
Point4_<_Tp>::operator Vec<_Tp2, 4>() const
{
    return Vec<_Tp2, 4>(u, v, x, y);
}
#else
template<typename _Tp> inline
Point4_<_Tp>::operator Vec<_Tp, 4>() const
{
    return Vec<_Tp, 4>(u, v, x, y);
}
#endif

template<typename _Tp> inline
Point4_<_Tp>& Point4_<_Tp>::operator = (const Point4_& pt)
{
    u = pt.u; v = pt.v; x = pt.x; y = pt.y;
    return *this;
}

template<typename _Tp> inline
_Tp Point4_<_Tp>::dot(const Point4_& pt) const
{
    return saturate_cast<_Tp>(u*pt.u + v*pt.v + x*pt.x + y*pt.y);
}

template<typename _Tp> inline
double Point4_<_Tp>::ddot(const Point4_& pt) const
{
    return (double)u*pt.u + (double)v*pt.v + (double)x*pt.x + (double)y*pt.y;
}
  
} // cv

#endif //CYL_TYPES_HPP
