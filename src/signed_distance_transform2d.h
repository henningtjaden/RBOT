/**
 *   #, #,         CCCCCC  VV    VV MM      MM RRRRRRR
 *  %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR
 *  %    %## #    CC        V    V  MM M  M MM RR    RR
 *   ,%      %    CC        VV  VV  MM  MM  MM RRRRRR
 *   (%      %,   CC    CC   VVVV   MM      MM RR   RR
 *     #%    %*    CCCCCC     VV    MM      MM RR    RR
 *    .%    %/
 *       (%.      Computer Vision & Mixed Reality Group
 *                For more information see <http://cvmr.info>
 *
 * This file is part of RBOT.
 *
 *  @copyright:   RheinMain University of Applied Sciences
 *                Wiesbaden Rüsselsheim
 *                Germany
 *     @author:   Henning Tjaden
 *                <henning dot tjaden at gmail dot com>
 *    @version:   1.0
 *       @date:   30.08.2018
 *
 * RBOT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RBOT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with RBOT. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SIGNED_DISTANCE_TRANSFORM2D_H
#define SIGNED_DISTANCE_TRANSFORM2D_H

#include <iostream>

#include <emmintrin.h>

#include <opencv2/core.hpp>

/**
 *  This class implements a signed 2D Euclidean distance transform
 *  of an arbitrary binary image (e.g. an object silhouette mask).
 *  Parts of this code are based on an open souce C-implementation
 *  available here:
 *  https://people.xiph.org/~tterribe/notes/edt.c
 *  It was originally published on this website:
 *  https://people.xiph.org/~tterribe/notes/edt.html
 *  However, the original code has been strongly modified, parallelized
 *  and extended, such that also the closest contour point is computed
 *  for each pixel in addition to its signed distance to it.
 */
class SignedDistanceTransform2D
{
public:
    /**
     *  Initializes the an instance with a specified maximum distance at wich the
     *  clostest contour points for every pixel are still computed.
     *
     *  @param  maxDist The maximal absolute distance at which the closest contour points are being comuted.
     */
    SignedDistanceTransform2D(float maxDist);
    
    ~SignedDistanceTransform2D();
    
    /**
     *  Computes the 2D Euclidean signed distance transform of a given input image as
     *  well as the coordinates of the clostest contour location for every pixel with
     *  CPU multi-threading.
     *
     *  @param  src The input image of which the distance transform shall be computed (single channel, float of uchar).
     *  @param  sdt The output 2D Euclidean signed distance transform of src.
     *  @param  xyPos The per pixel 2D coordinates of the closest contour points (two channel, integer).
     *  @param  threads The number of threads to be used for parallelization.
     *  @param  key In case of a uchar input image that is not binary, the value specidfies the intensitiy to be considered foregorund (default = 0, i.e. anything not equal to 0 is considered foreground).
     */
    void computeTransform(const cv::Mat &src, cv::Mat &sdt, cv::Mat &xyPos, int threads, uchar key = 0);
    
    /**
     *  Computes the first order derivatives of a given 2D Euclidean signed distance
     *  level-set in x- and y- direction at each pixel using central differences with
     *  CPU multi-threading.
     *
     *  @param  sdt The input 2D Euclidean signed distance transform of which the
     *  derivatives shall be computed (single channel, float).
     *  @param  dX The output derivatives in x-direction (single channel, float).
     *  @param  dY The output derivatives in y-direction (single channel, float).
     *  @param  threads The number of threads to be used for parallelization.
     */
    void computeDerivatives(const cv::Mat &sdt, cv::Mat &dX, cv::Mat &dY, int threads);
    
private:
    float maxDist;
};


/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, for every row in a binary input image
 *  the per pixel 1D signed distance transform is computed. Here, also the x locations of
 *  the closest contour points per pixel are calculated.
 */
template <class type>

class Parallel_For_distanceTransformRows: public cv::ParallelLoopBody
{
private:
    cv::Mat _src;
    cv::Mat _dd;
    cv::Mat _xPos;
    
    int *_v;
    int *_z;

    int _threads;
    
public:
    Parallel_For_distanceTransformRows(const cv::Mat &src, cv::Mat &dd, cv::Mat &xPos, int *v, int *z, int threads)
    {
        _src = src;
        _dd = dd;
        
        _xPos = xPos;
        
        _threads = threads;
        
        _v = v;
        _z = z;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        type *src_pixels = (type *)_src.ptr<type>();
        int *dd = (int *)_dd.ptr<int>();
        int *xPos = (int *)_xPos.ptr<int>();
        
        int range = _src.rows/_threads;
        
        int yEnd = r.end*range;
        if(r.end == _threads)
        {
            yEnd = _src.rows;
        }
        
        for(int y = r.start*range; y < yEnd; y++)
        {
            type *src_row = src_pixels + y * _src.cols;
            int *v = _v + r.start * _src.cols;
            int *z = _z + r.start * (_src.cols + 1);
            
            int j;
            int k =-1;
            for(j = 1; j < _src.cols; j++)
            {
                if(!!src_row[j-1] != !!src_row[j])
                {
                    int q;
                    int s;
                    q=(j<<1)-1;
                    s=k<0?0:((v[k]+q)>>2)+1;
                    v[++k]=q;
                    z[k]=s;
                }
            }
            if(k<0)
            {
                for(j = 0; j < _src.cols; j++)
                {
                    dd[j * _src.rows + y] = INT_MAX + !!src_row[j];
                    xPos[y * _src.cols + j] = -1;
                }
            }
            else
            {
                int zk;
                z[k+1] = _src.cols;
                j = k = 0;
                do{
                    int d1;
                    int d2;
                    d1=(j<<1)-v[k];
                    
                    int zeroPosX = (v[k]+1)/2;
                    bool bg = !src_row[zeroPosX];
                    if(bg)
                        zeroPosX -=1;
                    
                    d2=d1*d1;
                    d1=(d1+1)<<2;
                    zk=z[++k];
                    for(;;)
                    {
                        dd[j * _src.rows + y] = !src_row[j] ? d2 : -d2;
                        xPos[y * _src.cols + j] = zeroPosX;
                        
                        if(++j >= zk) break;
                        d2+=d1;
                        d1+=8;
                    }
                }
                while(zk < _src.cols);
            }
        }
    }
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, for every row in a input image
 *  the per pixel 1D signed distance transform is computed based on a given key value
 *  that specifies the intensitiy of the foreground region. Here, also the x locations of
 *  the closest contour points per pixel are calculated.
 */
class Parallel_For_distanceTransformRowsWithKey: public cv::ParallelLoopBody
{
private:
    cv::Mat _src;
    
    uchar _key;
    
    cv::Mat _dd;
    cv::Mat _xPos;
    
    int *_v;
    int *_z;
    
    int _threads;
    
public:
    Parallel_For_distanceTransformRowsWithKey(const cv::Mat &src, uchar key, cv::Mat &dd, cv::Mat &xPos, int *v, int *z, int threads)
    {
        _src = src;
        
        _key = key;
        
        _dd = dd;
        
        _xPos = xPos;
        
        _threads = threads;
        
        _v = v;
        _z = z;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        uchar *src_pixels = (uchar *)_src.ptr<uchar>();
        int *dd = (int *)_dd.ptr<int>();
        int *xPos = (int *)_xPos.ptr<int>();
        
        int range = _src.rows/_threads;
        
        int yEnd = r.end*range;
        if(r.end == _threads)
        {
            yEnd = _src.rows;
        }
        
        for(int y = r.start*range; y < yEnd; y++)
        {
            uchar *src_row = src_pixels + y * _src.cols;
            int *v = _v + r.start * _src.cols;
            int *z = _z + r.start * (_src.cols + 1);
            
            int j;
            int k =-1;
            for(j = 1; j < _src.cols; j++)
            {
                if((src_row[j-1] == _key) != (src_row[j] == _key))
                {
                    int q;
                    int s;
                    q=(j<<1)-1;
                    s=k<0?0:((v[k]+q)>>2)+1;
                    v[++k]=q;
                    z[k]=s;
                }
            }
            if(k<0)
            {
                for(j = 0; j < _src.cols; j++)
                {
                    dd[j * _src.rows + y] = INT_MAX + (src_row[j] == _key);
                    xPos[y * _src.cols + j] = -1;
                }
            }
            else
            {
                int zk;
                z[k+1] = _src.cols;
                j = k = 0;
                do{
                    int d1;
                    int d2;
                    d1=(j<<1)-v[k];
                    
                    int zeroPosX = (v[k]+1)/2;
                    bool bg = (src_row[zeroPosX] != _key);
                    if(bg)
                        zeroPosX -=1;
                    
                    d2=d1*d1;
                    d1=(d1+1)<<2;
                    zk=z[++k];
                    for(;;)
                    {
                        dd[j * _src.rows + y] = (src_row[j] != _key) ? d2 : -d2;
                        xPos[y * _src.cols + j] = zeroPosX;
                        
                        if(++j >= zk) break;
                        d2+=d1;
                        d1+=8;
                    }
                }
                while(zk < _src.cols);
            }
        }
    }
};


/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, the per pixel 2D signed distance
 *  transform is computed for every column based on the previously transformed rows.
 *  Here, also the 2D locations of the closest contour points per pixel are calculated.
 */
class Parallel_For_distanceTransformCols: public cv::ParallelLoopBody
{
private:
    cv::Mat _src;
    cv::Mat _dst;
    
    cv::Mat _xPos;
    cv::Mat _xyPos;
    
    int *_v;
    int *_z;
    int *_f;
    
    float _maxDist;
    
    int _threads;
    
public:
    Parallel_For_distanceTransformCols(const cv::Mat &src, cv::Mat &dst, const cv::Mat &xPos, cv::Mat &xyPos, float maxDist, int *v, int *z, int *f, int threads)
    {
        _src = src;
        _dst = dst;
        
        _xPos = xPos;
        _xyPos = xyPos;
        
        _maxDist = maxDist;
        
        _threads = threads;
        
        _v = v;
        _z = z;
        _f = f;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        int *dd = (int*)_src.ptr<int>();
        float *_d = (float*)_dst.ptr<float>();
        
        int *xPos = (int*)_xPos.ptr<int>();
        int *xyPos = (int*)_xyPos.ptr<int>();
        
        int range = _src.cols/_threads;
        
        int xEnd = r.end*range;
        if(r.end == _threads)
        {
            xEnd = _src.cols;
        }
        
        for(int x = r.start*range; x < xEnd; x++)
        {
            int *v = _v + r.start * _src.rows;
            int *z = _z + r.start * (_src.rows + 1);
            int *f = _f + r.start * _src.rows;
            
            int psign;
            int v2;
            int q2;
            int k=-1;
            int i;
            
            psign=dd[x*_src.rows+0]<0;
            
            for(i=0,q2=1;i<_src.rows;i++)
            {
                int sign;
                int d;
                d=dd[x*_src.rows+i];
                sign=d<0;
                if(sign!=psign)
                {
                    int q;
                    int s;
                    q=(i<<1)-1;
                    if(k<0)
                    {
                        s=0;
                    }
                    else
                    {
                        for(;;)
                        {
                            s=q2-v2-f[k];
                            if(s>0)
                            {
                                s=s/((q-v[k])<<2)+1;
                                if(s>z[k])
                                    break;
                                }
                                else
                                {
                                    s=0;
                                }
                            if(--k<0)
                                break;
                            v2=v[k]*v[k];
                        }
                    }
                    v[++k]=q;
                    f[k]=0;
                    z[k]=s;
                    v2=q2;
                }
                if(sign==d-sign+!sign<0)
                {
                    int fq;
                    int q;
                    int s;
                    int t;
                    fq=abs(d);
                    q=(i<<1)-1;
                    if(k<0)
                    {
                        s=0;
                        t=1;
                    }
                    else
                    {
                        for(;;)
                        {
                            t=(q+1-v[k])*(q+1-v[k])+f[k]-fq;
                            if(t>0)
                            {
                                s=q2-v2+fq-f[k];
                                s=s<=0?0:s/((q-v[k])<<2)+1;
                            }
                            else
                            {
                                s=(q2+(i<<3)-v2+fq-f[k])/((q+2-v[k])<<2)+1;
                            }
                            if(s>z[k]||--k<0)
                                break;
                            v2=v[k]*v[k];
                        }
                    }
                    if(t>0)
                    {
                        if(s<i)
                        {
                            v[++k]=q;
                            f[k]=fq;
                            z[k]=s;
                        }
                        v[++k]=q+1;
                        f[k]=fq;
                        z[k]=i;
                        s=i+1;
                    }
                    if(s<_src.rows)
                    {
                        v[++k]=q+2;
                        f[k]=fq;
                        z[k]=s;
                        v2=q2+(i<<3);
                    }
                }
                psign=sign;
                q2+=i<<3;
            }
            if(k<0) // NOT A SINGLE FOREGROUND PIXEL!!
            {
                for(i = 0; i < _src.rows; i++)
                {
                    _d[i*_src.cols+x] = INT_MAX;
                }
                break;
            }
            else
            {
                int zk;
                z[k+1]=_src.rows;
                i=k=0;
                do{
                    int d2;
                    int d1;
                    d1=(i<<1)-v[k];
                    d2=d1*d1+f[k];
                    
                    int zeroPosY = (v[k]+1)/2;
                    bool isSameX = f[k] == 0;
                    
                    d1=(d1+1)<<2;
                    zk=z[++k];
                    for(;;)
                    {
                        if(i >= _src.rows)
                            break;
                        float ds = sqrt(d2);
                        
                        bool bg = dd[x*_src.rows+i] > 0;
                        ds = bg ? ds : -ds;
                        ds = (ds+1)/2;
                        
                        _d[i*_src.cols+x] = ds;
                        
                        if(fabs(ds) <= _maxDist)
                        {
                            int py = (i < zeroPosY) ? zeroPosY-!bg : zeroPosY-bg;
                            
                            if(i == zeroPosY && bg && !isSameX)
                                py += 1;
                            
                            int px = 0;
                            if(isSameX)
                            {
                                px = x;
                            }
                            else
                            {
                                px = xPos[py*_xPos.cols + x];
                                
                                if(i >= zeroPosY && py > 0)
                                {
                                    int px2 = xPos[(py-1)*_xPos.cols + x];
                                    if(!(abs(x-px) <= abs(x-px2) || px2 == 0))
                                    {
                                        px = px2;
                                        py -= bg;
                                    }
                                }
                                if(i < zeroPosY && py < _xPos.rows-1)
                                {
                                    int px2 = xPos[(py+1)*_xPos.cols + x];
                                    if(!(abs(x-px) <= abs(x-px2) || px2 == 0))
                                    {
                                        px = px2;
                                        py += bg;
                                    }
                                }
                            }
                            
                            xyPos[2*(i*_xyPos.cols+x) + 0] = px;
                            xyPos[2*(i*_xyPos.cols+x) + 1] = py;
                        }
                        if(++i>=zk)break;
                        d2+=d1;
                        d1+=8;
                    }
                }
                while(zk<_src.rows);
            }
        }
    }
};


/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, for each pixel the central differences
 *  in x-direction are computed from a given 2D Euclidean singed distance transform.
 */
template <class type>
class Parallel_For_distanceTransformDX: public cv::ParallelLoopBody
{
private:
    cv::Mat _sdt;
    cv::Mat _dX;
    
    type div;
    
    int stepSize;
    
    __m128 v_div;
    
    int _threads;
    
public:
    Parallel_For_distanceTransformDX(const cv::Mat &sdt, cv::Mat &dX, int threads)
    {
        _sdt = sdt;
        _dX = dX;
        
        div = 2;
        
        stepSize = sizeof(__m128)/sizeof(type);
        
        v_div =  _mm_set1_ps(2.0);
        
        _threads = threads;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        type *sdt = (type *)_sdt.ptr<type>();
        type *dX = (type *)_dX.ptr<type>();
        
        int range = _sdt.rows/_threads;
        
        int yEnd = r.end*range;
        if(r.end == _threads)
        {
            yEnd = _sdt.rows;
        }
        
        for(int y = r.start*range; y < yEnd; y++)
        {
            int row_idx = y*_sdt.cols;
            int x;
            
            for(x = 1; x < _sdt.cols-1-stepSize; x+=stepSize)
            {
                __m128 v_b = _mm_loadu_ps(&sdt[row_idx + x - 1]);
                __m128 v_f = _mm_loadu_ps(&sdt[row_idx + x + 1]);
                
                __m128 v_diff = _mm_sub_ps(v_f, v_b);
                _mm_storeu_ps(&dX[row_idx + x], _mm_div_ps(v_diff, v_div));
            }
            
            while(x < _sdt.cols-1)
            {
                dX[row_idx + x] = (sdt[row_idx + x + 1] - sdt[row_idx + x - 1])/div;
                x++;
            }
        }
    }
};


/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, for each pixel the central differences
 *  in y-direction are computed from a given 2D Euclidean singed distance transform.
 */
template <class type>
class Parallel_For_distanceTransformDY: public cv::ParallelLoopBody
{
private:
    cv::Mat _sdt;
    cv::Mat _dY;
    
    type div;
    
    int stepSize;
    
    __m128 v_div;
    
    int _threads;
    
public:
    Parallel_For_distanceTransformDY(const cv::Mat &sdt, cv::Mat &dY, int threads)
    {
        _sdt = sdt;
        _dY = dY;
        
        div = 2;
        
        stepSize = sizeof(__m128)/sizeof(type);
        
        v_div =  _mm_set1_ps(2.0);
        
        _threads = threads;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        type *sdt = (type *)_sdt.ptr<type>();
        type *dY = (type *)_dY.ptr<type>();
        
        int range = _sdt.rows/_threads;
        
        int yEnd = r.end*range;
        if(r.end == _threads)
        {
            yEnd = _sdt.rows-1;
        }
        
        int yStart = r.start*range;
        if(r.start == 0)
        {
            yStart = 1;
        }
        
        for(int y = yStart; y < yEnd; y++)
        {
            int row_idx_prev = (y-1)*_sdt.cols;
            int row_idx = y*_sdt.cols;
            int row_idx_next = (y+1)*_sdt.cols;
            
            int x;
            for(x = 0; x < _sdt.cols-stepSize; x+=stepSize)
            {
                __m128 v_b = _mm_loadu_ps(&sdt[row_idx_prev + x]);
                __m128 v_f = _mm_loadu_ps(&sdt[row_idx_next + x]);
                
                __m128 v_diff = _mm_sub_ps(v_f, v_b);
                _mm_storeu_ps(&dY[row_idx + x], _mm_div_ps(v_diff, v_div));
            }
            
            while(x < _sdt.cols)
            {
                dY[row_idx + x] = (sdt[row_idx_next + x] - sdt[row_idx_prev + x])/div;
                x++;
            }

        }
    }
};

#endif //SIGNED_DISTANCE_TRANSFORM2D_H
