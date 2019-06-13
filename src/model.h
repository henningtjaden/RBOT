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

#ifndef MODEL_H
#define MODEL_H

#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "transformations.h"

/**
 *  A 3d model class based on the ASSIMP library mostly implemented
 *  wrt the OBJ/PLY file formats. The class provides functions to load the
 *  model data from a specified file, drawing the model with OpenGL
 *  as well as calculating the bounding box of the model and setting
 *  individual vertex colors. The model data is uploaded to the GPU in
 *  form of VertexBufferObjects.
 */
class Model
{
    
public:
    /**
     *  Constructor loading the 3d model data from a given OBJ/PLY file.
     *  The initial pose is computed from 3 translation parameters tx, ty, tz
     *  and 3 Euler angles alpha, beta, gamma. Here, the overall rotation
     *  matrix is composed as R(alpha)*R(beta)*R(gamma).
     *  The bounding box is also calculated during initialization.
     *
     *  @param objFilename  The relative path to an OBJ/PLY file describing the model.
     *  @param tx  The models initial translation in X-direction relative to the camera.
     *  @param ty  The models initial translation in Y-direction relative to the camera.
     *  @param tz  The models initial translation in Z-direction relative to the camera.
     *  @param alpha  The models initial Euler angle rotation about X-axis of the camera.
     *  @param beta  The models initial Euler angle rotation about Y-axis of the camera.
     *  @param gamma  The models initial Euler angle rotation about Z-axis of the camera.
     *  @param scale  A scaling factor applied to the model in order change its size independent of the original data.
     */
    Model(const std::string modelFilename, float tx, float ty, float tz, float alpha, float beta, float gamma, float scale);
    
    ~Model();
    
    /**
     *  Draws the model with a given shader programm and
     *  a specified OpenGL data primitive type using VBOs.
     *
     *  @param  program    The shader programm to be used.
     *  @param  primitives The primitive type that shall be used for drawing (e.g. GL_POINTS, GL_LINES,...). The default value is set to GL_TRIANGLES.
     */
    void draw(QOpenGLShaderProgram *program, GLint primitives = GL_TRIANGLES);
    
    /**
     *  The 3d data is packed into VOBs and uploaded to the GPU.
     *  Should be called after a valid OpenGL context exists.
     *  Must be called before a model can get rendered!
     */
    void initBuffers();
    
    /**
     *  Must be called to start pose tracking, in order to indicate
     *  that a set of tclc-histograms has been filled, such that
     *  the pose of the corresponding 3D Object can be estimated.
     *  It is also important for rendering a common scene mask in within
     *  the rendering engine. Here only initialized models are drawn.
     */
    void initialize();
    
    /**
     *  Tells whether the model has been initilaized for tracking.
     *
     *  @return True if it has been initialized and false otherwise.
     */
    bool isInitialized();
    
    /**
     *  Returns the current 6DOF rigid body transformation
     *  of the model in form of a 4x4 float matrix
     *  T_cm = [r11 r12 r13 tx]
     *         [r21 r22 r23 ty]
     *         [r31 r32 r33 tz]
     *         [  0   0   0  1],
     *  describing the transformation from model coordinates X_m
     *  into camera coordinates X_c.
     *
     *  @return  The current 6DOF pose of the model.
     */
    cv::Matx44f getPose();
    
    /**
     *  Sets the current model pose to a given 6DOF rigid body
     *  transformation in form of a 4x4 float matrix
     *  T_cm = [r11 r12 r13 tx]
     *         [r21 r22 r23 ty]
     *         [r31 r32 r33 tz]
     *         [  0   0   0  1],
     *  describing the trandformation from object coordinates X_m
     *  into camera coordinates X_c.
     *
     *  @param  T_cm The new 6DOF pose of the model.
     */
    void setPose(const cv::Matx44f &T_cm);
    
    /**
     *  Sets a new initial model pose to a given 6DOF rigid body
     *  transformation in form of a 4x4 float matrix
     *  T_cm = [r11 r12 r13 tx]
     *         [r21 r22 r23 ty]
     *         [r31 r32 r33 tz]
     *         [  0   0   0  1],
     *  describing the trandformation from object coordinates X_m
     *  into camera coordinates X_c.This pose is applied when reset()
     *  is called.
     *
     *  @param  T_cm The new initial 6DOF pose of the model.
     */
    void setInitialPose(const cv::Matx44f &T_cm);
    
    /**
     *  Returns the normalization matrix of the model. In the current
     *  implementation this matrix translates the model such that the
     *  center of its 3D bounding box is its origin and applies the
     *  prescibed scaling factor.
     *
     *  @return The normalization matrix of the model, applied before its 6DOF pose.
     */
    cv::Matx44f getNormalization();
    
    
    /**
     *  Returns the left (min(X0,... Xn-1)) bottom (min(Y0,... Yn-1))
     *  near (min(Z0,... Zn-1)) corner of the unnormalized bounding box
     *  of the model.
     *
     *  @return  The left bottom near corner of the bounding box of the model.
     */
    cv::Vec3f getLBN();
    
    /**
     *  Returns the right (max(X0,... Xn-1)) top (max(Y0,... Yn-1))
     *  far (max(Z0,... Zn-1)) corner of the unnormalized bounding box
     *  of the model.
     *
     *  @return  The right top far corner of the bounding box of the model.
     */
    cv::Vec3f getRTF();
    
    /**
     *  Returns the scaling factor specified in the contructor.
     *
     *  @return  The prescibed scaling factor.
     */
    float getScaling();
    
    /**
     *  Returns a vector containing all unnormalized 3D model
     *  verticies [X_m, Y_m, Z_m].
     *
     *  @return  A vector containing all unnormalized 3D model verticies.
     */
    std::vector<cv::Vec3f> getVertices();
    
    /**
     *  Returns the total number of 3D model verticies.
     *
     *  @return  The total number of 3D model verticies.
     */
    int getNumVertices();
    
    /**
     *  Returns the index of the model. These indices should be
     *  unique and within [1,255] as they also define the rendering
     *  intensity within the common silhouette mask.
     *
     *  @return  The index of the 3D model.
     */
    int getModelID();
    
    /**
     *  Sets the index of the model. These indices should be
     *  unique and within [1,255] as they also define the rendering
     *  intensity within the common silhouette mask.
     *
     *  @param  The index of the 3D model.
     */
    void setModelID(int i);
    
    /**
     *  Sets the current pose to previously defined the initial pose
     *  and initialization state to false.
     */
    void reset();
    
private:
    int m_id;
    
    cv::Matx44f T_i;
    
    cv::Matx44f T_cm;
    
    cv::Matx44f T_n;
    
    bool initialized;
    
    bool hasNormals;
    
    std::vector<cv::Vec3f> vertices;
    std::vector<cv::Vec3f> normals;
    std::vector<GLuint> indices;
    std::vector<GLuint> offsets;
    
    QOpenGLBuffer vertexBuffer;
    QOpenGLBuffer normalBuffer;
    QOpenGLBuffer indexBuffer;
    
    bool buffersInitialsed;
    
    cv::Vec3f lbn;
    cv::Vec3f rtf;
    float scaling;
    
    
    /**
     *  Loads the model data from the specified file.
     *
     *  @param  objFilename The relative path to the OBJ/PLY file.
     */
    void loadModel(const std::string modelFilename);
};

#endif /* MODEL_H */
