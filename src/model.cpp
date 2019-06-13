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

#include "model.h"
#include "tclc_histograms.h"

#include <limits>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>

using namespace std;
using namespace cv;

Model::Model(const string modelFilename, float tx, float ty, float tz, float alpha, float beta, float gamma, float scale)
{
    m_id = 0;
    
    initialized = false;
    
    buffersInitialsed = false;
    
    T_i = Transformations::translationMatrix(tx, ty, tz)
    *Transformations::rotationMatrix(alpha, Vec3f(1, 0, 0))
    *Transformations::rotationMatrix(beta, Vec3f(0, 1, 0))
    *Transformations::rotationMatrix(gamma, Vec3f(0, 0, 1))
    *Matx44f::eye();
    
    T_cm = T_i;
    
    scaling = scale;
    
    T_n = Matx44f::eye();
    
    hasNormals = false;
    
    vertexBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    normalBuffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    indexBuffer = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
    
    loadModel(modelFilename);
}


Model::~Model()
{
    vertices.clear();
    normals.clear();
    
    indices.clear();
    offsets.clear();
    
    if(buffersInitialsed)
    {
        vertexBuffer.release();
        vertexBuffer.destroy();
        normalBuffer.release();
        normalBuffer.destroy();
        
        indexBuffer.release();
        indexBuffer.destroy();
    }
}

void Model::initBuffers()
{
    vertexBuffer.create();
    vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexBuffer.bind();
    vertexBuffer.allocate(vertices.data(), (int)vertices.size() * sizeof(Vec3f));
    
    normalBuffer.create();
    normalBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    normalBuffer.bind();
    normalBuffer.allocate(normals.data(), (int)normals.size() * sizeof(Vec3f));
    
    indexBuffer.create();
    indexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    indexBuffer.bind();
    indexBuffer.allocate(indices.data(), (int)indices.size() * sizeof(int));
    
    buffersInitialsed = true;
}


void Model::initialize()
{
    initialized = true;
}


bool Model::isInitialized()
{
    return initialized;
}


void Model::draw(QOpenGLShaderProgram *program, GLint primitives)
{
    vertexBuffer.bind();
    program->enableAttributeArray("aPosition");
    program->setAttributeBuffer("aPosition", GL_FLOAT, 0, 3, sizeof(Vec3f));
    
    normalBuffer.bind();
    program->enableAttributeArray("aNormal");
    program->setAttributeBuffer("aNormal", GL_FLOAT, 0, 3, sizeof(Vec3f));
    
    program->enableAttributeArray("aColor");
    program->setAttributeBuffer("aColor", GL_UNSIGNED_BYTE, 0, 3, sizeof(Vec3b));
    
    indexBuffer.bind();
    
    for (uint i = 0; i < offsets.size() - 1; i++) {
        GLuint size = offsets.at(i + 1) - offsets.at(i);
        GLuint offset = offsets.at(i);
        
        glDrawElements(primitives, size, GL_UNSIGNED_INT, (GLvoid*)(offset*sizeof(GLuint)));
    }
}


Matx44f Model::getPose()
{
    return T_cm;
}

void Model::setPose(const Matx44f &T_cm)
{
    this->T_cm = T_cm;
}

void Model::setInitialPose(const Matx44f &T_cm)
{
    T_i = T_cm;
}

Matx44f Model::getNormalization()
{
    return T_n;
}


Vec3f Model::getLBN()
{
    return lbn;
}

Vec3f Model::getRTF()
{
    return rtf;
}

float Model::getScaling() {
    
    return scaling;
}


vector<Vec3f> Model::getVertices()
{
    return vertices;
}

int Model::getNumVertices()
{
    return (int)vertices.size();
}


int Model::getModelID()
{
    return m_id;
}


void Model::setModelID(int i)
{
    m_id = i;
}


void Model::reset()
{
    initialized = false;
    
    T_cm = T_i;
}


void Model::loadModel(const string modelFilename)
{
    Assimp::Importer importer;
    
    const aiScene* scene = importer.ReadFile(modelFilename, aiProcessPreset_TargetRealtime_Fast);
    
    aiMesh *mesh = scene->mMeshes[0];
    
    hasNormals = mesh->HasNormals();
    
    float inf = numeric_limits<float>::infinity();
    lbn = Vec3f(inf, inf, inf);
    rtf = Vec3f(-inf, -inf, -inf);
    
    for(int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace f = mesh->mFaces[i];
        
        indices.push_back(f.mIndices[0]);
        indices.push_back(f.mIndices[1]);
        indices.push_back(f.mIndices[2]);
    }
    
    for(int i = 0; i < mesh->mNumVertices; i++)
    {
        aiVector3D v = mesh->mVertices[i];
        
        Vec3f p(v.x, v.y, v.z);
        
        // compute the 3D bounding box of the model
        if (p[0] < lbn[0]) lbn[0] = p[0];
        if (p[1] < lbn[1]) lbn[1] = p[1];
        if (p[2] < lbn[2]) lbn[2] = p[2];
        if (p[0] > rtf[0]) rtf[0] = p[0];
        if (p[1] > rtf[1]) rtf[1] = p[1];
        if (p[2] > rtf[2]) rtf[2] = p[2];
        
        vertices.push_back(p);
    }
    
    if(hasNormals)
    {
        for(int i = 0; i < mesh->mNumVertices; i++)
        {
            aiVector3D n = mesh->mNormals[i];
            
            Vec3f vn = Vec3f(n.x, n.y, n.z);
            
            normals.push_back(vn);
        }
    }
    
    offsets.push_back(0);
    offsets.push_back(mesh->mNumFaces*3);
    
    // the center of the 3d bounding box
    Vec3f bbCenter = (rtf + lbn)/2;
    
    // compute a normalization transform that moves the object to the center of its bounding box and scales it according to the prescribed factor
    T_n = Transformations::scaleMatrix(scaling)*Transformations::translationMatrix(-bbCenter[0], -bbCenter[1], -bbCenter[2]);
    
    //T_n = Transformations::scaleMatrix(scaling);
}
