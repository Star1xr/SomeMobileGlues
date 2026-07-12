// MobileGlues - gl/drawing.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include "drawing.h"
#include "buffer.h"
#include "framebuffer.h"
#include "mg.h"
#include "texture.h"
#include <ankerl/unordered_dense.h>

#define DEBUG 0

GLuint bufSampelerProg;
GLuint bufSampelerLoc;
std::string bufSampelerName;

extern UnorderedMap<GLuint, bool> program_map_is_sampler_buffer_emulated;
extern UnorderedMap<GLuint, bool> program_map_is_atomic_counter_emulated;

UnorderedMap<GLuint, SamplerInfo> g_samplerCacheForSamplerBuffer;

struct BufferTexUniformCache {
    GLint lastTexId = 0;
    GLint lastWidth = 0;
    GLint lastHeight = 0;
};

static ankerl::unordered_dense::map<GLuint, BufferTexUniformCache> s_buffer_tex_cache;

void setupBufferTextureUniforms(GLuint program) {
    LOG_D("setupBufferTextureUniforms, program: %d", program);

    if (!program_map_is_sampler_buffer_emulated[program]) return;

    if (g_samplerCacheForSamplerBuffer.find(program) == g_samplerCacheForSamplerBuffer.end()) {
        auto& progSamplerInfo = g_samplerCacheForSamplerBuffer[program];
        GLint locWidth = GLES.glGetUniformLocation(program, "u_BufferTexWidth");
        GLint locHeight = GLES.glGetUniformLocation(program, "u_BufferTexHeight");
        if (locWidth == -1) {
            LOG_W("u_BufferTexWidth uniform not found in program %d", program);
            return;
        }

        progSamplerInfo.locHeight = locHeight;
        progSamplerInfo.locWidth = locWidth;
        progSamplerInfo.samplers.clear();

        GLint numUniforms = 0;
        GLES.glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &numUniforms);
        LOG_D("Program %d has %d active uniforms", program, numUniforms);

        for (GLint i = 0; i < numUniforms; ++i) {
            const GLsizei bufSize = 256;
            GLchar name[bufSize];
            GLsizei length = 0;
            GLint size = 0;
            GLenum type = 0;
            GLES.glGetActiveUniform(program, i, bufSize, &length, &size, &type, name);

            if (type == GL_SAMPLER_2D || type == GL_INT_SAMPLER_2D) {
                GLint locSampler = GLES.glGetUniformLocation(program, name);
                progSamplerInfo.samplers.push_back(locSampler);
            }
        }
    }

    auto& progSamplerInfo = g_samplerCacheForSamplerBuffer[program];

    GLint locWidth = progSamplerInfo.locWidth;
    GLint locHeight = progSamplerInfo.locHeight;

    GLuint prev_unit = gl_state->current_tex_unit;
    const GLint unit = 15;

    GLES.glActiveTexture(GL_TEXTURE0 + unit);
    GLint texId = 0;
    GLES.glGetIntegerv(GL_TEXTURE_BINDING_2D, &texId);

    if (texId == 0) {
        GLES.glActiveTexture(GL_TEXTURE0 + prev_unit);
        return;
    }

    auto& cache = s_buffer_tex_cache[program];
    auto texObject = mgGetTexObjectByID(texId);

    bool changed = (cache.lastTexId != texId) || (cache.lastWidth != texObject->width) ||
                   (cache.lastHeight != texObject->height);

    if (changed) {
        cache.lastTexId = texId;
        cache.lastWidth = texObject->width;
        cache.lastHeight = texObject->height;

        for (auto locSampler : progSamplerInfo.samplers) {
            if (locSampler >= 0) {
                GLES.glUniform1i(locSampler, unit);
            }
        }
        GLES.glUniform1i(locWidth, texObject->width);
        GLES.glUniform1i(locHeight, texObject->height);
    }

    GLES.glActiveTexture(GL_TEXTURE0 + prev_unit);
}

QuadConvBuf s_quad_conv;

extern "C" void ensureQuadConvBuffer(GLsizeiptr needed) {
    if (!s_quad_conv.ebo) {
        GLES.glGenBuffers(1, &s_quad_conv.ebo);
    }
    if (needed > s_quad_conv.capacity) {
        GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_quad_conv.ebo);
        GLES.glBufferData(GL_ELEMENT_ARRAY_BUFFER, needed, nullptr, GL_DYNAMIC_DRAW);
        s_quad_conv.capacity = needed;
    }
}

// Converts a GL_QUADS draw to GL_TRIANGLES for non-indexed draws
extern "C" void drawArraysQuads(GLenum mode, GLint first, GLsizei count) {
    if (count < 4 || (count % 4) != 0) return;
    GLsizei quadCount = count / 4;
    GLsizei triCount = quadCount * 6;
    GLsizeiptr bufSize = triCount * sizeof(GLuint);

    ensureQuadConvBuffer(bufSize);

    void* mapped = GLES.glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0, bufSize,
                                          GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if (!mapped) return;

    GLuint* out = (GLuint*)mapped;
    for (GLsizei q = 0; q < quadCount; ++q) {
        GLuint base = first + q * 4;
        out[q * 6 + 0] = base + 0;
        out[q * 6 + 1] = base + 1;
        out[q * 6 + 2] = base + 2;
        out[q * 6 + 3] = base + 0;
        out[q * 6 + 4] = base + 2;
        out[q * 6 + 5] = base + 3;
    }
    GLES.glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    GLES.glDrawElements(GL_TRIANGLES, triCount, GL_UNSIGNED_INT, 0);
}

// Converts a GL_QUADS indexed draw to GL_TRIANGLES
template<typename T>
static void convertQuadIndicesT(const void* srcIndices, GLsizei count, void* dst,
                                 GLint basevertex, bool hasBaseVertex) {
    const T* src = (const T*)srcIndices;
    T* dst_ = (T*)dst;
    GLsizei quadCount = count / 4;
    for (GLsizei q = 0; q < quadCount; ++q) {
        T i0 = src[q * 4 + 0];
        T i1 = src[q * 4 + 1];
        T i2 = src[q * 4 + 2];
        T i3 = src[q * 4 + 3];
        if (hasBaseVertex) {
            i0 += basevertex; i1 += basevertex; i2 += basevertex; i3 += basevertex;
        }
        dst_[q * 6 + 0] = i0;
        dst_[q * 6 + 1] = i1;
        dst_[q * 6 + 2] = i2;
        dst_[q * 6 + 3] = i0;
        dst_[q * 6 + 4] = i2;
        dst_[q * 6 + 5] = i3;
    }
}

extern "C" void drawElementsQuads(GLenum mode, GLsizei count, GLenum type, const void* indices,
                               GLint basevertex, bool hasBaseVertex) {
    if (count < 4 || (count % 4) != 0) return;
    GLsizei quadCount = count / 4;
    GLsizei triCount = quadCount * 6;

    size_t indexSize;
    switch (type) {
    case GL_UNSIGNED_BYTE:  indexSize = 1; break;
    case GL_UNSIGNED_SHORT: indexSize = 2; break;
    case GL_UNSIGNED_INT:   indexSize = 4; break;
    default: return;
    }

    GLsizeiptr bufSize = triCount * indexSize;
    ensureQuadConvBuffer(bufSize);

    GLint prevEbo;
    GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &prevEbo);
    GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_quad_conv.ebo);

    void* mapped = GLES.glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0, bufSize,
                                          GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if (!mapped) {
        GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevEbo);
        return;
    }

    if (type == GL_UNSIGNED_BYTE)
        convertQuadIndicesT<GLubyte>(indices, count, mapped, basevertex, hasBaseVertex);
    else if (type == GL_UNSIGNED_SHORT)
        convertQuadIndicesT<GLushort>(indices, count, mapped, basevertex, hasBaseVertex);
    else
        convertQuadIndicesT<GLuint>(indices, count, mapped, basevertex, hasBaseVertex);

    GLES.glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    GLES.glDrawElements(GL_TRIANGLES, triCount, type, 0);

    GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevEbo);
}

void prepareForDraw() {
    LOG_D("prepareForDraw...")
    if (hardware->emulate_texture_buffer) {
        setupBufferTextureUniforms(gl_state->current_program);
    }
}

void glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei primcount) {
    LOG()
    LOG_D("glDrawElementsInstanced, mode: %d, count: %d, type: %d, indices: %p, primcount: %d", mode, count, type,
          indices, primcount)
    prepareForDraw();
    if (mode == GL_QUADS) {
        if (count < 4 || (count % 4) != 0) return;
        GLsizei quadCount = count / 4;
        GLsizei triCount = quadCount * 6;
        size_t indexSize;
        switch (type) {
        case GL_UNSIGNED_BYTE:  indexSize = 1; break;
        case GL_UNSIGNED_SHORT: indexSize = 2; break;
        case GL_UNSIGNED_INT:   indexSize = 4; break;
        default: return;
        }
        GLsizeiptr bufSize = triCount * indexSize;
        ensureQuadConvBuffer(bufSize);
        GLint prevEbo;
        GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &prevEbo);
        GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_quad_conv.ebo);
        void* mapped = GLES.glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0, bufSize,
                                              GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        if (!mapped) { GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevEbo); return; }
        if (type == GL_UNSIGNED_BYTE)
            convertQuadIndicesT<GLubyte>(indices, count, mapped, 0, false);
        else if (type == GL_UNSIGNED_SHORT)
            convertQuadIndicesT<GLushort>(indices, count, mapped, 0, false);
        else
            convertQuadIndicesT<GLuint>(indices, count, mapped, 0, false);
        GLES.glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
        GLES.glDrawElementsInstanced(GL_TRIANGLES, triCount, type, 0, primcount);
        GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevEbo);
        return;
    }
    GLES.glDrawElementsInstanced(mode, count, type, indices, primcount);
    CHECK_GL_ERROR
}

void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) {
    LOG()
    LOG_D("glDrawElements, mode: %d, count: %d, type: %d, indices: %p", mode, count, type, indices)
    prepareForDraw();
    if (mode == GL_QUADS) {
        drawElementsQuads(mode, count, type, indices, 0, false);
        return;
    }
    GLES.glDrawElements(mode, count, type, indices);
    CHECK_GL_ERROR
}

void glBindImageTexture(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access,
                        GLenum format) {
    LOG()
    LOG_D("glBindImageTexture, unit: %d, texture: %d, level: %d, layered: %d, layer: %d, access: %d, format: %d", unit,
          texture, level, layered, layer, access, format)
    GLES.glBindImageTexture(unit, texture, level, layered, layer, access, format);
    CHECK_GL_ERROR
}

void glUniform1i(GLint location, GLint v0) {
    LOG()
    LOG_D("glUniform1i, location: %d, v0: %d", location, v0)
    GLES.glUniform1i(location, v0);
    CHECK_GL_ERROR
}

void bindAllAtomicCounterAsSSBO();
void glDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {
    LOG()
    LOG_D("glDispatchCompute, num_groups_x: %d, num_groups_y: %d, num_groups_z: %d", num_groups_x, num_groups_y,
          num_groups_z)
    if (program_map_is_atomic_counter_emulated[gl_state->current_program]) {
        bindAllAtomicCounterAsSSBO();
        LOG_D("Atomic counters bound as SSBOs for program %d", gl_state->current_program);
    } else {
        LOG_D("No atomic counters bound as SSBOs for program %d", gl_state->current_program);
    }
    GLES.glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
    CHECK_GL_ERROR
}

void glMemoryBarrier(GLbitfield barriers) {
    LOG()
    LOG_D("glMemoryBarrier, barriers: %d", barriers)
    if (program_map_is_atomic_counter_emulated[gl_state->current_program]) {
        barriers |= GL_ATOMIC_COUNTER_BARRIER_BIT;
        barriers |= GL_SHADER_STORAGE_BARRIER_BIT;
    }
    GLES.glMemoryBarrier(barriers);
    CHECK_GL_ERROR
}

void glDrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLint basevertex) {
    LOG()
    LOG_D("glDrawElementsBaseVertex, mode: %d, count: %d, type: %d, indices: %p, basevertex: %d", mode, count, type,
          indices, basevertex);
    prepareForDraw();
    if (mode == GL_QUADS) {
        drawElementsQuads(mode, count, type, indices, basevertex, true);
        return;
    }
    if (hardware->es_version < 320 && !g_gles_caps.GL_EXT_draw_elements_base_vertex &&
        !g_gles_caps.GL_OES_draw_elements_base_vertex) {
        LOG_D("Emulating glDrawElementsBaseVertex")
        GLint prevElementBuffer;
        GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &prevElementBuffer);

        if (basevertex == 0) {
            GLES.glDrawElements(mode, count, type, indices);
            return;
        }

        size_t indexSize;
        switch (type) {
        case GL_UNSIGNED_INT:
            indexSize = sizeof(GLuint);
            break;
        case GL_UNSIGNED_SHORT:
            indexSize = sizeof(GLushort);
            break;
        case GL_UNSIGNED_BYTE:
            indexSize = sizeof(GLubyte);
            break;
        default:
            return;
        }

        size_t dataSize = count * indexSize;

        static GLuint s_temp_buffer = 0;
        static GLsizeiptr s_temp_buffer_size = 0;
        if (!s_temp_buffer) {
            GLES.glGenBuffers(1, &s_temp_buffer);
        }

        if (dataSize > (size_t)s_temp_buffer_size) {
            GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_temp_buffer);
            GLES.glBufferData(GL_ELEMENT_ARRAY_BUFFER, dataSize, nullptr, GL_DYNAMIC_DRAW);
            s_temp_buffer_size = dataSize;
        }

        void* mapped =
            GLES.glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0, dataSize, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        if (!mapped) {
            GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);
            return;
        }

        if (prevElementBuffer != 0) {
            GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);
            void* srcData =
                GLES.glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, (GLintptr)indices, dataSize, GL_MAP_READ_BIT);
            if (srcData) {
                memcpy(mapped, srcData, dataSize);
                GLES.glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
            } else {
                GLES.glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
                GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);
                return;
            }
            GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_temp_buffer);
        } else {
            memcpy(mapped, indices, dataSize);
        }

        switch (type) {
        case GL_UNSIGNED_INT:
            for (int j = 0; j < count; ++j) {
                ((GLuint*)mapped)[j] += basevertex;
            }
            break;
        case GL_UNSIGNED_SHORT:
            for (int j = 0; j < count; ++j) {
                ((GLushort*)mapped)[j] += basevertex;
            }
            break;
        case GL_UNSIGNED_BYTE:
            for (int j = 0; j < count; ++j) {
                ((GLubyte*)mapped)[j] += basevertex;
            }
            break;
        }

        GLES.glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
        GLES.glDrawElements(mode, count, type, 0);
        GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);

        CHECK_GL_ERROR
    } else {
        GLES.glDrawElementsBaseVertex(mode, count, type, indices, basevertex);
    }
    CHECK_GL_ERROR
}
