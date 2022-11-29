/*
 * subr.h
 *
 * Copyright (c) 2012, Seiichi Uchida. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SUBR_H
#define SUBR_H

/* prototypes */
short  Contour_Detect(IplImage*);
void   Contour_To_Directional_Pattern(short);
void   Blurring();
void   Equalize_Intensity();
void   Equalize_Directional_Pattern();
int    Compare(const void*, const void*);
void   Make_Intensity(IplImage*);
void   Blur_Intensity();
double DIRP_Dist(DIRP (*)[N][N], DIRP (*)[N][N]);
int    extract_feature(IplImage*, datafolder**);
void   extract_feature_wrapper(char*, datafolder**);
int    db_save(char*, feature_db*);

#ifdef __cplusplus
#define _EX_DECL
#else
#define _EX_DECL extern
#endif

#ifdef __cplusplus
extern "C"
#endif
    _EX_DECL feature_db*
    db_load(char*);

#endif /* SUBR_H */
