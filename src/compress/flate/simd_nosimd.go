// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(goexperiment.simd && (amd64 || arm64))

package flate

// When built without GOEXPERIMENT=simd or on architectures other than amd64
// and arm64, haveSIMD is a compile-time false constant. The compiler folds
// "if !haveSIMD { goto fallbackCopy }" in huffmanBlock into an unconditional
// jump and dead-code-eliminates the entire SIMD block, while these no-op stubs
// allow inflate.go to typecheck without duplicating huffmanBlock across build
// tags.
const haveSIMD = false

type simdVec struct{}

var distStep [16]uint8

func load16(p *[maxMatchOffset]byte, i int) simdVec                      { return simdVec{} }
func store16(p *[maxMatchOffset]byte, i int, v simdVec)                  {}
func permuteDist16(v simdVec, dist int) simdVec                          { return simdVec{} }
func blendStore16(p *[maxMatchOffset]byte, i int, v simdVec, length int) {}
