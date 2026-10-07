// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64)

package flate

import (
	"simd/archsimd"
	"unsafe"
)

// simdVec aliases the 128-bit byte vector type used for in-window LZ77 match
// copying. The SIMD operations below are split into four tiny leaf functions
// (load16, store16, permuteDist16, blendStore16) rather than a single
// match-copy function so that each helper stays well below the Go compiler's
// 80-point inlining budget (blendStore16 alone costs 55 because IfElse is a Go
// method wrapper in simd/archsimd). Wrapping the entire SIMD copy loop in one
// function would exceed the inlining budget and force a non-inlined CALL on
// every match, spilling huffmanBlock's hot loop registers (b, nb, wrPos) to
// the stack.
type simdVec = archsimd.Uint8x16

// distStep[d] is the largest multiple of d <= 16: (16 / d) * d.
// Advancing dstPos by distStep[d] preserves the phase of a period-d pattern,
// allowing the same shuffled vector to be stored repeatedly without reloading.
var distStep = [16]uint8{0, 16, 16, 15, 16, 15, 12, 14, 16, 9, 10, 11, 12, 13, 14, 15}

var lenMaskTables [32]archsimd.Mask8x16

func init() {
	// On amd64, 128-bit archsimd operations emit VEX-encoded AVX instructions
	// (VMOVDQU, VPSHUFB, VPBLENDVB), whereas default GOAMD64=v1 only guarantees
	// SSE2 at compile time. Guard vector table initialization with haveSIMD so
	// package init does not SIGILL on a CPU without AVX.
	if !haveSIMD {
		return
	}
	initDistMasks()
	for k := 0; k <= 16; k++ {
		var arr [16]int8
		for i := range k {
			arr[i] = -1
		}
		lenMaskTables[k] = archsimd.LoadInt8x16Array(&arr).ToMask()
	}
}

// The preceding wrPos+length+16 and dist bounds checks in huffmanBlock prove
// 0 <= i <= maxMatchOffset-16. Using LoadUint8x16Array/StoreArray with
// unsafe.Add avoids slice header construction, Spectre slice-base masking, and
// (*[16]byte)(slice) bounds checks without growing dd.hist past Go's 32 KB
// malloc size class.
func load16(p *[maxMatchOffset]byte, i int) simdVec {
	return archsimd.LoadUint8x16Array((*[16]byte)(unsafe.Add(unsafe.Pointer(p), i)))
}

func store16(p *[maxMatchOffset]byte, i int, v simdVec) {
	v.StoreArray((*[16]byte)(unsafe.Add(unsafe.Pointer(p), i)))
}

func blendStore16(p *[maxMatchOffset]byte, i int, v simdVec, length int) {
	orig := load16(p, i)
	store16(p, i, v.IfElse(lenMaskTables[length&31], orig))
}
