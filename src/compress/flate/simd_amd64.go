// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package flate

import "simd/archsimd"

// haveSIMD reports whether the CPU supports AVX, which is required at runtime
// because archsimd's 128-bit operations emit VEX-encoded instructions
// (VMOVDQU, VPSHUFB, VPBLENDVB) whereas default GOAMD64=v1 only guarantees SSE2.
var haveSIMD = archsimd.X86.AVX()

var distMaskTables [16]archsimd.Int8x16

func initDistMasks() {
	for d := 1; d < 16; d++ {
		var arr [16]int8
		for i := range arr {
			arr[i] = int8(i % d)
		}
		distMaskTables[d] = archsimd.LoadInt8x16Array(&arr)
	}
}

// permuteDist16 replicates the first dist bytes (1 <= dist < 16) across all
// 16 lanes using VPSHUFB.
func permuteDist16(v simdVec, dist int) simdVec {
	return v.PermuteOrZero(distMaskTables[dist&15])
}
