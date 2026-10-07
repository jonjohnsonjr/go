// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package flate

import "simd/archsimd"

// haveSIMD is a compile-time constant on arm64 because 128-bit NEON is
// mandatory in baseline ARMv8-A, allowing the compiler to fold away the
// !haveSIMD fallback branch.
const haveSIMD = true

var distMaskTables [16]archsimd.Uint8x16

func initDistMasks() {
	for d := 1; d < 16; d++ {
		var arr [16]uint8
		for i := range arr {
			arr[i] = uint8(i % d)
		}
		distMaskTables[d] = archsimd.LoadUint8x16Array(&arr)
	}
}

// permuteDist16 replicates the first dist bytes (1 <= dist < 16) across all
// 16 lanes using NEON VTBL.
func permuteDist16(v simdVec, dist int) simdVec {
	return v.LookupOrZero(distMaskTables[dist&15])
}
