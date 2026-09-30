// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package flate

import "simd/archsimd"

// distStep[d] is the largest multiple of d <= 16: (16 / d) * d.
// Advancing dstPos by distStep[d] preserves the phase of a period-d pattern,
// allowing the same PermuteOrZero vector to be stored repeatedly without reloading.
var distStep = [16]uint8{0, 16, 16, 15, 16, 15, 12, 14, 16, 9, 10, 11, 12, 13, 14, 15}

var distMaskTables = func() (masks [17]archsimd.Int8x16) {
	for d := 1; d <= 16; d++ {
		var arr [16]int8
		for i := range arr {
			arr[i] = int8(i % d) // for d == 16, i % 16 == i (identity)
		}
		masks[d] = archsimd.LoadInt8x16Array(&arr)
	}
	return masks
}()

var lenMaskTables = func() (masks [17]archsimd.Mask8x16) {
	for k := 0; k <= 16; k++ {
		var arr [16]int8
		for i := range k {
			arr[i] = -1
		}
		masks[k] = archsimd.LoadInt8x16Array(&arr).ToMask()
	}
	return masks
}()
