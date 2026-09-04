// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
	"testing/iotest"
)

// byteReaderOnly hides the concrete type of a reader, so that the
// decompressor cannot peek at its data and must use ReadByte.
type byteReaderOnly struct {
	io.Reader
	io.ByteReader
}

// errPastEnd is returned by the reader that follows the DEFLATE stream in
// the input sources below.
var errPastEnd = errors.New("read past end of stream")

type errReader struct{ err error }

func (r errReader) Read([]byte) (int, error) { return 0, r.err }

// inputSources builds the input readers to test the decompressor with,
// each positioned at the start of a stream followed by trailing data.
// rest returns the data left unread after the stream has been decompressed,
// or nil if the reader is expected to buffer input beyond the stream.
func inputSources(stream, trailing string) []struct {
	name string
	r    io.Reader
	rest func() (string, error)
} {
	type source = struct {
		name string
		r    io.Reader
		rest func() (string, error)
	}
	readAll := func(r io.Reader) func() (string, error) {
		return func() (string, error) {
			b, err := io.ReadAll(r)
			return string(b), err
		}
	}
	// stream followed by trailing data, one byte per Read call,
	// followed by a reader that reports reads past the end.
	slow := func() io.Reader {
		return io.MultiReader(iotest.OneByteReader(strings.NewReader(stream+trailing)), errReader{errPastEnd})
	}

	var sources []source
	br := bytes.NewReader([]byte(stream + trailing))
	sources = append(sources, source{"bytes.Reader", br, readAll(br)})
	bb := bytes.NewBufferString(stream + trailing)
	sources = append(sources, source{"bytes.Buffer", bb, func() (string, error) { return bb.String(), nil }})
	sr := strings.NewReader(stream + trailing)
	sources = append(sources, source{"strings.Reader", sr, readAll(sr)})
	bufr := bufio.NewReaderSize(strings.NewReader(stream+trailing), 64)
	sources = append(sources, source{"bufio.Reader", bufr, readAll(bufr)})
	bufr1 := bufio.NewReaderSize(slow(), 16)
	sources = append(sources, source{"bufio.Reader/OneByte", bufr1, readAll(bufr1)})
	sr2 := strings.NewReader(stream + trailing)
	sources = append(sources, source{"ByteReader", byteReaderOnly{sr2, sr2}, readAll(sr2)})
	sources = append(sources, source{"io.Reader", struct{ io.Reader }{slow()}, nil})
	return sources
}

// TestReaderInputPosition verifies that the decompressor consumes exactly
// the DEFLATE stream from the underlying reader, regardless of the
// read-ahead strategy used for the source type: data following the stream
// (such as a gzip trailer or another stream) can be read afterwards, and
// the underlying reader is never read beyond what is needed.
func TestReaderInputPosition(t *testing.T) {
	const trailing = "hello trailing data"
	for _, n := range []int{0, 1, 3, 10, 1 << 10, 1 << 16, 1 << 20} {
		data := make([]byte, n)
		for i := range data {
			data[i] = byte(i%251) & byte(i%17<<3)
		}
		for _, level := range []int{HuffmanOnly, NoCompression, BestSpeed, DefaultCompression, BestCompression} {
			var compressed bytes.Buffer
			w, err := NewWriter(&compressed, level)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := w.Write(data); err != nil {
				t.Fatal(err)
			}
			if err := w.Close(); err != nil {
				t.Fatal(err)
			}
			stream := compressed.String()

			for _, src := range inputSources(stream, trailing) {
				r := NewReader(src.r)
				got, err := io.ReadAll(r)
				if err != nil {
					t.Errorf("n=%d level=%d %s: ReadAll: %v", n, level, src.name, err)
					continue
				}
				if !bytes.Equal(got, data) {
					t.Errorf("n=%d level=%d %s: decompressed data mismatch", n, level, src.name)
					continue
				}
				if src.rest == nil {
					continue
				}
				rest, err := src.rest()
				if rest != trailing {
					t.Errorf("n=%d level=%d %s: trailing data = %q (len %d), want %q",
						n, level, src.name, rest, len(rest), trailing)
				}
				if err != nil && err != errPastEnd {
					t.Errorf("n=%d level=%d %s: reading trailing data: %v", n, level, src.name, err)
				}
			}
		}
	}
}

// TestReaderCorruptOffset verifies that the offset reported for corrupt
// input lies within the stream, for every input source type.
func TestReaderCorruptOffset(t *testing.T) {
	data := make([]byte, 1<<16)
	for i := range data {
		data[i] = byte(i%251) & byte(i%17<<3)
	}
	for _, level := range []int{HuffmanOnly, BestSpeed, BestCompression} {
		var compressed bytes.Buffer
		w, _ := NewWriter(&compressed, level)
		w.Write(data)
		w.Close()
		stream := compressed.Bytes()
		for pos := 1; pos < len(stream); pos += len(stream) / 7 {
			corrupt := bytes.Clone(stream)
			corrupt[pos] ^= 0x55
			for _, src := range inputSources(string(corrupt), "") {
				_, err := io.Copy(io.Discard, NewReader(src.r))
				var cerr CorruptInputError
				if errors.As(err, &cerr) && (cerr < 0 || int(cerr) > len(stream)) {
					t.Errorf("level=%d pos=%d %s: %v, want offset in [0, %d]", level, pos, src.name, err, len(stream))
				}
			}
		}
	}
}
