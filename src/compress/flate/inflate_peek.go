// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bufio"
	"bytes"
	"io"
	"strings"
)

// A peeker provides read-ahead access to a reader's pending data without
// consuming it. It is used by the decompressor to refill its bit buffer
// from a slice instead of calling ReadByte for every input byte.
//
// The contract matches bufio.Reader: Peek returns the next n bytes without
// advancing the reader, and Discard consumes n bytes. Bytes are only
// consumed from the underlying reader via Discard, so the decompressor
// still never consumes input beyond the end of the DEFLATE stream.
type peeker interface {
	// Buffered returns the number of bytes that can be peeked without
	// blocking or, for in-memory readers, the number of remaining bytes.
	Buffered() int
	Peek(n int) ([]byte, error)
	Discard(n int) (int, error)
}

// peekWindow is the maximum number of bytes peeked at a time.
// peekMore never asks a peeker for more than what is already buffered,
// so this only bounds the window (and the readerAtPeeker's copy), and
// does not need to fit in a bufio.Reader's buffer.
const peekWindow = 4096

// peekMore discards the input bytes consumed so far and peeks the next
// window of input. It reports an error only if no input byte is available.
// To avoid blocking on partial input, it peeks no more than what is
// already buffered, except that it requests a single byte (which may
// block, just as ReadByte would) if nothing is buffered.
func (f *decompressor) peekMore() error {
	f.discardFed()
	n := f.pr.Buffered()
	if n == 0 {
		n = 1
	} else if n > peekWindow {
		n = peekWindow
	}
	p, err := f.pr.Peek(n)
	if len(p) == 0 {
		return noEOF(err)
	}
	f.pb = p
	return nil
}

// discardFed consumes the peeked bytes whose bits have been fed into the
// bit buffer, and invalidates the current peek window. It must be called
// before any direct read from f.r, and before returning to the caller, so
// that the reader's position is exactly as if every byte had been read
// with ReadByte.
func (f *decompressor) discardFed() {
	if f.nfed > 0 {
		if _, err := f.pr.Discard(f.nfed); err != nil && f.err == nil {
			f.err = err
		}
		f.roffset += int64(f.nfed)
		f.nfed = 0
	}
	f.pb = nil
}

// makePeeker returns a peeker for r, or nil if r does not support peeking.
// Only well-known reader types are used, since the peeker contract must be
// guaranteed by the implementation.
func (f *decompressor) makePeeker(r Reader) peeker {
	switch r := r.(type) {
	case *bufio.Reader:
		return r
	case *bytes.Buffer:
		// Reuse a previously allocated adapter if possible.
		if p, ok := f.pr.(*bytesBufferPeeker); ok {
			p.b = r
			return p
		}
		return &bytesBufferPeeker{b: r}
	case *bytes.Reader:
		return f.makeReaderAtPeeker(r)
	case *strings.Reader:
		return f.makeReaderAtPeeker(r)
	}
	return nil
}

func (f *decompressor) makeReaderAtPeeker(r sizedReaderAt) peeker {
	if p, ok := f.pr.(*readerAtPeeker); ok {
		p.r = r
		return p
	}
	return &readerAtPeeker{r: r}
}

// bytesBufferPeeker adapts a bytes.Buffer to the peeker interface.
// The buffer's unread bytes are peeked directly, without copying.
type bytesBufferPeeker struct {
	b *bytes.Buffer
}

func (p *bytesBufferPeeker) Buffered() int { return p.b.Len() }

func (p *bytesBufferPeeker) Peek(n int) ([]byte, error) {
	d := p.b.Bytes()
	if len(d) < n {
		return d, io.EOF
	}
	return d[:n], nil
}

func (p *bytesBufferPeeker) Discard(n int) (int, error) {
	return len(p.b.Next(n)), nil
}

// sizedReaderAt is implemented by bytes.Reader and strings.Reader.
type sizedReaderAt interface {
	io.ReaderAt
	io.Seeker
	Len() int
}

// readerAtPeeker adapts a bytes.Reader or strings.Reader to the peeker
// interface using ReadAt and Seek, which never block.
type readerAtPeeker struct {
	r   sizedReaderAt
	buf [peekWindow]byte
}

func (p *readerAtPeeker) Buffered() int { return p.r.Len() }

func (p *readerAtPeeker) Peek(n int) ([]byte, error) {
	n = min(n, len(p.buf))
	pos, err := p.r.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}
	m, err := p.r.ReadAt(p.buf[:n], pos)
	return p.buf[:m], err
}

func (p *readerAtPeeker) Discard(n int) (int, error) {
	if _, err := p.r.Seek(int64(n), io.SeekCurrent); err != nil {
		return 0, err
	}
	return n, nil
}
