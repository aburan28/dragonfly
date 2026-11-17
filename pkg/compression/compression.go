/*
 *     Copyright 2026 The Dragonfly Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Package compression provides streaming compression and decompression of data
// distributed through Dragonfly, with format detection and a bound on the
// decompressed size so untrusted input cannot exhaust memory or disk.
package compression

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/url"
	"path"
	"strings"

	"github.com/klauspost/compress/gzip"
	"github.com/klauspost/compress/zstd"
)

// Algorithm is a compression format.
type Algorithm string

const (
	// None is uncompressed data.
	None Algorithm = "none"

	// Gzip is RFC 1952 gzip. Concatenated members are decompressed as one stream.
	Gzip Algorithm = "gzip"

	// Zstd is RFC 8878 Zstandard. Concatenated and skippable frames are supported.
	Zstd Algorithm = "zstd"
)

// Algorithms lists the supported algorithms.
var Algorithms = []Algorithm{None, Gzip, Zstd}

// SniffLen is the number of leading bytes Detect needs.
const SniffLen = 4

// zstdMaxWindow bounds the memory a zstd frame can make the decoder allocate. It
// matches the window the zstd CLI accepts by default without --long.
const zstdMaxWindow = 128 << 20

var (
	// ErrUnsupported is returned for an unknown algorithm.
	ErrUnsupported = errors.New("unsupported compression algorithm")

	// ErrSizeLimit is returned when decompressed data exceeds the configured limit.
	ErrSizeLimit = errors.New("decompressed size exceeds limit")
)

// ParseAlgorithm parses an algorithm name. The empty string is None.
func ParseAlgorithm(s string) (Algorithm, error) {
	switch Algorithm(strings.ToLower(strings.TrimSpace(s))) {
	case None, "":
		return None, nil
	case Gzip, "gz":
		return Gzip, nil
	case Zstd, "zst":
		return Zstd, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrUnsupported, s)
	}
}

// Detect identifies the algorithm from the leading bytes of the data. It returns
// None when no known magic number matches, including when header is shorter than
// SniffLen.
func Detect(header []byte) Algorithm {
	switch {
	case len(header) >= 2 && header[0] == 0x1f && header[1] == 0x8b:
		return Gzip
	case len(header) >= 4 && header[0] == 0x28 && header[1] == 0xb5 && header[2] == 0x2f && header[3] == 0xfd:
		return Zstd
	case len(header) >= 4 && header[0]&0xf0 == 0x50 && header[1] == 0x2a && header[2] == 0x4d && header[3] == 0x18:
		// A zstd skippable frame (magic 0x184D2A5?), which only appears in zstd streams.
		return Zstd
	default:
		return None
	}
}

// FromPath infers the algorithm from the extension of a path or URL. Query strings
// and fragments are ignored. It returns None for unknown extensions.
func FromPath(p string) Algorithm {
	if u, err := url.Parse(p); err == nil && u.Path != "" {
		p = u.Path
	}

	switch strings.ToLower(path.Ext(p)) {
	case ".gz", ".tgz", ".gzip":
		return Gzip
	case ".zst", ".zstd", ".tzst":
		return Zstd
	default:
		return None
	}
}

// Level is a compression level, mapped onto each algorithm's own scale.
type Level int

const (
	// LevelDefault balances speed and ratio.
	LevelDefault Level = iota

	// LevelFastest favors speed.
	LevelFastest

	// LevelBetter favors ratio at moderate cost.
	LevelBetter

	// LevelBest favors ratio regardless of cost.
	LevelBest
)

// String returns the name of the level.
func (l Level) String() string {
	switch l {
	case LevelFastest:
		return "fastest"
	case LevelBetter:
		return "better"
	case LevelBest:
		return "best"
	default:
		return "default"
	}
}

func gzipLevel(l Level) int {
	switch l {
	case LevelFastest:
		return gzip.BestSpeed
	case LevelBetter:
		return 7
	case LevelBest:
		return gzip.BestCompression
	default:
		return gzip.DefaultCompression
	}
}

func zstdLevel(l Level) zstd.EncoderLevel {
	switch l {
	case LevelFastest:
		return zstd.SpeedFastest
	case LevelBetter:
		return zstd.SpeedBetterCompression
	case LevelBest:
		return zstd.SpeedBestCompression
	default:
		return zstd.SpeedDefault
	}
}

type nopWriteCloser struct{ io.Writer }

func (nopWriteCloser) Close() error { return nil }

// NewWriter returns a writer compressing into w. Close flushes the trailer but does
// not close w.
func NewWriter(w io.Writer, alg Algorithm, level Level) (io.WriteCloser, error) {
	switch alg {
	case None, "":
		return nopWriteCloser{w}, nil
	case Gzip:
		return gzip.NewWriterLevel(w, gzipLevel(level))
	case Zstd:
		return zstd.NewWriter(w, zstd.WithEncoderLevel(zstdLevel(level)))
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnsupported, alg)
	}
}

type readerOptions struct {
	maxSize int64
}

// ReaderOption configures a decompressing reader.
type ReaderOption func(*readerOptions)

// WithMaxSize fails the read with ErrSizeLimit once more than n decompressed bytes
// are produced. n <= 0 means no limit.
func WithMaxSize(n int64) ReaderOption {
	return func(o *readerOptions) { o.maxSize = n }
}

// NewReader returns a reader decompressing alg data from r. Close releases decoder
// resources but does not close r.
func NewReader(r io.Reader, alg Algorithm, opts ...ReaderOption) (io.ReadCloser, error) {
	var o readerOptions
	for _, opt := range opts {
		opt(&o)
	}

	var rc io.ReadCloser
	switch alg {
	case None, "":
		rc = io.NopCloser(r)
	case Gzip:
		gr, err := gzip.NewReader(r)
		if err != nil {
			return nil, fmt.Errorf("gzip: %w", err)
		}

		rc = gr
	case Zstd:
		zr, err := zstd.NewReader(r,
			zstd.WithDecoderConcurrency(1),
			zstd.WithDecoderMaxWindow(zstdMaxWindow),
			zstd.WithDecoderMaxMemory(uint64(zstdMaxWindow)*2),
		)
		if err != nil {
			return nil, fmt.Errorf("zstd: %w", err)
		}

		rc = zr.IOReadCloser()
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnsupported, alg)
	}

	if o.maxSize > 0 {
		rc = &limitReader{rc: rc, remaining: o.maxSize}
	}

	return rc, nil
}

// NewAutoReader detects the algorithm from the leading bytes of r and returns a
// reader decompressing it, or passing it through unchanged when it is not
// compressed.
func NewAutoReader(r io.Reader, opts ...ReaderOption) (io.ReadCloser, Algorithm, error) {
	br := bufio.NewReader(r)
	header, err := br.Peek(SniffLen)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, "", err
	}

	alg := Detect(header)
	rc, err := NewReader(br, alg, opts...)
	if err != nil {
		return nil, "", err
	}

	return rc, alg, nil
}

// limitReader fails with ErrSizeLimit instead of silently truncating, so a
// decompression bomb is reported rather than producing a short file.
type limitReader struct {
	rc        io.ReadCloser
	remaining int64
}

func (l *limitReader) Read(p []byte) (int, error) {
	if l.remaining < 0 {
		return 0, ErrSizeLimit
	}

	// Read one byte past the limit so data ending exactly at it is accepted.
	if int64(len(p)) > l.remaining+1 {
		p = p[:l.remaining+1]
	}

	n, err := l.rc.Read(p)
	l.remaining -= int64(n)
	if l.remaining < 0 {
		return n + int(l.remaining), ErrSizeLimit
	}

	return n, err
}

func (l *limitReader) Close() error { return l.rc.Close() }

// Compress compresses data in memory.
func Compress(data []byte, alg Algorithm, level Level) ([]byte, error) {
	if alg == None || alg == "" {
		return append([]byte(nil), data...), nil
	}

	var buf bytes.Buffer
	w, err := NewWriter(&buf, alg, level)
	if err != nil {
		return nil, err
	}

	if _, err := w.Write(data); err != nil {
		w.Close()
		return nil, err
	}

	if err := w.Close(); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// Decompress decompresses data in memory. maxSize <= 0 means no limit.
func Decompress(data []byte, alg Algorithm, maxSize int64) ([]byte, error) {
	r, err := NewReader(bytes.NewReader(data), alg, WithMaxSize(maxSize))
	if err != nil {
		return nil, err
	}
	defer r.Close()

	return io.ReadAll(r)
}
