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

package compression

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// corpus returns n bytes of semi-structured text, compressible like logs or JSON.
func corpus(n int, seed int64) []byte {
	rng := rand.New(rand.NewSource(seed))
	words := []string{"piece", "task", "peer", "seed", "scheduler", "manager", "digest", "sha256", "layer", "blob"}
	var buf bytes.Buffer
	for buf.Len() < n {
		fmt.Fprintf(&buf, `{"id":%d,"kind":%q,"size":%d,"ok":%t}`+"\n",
			rng.Intn(1<<20), words[rng.Intn(len(words))], rng.Int63n(1<<32), rng.Intn(2) == 0)
	}

	return buf.Bytes()[:n]
}

func randomBytes(n int, seed int64) []byte {
	b := make([]byte, n)
	rand.New(rand.NewSource(seed)).Read(b)
	return b
}

func TestRoundTrip(t *testing.T) {
	inputs := map[string][]byte{
		"empty":  {},
		"one":    {0x42},
		"text":   corpus(1<<20, 1),
		"random": randomBytes(256<<10, 2),
	}

	for _, alg := range Algorithms {
		for _, level := range []Level{LevelFastest, LevelDefault, LevelBetter, LevelBest} {
			for name, in := range inputs {
				t.Run(fmt.Sprintf("%s/%s/%s", alg, level, name), func(t *testing.T) {
					compressed, err := Compress(in, alg, level)
					require.NoError(t, err)

					if alg != None {
						assert.Equal(t, alg, Detect(compressed), "compressed output must be detectable")
					}

					out, err := Decompress(compressed, alg, 0)
					require.NoError(t, err)
					assert.True(t, bytes.Equal(in, out))

					rc, detected, err := NewAutoReader(bytes.NewReader(compressed))
					require.NoError(t, err)
					defer rc.Close()

					auto, err := io.ReadAll(rc)
					require.NoError(t, err)
					assert.True(t, bytes.Equal(in, auto))
					if alg != None && len(in) > 0 {
						assert.Equal(t, alg, detected)
					}
				})
			}
		}
	}
}

func TestConcatenatedStreams(t *testing.T) {
	a, b := corpus(10000, 3), corpus(20000, 4)
	for _, alg := range []Algorithm{Gzip, Zstd} {
		t.Run(string(alg), func(t *testing.T) {
			ca, err := Compress(a, alg, LevelDefault)
			require.NoError(t, err)
			cb, err := Compress(b, alg, LevelDefault)
			require.NoError(t, err)

			out, err := Decompress(append(ca, cb...), alg, 0)
			require.NoError(t, err)
			assert.True(t, bytes.Equal(append(append([]byte{}, a...), b...), out))
		})
	}
}

func TestSizeLimit(t *testing.T) {
	// 64 MiB of zeros compresses to a few KiB: a small bomb.
	bomb := make([]byte, 64<<20)
	for _, alg := range []Algorithm{Gzip, Zstd} {
		t.Run(string(alg), func(t *testing.T) {
			compressed, err := Compress(bomb, alg, LevelDefault)
			require.NoError(t, err)
			assert.Less(t, len(compressed), 1<<20)

			_, err = Decompress(compressed, alg, 1<<20)
			assert.ErrorIs(t, err, ErrSizeLimit)

			out, err := Decompress(compressed, alg, int64(len(bomb)))
			require.NoError(t, err, "data ending exactly at the limit is accepted")
			assert.Len(t, out, len(bomb))
		})
	}

	_, err := Decompress([]byte("hello"), None, 4)
	assert.ErrorIs(t, err, ErrSizeLimit)
}

func TestCorruptInput(t *testing.T) {
	in := corpus(100000, 5)
	for _, alg := range []Algorithm{Gzip, Zstd} {
		t.Run(string(alg), func(t *testing.T) {
			compressed, err := Compress(in, alg, LevelDefault)
			require.NoError(t, err)

			truncated := compressed[:len(compressed)/2]
			_, err = Decompress(truncated, alg, 0)
			assert.Error(t, err, "truncated stream")

			flipped := append([]byte{}, compressed...)
			flipped[len(flipped)-6] ^= 0xff
			_, err = Decompress(flipped, alg, 0)
			assert.Error(t, err, "checksum mismatch")
		})
	}

	_, err := Decompress([]byte("not gzip"), Gzip, 0)
	assert.Error(t, err)
}

func TestDetect(t *testing.T) {
	cases := []struct {
		header []byte
		want   Algorithm
	}{
		{nil, None},
		{[]byte{0x1f}, None},
		{[]byte{0x1f, 0x8b, 0x08, 0x00}, Gzip},
		{[]byte{0x28, 0xb5, 0x2f, 0xfd}, Zstd},
		{[]byte{0x50, 0x2a, 0x4d, 0x18}, Zstd},
		{[]byte{0x5f, 0x2a, 0x4d, 0x18}, Zstd},
		{[]byte("PK\x03\x04"), None},
		{[]byte("{\"a\""), None},
	}

	for _, c := range cases {
		assert.Equal(t, c.want, Detect(c.header), "%x", c.header)
	}
}

func TestFromPath(t *testing.T) {
	cases := map[string]Algorithm{
		"https://example.com/data.tar.gz":           Gzip,
		"https://example.com/data.tgz?sig=abc.zst":  Gzip,
		"s3://bucket/model/weights.safetensors.zst": Zstd,
		"/local/file.TZST":                          Zstd,
		"https://example.com/data.tar":              None,
		"https://example.com/gz":                    None,
		"https://example.com/a.zst#frag.gz":         Zstd,
	}

	for p, want := range cases {
		assert.Equal(t, want, FromPath(p), p)
	}
}

func TestParseAlgorithm(t *testing.T) {
	for in, want := range map[string]Algorithm{"": None, "none": None, "GZIP": Gzip, "gz": Gzip, " zstd ": Zstd, "zst": Zstd} {
		got, err := ParseAlgorithm(in)
		require.NoError(t, err, in)
		assert.Equal(t, want, got, in)
	}

	_, err := ParseAlgorithm("brotli")
	assert.True(t, errors.Is(err, ErrUnsupported))

	_, err = NewWriter(io.Discard, "lz4", LevelDefault)
	assert.ErrorIs(t, err, ErrUnsupported)
	_, err = NewReader(bytes.NewReader(nil), "lz4")
	assert.ErrorIs(t, err, ErrUnsupported)
}

// BenchmarkCompress and BenchmarkDecompress report throughput on the uncompressed
// size and the compression ratio, on compressible text and incompressible data.
func BenchmarkCompress(b *testing.B) {
	for _, data := range benchInputs() {
		for _, alg := range []Algorithm{Gzip, Zstd} {
			for _, level := range []Level{LevelFastest, LevelDefault, LevelBetter} {
				b.Run(fmt.Sprintf("%s/%s/%s", data.name, alg, level), func(b *testing.B) {
					var compressed []byte
					b.SetBytes(int64(len(data.data)))
					for b.Loop() {
						var err error
						if compressed, err = Compress(data.data, alg, level); err != nil {
							b.Fatal(err)
						}
					}

					b.ReportMetric(float64(len(data.data))/float64(len(compressed)), "ratio")
				})
			}
		}
	}
}

func BenchmarkDecompress(b *testing.B) {
	for _, data := range benchInputs() {
		for _, alg := range []Algorithm{Gzip, Zstd} {
			compressed, err := Compress(data.data, alg, LevelDefault)
			if err != nil {
				b.Fatal(err)
			}

			b.Run(fmt.Sprintf("%s/%s", data.name, alg), func(b *testing.B) {
				b.SetBytes(int64(len(data.data)))
				for b.Loop() {
					r, err := NewReader(bytes.NewReader(compressed), alg)
					if err != nil {
						b.Fatal(err)
					}

					if _, err := io.Copy(io.Discard, r); err != nil {
						b.Fatal(err)
					}
					r.Close()
				}
			})
		}
	}
}

func benchInputs() []struct {
	name string
	data []byte
} {
	return []struct {
		name string
		data []byte
	}{
		{"text", corpus(16<<20, 10)},
		{"random", randomBytes(16<<20, 11)},
	}
}
