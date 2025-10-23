package main

import (
	"fmt"
	"go_library/image_go"
	"go_library/math_go"
	"math"
	"runtime"
	"sync"
	"time"
)

type QuantizationMatrix struct {
	Matrix  math_go.NDArray[int16]
	quality int
}

func NewQuantizationMatrix(quality int) *QuantizationMatrix {
	matrix := math_go.NewArray[int16](8, 8)

	/*
		Q-50 matrix
		[16, 11, 10, 16, 24, 40, 51, 61],
		[12, 12, 14, 19, 26, 58, 60, 55],
		[14, 13, 16, 24, 40, 57, 69, 56],
		[14, 17, 22, 29, 51, 87, 80, 62],
		[18, 22, 37, 56, 68, 109, 103, 77],
		[24, 35, 55, 64, 81, 104, 113, 92],
		[49, 64, 78, 87, 103, 121, 120, 101],
		[72, 92, 95, 98, 112, 100, 103, 99],

		Q-10 matrix
		[80, 60, 50, 80, 120, 200, 255, 255],
		[60, 60, 70, 95, 130, 255, 255, 255],
		[70, 65, 80, 120, 200, 255, 255, 255],
		[70, 85, 110, 145, 255, 255, 255, 255],
		[90, 110, 185, 255, 255, 255, 255, 255],
		[120, 175, 255, 255, 255, 255, 255, 255],
		[245, 255, 255, 255, 255, 255, 255, 255],
		[255, 255, 255, 255, 255, 255, 255, 255],
	*/

	Data := []int16{16, 11, 10, 16, 24, 40, 51, 61,
		12, 12, 14, 19, 26, 58, 60, 55,
		14, 13, 16, 24, 40, 57, 69, 56,
		14, 17, 22, 29, 51, 87, 80, 62,
		18, 22, 37, 56, 68, 109, 103, 77,
		24, 35, 55, 64, 81, 104, 113, 92,
		49, 64, 78, 87, 103, 121, 120, 101,
		72, 92, 95, 98, 112, 100, 103, 99,
	}

	if quality == 10 {
		Data = []int16{80, 60, 50, 80, 120, 200, 255, 255,
			60, 60, 70, 95, 130, 255, 255, 255,
			70, 65, 80, 120, 200, 255, 255, 255,
			70, 85, 110, 145, 255, 255, 255, 255,
			90, 110, 185, 255, 255, 255, 255, 255,
			120, 175, 255, 255, 255, 255, 255, 255,
			245, 255, 255, 255, 255, 255, 255, 255,
			255, 255, 255, 255, 255, 255, 255, 255,
		}
	}

	for i := range 8 {
		for j := range 8 {
			matrix.Set(Data[i*8+j], i, j)
		}
	}

	return &QuantizationMatrix{
		Matrix:  *matrix,
		quality: quality,
	}
}

func DivideImageIntoBlocks(image *math_go.NDArray[uint8], blockSize int) []math_go.NDArray[uint8] {
	height, width := image.Shape[0], image.Shape[1]
	var blocks []math_go.NDArray[uint8]

	for y := 0; y < height; y += blockSize {
		for x := 0; x < width; x += blockSize {
			block := math_go.NewArray[uint8](blockSize, blockSize)

			for by := range blockSize {
				for bx := range blockSize {
					if y+by < height && x+bx < width {
						intensity := image.At(y+by, x+bx)
						block.Set(intensity, by, bx)
					} else {
						block.Set(0, by, bx)
					}
				}
			}
			blocks = append(blocks, *block)
		}
	}

	return blocks
}

func LevelShiftBlock(block *math_go.NDArray[uint8]) *math_go.NDArray[int16] {
	shiftedBlock := math_go.NewArray[int16](block.Shape[0], block.Shape[1])
	for y := 0; y < block.Shape[0]; y++ {
		for x := 0; x < block.Shape[1]; x++ {
			shiftedValue := int16(block.At(y, x)) - 128
			shiftedBlock.Set(shiftedValue, y, x)
		}
	}

	return shiftedBlock
}

func DCTMatrix(N int) *math_go.NDArray[float64] {
	dctMatrix := math_go.NewArray[float64](N, N)

	for u := range N {
		for x := range N {
			cu := math.Sqrt(2.0 / float64(N))
			if u == 0 {
				cu = 1.0 / math.Sqrt(float64(N))
			}
			expression := cu * math.Cos((2.0*float64(x)+1.0)*float64(u)*math.Pi/(2.0*float64(N)))
			dctMatrix.Set(expression, u, x)
		}
	}

	return dctMatrix
}

func ApplyDCT(block *math_go.NDArray[int16], dctMatrix *math_go.NDArray[float64]) *math_go.NDArray[float64] {
	N := block.Shape[0]

	tmp := math_go.NewArray[float64](N, N)
	for i := range N {
		for j := range N {
			tmp.Set(float64(block.At(i, j)), i, j)
		}
	}

	return dctMatrix.Multiply(tmp.Multiply(dctMatrix.Transpose()))
}

func QuantizeBlock(block *math_go.NDArray[float64], qMatrix *QuantizationMatrix) *math_go.NDArray[int16] {
	qMatrixFloat := math_go.NewArray[float64](qMatrix.Matrix.Shape[0], qMatrix.Matrix.Shape[1])
	for i := range qMatrix.Matrix.Shape[0] {
		for j := range qMatrix.Matrix.Shape[1] {
			qMatrixFloat.Set(float64(qMatrix.Matrix.At(i, j)), i, j)
		}
	}
	quantizedBlock := block.DivideElementwise(qMatrixFloat)
	result := math_go.NewArray[int16](block.Shape[0], block.Shape[1])

	for i := range quantizedBlock.Shape[0] {
		for j := range quantizedBlock.Shape[1] {
			quantizedValue := int16(math.Round(quantizedBlock.At(i, j)))
			result.Set(quantizedValue, i, j)
		}
	}

	return result
}

func Encode(qBlock *math_go.NDArray[int16]) []int16 {
	var encoded []int16

	flatBlock := qBlock.Flatten()
	count := 1
	for i := 1; i < flatBlock.Size(); i++ {
		if flatBlock.At(i) == flatBlock.At(i-1) {
			count++
		} else {
			encoded = append(encoded, flatBlock.At(i-1))
			encoded = append(encoded, int16(count))
			count = 1
		}
	}
	encoded = append(encoded, flatBlock.At(flatBlock.Size()-1))
	encoded = append(encoded, int16(count))

	return encoded
}

func Decompress(encoded []int16, blockSize int, qMatrix *QuantizationMatrix) *math_go.NDArray[int16] {
	var flatBlock []int16
	for i := 0; i < len(encoded); i += 2 {
		value := encoded[i]
		count := int(encoded[i+1])
		for range count {
			flatBlock = append(flatBlock, value)
		}
	}

	quantizedBlock := math_go.NewArray[int16](blockSize, blockSize)
	for i := range blockSize {
		for j := range blockSize {
			quantizedBlock.Set(flatBlock[i*blockSize+j], i, j)
		}
	}

	return quantizedBlock.MultiplyElementwise(&qMatrix.Matrix)
}

func ApplyIDCT(dctBlock *math_go.NDArray[int16]) *math_go.NDArray[float64] {
	dctBlockFloat := math_go.NewArray[float64](dctBlock.Shape[0], dctBlock.Shape[1])
	for i := range dctBlock.Shape[0] {
		for j := range dctBlock.Shape[1] {
			dctBlockFloat.Set(float64(dctBlock.At(i, j)), i, j)
		}
	}

	N := dctBlock.Shape[0]
	dctMatrix := DCTMatrix(N)

	return dctMatrix.Transpose().Multiply(dctBlockFloat.Multiply(dctMatrix))
}

func PadImage(image *math_go.NDArray[uint8], blockSize int) *math_go.NDArray[uint8] {
	height, width := image.Shape[0], image.Shape[1]
	newHeight, newWidth := height, width
	if height%blockSize != 0 {
		padHeight := blockSize - (height % blockSize)
		newHeight = height + padHeight
	}
	if width%blockSize != 0 {
		padWidth := blockSize - (width % blockSize)
		newWidth = width + padWidth
	}

	paddedImage := math_go.NewArray[uint8](newHeight, newWidth)
	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			if y < height && x < width {
				paddedImage.Set(image.At(y, x), y, x)
			} else {
				paddedImage.Set(0, y, x)
			}
		}
	}

	return paddedImage
}

func JPEG(image *math_go.NDArray[uint8], quality int) *math_go.NDArray[uint8] {
	blockSize := 8
	qMatrix := NewQuantizationMatrix(quality)
	paddedImage := PadImage(image, blockSize)
	blocks := DivideImageIntoBlocks(paddedImage, blockSize)

	dctMatrix := DCTMatrix(blockSize)

	originalSize := paddedImage.Shape[0] * paddedImage.Shape[1]
	totalEncodedSize := 0
	for _, block := range blocks {
		levelShifted := LevelShiftBlock(&block)
		dctBlock := ApplyDCT(levelShifted, dctMatrix)
		quantizedBlock := QuantizeBlock(dctBlock, qMatrix)
		encodedData := Encode(quantizedBlock)
		totalEncodedSize += len(encodedData)
	}
	compressionRatio := float64(originalSize) / float64(totalEncodedSize)
	fmt.Printf("Compression Ratio: %.2f\n", compressionRatio)

	for idx, block := range blocks {
		levelShifted := LevelShiftBlock(&block)
		dctBlock := ApplyDCT(levelShifted, dctMatrix)
		quantizedBlock := QuantizeBlock(dctBlock, qMatrix)
		encodedData := Encode(quantizedBlock)
		decompressedDCTBlock := Decompress(encodedData, blockSize, qMatrix)
		idctBlock := ApplyIDCT(decompressedDCTBlock)
		idctBlock = idctBlock.AddScalar(128.0)
		idctBlock = idctBlock.Clip(0.0, 255.0)

		i := (idx * blockSize) / paddedImage.Shape[1] * blockSize
		j := (idx * blockSize) % paddedImage.Shape[1]
		for by := range blockSize {
			for bx := range blockSize {
				paddedImage.Set(uint8(math.Round(idctBlock.At(by, bx))), i+by, j+bx)
			}
		}
	}

	return paddedImage
}

func ApplyJPEGBlock(block *math_go.NDArray[uint8], idx int, qMatrix *QuantizationMatrix, dctMatrix *math_go.NDArray[float64], blockSize int, width int) (math_go.NDArray[uint8], int, int) {
	levelShifted := LevelShiftBlock(block)
	dctBlock := ApplyDCT(levelShifted, dctMatrix)
	quantizedBlock := QuantizeBlock(dctBlock, qMatrix)
	encodedData := Encode(quantizedBlock)
	decompressedDCTBlock := Decompress(encodedData, blockSize, qMatrix)
	idctBlock := ApplyIDCT(decompressedDCTBlock)
	idctBlock = idctBlock.AddScalar(128.0)
	idctBlock = idctBlock.Clip(0.0, 255.0)

	i := (idx * blockSize) / width * blockSize
	j := (idx * blockSize) % width

	resultBlock := math_go.NewArray[uint8](blockSize, blockSize)
	for by := range blockSize {
		for bx := range blockSize {
			resultBlock.Set(uint8(math.Round(idctBlock.At(by, bx))), by, bx)
		}
	}

	return *resultBlock, i, j
}

func JPEGConcurrent(image *math_go.NDArray[uint8], quality int) *math_go.NDArray[uint8] {
	blockSize := 8
	qMatrix := NewQuantizationMatrix(quality)
	paddedImage := PadImage(image, blockSize)
	blocks := DivideImageIntoBlocks(paddedImage, blockSize)

	dctMatrix := DCTMatrix(blockSize)

	// concurrent processing of blocks
	type result struct {
		block math_go.NDArray[uint8]
		i     int
		j     int
	}
	results := make(chan result, len(blocks))

	runtime.GOMAXPROCS(runtime.NumCPU())
	var wg sync.WaitGroup
	numJobs := runtime.NumCPU()

	wg.Add(numJobs)
	for w := range numJobs {
		go func(workerID int) {
			defer wg.Done()
			for idx := workerID; idx < len(blocks); idx += numJobs {
				block := blocks[idx]
				processedBlock, i, j := ApplyJPEGBlock(&block, idx, qMatrix, dctMatrix, blockSize, paddedImage.Shape[1])
				results <- result{block: processedBlock, i: i, j: j}
			}
		}(w)
	}

	wg.Wait()
	close(results)

	for res := range results {
		for by := range blockSize {
			for bx := range blockSize {
				paddedImage.Set(res.block.At(by, bx), res.i+by, res.j+bx)
			}
		}
	}

	return paddedImage
}

func main() {
	inputImagePath := "/home/vm/Documents/go-projects/go_library/einstein.jpg"
	imageArr := imagego.LoadImageNDArray(inputImagePath)

	startTime := time.Now()
	compressedImage := JPEGConcurrent(imageArr, 50)
	compressedData2D := [][]uint8{}
	height, width := compressedImage.Shape[0], compressedImage.Shape[1]
	for y := range height {
		row := []uint8{}
		for x := range width {
			row = append(row, compressedImage.At(y, x))
		}
		compressedData2D = append(compressedData2D, row)
	}
	imagego.DisplayImageGray(compressedData2D)
	endTime := time.Now()
	fmt.Printf("Custom JPEG time: %v\n", endTime.Sub(startTime))
}
