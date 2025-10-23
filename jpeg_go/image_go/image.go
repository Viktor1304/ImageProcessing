package imagego

import (
	"go_library/math_go"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"os"
)

func LoadImageGray(path string) [][]uint8 {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	image, _, err := image.Decode(file)
	if err != nil {
		panic(err)
	}

	bounds := image.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	grayValues := make([][]uint8, height)
	for y := range height {
		grayValues[y] = make([]uint8, width)
	}

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := color.GrayModel.Convert(image.At(x, y)).(color.Gray)
			grayValues[y-bounds.Min.Y][x-bounds.Min.X] = c.Y
		}
	}

	return grayValues
}

func LoadImageNDArray(path string) *math_go.NDArray[uint8] {
	data := LoadImageGray(path)
	height := len(data)
	width := len(data[0])

	array := math_go.NewArray[uint8](height, width)
	for y := range height {
		for x := range width {
			array.Set(data[y][x], y, x)
		}
	}

	return array
}

func DisplayImageGray(data [][]uint8) {
	height := len(data)
	width := len(data[0])

	img := image.NewGray(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			img.SetGray(x, y, color.Gray{Y: data[y][x]})
		}
	}

	draw.Draw(img, img.Bounds(), img, image.Point{}, draw.Src)

	file, err := os.Create("output.jpeg")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	err = png.Encode(file, img)
	if err != nil {
		panic(err)
	}
}
