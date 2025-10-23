package math_go

import (
	"fmt"
	"reflect"
)

type Number interface {
	int | int8 | int16 | int32 | int64 |
		uint | uint8 | uint16 | uint32 | uint64 |
		float32 | float64
}

type NDArray[T Number] struct {
	Shape   []int
	Strides []int
	Data    []T
	Type    reflect.Type
}

func NewArray[T Number](shape ...int) *NDArray[T] {
	size := 1
	strides := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = size
		size *= shape[i]
	}
	data := make([]T, size)

	return &NDArray[T]{
		Shape:   shape,
		Strides: strides,
		Data:    data,
		Type:    reflect.TypeOf(*new(T)),
	}
}

func (arr *NDArray[T]) flatIndex(dim_arr ...int) int {
	if len(dim_arr) != len(arr.Shape) {
		panic(fmt.Sprintf("expected %d dimensions, got %d", len(arr.Shape), len(dim_arr)))
	}

	flat_idx := 0
	for dim, size := range dim_arr {
		if size < 0 || size >= arr.Shape[dim] {
			panic(fmt.Sprintf("Dimension size %d out of range for dimension %d", size, dim))
		}
		flat_idx += size * arr.Strides[dim]
	}

	return flat_idx
}

func (arr *NDArray[T]) At(dim_arr ...int) T {
	return arr.Data[arr.flatIndex(dim_arr...)]
}

func (arr *NDArray[T]) Set(value T, dim_arr ...int) {
	valueType := reflect.TypeOf(value)
	if valueType != arr.Type {
		panic(fmt.Sprintf("Type mismatch: expected %s, got %s", arr.Type, valueType))
	}
	arr.Data[arr.flatIndex(dim_arr...)] = value
}

func (arr *NDArray[T]) Size() int {
	return len(arr.Data)
}

func (arr *NDArray[T]) DType() reflect.Type {
	return arr.Type
}

func (arr *NDArray[T]) Add(other *NDArray[T]) *NDArray[T] {
	if !reflect.DeepEqual(arr.Shape, other.Shape) {
		panic(fmt.Sprintf("Shape mismatch between %d and %d", arr.Shape, other.Shape))
	}

	result := NewArray[T](arr.Shape...)
	for i := 0; i < arr.Size(); i++ {
		result.Data[i] = arr.Data[i] + other.Data[i]
	}

	return result
}

func (arr *NDArray[T]) Substract(other *NDArray[T]) *NDArray[T] {
	if !reflect.DeepEqual(arr.Shape, other.Shape) {
		panic(fmt.Sprintf("Shape mismatch between %d and %d", arr.Shape, other.Shape))
	}

	result := NewArray[T](arr.Shape...)
	for i := 0; i < arr.Size(); i++ {
		result.Data[i] = arr.Data[i] - other.Data[i]
	}

	return result
}

func (arr *NDArray[T]) SubstractScalar(scalar T) *NDArray[T] {
	result := NewArray[T](arr.Shape...)
	for i := 0; i < arr.Size(); i++ {
		result.Data[i] = arr.Data[i] - scalar
	}

	return result
}

func (arr *NDArray[T]) AddScalar(scalar T) *NDArray[T] {
	result := NewArray[T](arr.Shape...)
	for i := 0; i < arr.Size(); i++ {
		result.Data[i] = arr.Data[i] + scalar
	}

	return result
}

func (arr *NDArray[T]) MultiplyScalar(scalar T) *NDArray[T] {
	result := NewArray[T](arr.Shape...)
	for i := 0; i < arr.Size(); i++ {
		result.Data[i] = arr.Data[i] * scalar
	}

	return result
}

func (arr *NDArray[T]) MultiplyElementwise(other *NDArray[T]) *NDArray[T] {
	if !reflect.DeepEqual(arr.Shape, other.Shape) {
		panic(fmt.Sprintf("Shape mismatch during elementwise multiplication between %d and %d", arr.Shape, other.Shape))
	}

	result := NewArray[T](arr.Shape...)
	for i := 0; i < arr.Size(); i++ {
		result.Data[i] = arr.Data[i] * other.Data[i]
	}

	return result
}

func (arr *NDArray[T]) DivideElementwise(other *NDArray[T]) *NDArray[T] {
	if !reflect.DeepEqual(arr.Shape, other.Shape) {
		panic(fmt.Sprintf("Shape mismatch during elementwise multiplication between %d and %d", arr.Shape, other.Shape))
	}

	result := NewArray[T](arr.Shape...)
	for i := 0; i < arr.Size(); i++ {
		result.Data[i] = arr.Data[i] / other.Data[i]
	}

	return result
}

func (arr *NDArray[T]) Flatten() *NDArray[T] {
	result := NewArray[T](arr.Size())
	copy(result.Data, arr.Data)

	return result
}

func (arr *NDArray[T]) Transpose() *NDArray[T] {
	if len(arr.Shape) != 2 {
		panic("Transpose only supported for 2D arrays")
	}

	newShape := []int{arr.Shape[1], arr.Shape[0]}
	result := NewArray[T](newShape...)
	for i := 0; i < arr.Shape[0]; i++ {
		for j := 0; j < arr.Shape[1]; j++ {
			result.Set(arr.At(i, j), j, i)
		}
	}

	return result
}

func (arr *NDArray[T]) Multiply(other *NDArray[T]) *NDArray[T] {
	if len(arr.Shape) != 2 || len(other.Shape) != 2 {
		panic("Matrix multiplication only supported for 2D arrays")
	}
	if arr.Shape[1] != other.Shape[0] {
		panic(fmt.Sprintf("Shape mismatch for matrix multiplication between %d and %d", arr.Shape, other.Shape))
	}

	newShape := []int{arr.Shape[0], other.Shape[1]}
	result := NewArray[T](newShape...)
	for i := range arr.Shape[0] {
		for j := range other.Shape[1] {
			var sum T
			for k := range arr.Shape[1] {
				sum += arr.At(i, k) * other.At(k, j)
			}
			result.Set(sum, i, j)
		}
	}

	return result
}

func (arr *NDArray[T]) Reshape(newShape ...int) *NDArray[T] {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}

	if newSize != arr.Size() {
		panic(fmt.Sprintf("Cannot reshape array of size %d to shape %d", arr.Size(), newShape))
	}

	result := NewArray[T](newShape...)
	copy(result.Data, arr.Data)

	return result
}

func (arr *NDArray[T]) Clip(minValue, maxValue int) *NDArray[T] {
	result := NewArray[T](arr.Shape...)
	for i := range arr.Size() {
		var value T
		if arr.Data[i] < T(minValue) {
			value = T(minValue)
		} else if arr.Data[i] > T(maxValue) {
			value = T(maxValue)
		} else {
			value = arr.Data[i]
		}
		result.Data[i] = value
	}

	return result
}

func (arr *NDArray[T]) Extend(other *NDArray[T]) *NDArray[T] {
	if len(arr.Shape) != 1 || len(other.Shape) != 1 {
		panic("Extend only supported for 1D arrays")
	}

	newShape := []int{arr.Shape[0] + other.Shape[0]}
	result := NewArray[T](newShape...)
	copy(result.Data, arr.Data)
	copy(result.Data[arr.Shape[0]:], other.Data)

	return result
}

func (arr *NDArray[T]) String() string {
	return fmt.Sprintf("NDArray(shape=%v, dtype=%s, data=%v)", arr.Shape, arr.Type, arr.Data)
}

func (arr *NDArray[T]) Print() {
	fmt.Println(arr.String())
}
