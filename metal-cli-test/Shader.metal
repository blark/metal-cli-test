//
//  Shader.metal
//  metal-cli-test
//
//  Created by Mark Baseggio on 2022-12-19.
//

#include <metal_stdlib>
using namespace metal;

// All data types must be the same as defined in Swift
typedef unsigned int uint;
typedef int DataType;

kernel void parsum(const device DataType* data [[ buffer(0) ]],
                   const device uint& dataLength [[ buffer(1) ]],
                   device DataType* sums [[ buffer(2) ]],
                   const device uint& elementsPerSum [[ buffer(3) ]],
                   
                   const uint tgPos [[ threadgroup_position_in_grid ]],
                   const uint tPerTg [[ threads_per_threadgroup ]],
                   const uint tPos [[ thread_position_in_threadgroup ]]) {
    // Calculate the total index value based on the group index, batch index, and position in the group, which is unique
    uint resultIndex = tgPos * tPerTg + tPos;
    // Calculate the starting and ending positions of the data for this batch
    uint dataIndex = resultIndex * elementsPerSum; // Where the summation should begin
    uint endIndex = dataIndex + elementsPerSum < dataLength ? dataIndex + elementsPerSum : dataLength; // The index where summation should end
    // Sum the data for this batch
    for (; dataIndex < endIndex; dataIndex++)
        sums[resultIndex] += data[dataIndex];
}
