//
//  main.swift
//  metal-cli-test
//
//  Adapted from [this Stackoverflow post](https://stackoverflow.com/questions/38164634/compute-sum-of-array-values-in-parallel-with-metal-swift)
//  Article [in Chinese here](https://article.itxueyuan.com/rQ4n2)
//
//  Credit to ChatGPT for the Chinese comment translations ;)
//

import Metal
import Foundation


// Define the length of the dataset, a total of count data to be summed
// In swift, underscores can be added to numeric literals to represent scientific notation, which is very characteristic
let count = 10_000_000

// Each group of elementsPerSum data is assigned to a kernel for summary
let elementsPerSum = 10_000

// Every data type must be a type compatible with C because the shader language that the GPU runs is derived from C++14.
typealias DataType = CInt // Data type, has to be the same as in the shader

//- device and kernel ----------------------------------------------------------------

let device = MTLCopyAllDevices()[0] // compiler complains if we use MTLCreateSystemDefaultDevice()!
// Load the default.metallib (compiled shader) in the current directory and use the parsum kernel function
let parsum = device.makeDefaultLibrary()!.makeFunction(name: "parsum")!
// Create a pipeline for GPU computation
let pipeline = try! device.makeComputePipelineState(function: parsum)

//- data -----------------------------------------------------------------------------

// Generate a random dataset
var data = (0..<count).map{ _ in DataType(arc4random_uniform(100)) }
// The total number of data passed to the kernel function, so it is also compatible with C
var dataCount = CUnsignedInt(count)
// The number of summaries per group passed to the kernel function, as above
var elementsPerSumC = CUnsignedInt(elementsPerSum)
// Return the number of batch summaries results
let resultsCount = (count + elementsPerSum - 1) / elementsPerSum


//-----------------------------------------------------------------------------

// Create two buffers that communicate with the GPU, one for input to the kernel function and one for the kernel function to return the result
let dataBuffer = device.makeBuffer(bytes: &data, length: MemoryLayout<DataType>.stride * count, options: [])!
let resultsBuffer = device.makeBuffer(length: MemoryLayout<DataType>.stride * resultsCount, options: [])!

// The result is a C pointer, which needs to be converted to a data type that can be accessed in Swift
let pointer = resultsBuffer.contents().bindMemory(to: DataType.self, capacity: resultsCount)
let results = UnsafeBufferPointer<DataType>(start: pointer, count: resultsCount)

// Create a GPU command queue
let queue = device.makeCommandQueue()!

// A GPU command buffer, where multiple computations are typically placed and submitted for execution at once
let cmds = queue.makeCommandBuffer()!

// A command encoder is used to package a call to a GPU kernel function, its arguments, etc. into one command
let encoder = cmds.makeComputeCommandEncoder()!

//------------------------------------------------------------------------------

// Set the function and its related parameters for a call to a GPU kernel function, as mentioned earlier, must use a type compatible with C.
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(dataBuffer, offset: 0, index: 0)
encoder.setBytes(&dataCount, length: MemoryLayout<CUnsignedInt>.size, index: 1)
encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
encoder.setBytes(&elementsPerSumC, length: MemoryLayout<CUnsignedInt>.size, index: 3)

// Set the number of tasks per group
// We have to calculate the sum `resultCount` times => amount of threadgroups is `resultsCount` / `threadExecutionWidth` (rounded up) because each threadgroup will process `threadExecutionWidth` threads
let threadgroupsPerGrid = MTLSize(width: (resultsCount + pipeline.threadExecutionWidth - 1) / pipeline.threadExecutionWidth, height: 1, depth: 1)
print(pipeline.threadExecutionWidth)

// Set the number of tasks per batch, must be a multiple of the group number above
// Here we set that each threadgroup should process `threadExecutionWidth` threads, the only important thing for performance is that this number is a multiple of `threadExecutionWidth` (here 1 times)
let threadsPerThreadgroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)

// Assign task threads
encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

// Finish all settings for a call
encoder.endEncoding()

//------------------------------------------------------------------------------

var start, end: UInt64
var result: DataType = 0

//------------------------------------------------------------------------------

start = mach_absolute_time()

// Actually submit the task
cmds.commit()

// Wait for the GPU calculation to complete
cmds.waitUntilCompleted()

// The GPU calculation is summarized in batches, the number is very small, and the CPU is used to perform a complete summary at the end
for elem in results {
    result += elem
}

end = mach_absolute_time()

print("Metal result: \(result), time: \(Double(end - start) / Double(NSEC_PER_SEC))")

//------------------------------------------------------------------------------

result = 0

// The following is a complete calculation using the CPU and displays the result and time consumed
start = mach_absolute_time()
data.withUnsafeBufferPointer { buffer in
    for elem in buffer {
        result += elem
    }
}
end = mach_absolute_time()
print("CPU result: \(result), time: \(Double(end - start) / Double(NSEC_PER_SEC))")
