#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>


class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char * msg) noexcept override {
	(void)severity;
	std::cout << msg << std::endl;
  }
};

uint8_t sizeof_dtype(nvinfer1::DataType dtype) {
	switch (dtype) {
		case nvinfer1::DataType::kFLOAT: return 4;
		case nvinfer1::DataType::kHALF: return 2;
		case nvinfer1::DataType::kINT8: return 1;
		case nvinfer1::DataType::kINT32: return 4;
		case nvinfer1::DataType::kBOOL: return 1;
		case nvinfer1::DataType::kUINT8: return 1;
		case nvinfer1::DataType::kFP8: return 1;
		default: throw std::invalid_argument("dtype must be variant of DataType");
	}
}

size_t sizeof_tensor(std::unique_ptr<nvinfer1::IExecutionContext> & context, char const * name) {
	auto shape = context->getTensorShape(name);
	auto dtype = context->getEngine().getTensorDataType(name);
	auto format = context->getEngine().getTensorFormat(name);

	if (format != nvinfer1::TensorFormat::kLINEAR) {
		auto vectorized_dimension = context->getEngine().getTensorComponentsPerElement(name);
		auto components_per_element = context->getEngine().getTensorComponentsPerElement(name);
		shape.d[vectorized_dimension] += components_per_element - (shape.d[vectorized_dimension] % components_per_element);
	}

	auto size = 1;
	for (int dim = 0; dim < shape.nbDims; dim++) {
		size *= shape.d[dim];
	}
	size *= sizeof_dtype(dtype);

	return size;
}

class HostMemory {
	public:
		HostMemory(size_t size) {
			size_ = size;
			auto err = cudaMallocHost(&ptr_, size);
			if (err != cudaSuccess) {
				throw std::bad_alloc();
			}
		}

		~HostMemory() {
			auto err = cudaFreeHost(ptr_);
			if (err != cudaSuccess) {
				std::abort();
			}
		}

		HostMemory(const HostMemory& other) = delete;
		HostMemory& operator= (const HostMemory&) = delete;

		HostMemory(HostMemory&& o) noexcept {
			ptr_ = o.ptr_;
			size_ = o.size_;
			o.ptr_ = nullptr;
			o.size_ = 0;
		}

		void * data() {
			return ptr_;
		}

		void randomize(std::default_random_engine & re) {
			typedef std::default_random_engine::result_type rand_t;
			std::generate((rand_t *)ptr_, (rand_t *)ptr_ + (size_ / sizeof(rand_t)), std::ref(re));
		}

	private:
		void * ptr_;
		size_t size_;
};

struct IOBuffers {
	std::vector<HostMemory> inputs;
	std::vector<HostMemory> outputs;

	void randomize(std::default_random_engine & re) {
		std::for_each(inputs.begin(), inputs.end(), [&re](HostMemory & m ) { m.randomize(re); });
		std::for_each(outputs.begin(), outputs.end(), [&re](HostMemory & m ) { m.randomize(re); });
	}
};

IOBuffers set_io(std::unique_ptr<nvinfer1::IExecutionContext> & context) {
	IOBuffers buffers;

	int32_t num_io = context->getEngine().getNbIOTensors();
	for (int32_t idx = 0; idx < num_io; idx++) {
		auto name = context->getEngine().getIOTensorName(idx);
		auto mode = context->getEngine().getTensorIOMode(name);
		auto size = sizeof_tensor(context, name);
		HostMemory buffer(size);

		switch (mode) {
			case nvinfer1::TensorIOMode::kINPUT:
				context->setInputTensorAddress(name, buffer.data());
				buffers.inputs.push_back(std::move(buffer));
				break;
			case nvinfer1::TensorIOMode::kOUTPUT:
				context->setTensorAddress(name, buffer.data());
				buffers.outputs.push_back(std::move(buffer));
				break;
			default: std::abort();
		}
	}

	return buffers;
}

class Stream {
	public:
		Stream() {
			auto err = cudaStreamCreate(&ptr_);
			if (err != cudaSuccess) {
				throw std::runtime_error("failed to create stream");
			}
		}

		~Stream() {
			if (cudaStreamSynchronize(ptr_) != cudaSuccess) {
				std::abort();
			}
			if (cudaStreamDestroy(ptr_) != cudaSuccess) {
				std::abort();
			}
		}

		void synchronize() {
			if (cudaStreamSynchronize(ptr_) != cudaSuccess) {
				throw std::runtime_error("failed to synchronize stream");
			}
		}

		void begin_capture() {
			if (cudaStreamBeginCapture(ptr_, cudaStreamCaptureModeGlobal) != cudaSuccess) {
				throw std::runtime_error("failed to begin capture");
			}
		}

		cudaStream_t inner() {
			return ptr_;
		}
	
	private:
		cudaStream_t ptr_;
};

class Graph {
	public:
		Graph(Stream & stream) {
			if (cudaStreamEndCapture(stream.inner(), &ptr_) != cudaSuccess) {
				throw std::runtime_error("failed to end capture");
			}
		}

		cudaGraph_t inner() {
			return ptr_;
		}

	private:
		cudaGraph_t ptr_;
};


class GraphExec {
	public:
		GraphExec(Graph & graph) {
			auto err = cudaGraphInstantiate(&ptr_, graph.inner(), nullptr, nullptr, 0);
			if (err != cudaSuccess) {
				throw std::runtime_error("could not instantiate graph as executable");
			}
		}

		void launch(Stream & stream) {
			if (cudaGraphLaunch(ptr_, stream.inner()) != cudaSuccess) {
				throw std::runtime_error("failed to launch graph instance");
			}
		}
	
	private:
		cudaGraphExec_t ptr_;
};

void try_enqueue(std::unique_ptr<nvinfer1::IExecutionContext> & context, Stream & stream) {
	if (!context->enqueueV3(stream.inner())) {
		throw std::runtime_error("failed to enqueue ExecutionContext");
	}
}


int main() {
    Logger logger;

	auto runtime_inner = nvinfer1::createInferRuntime(logger);
	if (!runtime_inner) { throw std::runtime_error("failed to create runtime"); }
    std::unique_ptr<nvinfer1::IRuntime> runtime(std::move(runtime_inner));

	std::string plan0_path = "model0.plan";
	std::string plan1_path = "model1.plan";

	// open plan files
	std::ifstream plan0_file(plan0_path);
	if (!plan0_file.is_open()) { throw std::runtime_error("failed to open plan0_path"); }

	std::ifstream plan1_file(plan1_path);
	if (!plan1_file.is_open()) { throw std::runtime_error("failed to open plan1_path"); }

	// load plan data
	auto plan0 = std::vector<char>(std::istreambuf_iterator<char>{plan0_file}, {});
	auto plan1 = std::vector<char>(std::istreambuf_iterator<char>{plan1_file}, {});

	// construct engines from plans
	auto engine0_inner = runtime->deserializeCudaEngine(plan0.data(), plan0.size());
	if (!engine0_inner) { throw std::runtime_error("failed to deserialize plan0"); }
	std::unique_ptr<nvinfer1::ICudaEngine> engine0(std::move(engine0_inner));

	auto engine1_inner = runtime->deserializeCudaEngine(plan1.data(), plan1.size());
	if (!engine1_inner) { throw std::runtime_error("failed to deserialize plan1"); }
	std::unique_ptr<nvinfer1::ICudaEngine> engine1(std::move(engine1_inner));

	// construct contexts from engines
	auto context0_inner = engine0->createExecutionContext();
	if (!context0_inner) { throw std::runtime_error("failed to create context0 from engine0"); }
	std::unique_ptr<nvinfer1::IExecutionContext> context0(std::move(context0_inner));

	auto context1_inner = engine1->createExecutionContext();
	if (!context1_inner) { throw std::runtime_error("failed to create context1 from engine1"); }
	std::unique_ptr<nvinfer1::IExecutionContext> context1(std::move(context1_inner));

	// set io buffers
	auto buffers0 = set_io(context0);
	auto buffers1 = set_io(context1);

	// set up streams
	Stream stream0;
	Stream stream1;

	// test enqueue
	try_enqueue(context0, stream0);
	stream0.synchronize();

	try_enqueue(context1, stream1);
	stream1.synchronize();

	// capture graphs
	stream0.begin_capture();
	try_enqueue(context0, stream0);
	Graph graph0(stream0);
	GraphExec graph_exec0(graph0);
	stream0.synchronize();

	stream1.begin_capture();
	try_enqueue(context1, stream1);
	Graph graph1(stream1);
	GraphExec graph_exec1(graph1);
	stream1.synchronize();

	// run fuzz
	std::default_random_engine re;
	for (uint64_t i = 0; true; i++) {
		std::cout << "\r" << i << std::flush;
		buffers0.randomize(re);
		buffers1.randomize(re);
		graph_exec0.launch(stream0);
		graph_exec1.launch(stream1);
		stream0.synchronize();
		stream1.synchronize();
	}

	return 0;
}

