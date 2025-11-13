import tensorrt as trt

def build_engine(onnx_path, engine_path, fp16=True):
    """
    Convert ONNX model to TensorRT engine
    
    Args:
        onnx_path: path to ONNX model
        engine_path: path to save engine file
        fp16: use FP16 precision (faster, slightly less accurate)
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX file
    print(f"[INFO] Parsing ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("[ERROR] Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(f"  Error {error}: {parser.get_error(error)}")
            return False
    
    # Configure builder
    config = builder.create_builder_config()
    
    if fp16:
        print("[INFO] Using FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("[INFO] Using FP32 precision")
    
    # Set workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Build engine
    print("[INFO] Building TensorRT engine...")
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        print("[ERROR] Failed to build engine")
        return False
    
    # Save engine
    print(f"[INFO] Saving engine to: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    print("[INFO] Engine built successfully!")
    return True


if __name__ == "__main__":
    # Convert ONNX to engine
    onnx_file = "ONNX/yolov8n.onnx"
    engine_file = "ONNX/yolov8n.engine"
    
    build_engine(onnx_file, engine_file, fp16=True)