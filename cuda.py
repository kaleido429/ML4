import tensorflow as tf
import os

print("TensorFlow 버전:", tf.__version__)
print("CUDA 지원 빌드:", tf.test.is_built_with_cuda())
print("사용 가능한 GPU:", tf.config.list_physical_devices('GPU'))
print("CUDA_PATH 환경변수:", os.environ.get('CUDA_PATH', 'None'))

# CUDA 라이브러리 찾기 시도
try:
    import ctypes
    ctypes.CDLL("nvcuda.dll")
    print("CUDA 드라이버 라이브러리: 찾음")
except OSError:
    print("CUDA 드라이버 라이브러리: 찾을 수 없음")

try:
    ctypes.CDLL("cudart64_12.dll")  # 버전에 따라 숫자 변경
    print("CUDA 런타임 라이브러리: 찾음")
except OSError:
    print("CUDA 런타임 라이브러리: 찾을 수 없음")

# Check if GPU is available
tf.debugging.set_log_device_placement(True)
def is_gpu_available():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs available: {[gpu.name for gpu in gpus]}")
            return True
        else:
            print("No GPUs found.")
            return False
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        return False