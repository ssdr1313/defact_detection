import sys


if __name__=="__main__":
    print("开始检查运行环境")
    print(f"Python 版本")
    print(f"ython: {sys.version}")

    print(f"PyTorch 与 CUDA 检查")
    try:
        import torch
        torch_ver = torch.__version__
        print(f"PyTorch 版本: {torch_ver}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 可用: {cuda_available}")
    except ImportError:
        print("错误：无法导入 torch，请安装 PyTorch！")
    except Exception as e:
        print(f"Torch 检查出错: {e}")


