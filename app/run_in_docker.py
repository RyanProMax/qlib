#!/usr/bin/env python3
"""
Docker中执行Python命令的工具脚本
连接到现有的qlib_jupyter容器
"""
import subprocess
import sys
import os
from pathlib import Path

def run_in_docker(script_path=None, command=None, jupyter=False):
    """
    在现有的Docker容器中执行Python脚本或命令
    
    Args:
        script_path: Python脚本路径（相对于app目录）
        command: 直接执行的命令
        jupyter: 是否启动Jupyter notebook（容器已有端口映射）
    """
    # 使用现有的qlib_jupyter容器
    container_name = "qlib_jupyter"
    
    if jupyter:
        print("Jupyter notebook应该已经在运行中...")
        print("请访问 http://localhost:8888 来使用Jupyter")
        print("如果没有运行，请在容器中手动启动：")
        print(f"docker exec -it {container_name} bash -c 'cd /workspace && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'")
        return 0
    elif script_path:
        # 检查脚本是否存在
        if not Path(script_path).exists():
            print(f"❌ 错误: 脚本文件不存在: {script_path}")
            return 1
        
        docker_cmd = [
            "docker", "exec", "-it",
            container_name,
            "bash", "-c", 
            f"python -m pip install matplotlib seaborn --quiet && "
            f"cd /workspace/app && "
            f"python {script_path}"
        ]
    elif command:
        # 在现有容器中执行自定义命令
        docker_cmd = [
            "docker", "exec", "-it",
            container_name,
            "bash", "-c", f"cd /workspace && {command}"
        ]
    else:
        # 连接到现有容器的交互式shell
        docker_cmd = [
            "docker", "exec", "-it",
            container_name,
            "bash", "-c", "cd /workspace && bash"
        ]
    
    print(f"🚀 执行Docker命令: {' '.join(docker_cmd)}")
    
    try:
        result = subprocess.run(docker_cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker命令执行失败: {e}")
        print(f"请确保容器 {container_name} 正在运行")
        print("可以使用以下命令检查容器状态：docker ps")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        return 130

def check_container_status():
    """检查容器状态"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=qlib_jupyter", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True, text=True, check=True
        )
        print("📊 容器状态:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError:
        print("❌ 无法获取容器状态，请检查Docker是否运行")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="在现有的qlib_jupyter容器中执行命令")
    parser.add_argument("--script", "-s", help="要执行的Python脚本路径（相对于app目录）")
    parser.add_argument("--command", "-c", help="要执行的自定义命令")
    parser.add_argument("--jupyter", "-j", action="store_true", help="检查Jupyter notebook状态")
    parser.add_argument("--demo", action="store_true", help="运行基本demo")
    parser.add_argument("--status", action="store_true", help="检查容器状态")
    
    args = parser.parse_args()
    
    if args.status:
        check_container_status()
    elif args.jupyter:
        print("检查Jupyter notebook状态...")
        run_in_docker(jupyter=True)
    elif args.demo:
        print("运行qlib基本demo...")
        # 基本demo使用容器中预安装的qlib
        demo_cmd = "cd /tmp && python -c \"" + """
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK

print('=== Qlib基本demo ===')
provider_uri = '~/.qlib/qlib_data/cn_data'
GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
qlib.init(provider_uri=provider_uri, region=REG_CN)

model = init_instance_by_config(CSI300_GBDT_TASK['model'])
dataset = init_instance_by_config(CSI300_GBDT_TASK['dataset'])

example_df = dataset.prepare('train')
print(example_df.head())
print('基本demo运行成功！')
""" + "\""
        run_in_docker(command=demo_cmd)
    elif args.script:
        print(f"📊 执行脚本: {args.script}")
        run_in_docker(script_path=args.script)
    elif args.command:
        print(f"📊 执行命令: {args.command}")
        run_in_docker(command=args.command)
    else:
        print("🔗 连接到qlib_jupyter容器...")
        run_in_docker() 
