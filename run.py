import os
import sys
import subprocess


def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'jieba', 'nltk',
        'gensim', 'tensorflow', 'flask', 'matplotlib', 'seaborn', 'joblib'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")

    return missing_packages


def main():
    print("=== 文本情感分析系统 ===")
    print("1. 检查依赖")
    print("2. 训练模型")
    print("3. 命令行预测")
    print("4. 启动Web服务")
    print("5. 退出")

    while True:
        choice = input("\n请选择操作 (1-5): ").strip()

        if choice == '1':
            print("\n检查依赖...")
            missing = check_dependencies()
            if missing:
                print(f"\n缺少以下包: {', '.join(missing)}")
                install = input("是否安装? (y/n): ").lower()
                if install == 'y':
                    for package in missing:
                        if package == 'sklearn':
                            package = 'scikit-learn'
                        os.system(f"pip install {package}")
            else:
                print("所有依赖都已安装!")

        elif choice == '2':
            print("开始训练模型...")
            try:
                from train import main as train_main
                train_main()
            except Exception as e:
                print(f"训练失败: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '3':
            print("启动命令行预测...")
            try:
                from predict import main as predict_main
                predict_main()
            except Exception as e:
                print(f"预测失败: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '4':
            print("启动Web服务...")
            print("服务将在 http://localhost:5000 启动")
            try:
                os.system("python app.py")
            except KeyboardInterrupt:
                print("\nWeb服务已停止")

        elif choice == '5':
            print("再见！")
            break

        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()