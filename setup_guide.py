"""
Quick Setup and Usage Guide for QCNN Implementation
Run this script to check your environment and get started
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  Warning: Python 3.8+ recommended")
    return True


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_dependencies():
    """Check all required dependencies"""
    print("\nChecking dependencies...")
    
    packages = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('networkx', 'networkx'),
        ('streamlit', 'streamlit'),
        ('pennylane', 'pennylane'),
        ('plotly', 'plotly'),
        ('pandas', 'pandas'),
        ('seaborn', 'seaborn'),
    ]
    
    all_installed = True
    missing = []
    
    for pkg_name, import_name in packages:
        if check_package(pkg_name, import_name):
            print(f"  ✓ {pkg_name}")
        else:
            print(f"  ✗ {pkg_name} - NOT INSTALLED")
            all_installed = False
            missing.append(pkg_name)
    
    if not all_installed:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install -r requirements.txt")
    else:
        print("\n✓ All dependencies installed!")
    
    return all_installed


def check_files():
    """Check if required files exist"""
    print("\nChecking project files...")
    
    required_files = [
        'chembur_network.py',
        'traffic_data_generator.py',
        'spatiotemporal_gcn.py',
        'train_model.py',
        'qcnn_model.py',
        'train_qcnn.py',
        'compare_models.py',
        'comparison_dashboard.py',
        'requirements.txt'
    ]
    
    all_present = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            all_present = False
    
    return all_present


def check_models():
    """Check if models are trained"""
    print("\nChecking trained models...")
    
    gcn_exists = Path('traffic_gcn_model.pth').exists()
    qcnn_exists = Path('qcnn_traffic_model.pth').exists()
    
    if gcn_exists:
        print("  ✓ GCN model (traffic_gcn_model.pth)")
    else:
        print("  ✗ GCN model not trained")
    
    if qcnn_exists:
        print("  ✓ QCNN model (qcnn_traffic_model.pth)")
    else:
        print("  ✗ QCNN model not trained")
    
    return gcn_exists, qcnn_exists


def print_usage_guide():
    """Print usage guide"""
    print("\n" + "=" * 80)
    print("QCNN PROJECT USAGE GUIDE")
    print("=" * 80)
    
    print("\n📝 WORKFLOW:")
    print("-" * 80)
    
    print("\n1️⃣  Train Classical GCN (if not already done):")
    print("   python train_model.py")
    print("   or: run_dashboard.bat (Windows)")
    
    print("\n2️⃣  Train Quantum CNN:")
    print("   python train_qcnn.py")
    print("   or: train_qcnn.bat (Windows)")
    
    print("\n3️⃣  Compare Models:")
    print("   python compare_models.py")
    print("   or: run_comparison.bat (Windows)")
    
    print("\n4️⃣  Launch Comparison Dashboard:")
    print("   streamlit run comparison_dashboard.py")
    print("   or: run_comparison_dashboard.bat (Windows)")
    
    print("\n" + "=" * 80)
    print("🔬 WHAT EACH SCRIPT DOES:")
    print("=" * 80)
    
    print("\n📦 qcnn_model.py")
    print("   - Defines Quantum CNN architecture using PennyLane")
    print("   - Implements quantum feature encoding")
    print("   - Creates quantum convolutional layers")
    print("   - Handles quantum circuit execution")
    
    print("\n🎯 train_qcnn.py")
    print("   - Trains the QCNN model")
    print("   - Generates performance metrics")
    print("   - Saves trained model and visualizations")
    print("   - Outputs: qcnn_traffic_model.pth, qcnn_metrics.json")
    
    print("\n📊 compare_models.py")
    print("   - Loads both GCN and QCNN models")
    print("   - Evaluates on same test data")
    print("   - Compares metrics and latency")
    print("   - Outputs: model_comparison.json")
    
    print("\n📈 comparison_dashboard.py")
    print("   - Interactive Streamlit dashboard")
    print("   - Side-by-side model comparison")
    print("   - Visualizations and insights")
    print("   - Performance analysis")
    
    print("\n" + "=" * 80)
    print("💡 TIPS:")
    print("=" * 80)
    print("  • QCNN training is slower due to quantum simulation")
    print("  • Reduce batch_size or epochs if memory issues occur")
    print("  • Use the dashboard for interactive exploration")
    print("  • Check QCNN_README.md for detailed documentation")
    
    print("\n" + "=" * 80)


def main():
    """Main function"""
    print("=" * 80)
    print("QCNN SETUP CHECKER")
    print("=" * 80)
    
    # Check Python
    check_python_version()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check files
    files_ok = check_files()
    
    # Check models
    gcn_exists, qcnn_exists = check_models()
    
    # Summary
    print("\n" + "=" * 80)
    print("SYSTEM STATUS")
    print("=" * 80)
    
    if deps_ok and files_ok:
        print("\n✅ Environment ready!")
        
        if not gcn_exists and not qcnn_exists:
            print("\n🚀 NEXT STEP: Train models")
            print("   Start with: python train_model.py")
            print("   Then run: python train_qcnn.py")
        elif gcn_exists and not qcnn_exists:
            print("\n🚀 NEXT STEP: Train QCNN")
            print("   Run: python train_qcnn.py")
        elif gcn_exists and qcnn_exists:
            print("\n🚀 NEXT STEP: Compare models")
            print("   Run: python compare_models.py")
            print("   Then: streamlit run comparison_dashboard.py")
        else:
            print("\n🚀 NEXT STEP: Train GCN model first")
            print("   Run: python train_model.py")
    else:
        print("\n❌ Setup incomplete!")
        if not deps_ok:
            print("   Install dependencies: pip install -r requirements.txt")
        if not files_ok:
            print("   Some project files are missing")
    
    # Print guide
    print_usage_guide()
    
    print("\n📖 For detailed information, see: QCNN_README.md")
    print("=" * 80)


if __name__ == '__main__':
    main()
