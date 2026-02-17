@echo off
echo ============================================
echo   Training Quantum CNN for Traffic Prediction
echo ============================================
echo.

python train_qcnn.py

echo.
echo ============================================
echo   QCNN Training Complete!
echo ============================================
echo.
echo Next steps:
echo   1. Run: run_comparison.bat to compare with GCN
echo   2. Or run: streamlit run comparison_dashboard.py
echo.
pause
