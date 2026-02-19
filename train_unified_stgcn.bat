@echo off
echo ========================================
echo Training Unified Spatiotemporal GCN
echo ========================================
call .venv\Scripts\activate.bat
python train_unified_stgcn.py
pause
