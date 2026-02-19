@echo off
echo ========================================
echo Launching Spatiotemporal GCN Visualization
echo ========================================
call .venv\Scripts\activate.bat
streamlit run streamlit_spatiotemporalGCN.py
pause
