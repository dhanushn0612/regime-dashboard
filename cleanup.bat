@echo off
echo ============================================
echo  REGIME DASHBOARD — CLEANUP OBSOLETE FILES
echo ============================================

set DIR=C:\Users\Dhanush\regime-dashboard\data_pipeline

echo Deleting obsolete files...

if exist "%DIR%\backtest.py" (
    del "%DIR%\backtest.py"
    echo   Deleted: backtest.py
)

if exist "%DIR%\backtest_extended.py" (
    del "%DIR%\backtest_extended.py"
    echo   Deleted: backtest_extended.py
)

if exist "%DIR%\diagnose.py" (
    del "%DIR%\diagnose.py"
    echo   Deleted: diagnose.py
)

if exist "%DIR%\diagnose_screener.py" (
    del "%DIR%\diagnose_screener.py"
    echo   Deleted: diagnose_screener.py
)

if exist "%DIR%\fii_dii_scraper.py" (
    del "%DIR%\fii_dii_scraper.py"
    echo   Deleted: fii_dii_scraper.py
)

if exist "%DIR%\build_fundamental_snapshots_old.py" (
    del "%DIR%\build_fundamental_snapshots_old.py"
    echo   Deleted: build_fundamental_snapshots_old.py
)

echo.
echo Final file list:
echo ================
dir "%DIR%" /B /A-D

echo.
echo ============================================
echo  CLEANUP COMPLETE
echo ============================================
pause
