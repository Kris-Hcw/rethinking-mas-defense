@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_bailian_reproduction.ps1" %*
if errorlevel 1 (
  echo.
  echo 实验运行失败，请查看上面的错误信息。
  pause
  exit /b 1
)
echo.
echo 实验已完成。
pause
