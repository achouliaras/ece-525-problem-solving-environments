@echo off
cd /D D:\Projects\Sxolh\PSEs\Final_project\scripts

start cmd.exe /c "predictor.bat"
start cmd.exe /c "trainer.bat"
timeout /t 5 /nobreak
start cmd.exe /c "sample_app.bat"