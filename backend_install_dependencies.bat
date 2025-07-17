@echo on
cd api || goto error
pip install -r requirements.txt || goto error
goto end

:error
echo.
echo [ERROR] Something went wrong. Please check the output above.
echo.

:end
pause
