@echo on
cd api || goto error
python app.py || goto error
goto end

:error
echo.
echo [ERROR] Something went wrong. Please check the output above.
echo.

:end
pause
