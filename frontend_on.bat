@echo on
cd client || goto error
npm run dev || goto error
goto end

:error
echo.
echo [ERROR] Something went wrong. Please check the output above.
echo.

:end
pause
