@echo on
cd api || goto error
python generate_train_predict.py || goto error
goto end

:error
echo.
echo [ERROR] Something went wrong. Please check the output above.
echo.

:end
pause
