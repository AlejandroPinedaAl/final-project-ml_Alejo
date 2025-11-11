@echo off
echo.
echo === Python Virtual Environment Setup ===
echo.

REM Desactivar el ambiente virtual actual si está activo
if defined VIRTUAL_ENV (
    echo Desactivando ambiente virtual actual: %VIRTUAL_ENV%
    call deactivate
)

echo Creando nuevo ambiente virtual: final-project-ml-venv
py -m venv final-project-ml-venv

echo Activando ambiente virtual...
call final-project-ml-venv\Scripts\activate

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Ambiente virtual creado con exito!
    echo Python actual: 
    where python
    
    echo.
    echo === Instalando requisitos ===
    if exist requirements.txt (
        echo requirements.txt encontrado, instalando librerias...
        pip install --no-cache-dir -r requirements.txt
        
        if %ERRORLEVEL% EQU 0 (
            echo.
            echo Todas las librerías instaladas correctamente.

            echo.
            echo === Registrando ambiente virtual con Jupyter ===
            echo Registrando kernel con Jupyter...
            python -m ipykernel install --user --name=final-project-ml-venv --display-name="Final Project ML - Marketing Campaign"
            
            if %ERRORLEVEL% EQU 0 (
                echo Ambiente virtual registrado como kernel de Jupyter correctamente.
                echo Ahora puedes seleccionar "Final Project ML - Marketing Campaign" en Jupyter notebook.
            ) else (
                echo Advertencia: Fallo al registrar el ambiente virtual como kernel de Jupyter.
            )

        ) else (
            echo.
            echo Error instalando las librerías desde requirements.txt.
        )
    ) else (
        echo.
        echo Advertencia: requirements.txt no fue encontrado en el directorio actual.
    )
) else (
    echo.
    echo Error activando el ambiente virtual.
)

echo.
echo Setup completado!
pause