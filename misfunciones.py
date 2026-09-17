
import pandas as pd
import numpy as np

#Extrae del codebook de la ESS los cod missing para cada variable
def extraer_cod_missing(soup, columnas):

    cod_missing = {}

    for col in columnas:

        seccion = soup.find("h3", id=col)

        if seccion is None:
            continue

        tabla = seccion.find_next("table")

        if tabla is None:
            continue

        missing_var = []

        for fila in tabla.find_all("tr")[1:]:

            celdas = fila.find_all("td")

            if len(celdas) < 2:
                continue

            valor = celdas[0].get_text(strip=True)
            categoria = celdas[1].get_text(strip=True)

            if "*" in categoria:
                missing_var.append(valor)

        if missing_var:
            cod_missing[col] = missing_var

    return cod_missing



#2. Se comprueba qué códigos missing están en df y su frecuencia:
def comprobar_cod_missing(df, cod_missing):

    resultados = []

    for variable, codigos in cod_missing.items():

        for codigo in codigos:

            # Convertimos el código al tipo de la variable cuando sea posible
            try:
                cod_convertido = float(codigo)
            except ValueError:
                cod_convertido = codigo

            n = (df[variable] == cod_convertido).sum()

            if n > 0:
                resultados.append({
                    "variable": variable,
                    "codigo": codigo,
                    "frecuencia": n,
                    "porcentaje": round(n / len(df) * 100, 3)
                })

    return pd.DataFrame(resultados)

# 3. Resume los códigos missing por variable y calcula su peso
# sobre las observaciones disponibles
def resumir_cod_missing(df, resultado_cod):

    resumen_cod = (
        resultado_cod
        .groupby("variable", as_index=False)
        .agg(
            n_codigos_especiales=("frecuencia", "sum")
        )
    )

    # Número de observaciones disponibles de cada variable
    resumen_cod["n_disponibles"] = resumen_cod["variable"].apply(
        lambda var: df[var].notna().sum()
    )

    # Porcentaje de códigos especiales sobre los datos disponibles
    resumen_cod["pct_codigos_especiales"] = (
        resumen_cod["n_codigos_especiales"]
        / resumen_cod["n_disponibles"]
        * 100
    )

    # Ordenamos de mayor a menor
    resumen_cod = resumen_cod.sort_values(
        "pct_codigos_especiales",
        ascending=False
    )

    return resumen_cod




#4. Asocia los códigos missing con su significado según el codebook
def asociar_tipo_cod_missing(soup, cod_missing):

    tipos_cod_missing = []

    for variable, codigos in cod_missing.items():

        seccion = soup.find("h3", id=variable)

        if seccion is None:
            continue

        tabla = seccion.find_next("table")

        if tabla is None:
            continue

        for fila in tabla.find_all("tr")[1:]:

            celdas = fila.find_all("td")

            if len(celdas) < 2:
                continue

            codigo = celdas[0].get_text(strip=True)
            respuesta = celdas[1].get_text(strip=True)

            if codigo in codigos:

                tipos_cod_missing.append({
                    "variable": variable,
                    "codigo": codigo,
                    "tipo_respuesta": (
                        respuesta
                        .replace("*", "")
                        .strip()
                    )
                })

    return pd.DataFrame(tipos_cod_missing)


#5. Recodifica como NaN los códigos missing definidos en el codebook
def recodificar_cod_missing(df, cod_missing):

    # Trabajamos sobre una copia para no modificar el df original
    df_limpio = df.copy()

    for variable, codigos in cod_missing.items():

        if variable not in df_limpio.columns:
            continue

        cod_convertidos = []

        for codigo in codigos:

            # Adaptamos el código al tipo de dato cuando sea posible
            try:
                cod_convertidos.append(float(codigo))

            except ValueError:
                cod_convertidos.append(codigo)

        # Sustituimos los códigos especiales por NaN
        df_limpio[variable] = df_limpio[variable].replace(
            cod_convertidos,
            np.nan
        )

    return df_limpio


#6. Tabla de diagnóstico de las variales con sus misings y rondas
def diagnostico_variables(df_original, df_limpio):

    resultados = []

    rondas = sorted(df_original["essround"].unique())

    for variable in df_original.columns:

        missing_antes = df_original[variable].isna().mean() * 100
        missing_despues = df_limpio[variable].isna().mean() * 100

        n_validos = df_limpio[variable].notna().sum()

        rondas_validas = []

        for ronda in rondas:

            datos = df_limpio.loc[
                df_limpio["essround"] == ronda,
                variable
            ]

            if datos.notna().any():
                rondas_validas.append(int(ronda))

        resultados.append({
            "variable": variable,
            "n_rondas": len(rondas_validas),
            "rondas": ", ".join(map(str, rondas_validas)),
            "n_validos": n_validos,
            "pct_validos": n_validos / len(df_limpio) * 100,
            "pct_missing_antes": missing_antes,
            "pct_missing_despues": missing_despues,
            "incremento": missing_despues - missing_antes
        })

    return (
        pd.DataFrame(resultados)
        .sort_values("pct_missing_despues", ascending=False)
        .reset_index(drop=True)
    )
