"""
data_loader.py
===============

Funções utilitárias para carregar o dataset e prepará-lo para os módulos
da aplicação. Aqui usamos Pandas/NumPy livremente para *carregar e
manipular* os dados (isso é permitido pelo enunciado) — mas nenhuma
medida estatística é calculada aqui com funções prontas; isso é papel
exclusivo de `core/minhastats.py`.
"""

from __future__ import annotations

import math
import os

import pandas as pd

from . import minhastats as ms

CAMINHO_DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


def carregar_dataset(caminho: str = CAMINHO_DATASET) -> pd.DataFrame:
    """Carrega o dataset CSV em um DataFrame do Pandas."""
    df = pd.read_csv(caminho)
    return df


LIMIAR_CARDINALIDADE_ID = 0.5  # colunas com mais de 50% de valores únicos são
                                # tratadas como identificadores (ex.: Rank, Name)
                                # e ficam de fora das listas de análise, pois não
                                # são variáveis estatísticas de verdade.


def _e_coluna_identificadora(df: pd.DataFrame, coluna: str) -> bool:
    n = len(df)
    if n == 0:
        return False
    return (df[coluna].nunique() / n) > LIMIAR_CARDINALIDADE_ID


def colunas_numericas(df: pd.DataFrame) -> list:
    """Retorna os nomes das colunas numéricas (int/float) do DataFrame,
    excluindo colunas identificadoras (ex.: uma coluna 'Rank' que é apenas
    um índice de 1 a n, sem significado estatístico como variável)."""
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not _e_coluna_identificadora(df, c)]


def colunas_categoricas(df: pd.DataFrame) -> list:
    """Retorna os nomes das colunas categóricas (texto/objeto/categoria),
    excluindo colunas identificadoras (ex.: um nome único por linha, que
    funciona como ID e não como categoria)."""
    return [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and not _e_coluna_identificadora(df, c)]


def serie_numerica_limpa(df: pd.DataFrame, coluna: str) -> list:
    """Extrai uma coluna numérica como lista de floats, removendo valores
    ausentes (NaN)."""
    serie = df[coluna].dropna()
    return [float(v) for v in serie.tolist()]


def serie_categorica_limpa(df: pd.DataFrame, coluna: str) -> list:
    """Extrai uma coluna categórica como lista de strings, removendo
    valores ausentes."""
    serie = df[coluna].dropna()
    return [str(v) for v in serie.tolist()]


def numero_classes_sturges(n: int) -> int:
    """Regra de Sturges para número de classes de uma tabela de frequências
    de variável contínua: k = 1 + log2(n), arredondado para cima.
    """
    if n <= 0:
        return 1
    return max(1, math.ceil(1 + math.log2(n)))


def tabela_frequencias_numerica(valores: list, n_classes: int | None = None) -> list:
    """Monta uma tabela de frequências em classes para uma variável
    contínua, usando a regra de Sturges para decidir o número de classes
    quando não informado.

    Retorna uma lista de dicionários com: classe (rótulo "li ⊢ ls"),
    limite_inferior, limite_superior, freq_absoluta, freq_relativa (%),
    freq_acumulada.
    """
    if not valores:
        return []
    minimo = min(valores)
    maximo = max(valores)
    n = len(valores)
    k = n_classes or numero_classes_sturges(n)

    amplitude_total = maximo - minimo
    if amplitude_total == 0:
        largura = 1.0
        k = 1
    else:
        largura = amplitude_total / k

    limites = [minimo + i * largura for i in range(k + 1)]
    contagens = [0] * k
    for v in valores:
        if v == maximo:
            idx = k - 1  # o último valor entra na última classe (fechada à direita)
        else:
            idx = int((v - minimo) / largura)
            idx = min(idx, k - 1)
        contagens[idx] += 1

    tabela = []
    acumulada = 0
    for i in range(k):
        acumulada += contagens[i]
        tabela.append({
            "classe": f"{limites[i]:.2f} ⊢ {limites[i + 1]:.2f}",
            "limite_inferior": limites[i],
            "limite_superior": limites[i + 1],
            "freq_absoluta": contagens[i],
            "freq_relativa": (contagens[i] / n) * 100.0,
            "freq_acumulada": acumulada,
        })
    return tabela


def tabela_frequencias_categorica(valores: list) -> list:
    """Monta uma tabela de frequências simples para uma variável
    categórica: categoria, frequência absoluta e relativa (%), ordenada
    da mais para a menos frequente.
    """
    n = len(valores)
    contagem: dict = {}
    for v in valores:
        contagem[v] = contagem.get(v, 0) + 1
    itens = sorted(contagem.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {
            "categoria": categoria,
            "freq_absoluta": freq,
            "freq_relativa": (freq / n) * 100.0,
        }
        for categoria, freq in itens
    ]


def interpretar_assimetria(skew: float) -> str:
    """Gera uma frase de interpretação automática da assimetria."""
    if abs(skew) < 0.5:
        return (f"A distribuição é aproximadamente simétrica "
                f"(coeficiente de assimetria = {skew:.3f}).")
    direcao = "à direita (cauda longa para valores altos)" if skew > 0 \
        else "à esquerda (cauda longa para valores baixos)"
    intensidade = "moderadamente" if abs(skew) < 1 else "fortemente"
    return (f"A distribuição é {intensidade} assimétrica {direcao} "
            f"(coeficiente de assimetria = {skew:.3f}).")


def interpretar_outliers(qtd_outliers: int, n: int) -> str:
    """Gera uma frase de interpretação automática sobre outliers."""
    if qtd_outliers == 0:
        return "Nenhum outlier foi detectado pela regra do IQR (1.5×IQR)."
    pct = (qtd_outliers / n) * 100.0
    return (f"Foram detectados {qtd_outliers} outlier(s) pela regra do IQR "
            f"(1.5×IQR), representando {pct:.2f}% das observações válidas.")


def interpretar_correlacao(r: float) -> str:
    """Gera uma frase de interpretação automática da força/direção da
    correlação de Pearson."""
    forca = abs(r)
    if forca < 0.1:
        intensidade = "praticamente nula"
    elif forca < 0.3:
        intensidade = "fraca"
    elif forca < 0.5:
        intensidade = "moderada"
    elif forca < 0.7:
        intensidade = "forte"
    else:
        intensidade = "muito forte"
    direcao = "positiva" if r >= 0 else "negativa"
    return f"Correlação {direcao} {intensidade} (r = {r:.3f})."
