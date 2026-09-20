"""
simulacao.py
=============

Módulo 3 — Probabilidade e Simulação.

Usa `random`/`numpy.random` apenas como GERADOR DE NÚMEROS ALEATÓRIOS
(sorteio de moedas / amostras) — isso não é uma "função pronta de
estatística", é a fonte de aleatoriedade da simulação de Monte Carlo.
Todas as médias e estatísticas calculadas sobre os dados simulados usam
as funções de `core/minhastats.py`.
"""

from __future__ import annotations

import random

from . import minhastats as ms


def lei_dos_grandes_numeros(n_max: int, p: float = 0.5, seed: int | None = None) -> dict:
    """Simula `n_max` lançamentos de uma moeda (Bernoulli(p)) e acompanha a
    frequência relativa de "cara" conforme o número de lançamentos cresce,
    demonstrando a Lei dos Grandes Números: a frequência relativa converge
    para p à medida que n aumenta.

    Retorna um dicionário com:
      - n: lista [1, 2, ..., n_max]
      - freq_relativa: lista com a frequência relativa acumulada em cada n
      - p: probabilidade teórica usada
    """
    if n_max < 1:
        raise ValueError("n_max deve ser >= 1.")
    if not (0 < p < 1):
        raise ValueError("p deve estar entre 0 e 1 (exclusive).")

    rng = random.Random(seed)
    sucessos = 0
    n_vals = []
    freq_vals = []
    for i in range(1, n_max + 1):
        resultado = 1 if rng.random() < p else 0
        sucessos += resultado
        n_vals.append(i)
        freq_vals.append(sucessos / i)

    return {"n": n_vals, "freq_relativa": freq_vals, "p": p}


def teorema_central_limite(populacao: list, tamanho_amostra: int,
                            n_repeticoes: int, seed: int | None = None) -> dict:
    """Sorteia `n_repeticoes` amostras (com reposição) de tamanho
    `tamanho_amostra` a partir de `populacao` (uma coluna numérica real do
    dataset) e calcula a média de cada amostra usando `minhastats.media`.

    Demonstra o Teorema Central do Limite: a distribuição das médias
    amostrais se aproxima de uma Normal conforme `tamanho_amostra` cresce,
    mesmo que a população original não seja normal.

    Retorna um dicionário com:
      - medias_amostrais: lista de médias (uma por repetição)
      - media_populacional: média de toda a população (via minhastats)
      - desvio_padrao_populacional: desvio padrão populacional (via minhastats)
      - erro_padrao_teorico: desvio_padrao_populacional / sqrt(tamanho_amostra)
    """
    if tamanho_amostra < 1:
        raise ValueError("tamanho_amostra deve ser >= 1.")
    if n_repeticoes < 1:
        raise ValueError("n_repeticoes deve ser >= 1.")
    if len(populacao) == 0:
        raise ValueError("A população não pode ser vazia.")

    rng = random.Random(seed)
    medias_amostrais = []
    for _ in range(n_repeticoes):
        amostra = [rng.choice(populacao) for _ in range(tamanho_amostra)]
        medias_amostrais.append(ms.media(amostra))

    media_pop = ms.media(populacao)
    dp_pop = ms.desvio_padrao(populacao, amostral=False)
    erro_padrao_teorico = dp_pop / (tamanho_amostra ** 0.5)

    return {
        "medias_amostrais": medias_amostrais,
        "media_populacional": media_pop,
        "desvio_padrao_populacional": dp_pop,
        "erro_padrao_teorico": erro_padrao_teorico,
    }
