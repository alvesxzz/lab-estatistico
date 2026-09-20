"""
graficos.py
============

Geração de gráficos com Matplotlib, renderizados no servidor e
devolvidos como imagens PNG codificadas em base64 (embutidas diretamente
no HTML). Essa abordagem evita depender de bibliotecas JavaScript
externas via CDN, então a aplicação funciona 100% offline.

Nenhuma medida estatística é calculada aqui — os gráficos apenas
desenham valores que já vêm prontos de `core/minhastats.py` e
`core/data_loader.py`.
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import minhastats as ms

# Paleta validada (skill de dataviz): ordem categórica fixa e segura para
# daltonismo (slot 1 azul, slot 2 laranja, slot 3 água, slot 8 vermelho para
# destaques/outliers), com cores de grade e tinta neutras.
COR_PRIMARIA = "#2a78d6"     # slot 1 - azul
COR_SECUNDARIA = "#eb6834"   # slot 2 - laranja
COR_TERCIARIA = "#1baf7a"    # slot 3 - água
COR_OUTLIER = "#e34948"      # slot 8 - vermelho (uso reservado a destaque/outlier)
COR_GRADE = "#e1e0d9"
COR_INK = "#0b0b0b"
COR_MUTED = "#898781"


def _fig_para_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _estilo_eixo(ax, titulo, xlabel="", ylabel=""):
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, color=COR_GRADE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def histograma(valores: list, titulo: str, coluna: str, n_classes: int | None = None,
               outliers: list | None = None) -> str:
    from .data_loader import numero_classes_sturges
    k = n_classes or numero_classes_sturges(len(valores))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(valores, bins=k, color=COR_PRIMARIA, edgecolor="white", zorder=3)
    if outliers:
        for v in set(outliers):
            ax.axvline(v, color=COR_OUTLIER, linestyle="--", linewidth=1, alpha=0.6, zorder=4)
    _estilo_eixo(ax, titulo, xlabel=coluna, ylabel="Frequência")
    return _fig_para_base64(fig)


def boxplot(valores: list, titulo: str, coluna: str) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    caixa = ax.boxplot(valores, vert=False, patch_artist=True,
                        boxprops=dict(facecolor=COR_PRIMARIA, alpha=0.7),
                        medianprops=dict(color="black"),
                        flierprops=dict(markerfacecolor=COR_OUTLIER,
                                         markeredgecolor=COR_OUTLIER, marker="o"))
    _estilo_eixo(ax, titulo, xlabel=coluna)
    ax.set_yticks([])
    return _fig_para_base64(fig)


def barras_categorica(tabela: list, titulo: str, coluna: str, top_n: int = 12) -> str:
    tabela = tabela[:top_n]
    categorias = [str(item["categoria"]) for item in tabela]
    valores = [item["freq_absoluta"] for item in tabela]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(categorias, valores, color=COR_PRIMARIA, zorder=3)
    _estilo_eixo(ax, titulo, xlabel=coluna, ylabel="Frequência")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    return _fig_para_base64(fig)


def pizza_categorica(tabela: list, titulo: str, top_n: int = 8) -> str:
    tabela_top = tabela[:top_n]
    outros = sum(item["freq_absoluta"] for item in tabela[top_n:])
    labels = [str(item["categoria"]) for item in tabela_top]
    valores = [item["freq_absoluta"] for item in tabela_top]
    if outros > 0:
        labels.append("Outros")
        valores.append(outros)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.pie(valores, labels=labels, autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 8})
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    return _fig_para_base64(fig)


def histograma_com_distribuicao(valores: list, titulo: str, coluna: str,
                                 curvas: list) -> str:
    """Histograma (densidade) com uma ou mais curvas teóricas sobrepostas.

    `curvas` é uma lista de dicionários: {"label": str, "pdf": callable, "cor": str}
    onde `pdf(x)` retorna a densidade no ponto x (funções de minhastats.py).
    """
    from .data_loader import numero_classes_sturges
    k = numero_classes_sturges(len(valores))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.hist(valores, bins=k, density=True, color=COR_PRIMARIA, alpha=0.55,
             edgecolor="white", zorder=3, label="Dados observados")

    minimo, maximo = min(valores), max(valores)
    margem = (maximo - minimo) * 0.05 if maximo > minimo else 1.0
    n_pontos = 300
    passo = (maximo - minimo + 2 * margem) / n_pontos
    xs = [minimo - margem + i * passo for i in range(n_pontos + 1)]

    cores_padrao = [COR_SECUNDARIA, COR_TERCIARIA, "#4a3aa7"]
    for i, curva in enumerate(curvas):
        cor = curva.get("cor") or cores_padrao[i % len(cores_padrao)]
        ys = [curva["pdf"](x) for x in xs]
        ax.plot(xs, ys, color=cor, linewidth=2.2, label=curva["label"], zorder=4)

    _estilo_eixo(ax, titulo, xlabel=coluna, ylabel="Densidade")
    ax.legend(fontsize=9, frameon=False)
    return _fig_para_base64(fig)


def dispersao_regressao(x: list, y: list, b0: float, b1: float,
                         titulo: str, x_label: str, y_label: str,
                         ponto_predito: tuple | None = None) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(x, y, color=COR_PRIMARIA, alpha=0.55, s=22, zorder=3, label="Observações")
    x_min, x_max = min(x), max(x)
    xs_linha = [x_min, x_max]
    ys_linha = [ms.prever(v, b0, b1) for v in xs_linha]
    ax.plot(xs_linha, ys_linha, color=COR_OUTLIER, linewidth=2.2, zorder=4,
             label=f"ŷ = {b0:.3f} + {b1:.3f}·x")
    if ponto_predito is not None:
        ax.scatter([ponto_predito[0]], [ponto_predito[1]], color="#111827",
                    s=90, marker="*", zorder=5, label="Predição")
    _estilo_eixo(ax, titulo, xlabel=x_label, ylabel=y_label)
    ax.legend(fontsize=9, frameon=False)
    return _fig_para_base64(fig)


def convergencia_lgn(n: list, freq_relativa: list, p: float, titulo: str) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(n, freq_relativa, color=COR_PRIMARIA, linewidth=1.4, zorder=3,
             label="Frequência relativa observada")
    ax.axhline(p, color=COR_OUTLIER, linestyle="--", linewidth=1.6, zorder=4,
                label=f"Probabilidade teórica p = {p}")
    _estilo_eixo(ax, titulo, xlabel="Número de lançamentos (n)",
                 ylabel="Frequência relativa de sucesso")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, frameon=False)
    return _fig_para_base64(fig)


def histograma_medias_amostrais(medias: list, media_pop: float, erro_padrao: float,
                                 tamanho_amostra: int, titulo: str) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    k = max(10, min(40, len(medias) // 20))
    ax.hist(medias, bins=k, density=True, color=COR_PRIMARIA, alpha=0.6,
             edgecolor="white", zorder=3, label="Médias amostrais simuladas")

    minimo, maximo = min(medias), max(medias)
    margem = (maximo - minimo) * 0.1 if maximo > minimo else 1.0
    n_pontos = 300
    passo = (maximo - minimo + 2 * margem) / n_pontos
    xs = [minimo - margem + i * passo for i in range(n_pontos + 1)]
    ys = [ms.normal_pdf(x, media_pop, erro_padrao) for x in xs]
    ax.plot(xs, ys, color=COR_OUTLIER, linewidth=2.2, zorder=4,
             label=f"Normal teórica (μ={media_pop:.2f}, erro padrão={erro_padrao:.3f})")

    _estilo_eixo(ax, titulo, xlabel=f"Média amostral (n={tamanho_amostra})",
                 ylabel="Densidade")
    ax.legend(fontsize=9, frameon=False)
    return _fig_para_base64(fig)
