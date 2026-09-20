"""
server.py
==========

Aplicação Flask do Laboratório Estatístico Interativo.

Cada rota corresponde a um módulo do enunciado:
  /                -> Visão geral do dataset
  /descritiva      -> Módulo 2: Estatística Descritiva Interativa
  /simulacao       -> Módulo 3: Probabilidade e Simulação (Monte Carlo)
  /distribuicoes   -> Módulo 4: Distribuições Teóricas
  /regressao       -> Módulo 5: Correlação e Regressão Linear

Toda medida estatística exibida ao usuário vem de `core/minhastats.py`.
Pandas/NumPy só são usados para carregar/filtrar o dataset.
"""

from __future__ import annotations

import os
import sys

from flask import Flask, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import data_loader as dl  # noqa: E402
from core import graficos as gf  # noqa: E402
from core import minhastats as ms  # noqa: E402
from core import simulacao as sim  # noqa: E402

app = Flask(__name__)

# Cache simples do dataset em memória (carregado uma vez por processo).
_DATASET_CACHE = {}


def obter_dataset():
    if "df" not in _DATASET_CACHE:
        _DATASET_CACHE["df"] = dl.carregar_dataset()
    return _DATASET_CACHE["df"]


def contexto_base():
    df = obter_dataset()
    return {
        "colunas_numericas": dl.colunas_numericas(df),
        "colunas_categoricas": dl.colunas_categoricas(df),
    }


@app.route("/")
def home():
    df = obter_dataset()
    ctx = contexto_base()
    ctx.update({
        "n_linhas": len(df),
        "n_colunas": len(df.columns),
        "colunas": list(df.columns),
        "amostra": df.head(8).to_dict(orient="records"),
    })
    return render_template("index.html", **ctx)


@app.route("/descritiva")
def descritiva():
    df = obter_dataset()
    ctx = contexto_base()
    todas_colunas = ctx["colunas_numericas"] + ctx["colunas_categoricas"]

    coluna = request.args.get("coluna") or (todas_colunas[0] if todas_colunas else None)
    ctx["coluna_selecionada"] = coluna

    if coluna is None:
        return render_template("descritiva.html", **ctx)

    if coluna in ctx["colunas_numericas"]:
        valores = dl.serie_numerica_limpa(df, coluna)
        q1, q2, q3 = ms.quartis(valores)
        outliers = ms.detectar_outliers_iqr(valores)
        lim_inf, lim_sup = ms.limites_outliers_iqr(valores)
        skew = ms.assimetria(valores)

        estatisticas = {
            "n": len(valores),
            "media": ms.media(valores),
            "mediana": ms.mediana(valores),
            "moda": ms.moda(valores),
            "amplitude": ms.amplitude(valores),
            "variancia_amostral": ms.variancia(valores, amostral=True),
            "desvio_padrao_amostral": ms.desvio_padrao(valores, amostral=True),
            "variancia_populacional": ms.variancia(valores, amostral=False),
            "desvio_padrao_populacional": ms.desvio_padrao(valores, amostral=False),
            "cv": ms.coeficiente_variacao(valores),
            "q1": q1, "q2": q2, "q3": q3,
            "iqr": ms.iqr(valores),
            "assimetria": skew,
        }

        ctx.update({
            "tipo": "numerica",
            "estatisticas": estatisticas,
            "tabela_frequencias": dl.tabela_frequencias_numerica(valores),
            "qtd_outliers": len(outliers),
            "outliers_exemplos": sorted(outliers, key=abs, reverse=True)[:15],
            "limite_inferior": lim_inf,
            "limite_superior": lim_sup,
            "interpretacao_assimetria": dl.interpretar_assimetria(skew),
            "interpretacao_outliers": dl.interpretar_outliers(len(outliers), len(valores)),
            "grafico_histograma": gf.histograma(valores, f"Histograma — {coluna}", coluna,
                                                 limite_inferior=lim_inf,
                                                 limite_superior=lim_sup),
            "grafico_boxplot": gf.boxplot(valores, f"Boxplot — {coluna}", coluna),
        })
    else:
        valores = dl.serie_categorica_limpa(df, coluna)
        tabela = dl.tabela_frequencias_categorica(valores)
        ctx.update({
            "tipo": "categorica",
            "n": len(valores),
            "n_categorias": len(tabela),
            "tabela_frequencias_cat": tabela,
            "grafico_barras": gf.barras_categorica(tabela, f"Frequências — {coluna}", coluna),
            "grafico_pizza": gf.pizza_categorica(tabela, f"Proporções — {coluna}"),
        })

    return render_template("descritiva.html", **ctx)


@app.route("/simulacao")
def simulacao():
    df = obter_dataset()
    ctx = contexto_base()

    # --- Lei dos Grandes Números ---
    n_max = request.args.get("n_max", default=1000, type=int)
    p_moeda = request.args.get("p", default=0.5, type=float)
    seed_lgn = request.args.get("seed_lgn", default=42, type=int)
    n_max = max(10, min(n_max, 20000))
    p_moeda = min(max(p_moeda, 0.01), 0.99)

    resultado_lgn = sim.lei_dos_grandes_numeros(n_max, p=p_moeda, seed=seed_lgn)
    ctx.update({
        "n_max": n_max, "p_moeda": p_moeda, "seed_lgn": seed_lgn,
        "freq_final_lgn": resultado_lgn["freq_relativa"][-1],
        "grafico_lgn": gf.convergencia_lgn(
            resultado_lgn["n"], resultado_lgn["freq_relativa"], p_moeda,
            "Lei dos Grandes Números — convergência da frequência relativa"),
    })

    # --- Teorema Central do Limite ---
    colunas_numericas = ctx["colunas_numericas"]
    coluna_clt = request.args.get("coluna_clt") or (colunas_numericas[0] if colunas_numericas else None)
    tamanho_amostra = request.args.get("tamanho_amostra", default=30, type=int)
    n_repeticoes = request.args.get("n_repeticoes", default=1000, type=int)
    seed_clt = request.args.get("seed_clt", default=7, type=int)
    tamanho_amostra = max(1, min(tamanho_amostra, 2000))
    n_repeticoes = max(10, min(n_repeticoes, 20000))

    ctx.update({
        "coluna_clt": coluna_clt, "tamanho_amostra": tamanho_amostra,
        "n_repeticoes": n_repeticoes, "seed_clt": seed_clt,
    })

    if coluna_clt:
        populacao = dl.serie_numerica_limpa(df, coluna_clt)
        resultado_clt = sim.teorema_central_limite(
            populacao, tamanho_amostra, n_repeticoes, seed=seed_clt)
        ctx.update({
            "media_populacional": resultado_clt["media_populacional"],
            "desvio_padrao_populacional": resultado_clt["desvio_padrao_populacional"],
            "erro_padrao_teorico": resultado_clt["erro_padrao_teorico"],
            "media_das_medias": ms.media(resultado_clt["medias_amostrais"]),
            "desvio_das_medias": ms.desvio_padrao(resultado_clt["medias_amostrais"], amostral=True),
            "grafico_clt": gf.histograma_medias_amostrais(
                resultado_clt["medias_amostrais"], resultado_clt["media_populacional"],
                resultado_clt["erro_padrao_teorico"], tamanho_amostra,
                f"Teorema Central do Limite — médias amostrais de \"{coluna_clt}\""),
        })

    return render_template("simulacao.html", **ctx)


DISTRIBUICOES_DISPONIVEIS = {
    "uniforme": "Uniforme",
    "exponencial": "Exponencial",
    "poisson": "Poisson (aprox. discreta)",
    "binomial": "Binomial (aprox. discreta)",
}


@app.route("/distribuicoes")
def distribuicoes():
    df = obter_dataset()
    ctx = contexto_base()
    colunas_numericas = ctx["colunas_numericas"]

    coluna = request.args.get("coluna") or (colunas_numericas[0] if colunas_numericas else None)
    segunda_dist = request.args.get("distribuicao", default="uniforme")
    ctx.update({
        "coluna_selecionada": coluna,
        "distribuicao_selecionada": segunda_dist,
        "distribuicoes_disponiveis": DISTRIBUICOES_DISPONIVEIS,
    })

    if not coluna:
        return render_template("distribuicoes.html", **ctx)

    valores = dl.serie_numerica_limpa(df, coluna)
    media_amostra = ms.media(valores)
    dp_amostra = ms.desvio_padrao(valores, amostral=True)
    minimo, maximo = min(valores), max(valores)

    curvas = [{
        "label": f"Normal (μ={media_amostra:.2f}, σ={dp_amostra:.2f})",
        "pdf": lambda x: ms.normal_pdf(x, media_amostra, dp_amostra),
        "cor": gf.COR_SECUNDARIA,
    }]

    parametros_texto = ""
    if segunda_dist == "uniforme":
        curvas.append({
            "label": f"Uniforme (a={minimo:.2f}, b={maximo:.2f})",
            "pdf": lambda x: ms.uniforme_pdf(x, minimo, maximo),
            "cor": gf.COR_TERCIARIA,
        })
        parametros_texto = (f"Parâmetros estimados: a = mínimo observado = {minimo:.3f}; "
                             f"b = máximo observado = {maximo:.3f}.")
    elif segunda_dist == "exponencial":
        lam = 1.0 / media_amostra if media_amostra != 0 else 1.0
        curvas.append({
            "label": f"Exponencial (λ={lam:.4f})",
            "pdf": lambda x: ms.exponencial_pdf(x, lam),
            "cor": gf.COR_TERCIARIA,
        })
        parametros_texto = f"Parâmetro estimado: λ = 1 / média = {lam:.4f}."
    elif segunda_dist == "poisson":
        lam = media_amostra
        curvas.append({
            "label": f"Poisson (λ={lam:.2f}) — massa discreta, plotada como densidade",
            "pdf": lambda x: ms.poisson_pmf(round(x), lam),
            "cor": gf.COR_TERCIARIA,
        })
        parametros_texto = (f"Parâmetro estimado: λ = média observada = {lam:.3f} "
                             f"(adequado a variáveis de contagem).")
    elif segunda_dist == "binomial":
        n_bin = max(1, round(maximo))
        p_bin = media_amostra / n_bin if n_bin > 0 else 0.5
        p_bin = min(max(p_bin, 0.0001), 0.9999)
        curvas.append({
            "label": f"Binomial (n={n_bin}, p={p_bin:.3f})",
            "pdf": lambda x: ms.binomial_pmf(round(x), n_bin, p_bin),
            "cor": gf.COR_TERCIARIA,
        })
        parametros_texto = (f"Parâmetros estimados: n = máximo observado = {n_bin}; "
                             f"p = média / n = {p_bin:.4f}.")

    ctx.update({
        "media_amostra": media_amostra,
        "dp_amostra": dp_amostra,
        "parametros_texto": parametros_texto,
        "grafico_distribuicao": gf.histograma_com_distribuicao(
            valores, f"Ajuste de distribuições — {coluna}", coluna, curvas),
    })
    return render_template("distribuicoes.html", **ctx)


@app.route("/regressao")
def regressao():
    df = obter_dataset()
    ctx = contexto_base()
    colunas_numericas = ctx["colunas_numericas"]

    var_x = request.args.get("var_x") or (colunas_numericas[0] if colunas_numericas else None)
    var_y = request.args.get("var_y") or (colunas_numericas[1] if len(colunas_numericas) > 1 else var_x)
    x_predito_raw = request.args.get("x_predito")

    ctx.update({"var_x": var_x, "var_y": var_y, "x_predito_raw": x_predito_raw or ""})

    if not var_x or not var_y:
        return render_template("regressao.html", **ctx)

    df_par = df[[var_x, var_y]].dropna()
    x_vals = [float(v) for v in df_par[var_x].tolist()]
    y_vals = [float(v) for v in df_par[var_y].tolist()]

    resultado = ms.regressao_linear_simples(x_vals, y_vals)
    cov = ms.covariancia(x_vals, y_vals)

    ponto_predito = None
    y_predito = None
    if x_predito_raw not in (None, ""):
        try:
            x_predito = float(x_predito_raw)
            y_predito = ms.prever(x_predito, resultado["b0"], resultado["b1"])
            ponto_predito = (x_predito, y_predito)
        except ValueError:
            y_predito = None

    sinal = "+" if resultado["b1"] >= 0 else "-"
    equacao = f"ŷ = {resultado['b0']:.4f} {sinal} {abs(resultado['b1']):.4f} · x"

    ctx.update({
        "n_pares": len(x_vals),
        "covariancia": cov,
        "correlacao": resultado["r"],
        "b0": resultado["b0"],
        "b1": resultado["b1"],
        "r2": resultado["r2"],
        "equacao": equacao,
        "interpretacao_correlacao": dl.interpretar_correlacao(resultado["r"]),
        "y_predito": y_predito,
        "grafico_regressao": gf.dispersao_regressao(
            x_vals, y_vals, resultado["b0"], resultado["b1"],
            f"{var_y} em função de {var_x}", var_x, var_y,
            ponto_predito=ponto_predito),
    })
    return render_template("regressao.html", **ctx)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
