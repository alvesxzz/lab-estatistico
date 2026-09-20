"""
test_minhastats.py
===================

Testes automatizados que validam cada função de `core/minhastats.py`
comparando o resultado com implementações consolidadas de NumPy, SciPy
e da biblioteca padrão `statistics`.

Tolerância numérica: usamos `math.isclose` com rel_tol=1e-9 (praticamente
igualdade de ponto flutuante) para operações puramente aritméticas, e
rel_tol=1e-6 para operações que envolvem iteração numérica (nenhuma neste
arquivo, mas mantemos a constante central para documentar o critério).

Este arquivo funciona de duas formas:
  1) Com pytest instalado:      pytest tests/test_minhastats.py -v
  2) Sem pytest (fallback):     python3 tests/test_minhastats.py
     (um mini executor de testes percorre todas as funções `test_*`)

As funções de teste usam `assert` simples, então são 100% compatíveis
com o formato de descoberta de testes do pytest.
"""

import math
import os
import sys

import numpy as np
from scipy import stats as scipy_stats
import statistics as pystats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import minhastats as ms  # noqa: E402

REL_TOL = 1e-9

# Conjunto de dados fixo usado na maioria dos testes (valores variados,
# incluindo repetição para testar moda).
DADOS_A = [4, 8, 15, 16, 23, 42, 8, 4, 15, 16, 16, 23, 42, 8, 4]
DADOS_B = [2.5, 3.1, 7.4, 1.9, 5.5, 6.6, 2.2, 9.9, 4.4, 3.3, 8.8, 5.1]

# Duas variáveis correlacionadas para testes de covariância/correlação/regressão
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.2, 17.9, 20.1]


def _iso(a, b, tol=REL_TOL, abs_tol=1e-9):
    return math.isclose(a, b, rel_tol=tol, abs_tol=abs_tol)


# ---------------------------------------------------------------------------
# Tendência central
# ---------------------------------------------------------------------------

def test_media():
    esperado = float(np.mean(DADOS_A))
    obtido = ms.media(DADOS_A)
    assert _iso(obtido, esperado), f"media: {obtido} != {esperado}"


def test_mediana_par():
    esperado = float(np.median(DADOS_A))
    obtido = ms.mediana(DADOS_A)
    assert _iso(obtido, esperado)


def test_mediana_impar():
    dados = DADOS_A[:-1]  # tamanho ímpar
    esperado = float(np.median(dados))
    obtido = ms.mediana(dados)
    assert _iso(obtido, esperado)


def test_moda():
    esperado = pystats.multimode(DADOS_A)
    obtido = ms.moda(DADOS_A)
    assert sorted(obtido) == sorted(esperado)


# ---------------------------------------------------------------------------
# Dispersão
# ---------------------------------------------------------------------------

def test_amplitude():
    esperado = max(DADOS_A) - min(DADOS_A)
    obtido = ms.amplitude(DADOS_A)
    assert _iso(obtido, esperado)


def test_variancia_amostral():
    esperado = float(np.var(DADOS_A, ddof=1))
    obtido = ms.variancia(DADOS_A, amostral=True)
    assert _iso(obtido, esperado)


def test_variancia_populacional():
    esperado = float(np.var(DADOS_A, ddof=0))
    obtido = ms.variancia(DADOS_A, amostral=False)
    assert _iso(obtido, esperado)


def test_desvio_padrao_amostral():
    esperado = float(np.std(DADOS_A, ddof=1))
    obtido = ms.desvio_padrao(DADOS_A, amostral=True)
    assert _iso(obtido, esperado)


def test_desvio_padrao_populacional():
    esperado = float(np.std(DADOS_A, ddof=0))
    obtido = ms.desvio_padrao(DADOS_A, amostral=False)
    assert _iso(obtido, esperado)


def test_percentil_25_50_75():
    for p in (10, 25, 50, 75, 90):
        esperado = float(np.percentile(DADOS_B, p))
        obtido = ms.percentil(DADOS_B, p)
        assert _iso(obtido, esperado, abs_tol=1e-6), f"p{p}: {obtido} != {esperado}"


def test_quartis():
    q1, q2, q3 = ms.quartis(DADOS_B)
    assert _iso(q1, float(np.percentile(DADOS_B, 25)), abs_tol=1e-6)
    assert _iso(q2, float(np.percentile(DADOS_B, 50)), abs_tol=1e-6)
    assert _iso(q3, float(np.percentile(DADOS_B, 75)), abs_tol=1e-6)


def test_iqr():
    esperado = float(np.percentile(DADOS_B, 75) - np.percentile(DADOS_B, 25))
    obtido = ms.iqr(DADOS_B)
    assert _iso(obtido, esperado, abs_tol=1e-6)


def test_coeficiente_variacao():
    esperado = (float(np.std(DADOS_B, ddof=1)) / float(np.mean(DADOS_B))) * 100.0
    obtido = ms.coeficiente_variacao(DADOS_B, amostral=True)
    assert _iso(obtido, esperado, abs_tol=1e-6)


def test_assimetria():
    esperado = float(scipy_stats.skew(DADOS_B, bias=True))
    obtido = ms.assimetria(DADOS_B)
    assert _iso(obtido, esperado, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Bivariadas
# ---------------------------------------------------------------------------

def test_covariancia_amostral():
    esperado = float(np.cov(X, Y, ddof=1)[0, 1])
    obtido = ms.covariancia(X, Y, amostral=True)
    assert _iso(obtido, esperado, abs_tol=1e-6)


def test_correlacao_pearson():
    esperado, _ = scipy_stats.pearsonr(X, Y)
    obtido = ms.correlacao_pearson(X, Y)
    assert _iso(obtido, float(esperado), abs_tol=1e-6)


def test_regressao_linear_simples():
    slope, intercept, r, p, se = scipy_stats.linregress(X, Y)
    resultado = ms.regressao_linear_simples(X, Y)
    assert _iso(resultado["b1"], float(slope), abs_tol=1e-6)
    assert _iso(resultado["b0"], float(intercept), abs_tol=1e-6)
    assert _iso(resultado["r2"], float(r) ** 2, abs_tol=1e-6)


def test_prever():
    resultado = ms.regressao_linear_simples(X, Y)
    y_pred = ms.prever(11, resultado["b0"], resultado["b1"])
    y_pred_manual = resultado["b0"] + resultado["b1"] * 11
    assert _iso(y_pred, y_pred_manual)


# ---------------------------------------------------------------------------
# Outliers (IQR)
# ---------------------------------------------------------------------------

def test_deteccao_outliers_iqr():
    dados = DADOS_B + [100.0, -50.0]  # dois outliers evidentes
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr_esperado = q3 - q1
    lim_inf = q1 - 1.5 * iqr_esperado
    lim_sup = q3 + 1.5 * iqr_esperado
    esperado = sorted([v for v in dados if v < lim_inf or v > lim_sup])
    obtido = sorted(ms.detectar_outliers_iqr(dados))
    assert obtido == esperado


# ---------------------------------------------------------------------------
# Distribuições teóricas
# ---------------------------------------------------------------------------

def test_normal_pdf():
    mu, sigma = 10.0, 2.0
    for x in (8.0, 10.0, 12.5):
        esperado = float(scipy_stats.norm.pdf(x, loc=mu, scale=sigma))
        obtido = ms.normal_pdf(x, mu, sigma)
        assert _iso(obtido, esperado, abs_tol=1e-9)


def test_binomial_pmf():
    n, p = 20, 0.3
    for k in (0, 5, 10, 20):
        esperado = float(scipy_stats.binom.pmf(k, n, p))
        obtido = ms.binomial_pmf(k, n, p)
        assert _iso(obtido, esperado, abs_tol=1e-9)


def test_poisson_pmf():
    lam = 4.5
    for k in (0, 2, 5, 10):
        esperado = float(scipy_stats.poisson.pmf(k, lam))
        obtido = ms.poisson_pmf(k, lam)
        assert _iso(obtido, esperado, abs_tol=1e-9)


def test_uniforme_pdf():
    a, b = 2.0, 10.0
    for x in (1.0, 5.0, 10.0, 11.0):
        esperado = float(scipy_stats.uniform.pdf(x, loc=a, scale=b - a))
        obtido = ms.uniforme_pdf(x, a, b)
        assert _iso(obtido, esperado, abs_tol=1e-9)


def test_exponencial_pdf():
    lam = 0.5
    for x in (0.0, 1.0, 5.0):
        esperado = float(scipy_stats.expon.pdf(x, scale=1 / lam))
        obtido = ms.exponencial_pdf(x, lam)
        assert _iso(obtido, esperado, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Casos de borda / validação de erros
# ---------------------------------------------------------------------------

def test_erro_lista_vazia():
    try:
        ms.media([])
        assert False, "deveria ter levantado ValueError"
    except ValueError:
        pass


def test_erro_tamanhos_diferentes():
    try:
        ms.covariancia([1, 2, 3], [1, 2])
        assert False, "deveria ter levantado ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Mini executor de testes (fallback sem pytest)
# ---------------------------------------------------------------------------

def _executar_sem_pytest():
    testes = [(nome, obj) for nome, obj in globals().items()
              if nome.startswith("test_") and callable(obj)]
    passou, falhou = 0, 0
    for nome, func in testes:
        try:
            func()
            print(f"  OK   {nome}")
            passou += 1
        except AssertionError as e:
            print(f"  FALHOU {nome}: {e}")
            falhou += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERRO  {nome}: {type(e).__name__}: {e}")
            falhou += 1
    total = passou + falhou
    print(f"\n{passou}/{total} testes passaram.")
    return falhou == 0


if __name__ == "__main__":
    ok = _executar_sem_pytest()
    sys.exit(0 if ok else 1)
