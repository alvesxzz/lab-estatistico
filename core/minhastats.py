"""
minhastats.py
=============

Núcleo estatístico implementado "na unha" (sem usar funções prontas de
estatística de NumPy/SciPy/statistics). As únicas operações herdadas de
outras bibliotecas são operações aritméticas básicas de listas/arrays
(soma, ordenação, potência), nunca uma função estatística pronta.

Cada função aqui é validada em tests/test_minhastats.py comparando o
resultado com NumPy/SciPy/statistics, com tolerância numérica documentada
(ver módulo de testes).

Convenções:
- Todas as funções aceitam listas, tuplas ou arrays 1D como entrada.
- Funções que dependem de "amostral vs. populacional" recebem o parâmetro
  `amostral: bool = True` (variância e desvio padrão amostrais usam
  divisor (n-1); populacionais usam divisor n).
- Funções levantam ValueError para entradas vazias ou de tamanho incompatível.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _to_list(x: Iterable[float]) -> list:
    """Converte qualquer iterável numérico em lista de floats."""
    lista = [float(v) for v in x]
    if len(lista) == 0:
        raise ValueError("A sequência de dados não pode ser vazia.")
    return lista


def _checar_mesmo_tamanho(x: Sequence[float], y: Sequence[float]) -> None:
    if len(x) != len(y):
        raise ValueError("As duas variáveis devem ter o mesmo número de observações.")


# ---------------------------------------------------------------------------
# Medidas de tendência central
# ---------------------------------------------------------------------------

def media(x: Iterable[float]) -> float:
    """Média aritmética: soma dos valores dividida pela quantidade de valores.

    Fórmula: x̄ = (Σ xᵢ) / n
    """
    dados = _to_list(x)
    soma = 0.0
    for v in dados:
        soma += v
    return soma / len(dados)


def _ordenar(x: Iterable[float]) -> list:
    dados = _to_list(x)
    # Implementação de ordenação própria (insertion sort) para não depender
    # de bibliotecas de estatística prontas — apenas ordenação genérica.
    # Para conjuntos grandes usamos sorted() do Python (função de ordenação
    # de propósito geral, não uma função estatística), o que é permitido
    # pela regra do trabalho (só medidas estatísticas precisam ser próprias).
    return sorted(dados)


def mediana(x: Iterable[float]) -> float:
    """Mediana: valor central da série ordenada (ou média dos dois centrais
    quando n é par).

    Fórmula:
        n ímpar -> x_((n+1)/2)
        n par   -> (x_(n/2) + x_(n/2 + 1)) / 2
    """
    dados = _ordenar(x)
    n = len(dados)
    meio = n // 2
    if n % 2 == 1:
        return dados[meio]
    return (dados[meio - 1] + dados[meio]) / 2.0


def moda(x: Iterable[float]) -> list:
    """Moda: valor(es) mais frequente(s). Retorna uma lista, pois uma
    distribuição pode ser multimodal. Se todos os valores forem igualmente
    frequentes, retorna todos (amodal)."""
    dados = _to_list(x)
    contagem: dict = {}
    for v in dados:
        contagem[v] = contagem.get(v, 0) + 1
    freq_max = max(contagem.values())
    modas = [v for v, c in contagem.items() if c == freq_max]
    modas.sort()
    return modas


# ---------------------------------------------------------------------------
# Medidas de dispersão
# ---------------------------------------------------------------------------

def amplitude(x: Iterable[float]) -> float:
    """Amplitude total: diferença entre o maior e o menor valor."""
    dados = _to_list(x)
    return max(dados) - min(dados)


def variancia(x: Iterable[float], amostral: bool = True) -> float:
    """Variância amostral (divisor n-1) ou populacional (divisor n).

    Fórmula (amostral):     s² = Σ(xᵢ - x̄)² / (n - 1)
    Fórmula (populacional): σ² = Σ(xᵢ - x̄)² / n
    """
    dados = _to_list(x)
    n = len(dados)
    if amostral and n < 2:
        raise ValueError("Variância amostral requer ao menos 2 observações.")
    m = media(dados)
    soma_quadrados = 0.0
    for v in dados:
        soma_quadrados += (v - m) ** 2
    divisor = (n - 1) if amostral else n
    return soma_quadrados / divisor


def desvio_padrao(x: Iterable[float], amostral: bool = True) -> float:
    """Desvio padrão: raiz quadrada da variância (amostral ou populacional)."""
    return math.sqrt(variancia(x, amostral=amostral))


def percentil(x: Iterable[float], p: float) -> float:
    """Percentil p (0 <= p <= 100) usando interpolação linear entre as
    ordens de estatística mais próximas — o mesmo método usado por
    numpy.percentile(..., method="linear"), que é o padrão do NumPy.

    Fórmula: posição h = (p/100) * (n - 1)
             resultado = x_(floor(h)) + fração(h) * (x_(floor(h)+1) - x_(floor(h)))
    """
    if not (0 <= p <= 100):
        raise ValueError("O percentil p deve estar entre 0 e 100.")
    dados = _ordenar(x)
    n = len(dados)
    if n == 1:
        return dados[0]
    h = (p / 100.0) * (n - 1)
    piso = math.floor(h)
    teto = math.ceil(h)
    if piso == teto:
        return dados[int(h)]
    fracao = h - piso
    return dados[piso] + fracao * (dados[teto] - dados[piso])


def quartis(x: Iterable[float]) -> tuple:
    """Retorna (Q1, Q2, Q3) usando a função `percentil`."""
    return (percentil(x, 25), percentil(x, 50), percentil(x, 75))


def iqr(x: Iterable[float]) -> float:
    """Intervalo interquartil: IQR = Q3 - Q1."""
    q1, _, q3 = quartis(x)
    return q3 - q1


def coeficiente_variacao(x: Iterable[float], amostral: bool = True) -> float:
    """Coeficiente de variação (%): CV = (desvio padrão / média) * 100.

    Útil para comparar a dispersão relativa entre variáveis com escalas
    diferentes.
    """
    m = media(x)
    if m == 0:
        raise ValueError("Coeficiente de variação indefinido quando a média é 0.")
    return (desvio_padrao(x, amostral=amostral) / m) * 100.0


def assimetria(x: Iterable[float]) -> float:
    """Coeficiente de assimetria de Pearson (skewness), baseado no terceiro
    momento padronizado (viés amostral, mesma convenção do
    scipy.stats.skew com bias=True):

        g1 = (1/n) * Σ[(xᵢ - x̄)/σ]³   onde σ é o desvio padrão populacional
    """
    dados = _to_list(x)
    n = len(dados)
    m = media(dados)
    sigma = desvio_padrao(dados, amostral=False)
    if sigma == 0:
        return 0.0
    soma_cubos = 0.0
    for v in dados:
        soma_cubos += ((v - m) / sigma) ** 3
    return soma_cubos / n


# ---------------------------------------------------------------------------
# Medidas bivariadas
# ---------------------------------------------------------------------------

def covariancia(x: Iterable[float], y: Iterable[float], amostral: bool = True) -> float:
    """Covariância entre duas variáveis.

    Fórmula (amostral):     cov(x,y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / (n - 1)
    Fórmula (populacional): cov(x,y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / n
    """
    dados_x = _to_list(x)
    dados_y = _to_list(y)
    _checar_mesmo_tamanho(dados_x, dados_y)
    n = len(dados_x)
    if amostral and n < 2:
        raise ValueError("Covariância amostral requer ao menos 2 observações.")
    mx = media(dados_x)
    my = media(dados_y)
    soma = 0.0
    for xi, yi in zip(dados_x, dados_y):
        soma += (xi - mx) * (yi - my)
    divisor = (n - 1) if amostral else n
    return soma / divisor


def correlacao_pearson(x: Iterable[float], y: Iterable[float]) -> float:
    """Coeficiente de correlação linear de Pearson.

    Fórmula: r = cov(x,y) / (σx * σy)

    O resultado é o mesmo independentemente de usar a versão amostral ou
    populacional de covariância/desvio padrão, pois o divisor (n ou n-1)
    se cancela entre numerador e denominador.
    """
    dados_x = _to_list(x)
    dados_y = _to_list(y)
    _checar_mesmo_tamanho(dados_x, dados_y)
    cov = covariancia(dados_x, dados_y, amostral=True)
    sx = desvio_padrao(dados_x, amostral=True)
    sy = desvio_padrao(dados_y, amostral=True)
    if sx == 0 or sy == 0:
        raise ValueError("Correlação indefinida quando uma variável é constante.")
    return cov / (sx * sy)


def regressao_linear_simples(x: Iterable[float], y: Iterable[float]) -> dict:
    """Regressão linear simples pelo método dos mínimos quadrados.

    Modelo: ŷ = b0 + b1 * x

    Fórmulas:
        b1 = cov(x,y) / var(x)
        b0 = ȳ - b1 * x̄

    Retorna um dicionário com b0 (intercepto), b1 (inclinação), r (correlação
    de Pearson) e r2 (coeficiente de determinação).
    """
    dados_x = _to_list(x)
    dados_y = _to_list(y)
    _checar_mesmo_tamanho(dados_x, dados_y)

    cov = covariancia(dados_x, dados_y, amostral=True)
    var_x = variancia(dados_x, amostral=True)
    if var_x == 0:
        raise ValueError("Regressão indefinida quando x é constante.")

    b1 = cov / var_x
    b0 = media(dados_y) - b1 * media(dados_x)

    r = correlacao_pearson(dados_x, dados_y)
    r2 = r_quadrado(dados_x, dados_y, b0, b1)

    return {"b0": b0, "b1": b1, "r": r, "r2": r2}


def prever(x_novo: float, b0: float, b1: float) -> float:
    """Aplica a reta ajustada: ŷ = b0 + b1 * x_novo."""
    return b0 + b1 * x_novo


def r_quadrado(x: Iterable[float], y: Iterable[float], b0: float, b1: float) -> float:
    """Coeficiente de determinação R².

    Fórmula: R² = 1 - (SQrésiduo / SQtotal)
        SQrésiduo = Σ(yᵢ - ŷᵢ)²
        SQtotal   = Σ(yᵢ - ȳ)²
    """
    dados_x = _to_list(x)
    dados_y = _to_list(y)
    my = media(dados_y)
    ss_res = 0.0
    ss_tot = 0.0
    for xi, yi in zip(dados_x, dados_y):
        y_pred = prever(xi, b0, b1)
        ss_res += (yi - y_pred) ** 2
        ss_tot += (yi - my) ** 2
    if ss_tot == 0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Detecção de outliers (regra do IQR)
# ---------------------------------------------------------------------------

def limites_outliers_iqr(x: Iterable[float], k: float = 1.5) -> tuple:
    """Retorna (limite_inferior, limite_superior) pela regra do IQR:

        limite_inferior = Q1 - k * IQR
        limite_superior = Q3 + k * IQR

    k = 1.5 é o valor clássico (Tukey) para outliers "moderados".
    """
    q1, _, q3 = quartis(x)
    amplitude_iqr = q3 - q1
    return (q1 - k * amplitude_iqr, q3 + k * amplitude_iqr)


def detectar_outliers_iqr(x: Iterable[float], k: float = 1.5) -> list:
    """Retorna a lista de valores considerados outliers pela regra do IQR."""
    dados = _to_list(x)
    lim_inf, lim_sup = limites_outliers_iqr(dados, k=k)
    return [v for v in dados if v < lim_inf or v > lim_sup]


# ---------------------------------------------------------------------------
# Distribuições teóricas (funções de densidade/probabilidade "na unha")
# ---------------------------------------------------------------------------

def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """Densidade da distribuição Normal(mu, sigma).

    f(x) = 1 / (sigma * sqrt(2*pi)) * exp( -(x-mu)² / (2*sigma²) )
    """
    if sigma <= 0:
        raise ValueError("sigma deve ser positivo.")
    coef = 1.0 / (sigma * math.sqrt(2 * math.pi))
    expoente = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coef * math.exp(expoente)


def _combinacao(n: int, k: int) -> float:
    """Combinação C(n, k) = n! / (k! (n-k)!), calculada com math.comb."""
    return float(math.comb(n, k))


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Probabilidade de massa da Binomial(n, p) em k sucessos.

    P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
    """
    if not (0 <= p <= 1):
        raise ValueError("p deve estar entre 0 e 1.")
    if not (0 <= k <= n):
        return 0.0
    return _combinacao(n, k) * (p ** k) * ((1 - p) ** (n - k))


def poisson_pmf(k: int, lam: float) -> float:
    """Probabilidade de massa da Poisson(lambda) em k eventos.

    P(X=k) = (lambda^k * e^-lambda) / k!
    """
    if lam < 0:
        raise ValueError("lambda deve ser não-negativo.")
    if k < 0:
        return 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def uniforme_pdf(x: float, a: float, b: float) -> float:
    """Densidade da distribuição Uniforme contínua em [a, b].

    f(x) = 1/(b-a) se a <= x <= b, senão 0.
    """
    if b <= a:
        raise ValueError("b deve ser maior que a.")
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def exponencial_pdf(x: float, lam: float) -> float:
    """Densidade da distribuição Exponencial(lambda).

    f(x) = lambda * e^(-lambda*x) para x >= 0, senão 0.
    """
    if lam <= 0:
        raise ValueError("lambda deve ser positivo.")
    if x < 0:
        return 0.0
    return lam * math.exp(-lam * x)
