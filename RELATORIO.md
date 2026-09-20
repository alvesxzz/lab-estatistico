# Relatório — Laboratório Estatístico Interativo

**Autor:** Andrey Alves Monteiro — matrícula 261027294

## 1. Dataset escolhido e justificativa

_(seção a finalizar após a definição do dataset — ver `data/FONTE.md` para o link
da fonte original. Justificativa considerará: tema, nº de registros, nº de
variáveis numéricas/categóricas e relevância para os módulos pedidos.)_

## 2. Núcleo estatístico — fórmulas utilizadas

Todas as funções abaixo estão implementadas em `core/minhastats.py`, sem uso de
funções prontas de estatística (apenas operações aritméticas básicas).

### 2.1 Tendência central

**Média aritmética**

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**Mediana** — valor central da série ordenada:

$$
\text{mediana} =
\begin{cases}
x_{\left(\frac{n+1}{2}\right)} & \text{se } n \text{ é ímpar} \\[4pt]
\dfrac{x_{(n/2)} + x_{(n/2+1)}}{2} & \text{se } n \text{ é par}
\end{cases}
$$

**Moda** — valor(es) de maior frequência absoluta.

### 2.2 Dispersão

**Amplitude:** $A = x_{\max} - x_{\min}$

**Variância amostral e populacional:**

$$s^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n-1} \qquad\qquad \sigma^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n}$$

**Desvio padrão:** $s = \sqrt{s^2}$ (amostral) e $\sigma = \sqrt{\sigma^2}$ (populacional)

**Percentil p (interpolação linear, mesmo método do `numpy.percentile`, method="linear"):**

$$h = \frac{p}{100}(n-1), \qquad \text{percentil} = x_{(\lfloor h \rfloor)} + \big(h - \lfloor h \rfloor\big)\left(x_{(\lceil h \rceil)} - x_{(\lfloor h \rfloor)}\right)$$

Quartis: $Q_1 = P_{25}$, $Q_2 = P_{50}$, $Q_3 = P_{75}$. IQR $= Q_3 - Q_1$.

**Coeficiente de variação:** $CV = \dfrac{s}{\bar{x}} \times 100\%$

**Assimetria (skewness, viés amostral — mesma convenção do `scipy.stats.skew(bias=True)`):**

$$g_1 = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{\sigma}\right)^3$$

### 2.3 Medidas bivariadas

**Covariância amostral:**

$$\text{cov}(x,y) = \frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{n-1}$$

**Correlação de Pearson:**

$$r = \frac{\text{cov}(x,y)}{s_x \, s_y}$$

**Regressão linear simples (mínimos quadrados):**

$$\hat{y} = b_0 + b_1 x, \qquad b_1 = \frac{\text{cov}(x,y)}{s_x^2}, \qquad b_0 = \bar{y} - b_1\bar{x}$$

**Coeficiente de determinação:**

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

### 2.4 Regra do IQR para outliers

$$\text{limite inferior} = Q_1 - 1.5 \times IQR \qquad \text{limite superior} = Q_3 + 1.5 \times IQR$$

Valores fora desse intervalo são classificados como outliers.

### 2.5 Distribuições teóricas

| Distribuição | Densidade / massa | Parâmetros estimados dos dados |
|---|---|---|
| Normal | $f(x)=\dfrac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu=\bar{x}$, $\sigma=s$ |
| Uniforme | $f(x)=\dfrac{1}{b-a}$ para $a\le x\le b$ | $a=x_{\min}$, $b=x_{\max}$ |
| Exponencial | $f(x)=\lambda e^{-\lambda x}$, $x\ge0$ | $\lambda = 1/\bar{x}$ |
| Poisson | $P(X=k)=\dfrac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda=\bar{x}$ |
| Binomial | $P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$ | $n=x_{\max}$, $p=\bar{x}/n$ |

## 3. Validação contra bibliotecas consolidadas

O arquivo `tests/test_minhastats.py` compara cada função própria com a
implementação equivalente de NumPy, SciPy e da biblioteca padrão `statistics`,
usando `math.isclose` com tolerância absoluta/relativa de 1e-6 a 1e-9 (dependendo
da operação). Resultado da última execução:

```
26/26 testes passaram
```

| Função própria | Referência usada na validação | Resultado |
|---|---|---|
| `media` | `numpy.mean` | ✅ |
| `mediana` | `numpy.median` | ✅ |
| `moda` | `statistics.multimode` | ✅ |
| `amplitude` | `max - min` | ✅ |
| `variancia` (amostral/populacional) | `numpy.var(ddof=1/0)` | ✅ |
| `desvio_padrao` (amostral/populacional) | `numpy.std(ddof=1/0)` | ✅ |
| `percentil` / `quartis` / `iqr` | `numpy.percentile` | ✅ |
| `coeficiente_variacao` | cálculo manual com `numpy` | ✅ |
| `assimetria` | `scipy.stats.skew(bias=True)` | ✅ |
| `covariancia` | `numpy.cov(ddof=1)` | ✅ |
| `correlacao_pearson` | `scipy.stats.pearsonr` | ✅ |
| `regressao_linear_simples` (b0, b1, R²) | `scipy.stats.linregress` | ✅ |
| `detectar_outliers_iqr` | regra do IQR calculada manualmente com `numpy.percentile` | ✅ |
| `normal_pdf` | `scipy.stats.norm.pdf` | ✅ |
| `binomial_pmf` | `scipy.stats.binom.pmf` | ✅ |
| `poisson_pmf` | `scipy.stats.poisson.pmf` | ✅ |
| `uniforme_pdf` | `scipy.stats.uniform.pdf` | ✅ |
| `exponencial_pdf` | `scipy.stats.expon.pdf` | ✅ |

## 4. Prints e explicação de cada módulo

_(seção a finalizar com capturas de tela reais após a definição do dataset —
ver pasta `docs/screenshots/`)_

### Módulo 2 — Estatística Descritiva
_(descrição + prints)_

### Módulo 3 — Probabilidade e Simulação
_(descrição + prints)_

### Módulo 4 — Distribuições Teóricas
_(descrição + prints)_

### Módulo 5 — Correlação e Regressão
_(descrição + prints)_

## 5. Descobertas (Módulo 6)

_(as três descobertas estatísticas mais interessantes serão documentadas aqui,
sustentadas pelos números e gráficos gerados pela própria aplicação, assim que o
dataset definitivo for carregado.)_

1. _Descoberta 1 — a preencher_
2. _Descoberta 2 — a preencher_
3. _Descoberta 3 — a preencher_
