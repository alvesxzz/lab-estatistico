# Fonte do dataset

- **Nome do dataset:** Video Game Sales
- **Fonte original (URL):** https://www.kaggle.com/datasets/gregorut/videogamesales
- **Descrição:** vendas globais de jogos eletrônicos (em milhões de unidades), agregando dados
  de mais de 16 mil títulos lançados entre 1980 e 2020, com vendas quebradas por região
  (América do Norte, Europa, Japão e demais mercados).
- **Nº de registros:** 16.598
- **Variáveis numéricas usadas na aplicação (6, mínimo exigido: 4):** `Year`, `NA_Sales`,
  `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales`
- **Variáveis categóricas usadas na aplicação (3, mínimo exigido: 2):** `Platform`, `Genre`,
  `Publisher`

## Observação sobre colunas excluídas da análise

O CSV original também traz as colunas `Rank` e `Name`. Ambas têm cardinalidade muito alta
(`Rank` é um índice de 1 a 16.598, `Name` tem ~69% de valores distintos) — ou seja, funcionam
como identificadores, não como variáveis estatísticas. A aplicação detecta isso
automaticamente (ver `core/data_loader.py::_e_coluna_identificadora`, que exclui qualquer
coluna com mais de 50% de valores únicos das listas de variáveis analisáveis) e por isso elas
não aparecem nos seletores dos módulos — mas continuam visíveis na tabela de amostra da página
inicial.

## Valores ausentes

`Year` possui 271 valores ausentes e `Publisher` possui 58. A aplicação remove esses valores
apenas no cálculo da variável em questão (`dropna` pontual), sem descartar a linha inteira do
dataset para as demais análises.
